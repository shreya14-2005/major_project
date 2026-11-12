"""Train the nutrient-only EfficientNetB0 model from prepared CSV manifests."""

import argparse
import csv
import json
import os
import random
from collections import Counter

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import tensorflow as tf

from training.nutrient_config import (
    CLASS_FOLDER_TO_LABEL,
    CLASS_ORDER,
    EVALUATION_ROOT,
    IMAGE_SIZE,
    MODEL_ROOT,
    RANDOM_SEED,
    SPLITS_ROOT,
)
from training.nutrient_preprocessing import preprocess_image_path


def read_manifest(path, allowed_splits):
    if not path.is_file():
        raise FileNotFoundError(f"Required split manifest is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Split manifest is empty: {path}")
    for row in rows:
        if row["split"] not in allowed_splits:
            raise ValueError(f"Unexpected split in {path}: {row['split']}")
        if row["label"] not in CLASS_ORDER:
            raise ValueError(f"Unknown label in {path}: {row['label']}")
        if not os.path.isfile(row["image_path"]):
            raise FileNotFoundError(f"Image referenced by manifest is missing: {row['image_path']}")
    return rows


def make_dataset(rows, tf, batch_size, training, seed):
    paths = np.asarray([row["image_path"] for row in rows], dtype=str)
    labels = np.asarray([CLASS_ORDER.index(row["label"]) for row in rows], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        return preprocess_image_path(path), label

    dataset = dataset.map(load, num_parallel_calls=4, deterministic=True)
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(rows),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    return dataset.batch(batch_size).prefetch(1)


def make_model(tf, seed):
    augmentation = tf.keras.Sequential(
        [
            tf.keras.layers.RandomRotation(0.05, seed=seed),
            tf.keras.layers.RandomFlip("horizontal", seed=seed),
            tf.keras.layers.RandomZoom(0.10, seed=seed),
            tf.keras.layers.RandomBrightness(
                0.12,
                value_range=(0, 255),
                seed=seed,
            ),
        ],
        name="training_only_augmentation",
    )
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(*IMAGE_SIZE, 3),
    )
    if not any(isinstance(layer, tf.keras.layers.Rescaling) for layer in base.layers):
        raise RuntimeError(
            "EfficientNetB0 internal rescaling layer is missing; refusing to train "
            "with an unverified input scale."
        )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="rgb_pixels_0_255")
    values = augmentation(inputs)
    values = base(values, training=False)
    values = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pool")(values)
    values = tf.keras.layers.Dropout(0.30, seed=RANDOM_SEED)(values)
    outputs = tf.keras.layers.Dense(len(CLASS_ORDER), activation="softmax", name="nutrient_class")(values)
    model = tf.keras.Model(inputs, outputs, name="nutrient_efficientnetb0")
    return model, base


class BestValidationCheckpoint(tf.keras.callbacks.Callback):
    """Small callback-compatible checkpoint that tracks across both fit phases."""

    def __init__(self, model_path):
        super().__init__()
        self.model_path = model_path
        self.best_loss = float("inf")

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_loss = logs.get("val_loss")
        if val_loss is None:
            raise RuntimeError("Validation loss was not reported; best model not saved.")
        if float(val_loss) < self.best_loss:
            self.best_loss = float(val_loss)
            self.model.save(self.model_path, overwrite=True)
            print(f"\nSaved best checkpoint (val_loss={self.best_loss:.6f})")


def append_history(path, phase, history, starting_epoch):
    fieldnames = [
        "phase",
        "epoch",
        "loss",
        "accuracy",
        "val_loss",
        "val_accuracy",
        "learning_rate",
    ]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        epochs = len(history.history.get("loss", []))
        for index in range(epochs):
            writer.writerow(
                {
                    "phase": phase,
                    "epoch": starting_epoch + index + 1,
                    "loss": history.history["loss"][index],
                    "accuracy": history.history["accuracy"][index],
                    "val_loss": history.history["val_loss"][index],
                    "val_accuracy": history.history["val_accuracy"][index],
                    "learning_rate": (
                        history.history.get("learning_rate", [None] * epochs)[index]
                    ),
                }
            )
    return len(history.history.get("loss", []))


def plot_history(history_path, output_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with history_path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        return
    epochs = [int(row["epoch"]) for row in rows]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for key, label in (("loss", "Training"), ("val_loss", "Validation")):
        axes[0].plot(epochs, [float(row[key]) for row in rows], label=label)
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    for key, label in (("accuracy", "Training"), ("val_accuracy", "Validation")):
        axes[1].plot(epochs, [float(row[key]) for row in rows], label=label)
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=20)
    parser.add_argument("--fine-tune-epochs", type=int, default=12)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.batch_size <= 0 or args.head_epochs <= 0 or args.fine_tune_epochs <= 0:
        parser.error("batch size and epoch counts must be positive")

    best_model_path = MODEL_ROOT / "best_model.keras"
    if best_model_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Model already exists: {best_model_path}. Use --overwrite to replace it."
        )
    if (EVALUATION_ROOT / "test_metrics.json").exists():
        raise FileExistsError(
            "A held-out test evaluation already exists. Refusing to train over "
            "its model/evaluation evidence."
        )

    from sklearn.utils.class_weight import compute_class_weight

    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)

    train_rows = read_manifest(SPLITS_ROOT / "train.csv", {"train"})
    val_rows = read_manifest(SPLITS_ROOT / "val.csv", {"val"})
    if {row["label"] for row in train_rows} != set(CLASS_ORDER):
        raise ValueError("Training manifest does not contain every expected class.")
    if {row["label"] for row in val_rows} != set(CLASS_ORDER):
        raise ValueError("Validation manifest does not contain every expected class.")
    if {row["related_group_id"] for row in train_rows} & {
        row["related_group_id"] for row in val_rows
    }:
        raise ValueError("A related-image group crosses training and validation.")

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    history_path = EVALUATION_ROOT / "training_history.csv"
    if history_path.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Training history already exists: {history_path}. Use --overwrite."
            )
        history_path.unlink()

    class_indices = np.asarray([CLASS_ORDER.index(row["label"]) for row in train_rows])
    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(CLASS_ORDER)),
        y=class_indices,
    )
    class_weights = {
        index: float(weight) for index, weight in enumerate(class_weights_array)
    }
    train_ds = make_dataset(train_rows, tf, args.batch_size, training=True, seed=args.seed)
    val_ds = make_dataset(val_rows, tf, args.batch_size, training=False, seed=args.seed)
    model, base = make_model(tf, args.seed)
    checkpoint = BestValidationCheckpoint(best_model_path)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        checkpoint,
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.head_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )
    completed_epochs = append_history(history_path, "head", history, 0)

    base.trainable = True
    fine_tune_layers = 30
    for layer in base.layers[:-fine_tune_layers]:
        layer.trainable = False
    for layer in base.layers[-fine_tune_layers:]:
        layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
    trainable_base_layers = sum(layer.trainable for layer in base.layers)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    fine_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.fine_tune_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=2,
    )
    completed_epochs += append_history(
        history_path, "fine_tune", fine_history, completed_epochs
    )

    if not best_model_path.is_file():
        raise RuntimeError("Training finished without producing a best checkpoint.")
    mapping = {
        "class_order": list(CLASS_ORDER),
        "folder_to_label": CLASS_FOLDER_TO_LABEL,
        "label_source_note": (
            "Labels are inherited from the supplied dataset folders and have not "
            "been independently scientifically verified."
        ),
    }
    (MODEL_ROOT / "class_mapping.json").write_text(
        json.dumps(mapping, indent=2), encoding="utf-8"
    )
    config = {
        "architecture": "EfficientNetB0",
        "weights": "ImageNet",
        "image_size": list(IMAGE_SIZE),
        "input_range": [0, 255],
        "preprocessing": (
            "RGB decode, bilinear resize with antialias to 224x224, float32; "
            "EfficientNetB0 internal rescaling"
        ),
        "class_order": list(CLASS_ORDER),
        "seed": args.seed,
        "batch_size": args.batch_size,
        "head_epochs_requested": args.head_epochs,
        "fine_tune_epochs_requested": args.fine_tune_epochs,
        "fine_tune_top_layers": fine_tune_layers,
        "trainable_base_layers_during_fine_tuning": trainable_base_layers,
        "augmentation": [
            "RandomRotation(0.05)",
            "RandomFlip(horizontal)",
            "RandomZoom(0.10)",
            "RandomBrightness(0.12, value_range=[0,255])",
        ],
        "early_stopping": {"monitor": "val_loss", "patience": 5, "restore_best_weights": True},
        "reduce_lr_on_plateau": {
            "monitor": "val_loss",
            "factor": 0.2,
            "patience": 2,
            "min_lr": 1e-6,
        },
        "class_weights": {
            CLASS_ORDER[index]: weight for index, weight in class_weights.items()
        },
        "train_image_count": len(train_rows),
        "validation_image_count": len(val_rows),
        "completed_epochs": completed_epochs,
        "best_validation_loss": checkpoint.best_loss,
        "test_set_used_during_training": False,
    }
    (MODEL_ROOT / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    with (MODEL_ROOT / "model_summary.txt").open("w", encoding="utf-8") as stream:
        model.summary(print_fn=lambda line: stream.write(line + "\n"))
    plot_history(history_path, EVALUATION_ROOT / "training_curves.png")
    report = [
        "# Nutrient EfficientNetB0 Training",
        "",
        "Only the nutrient EfficientNetB0 model was trained. No MobileNetV2 or disease "
        "model was loaded or changed.",
        "",
        f"- Best checkpoint: `{best_model_path}`",
        f"- Validation loss of saved checkpoint: {checkpoint.best_loss:.6f}",
        f"- Training images: {len(train_rows)}",
        f"- Validation images: {len(val_rows)}",
        f"- Epochs completed (head + fine-tuning): {completed_epochs}",
        f"- Trainable EfficientNet base layers during fine-tuning: {trainable_base_layers}",
        f"- Batch size: {args.batch_size}; seed: {args.seed}",
        f"- Input: RGB 224x224 float32 in [0,255], EfficientNetB0 internal rescaling.",
        "- Augmentation is included in the model and is active only during training.",
        "- The test manifest was not opened or used for checkpoint selection.",
        "- Labels reflect the supplied folder names with spelling-only mapping; "
        "biological correctness has not been independently verified.",
        "",
        "Detailed history, training configuration, class mapping, and model summary "
        "are saved beside the checkpoint and under `evaluation/nutrient_model/`.",
    ]
    (EVALUATION_ROOT / "training_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print(f"Training completed: {best_model_path}")
    print(f"Best validation loss: {checkpoint.best_loss:.6f}")
    print(f"History: {history_path}")


if __name__ == "__main__":
    main()
