"""Train an independent EfficientNetB0 maize disease classifier."""

import argparse
import csv
import json
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.efficientnetb0.training.common import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    RANDOM_SEED,
    class_mapping,
    inspect_split_directories,
    make_dataset,
)


HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 8
FINE_TUNE_LAST_LAYERS = 30
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5


def build_model(tf, weights):
    """Build EfficientNetB0 with raw-pixel input and a four-class head."""
    layers = tf.keras.layers
    inputs = layers.Input(
        shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
        name="image",
    )
    x = layers.RandomFlip("horizontal", name="training_horizontal_flip")(inputs)
    x = layers.RandomRotation(0.05, name="training_rotation")(x)
    x = layers.RandomZoom(0.1, name="training_zoom")(x)
    base_model = tf.keras.applications.EfficientNetB0(
        input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
        include_top=False,
        weights=weights,
        name="efficientnetb0",
    )
    base_model.trainable = False
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.3, name="head_dropout")(x)
    outputs = layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="class_probabilities",
    )(x)
    return tf.keras.Model(inputs, outputs, name="maize_efficientnetb0"), base_model


def compile_model(tf, model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def set_fine_tuning_layers(tf, base_model):
    """Unfreeze upper backbone layers while keeping all BatchNorm frozen."""
    base_model.trainable = True
    first_trainable_index = max(0, len(base_model.layers) - FINE_TUNE_LAST_LAYERS)
    for index, layer in enumerate(base_model.layers):
        layer.trainable = (
            index >= first_trainable_index
            and not isinstance(layer, tf.keras.layers.BatchNormalization)
        )


def make_callbacks(tf, checkpoint_path, patience):
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_loss",
            mode="min",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]


def make_class_weights(train_records):
    counts = Counter(record["class_name"] for record in train_records)
    total = len(train_records)
    return {
        index: total / (len(CLASS_NAMES) * counts[class_name])
        for index, class_name in enumerate(CLASS_NAMES)
    }


def save_training_graphs(output_directory, histories):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, title, ylabel, filename in (
        ("accuracy", "Accuracy", "Accuracy", "efficientnetb0_training_accuracy.png"),
        ("loss", "Loss", "Loss", "efficientnetb0_training_loss.png"),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        epoch_offset = 0
        for phase_name, history in histories:
            training_values = history.history.get(metric, [])
            validation_values = history.history.get(f"val_{metric}", [])
            epochs = range(epoch_offset + 1, epoch_offset + len(training_values) + 1)
            axis.plot(epochs, training_values, label=f"{phase_name} training")
            axis.plot(
                epochs,
                validation_values,
                linestyle="--",
                label=f"{phase_name} validation",
            )
            epoch_offset += len(training_values)
        axis.set_title(f"EfficientNetB0 training and validation {title.lower()}")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output_directory / filename, dpi=150)
        plt.close(figure)


def write_history(output_path, histories):
    keys = ("loss", "accuracy", "val_loss", "val_accuracy")
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["phase", "epoch", *keys])
        for phase_name, history in histories:
            epoch_count = len(history.history["loss"])
            for index in range(epoch_count):
                writer.writerow(
                    [
                        phase_name,
                        index + 1,
                        *[
                            history.history.get(key, [None] * epoch_count)[index]
                            for key in keys
                        ],
                    ]
                )


def expected_artifacts():
    model_directory = PROJECT_ROOT / "models" / "efficientnetb0"
    training_artifacts = model_directory / "training" / "artifacts"
    evaluation_directory = model_directory / "evaluation"
    return [
        model_directory / "model.keras",
        model_directory / "class_mapping.json",
        training_artifacts / "training_accuracy.png",
        training_artifacts / "training_loss.png",
        training_artifacts,
        evaluation_directory / "test_evaluation_report.txt",
        evaluation_directory / "classification_report.txt",
        evaluation_directory / "confusion_matrix.png",
        evaluation_directory / "misclassified_images",
        PROJECT_ROOT / "models" / "comparison" / "model_comparison_report.txt",
    ]


def train():
    dataset_directory = PROJECT_ROOT / "datasets" / "maize_disease_processed"
    if not dataset_directory.is_dir():
        raise FileNotFoundError(f"Prepared dataset is missing: {dataset_directory}")

    split_records, errors = inspect_split_directories(dataset_directory)
    for split_name in ("train", "val", "test"):
        records = split_records.get(split_name, [])
        print(f"{split_name}: {len(records)} images")
        counts = Counter(record["class_name"] for record in records)
        for class_name in CLASS_NAMES:
            print(f"  {class_name}: {counts[class_name]}")
    if errors:
        raise ValueError(
            "The existing dataset split failed validation:\n  - "
            + "\n  - ".join(errors)
        )

    training_output = (
        PROJECT_ROOT / "models" / "efficientnetb0" / "training" / "artifacts"
    )
    existing = [
        path
        for path in expected_artifacts()
        if path.exists()
        and not (
            path == training_output
            and path.is_dir()
            and not any(path.iterdir())
        )
    ]
    if existing:
        raise FileExistsError(
            "EfficientNetB0 output already exists; no files were overwritten. "
            "Move the existing EfficientNetB0 outputs before retraining:\n  - "
            + "\n  - ".join(str(path) for path in existing)
        )

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError("TensorFlow is required to train EfficientNetB0.") from error

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError) as error:
        print(f"Warning: TensorFlow deterministic ops were not enabled: {error}")

    class_to_index = {
        class_name: index for index, class_name in enumerate(CLASS_NAMES)
    }
    train_dataset = make_dataset(
        tf, split_records["train"], class_to_index, training=True
    )
    validation_dataset = make_dataset(
        tf, split_records["val"], class_to_index
    )
    class_weights = make_class_weights(split_records["train"])
    histories = []
    model_directory = PROJECT_ROOT / "models" / "efficientnetb0"
    output_directory = model_directory / "training" / "artifacts"
    training_output = output_directory
    model_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)

    print(f"TensorFlow: {tf.__version__}")
    print(f"Image input: {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]} RGB, raw [0, 255]")
    print(f"Batch size: {BATCH_SIZE}; seed: {RANDOM_SEED}")
    print(f"Class weights (training only): {class_weights}")
    print("Validation and test images are not augmented.")

    training_started = datetime.now().astimezone()
    elapsed_start = time.perf_counter()
    with tempfile.TemporaryDirectory(
        prefix=".efficientnetb0_training_",
        dir=model_directory,
    ) as temporary_directory:
        checkpoint_directory = Path(temporary_directory)
        initial_checkpoint = checkpoint_directory / "frozen_base_best.keras"
        model, _ = build_model(tf, weights="imagenet")
        compile_model(tf, model, HEAD_LEARNING_RATE)
        model.summary()
        initial_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=HEAD_EPOCHS,
            callbacks=make_callbacks(tf, initial_checkpoint, patience=4),
            class_weight=class_weights,
            shuffle=False,
            verbose=1,
        )
        histories.append(("frozen base", initial_history))
        if not initial_checkpoint.is_file():
            raise RuntimeError("Feature extraction did not save its best checkpoint.")

        selected_model = tf.keras.models.load_model(
            initial_checkpoint,
            compile=False,
        )
        initial_best_val_loss = min(initial_history.history["val_loss"])
        selected_best_val_loss = initial_best_val_loss
        selected_phase = "frozen EfficientNetB0 base"

        fine_tune_model, fine_tune_base = build_model(tf, weights=None)
        fine_tune_model.set_weights(selected_model.get_weights())
        set_fine_tuning_layers(tf, fine_tune_base)
        compile_model(tf, fine_tune_model, FINE_TUNE_LEARNING_RATE)
        fine_tune_checkpoint = checkpoint_directory / "fine_tuned_best.keras"
        fine_tune_history = fine_tune_model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=make_callbacks(tf, fine_tune_checkpoint, patience=3),
            class_weight=class_weights,
            shuffle=False,
            verbose=1,
        )
        histories.append(("fine-tuning", fine_tune_history))

        if not fine_tune_checkpoint.is_file():
            raise RuntimeError("Fine-tuning did not save a validation checkpoint.")
        candidate_val_loss = min(fine_tune_history.history["val_loss"])
        if candidate_val_loss < initial_best_val_loss:
            selected_model = tf.keras.models.load_model(
                fine_tune_checkpoint,
                compile=False,
            )
            selected_best_val_loss = candidate_val_loss
            selected_phase = "fine-tuned EfficientNetB0 upper layers"
        else:
            print(
                "Fine-tuning did not improve validation loss; "
                "the frozen-base checkpoint was selected."
            )

        save_training_graphs(output_directory, histories)
        training_output.mkdir(parents=True, exist_ok=True)
        write_history(training_output / "history.csv", histories)
        model_path = model_directory / "model.keras"
        selected_model.save(model_path)
        classes_path = model_directory / "class_mapping.json"
        classes_path.write_text(
            json.dumps(class_mapping(), indent=2) + "\n",
            encoding="utf-8",
        )

    training_seconds = time.perf_counter() - elapsed_start
    training_finished = datetime.now().astimezone()
    metadata = {
        "architecture": "EfficientNetB0",
        "tensorflow_version": tf.__version__,
        "python_version": sys.version.split()[0],
        "started_at": training_started.isoformat(timespec="seconds"),
        "finished_at": training_finished.isoformat(timespec="seconds"),
        "training_seconds_including_pretrained_initialization_and_saving": training_seconds,
        "selected_phase": selected_phase,
        "selected_validation_best_loss": selected_best_val_loss,
        "frozen_base_best_validation_loss": initial_best_val_loss,
        "trainable_parameter_count": int(
            sum(tf.keras.backend.count_params(weight) for weight in selected_model.trainable_weights)
        ),
        "parameter_count": int(selected_model.count_params()),
        "image_size": list(IMAGE_SIZE),
        "batch_size": BATCH_SIZE,
        "random_seed": RANDOM_SEED,
        "class_weights": {
            CLASS_NAMES[index]: weight for index, weight in class_weights.items()
        },
        "training_counts": {
            name: sum(record["class_name"] == name for record in split_records["train"])
            for name in CLASS_NAMES
        },
        "head_max_epochs": HEAD_EPOCHS,
        "fine_tuning_max_epochs": FINE_TUNE_EPOCHS,
        "fine_tuned_backbone_layers": FINE_TUNE_LAST_LAYERS,
        "batch_normalization_layers_frozen": True,
        "preprocessing": (
            "Input is raw RGB [0,255]. EfficientNetB0 has an internal Rescaling "
            "layer; no external normalization or preprocess_input is applied."
        ),
        "augmentation": (
            "Random horizontal flip, rotation 0.05, and zoom 0.1 in the model; "
            "active during training only."
        ),
        "training_device": [
            device.name for device in tf.config.list_physical_devices()
        ],
        "test_data_used_for_model_selection": False,
    }
    (training_output / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Selected model: {selected_phase}")
    print(f"Best validation loss: {selected_best_val_loss:.6f}")
    print(f"Training time: {training_seconds:.2f} seconds")
    print(f"Saved model: {model_path}")
    print(f"Saved class map: {classes_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Train the separate EfficientNetB0 maize disease model."
    )
    parser.parse_args()
    try:
        train()
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"EfficientNetB0 training failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
