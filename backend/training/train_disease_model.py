"""Train and evaluate a MobileNetV2 maize disease classifier."""

import argparse
import csv
import json
import re
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


CLASS_NAMES = ("Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy")
EXPECTED_SPLIT_COUNTS = {
    "train": {"Blight": 802, "Common_Rust": 914, "Gray_Leaf_Spot": 402, "Healthy": 814},
    "val": {"Blight": 172, "Common_Rust": 196, "Gray_Leaf_Spot": 86, "Healthy": 174},
    "test": {"Blight": 172, "Common_Rust": 196, "Gray_Leaf_Spot": 86, "Healthy": 174},
}
SUPPORTED_IMAGE_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42
HEAD_EPOCHS = 20
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LAST_LAYERS = 30
HEAD_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 1e-5


def inspect_split_directories(dataset_directory):
    """Validate all split folders, image files, and their expected class counts."""
    split_records = {}
    errors = []

    for split_name, expected_counts in EXPECTED_SPLIT_COUNTS.items():
        split_directory = dataset_directory / split_name
        if not split_directory.is_dir():
            errors.append(f"Missing split folder: {split_directory}")
            continue

        observed_classes = {
            path.name for path in split_directory.iterdir() if path.is_dir()
        }
        expected_classes = set(CLASS_NAMES)
        if observed_classes != expected_classes:
            errors.append(
                f"{split_name}: expected class folders {sorted(expected_classes)}, "
                f"found {sorted(observed_classes)}"
            )
            continue

        unexpected_files = [
            path
            for path in split_directory.iterdir()
            if path.is_file()
        ]
        if unexpected_files:
            errors.extend(
                f"Unexpected file outside a class folder: {path}"
                for path in unexpected_files
            )

        records = []
        counts = Counter()
        for class_name in CLASS_NAMES:
            class_directory = split_directory / class_name
            class_files = sorted(
                (path for path in class_directory.rglob("*") if path.is_file()),
                key=lambda path: str(path).casefold(),
            )
            for image_path in class_files:
                if image_path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
                    errors.append(f"Unsupported image file format: {image_path}")
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        image.load()
                except (OSError, ValueError, SyntaxError) as error:
                    errors.append(f"Unreadable image {image_path}: {error}")
                    continue

                records.append(
                    {
                        "path": image_path,
                        "class_name": class_name,
                    }
                )
                counts[class_name] += 1

        if dict(counts) != expected_counts:
            errors.append(
                f"{split_name}: expected counts {expected_counts}, "
                f"found {dict(counts)}"
            )
        split_records[split_name] = records

    return split_records, errors


def print_dataset_counts(split_records):
    """Print per-class counts and split totals before training."""
    print("Validated dataset image counts:")
    for split_name in ("train", "val", "test"):
        records = split_records.get(split_name, [])
        counts = Counter(record["class_name"] for record in records)
        print(f"  {split_name}: {len(records)} images")
        for class_name in CLASS_NAMES:
            print(f"    {class_name}: {counts[class_name]}")


def load_rgb_image(path_value):
    """Load one image with Pillow and return resized RGB pixel values."""
    image_path = path_value.numpy().decode("utf-8")
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def make_dataset(tf, records, class_to_index, training):
    """Create a streaming TensorFlow dataset from checked image paths."""
    paths = [str(record["path"]) for record in records]
    labels = [class_to_index[record["class_name"]] for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    def read_and_resize(path, label):
        image = tf.py_function(
            func=load_rgb_image,
            inp=[path],
            Tout=tf.uint8,
        )
        image.set_shape((IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
        image = tf.image.convert_image_dtype(image, tf.float32) * 255.0
        label = tf.one_hot(label, depth=len(CLASS_NAMES))
        return image, label

    dataset = dataset.map(
        read_and_resize,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.prefetch(tf.data.AUTOTUNE)


def build_model(tf, weights):
    """Build MobileNetV2 and a small classification head."""
    layers = tf.keras.layers

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3),
        include_top=False,
        weights=weights,
    )
    base_model.trainable = False

    inputs = layers.Input(shape=(IMAGE_SIZE[1], IMAGE_SIZE[0], 3), name="image")
    x = layers.RandomFlip("horizontal", name="training_horizontal_flip")(inputs)
    x = layers.RandomRotation(0.05, name="training_rotation")(x)
    x = layers.RandomZoom(0.1, name="training_zoom")(x)

    # MobileNetV2 expects image values scaled from [0, 255] to [-1, 1].
    x = layers.Rescaling(1.0 / 127.5, offset=-1, name="mobilenetv2_preprocessing")(x)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.25, name="head_dropout")(x)
    outputs = layers.Dense(
        len(CLASS_NAMES),
        activation="softmax",
        name="class_probabilities",
    )(x)
    model = tf.keras.Model(inputs, outputs, name="maize_mobilenetv2")
    return model, base_model


def compile_model(tf, model, learning_rate):
    """Compile the classifier with categorical cross-entropy and Adam."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )


def make_callbacks(tf, checkpoint_path, patience):
    """Create callbacks that monitor validation loss only."""
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
            min_lr=1e-6,
            verbose=1,
        ),
    ]


def make_balanced_class_weights(train_records, class_to_index):
    """Calculate inverse-frequency weights to support the smaller Gray class."""
    counts = Counter(record["class_name"] for record in train_records)
    total = len(train_records)
    class_count = len(CLASS_NAMES)
    return {
        class_to_index[class_name]: total / (class_count * counts[class_name])
        for class_name in CLASS_NAMES
    }


def set_fine_tuning_layers(tf, base_model):
    """Unfreeze only the upper MobileNetV2 layers, leaving BatchNorm frozen."""
    base_model.trainable = True
    first_trainable_index = max(0, len(base_model.layers) - FINE_TUNE_LAST_LAYERS)
    for index, layer in enumerate(base_model.layers):
        layer.trainable = (
            index >= first_trainable_index
            and not isinstance(layer, tf.keras.layers.BatchNormalization)
        )


def history_value(history, key):
    """Return one metric history or an empty list if Keras did not record it."""
    return list(history.history.get(key, []))


def save_training_graphs(output_directory, histories):
    """Save separate accuracy and loss charts for training and validation."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for metric, title, y_label in (
        ("accuracy", "Accuracy", "Accuracy"),
        ("loss", "Loss", "Loss"),
    ):
        for series_name, metric_prefix, line_style in (
            ("training", "", "-"),
            ("validation", "val_", "--"),
        ):
            figure, axis = plt.subplots(figsize=(8, 5))
            epoch_offset = 0
            for phase_name, history in histories:
                values = history_value(history, f"{metric_prefix}{metric}")
                epochs = range(epoch_offset + 1, epoch_offset + len(values) + 1)
                axis.plot(
                    epochs,
                    values,
                    linestyle=line_style,
                    label=phase_name,
                )
                epoch_offset += len(values)

            axis.set_title(f"{series_name.title()} {title.lower()}")
            axis.set_xlabel("Epoch")
            axis.set_ylabel(y_label)
            axis.grid(True, alpha=0.3)
            axis.legend()
            figure.tight_layout()
            figure.savefig(
                output_directory / f"{series_name}_{metric}.png",
                dpi=150,
            )
            plt.close(figure)


def save_confusion_matrix(output_path, matrix):
    """Save a labeled confusion matrix image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    figure, axis = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    )
    display.plot(ax=axis, cmap="Blues", colorbar=False, xticks_rotation=25)
    axis.set_title("Maize disease test-set confusion matrix")
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def write_evaluation_report(
    report_path,
    model,
    test_dataset,
    test_records,
    selected_phase,
    initial_best_val_loss,
    selected_best_val_loss,
    model_summary,
):
    """Evaluate the selected model once on test data and save the metrics."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    class_to_index = {name: index for index, name in enumerate(CLASS_NAMES)}
    true_labels = np.asarray(
        [class_to_index[record["class_name"]] for record in test_records],
        dtype=np.int64,
    )
    probabilities = model.predict(test_dataset, verbose=1)
    predicted_labels = np.argmax(probabilities, axis=1)
    if len(predicted_labels) != len(true_labels):
        raise RuntimeError(
            "The number of model predictions does not match the test image count."
        )

    accuracy = accuracy_score(true_labels, predicted_labels)
    precision = precision_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
    recall = recall_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
    f1 = f1_score(
        true_labels,
        predicted_labels,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    )
    matrix = confusion_matrix(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
    )

    with (report_path.parent / "confusion_matrix.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["actual/predicted", *CLASS_NAMES])
        for class_name, row in zip(CLASS_NAMES, matrix):
            writer.writerow([class_name, *row.tolist()])
    save_confusion_matrix(
        report_path.parent.parent / "outputs" / "confusion_matrix.png",
        matrix,
    )

    gray_index = class_to_index["Gray_Leaf_Spot"]
    gray_mask = true_labels == gray_index
    gray_recall = float(np.mean(predicted_labels[gray_mask] == gray_index))
    lines = [
        "MAIZE DISEASE MODEL TEST EVALUATION",
        "=" * 36,
        "",
        f"Evaluation date: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Selected model phase: {selected_phase}",
        f"Validation best loss (frozen base): {initial_best_val_loss:.6f}",
        f"Selected model best validation loss: {selected_best_val_loss:.6f}",
        f"Test image count: {len(test_records)}",
        "The test set was evaluated after model selection and was not used for training.",
        "",
        "Test metrics (macro average)",
        "----------------------------",
        f"Accuracy: {accuracy:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1-score: {f1:.4f}",
        f"Gray_Leaf_Spot recall: {gray_recall:.4f}",
        "",
        "Classification report",
        "---------------------",
        report,
        "Confusion matrix (rows are actual; columns are predicted)",
        "----------------------------------------------------------",
    ]
    lines.extend(
        f"{class_name}: {row.tolist()}" for class_name, row in zip(CLASS_NAMES, matrix)
    )
    lines.extend(["", "Model architecture", "------------------", *model_summary])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return accuracy, precision, recall, f1


def expected_artifacts(project_root):
    """Return generated output paths used to guard against accidental overwrite."""
    return [
        project_root / "models" / "maize_disease_model.keras",
        project_root / "models" / "maize_disease_classes.json",
        project_root / "evaluation" / "test_evaluation_report.txt",
        project_root / "evaluation" / "confusion_matrix.csv",
        project_root / "outputs" / "training_accuracy.png",
        project_root / "outputs" / "validation_accuracy.png",
        project_root / "outputs" / "training_loss.png",
        project_root / "outputs" / "validation_loss.png",
        project_root / "outputs" / "confusion_matrix.png",
    ]


def train_model(project_root, overwrite=False, skip_fine_tuning=False):
    """Validate data, train, choose by validation loss, and evaluate on test data."""
    dataset_directory = project_root / "datasets" / "maize_disease_processed"
    model_directory = project_root / "models"
    output_directory = project_root / "outputs"
    evaluation_directory = project_root / "evaluation"

    if not dataset_directory.is_dir():
        raise FileNotFoundError(
            f"Prepared dataset directory is missing: {dataset_directory}"
        )

    split_records, errors = inspect_split_directories(dataset_directory)
    print_dataset_counts(split_records)
    if errors:
        raise ValueError(
            "Dataset validation failed; training was not started:\n  - "
            + "\n  - ".join(errors)
        )

    split_report_path = project_root / "evaluation" / "dataset_split_report.txt"
    if split_report_path.is_file():
        split_report_text = split_report_path.read_text(encoding="utf-8")
        duplicate_match = re.search(
            r"Duplicate groups spanning different class labels:\s*(\d+)",
            split_report_text,
        )
        cross_class_duplicates = (
            int(duplicate_match.group(1)) if duplicate_match else 0
        )
        if cross_class_duplicates:
            print(
                f"Warning: the split report records {cross_class_duplicates} "
                "duplicate group(s) with "
                "different class labels. They remain in single splits, but their "
                "labels should be reviewed when interpreting results."
            )

    artifacts = expected_artifacts(project_root)
    existing_artifacts = [path for path in artifacts if path.exists()]
    if existing_artifacts and not overwrite:
        raise FileExistsError(
            "Training output files already exist and will not be overwritten. "
            "Pass --overwrite to replace them:\n  - "
            + "\n  - ".join(str(path) for path in existing_artifacts)
        )

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required for training. Install a TensorFlow build "
            "compatible with this Python environment, then rerun the script."
        ) from error

    tf.keras.utils.set_random_seed(RANDOM_SEED)
    try:
        tf.config.experimental.enable_op_determinism()
    except (AttributeError, RuntimeError):
        print("Warning: TensorFlow deterministic operations could not be enabled.")

    class_to_index = {
        class_name: index for index, class_name in enumerate(CLASS_NAMES)
    }
    train_dataset = make_dataset(
        tf,
        split_records["train"],
        class_to_index,
        training=True,
    )
    validation_dataset = make_dataset(
        tf,
        split_records["val"],
        class_to_index,
        training=False,
    )
    test_dataset = make_dataset(
        tf,
        split_records["test"],
        class_to_index,
        training=False,
    )
    class_weights = make_balanced_class_weights(
        split_records["train"],
        class_to_index,
    )

    model_directory.mkdir(parents=True, exist_ok=True)
    output_directory.mkdir(parents=True, exist_ok=True)
    evaluation_directory.mkdir(parents=True, exist_ok=True)

    histories = []
    model, _ = build_model(tf, weights="imagenet")
    compile_model(tf, model, HEAD_LEARNING_RATE)
    print("\nMobileNetV2 model architecture:")
    model.summary()
    print(f"\nTensorFlow version: {tf.__version__}")
    print(f"Image size: {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Initial epochs (maximum): {HEAD_EPOCHS}")
    print(f"Class weights (train only): {class_weights}")
    print("Validation and test images are not augmented.")

    with tempfile.TemporaryDirectory(
        prefix=".maize_training_",
        dir=model_directory,
    ) as temporary_directory:
        temporary_path = Path(temporary_directory)
        initial_checkpoint = temporary_path / "frozen_base_best.keras"
        initial_history = model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=HEAD_EPOCHS,
            callbacks=make_callbacks(tf, initial_checkpoint, patience=4),
            class_weight=class_weights,
            verbose=1,
        )
        histories.append(("frozen base", initial_history))
        if not initial_checkpoint.is_file():
            raise RuntimeError("Initial training did not create its best checkpoint.")

        selected_model = tf.keras.models.load_model(initial_checkpoint)
        initial_best_val_loss = min(initial_history.history["val_loss"])
        selected_best_val_loss = initial_best_val_loss
        selected_phase = "frozen MobileNetV2 base"

        if not skip_fine_tuning:
            fine_tune_model, fine_tune_base = build_model(
                tf,
                weights=None,
            )
            fine_tune_model.set_weights(selected_model.get_weights())
            set_fine_tuning_layers(tf, fine_tune_base)
            compile_model(tf, fine_tune_model, FINE_TUNE_LEARNING_RATE)
            fine_tune_checkpoint = temporary_path / "fine_tuned_best.keras"
            fine_tune_history = fine_tune_model.fit(
                train_dataset,
                validation_data=validation_dataset,
                epochs=FINE_TUNE_EPOCHS,
                callbacks=make_callbacks(tf, fine_tune_checkpoint, patience=3),
                class_weight=class_weights,
                verbose=1,
            )
            histories.append(("fine-tuning", fine_tune_history))

            if fine_tune_checkpoint.is_file():
                candidate_val_loss = min(fine_tune_history.history["val_loss"])
                if candidate_val_loss < initial_best_val_loss:
                    selected_model = tf.keras.models.load_model(fine_tune_checkpoint)
                    selected_best_val_loss = candidate_val_loss
                    selected_phase = "fine-tuned MobileNetV2 upper layers"
                else:
                    print(
                        "Fine-tuning did not improve best validation loss; "
                        "keeping the frozen-base model."
                    )

        save_training_graphs(output_directory, histories)
        model_path = model_directory / "maize_disease_model.keras"
        classes_path = model_directory / "maize_disease_classes.json"
        report_path = evaluation_directory / "test_evaluation_report.txt"

        # Test data is first passed through the model only after model selection.
        metrics = write_evaluation_report(
            report_path,
            selected_model,
            test_dataset,
            split_records["test"],
            selected_phase,
            initial_best_val_loss,
            selected_best_val_loss,
            [
                f"Architecture: MobileNetV2 (ImageNet weights, include_top=False)",
                f"Input: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB",
                "Head: GlobalAveragePooling2D -> Dropout(0.25) -> Dense(4, softmax)",
                "Training augmentation: horizontal flip, rotation(0.05), zoom(0.1)",
                f"Selected phase: {selected_phase}",
            ],
        )
        selected_model.save(model_path)
        classes_path.write_text(
            json.dumps(
                {
                    "class_to_index": class_to_index,
                    "index_to_class": {
                        str(index): class_name
                        for class_name, index in class_to_index.items()
                    },
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    print("\nFinal test-set metrics:")
    print(f"  Accuracy: {metrics[0]:.4f}")
    print(f"  Precision (macro): {metrics[1]:.4f}")
    print(f"  Recall (macro): {metrics[2]:.4f}")
    print(f"  F1-score (macro): {metrics[3]:.4f}")
    print(f"  Selected model: {model_path}")
    print(f"  Class mapping: {classes_path}")
    print(f"  Evaluation report: {report_path}")
    print(f"  Training graphs: {output_directory}")


def main():
    """Parse options, validate the dataset, and start model training."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train and evaluate a MobileNetV2 maize disease classifier."
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Check all split folders and image readability without training.",
    )
    parser.add_argument(
        "--skip-fine-tuning",
        action="store_true",
        help="Train only the frozen-base classifier and do not fine-tune.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing existing model and evaluation output files.",
    )
    arguments = parser.parse_args()

    dataset_directory = project_root / "datasets" / "maize_disease_processed"
    if arguments.validate_only:
        if not dataset_directory.is_dir():
            print(
                f"Prepared dataset directory is missing: {dataset_directory}",
                file=sys.stderr,
            )
            return 1
        split_records, errors = inspect_split_directories(dataset_directory)
        print_dataset_counts(split_records)
        if errors:
            print("Dataset validation errors:", file=sys.stderr)
            for error in errors:
                print(f"  - {error}", file=sys.stderr)
            return 1
        print("Dataset validation passed. No training was performed.")
        return 0

    try:
        train_model(
            project_root,
            overwrite=arguments.overwrite,
            skip_fine_tuning=arguments.skip_fine_tuning,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
