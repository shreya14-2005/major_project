"""Evaluate EfficientNetB0 once on the held-out test split and compare models."""

import argparse
import csv
import json
import re
import shutil
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from models.efficientnetb0.training.common import (
    BATCH_SIZE,
    CLASS_NAMES,
    IMAGE_SIZE,
    PROJECT_ROOT as DATA_PROJECT_ROOT,
    inspect_split_directories,
    make_dataset,
    read_rgb_pixels,
)


def load_class_mapping(path):
    if not path.is_file():
        raise FileNotFoundError(f"EfficientNetB0 class map is missing: {path}")
    mapping = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "class_to_index": {
            name: index for index, name in enumerate(CLASS_NAMES)
        },
        "index_to_class": {
            str(index): name for index, name in enumerate(CLASS_NAMES)
        },
    }
    if mapping != expected:
        raise ValueError("EfficientNetB0 class mapping does not match the dataset order.")
    return expected["class_to_index"]


def benchmark_model(tf, model, image_batches):
    """Measure warmed-up single-image model latency without disk preprocessing."""
    infer = tf.function(
        lambda images: model(images, training=False),
        reduce_retracing=True,
    )
    warmup = tf.convert_to_tensor(image_batches[0][np.newaxis, ...])
    for _ in range(3):
        infer(warmup).numpy()

    timings = []
    for image in image_batches:
        batch = tf.convert_to_tensor(image[np.newaxis, ...])
        started = time.perf_counter()
        infer(batch).numpy()
        timings.append(time.perf_counter() - started)
    return float(np.mean(timings)), len(timings)


def save_confusion_matrix(path, matrix):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    figure, axis = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=CLASS_NAMES,
    ).plot(ax=axis, cmap="Blues", colorbar=False, xticks_rotation=25)
    axis.set_title("EfficientNetB0 maize disease test-set confusion matrix")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def write_misclassifications(records, true_labels, predicted_labels, probabilities):
    evaluation_directory = PROJECT_ROOT / "models" / "efficientnetb0" / "evaluation"
    output_directory = evaluation_directory / "misclassified_images"
    output_directory.mkdir(parents=True)
    rows = []
    for record, actual_index, predicted_index, probability_row in zip(
        records, true_labels, predicted_labels, probabilities
    ):
        if actual_index == predicted_index:
            continue
        actual = CLASS_NAMES[int(actual_index)]
        predicted = CLASS_NAMES[int(predicted_index)]
        confidence = float(probability_row[predicted_index])
        target_directory = output_directory / actual
        target_directory.mkdir(exist_ok=True)
        output_name = (
            f"{len(rows) + 1:04d}__predicted_{predicted}__"
            f"confidence_{confidence:.4f}__{record['path'].name}"
        )
        shutil.copy2(record["path"], target_directory / output_name)
        rows.append(
            {
                "Image filename": record["path"].name,
                "Actual class": actual,
                "Predicted class": predicted,
                "Prediction confidence": confidence,
                "Misclassified image copy": str(
                    (Path(actual) / output_name).as_posix()
                ),
                "Source test image": str(record["path"].relative_to(PROJECT_ROOT)),
            }
        )

    csv_path = evaluation_directory / "test_predictions.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        columns = [
            "Image filename",
            "Actual class",
            "Predicted class",
            "Prediction confidence",
            "Misclassified image copy",
            "Source test image",
        ]
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        for record, actual_index, predicted_index, probability_row in zip(
            records, true_labels, predicted_labels, probabilities
        ):
            actual = CLASS_NAMES[int(actual_index)]
            predicted = CLASS_NAMES[int(predicted_index)]
            writer.writerow(
                {
                    "Image filename": record["path"].name,
                    "Actual class": actual,
                    "Predicted class": predicted,
                    "Prediction confidence": f"{probability_row[predicted_index]:.6f}",
                    "Misclassified image copy": next(
                        (
                            row["Misclassified image copy"]
                            for row in rows
                            if row["Source test image"]
                            == str(record["path"].relative_to(PROJECT_ROOT))
                        ),
                        "",
                    ),
                    "Source test image": str(record["path"].relative_to(PROJECT_ROOT)),
                }
            )
    return rows


def parse_mobilenet_report(report_path):
    if not report_path.is_file():
        raise FileNotFoundError(f"Existing MobileNetV2 report is missing: {report_path}")
    text = report_path.read_text(encoding="utf-8")
    scalar_values = {}
    for key, label in (
        ("accuracy", "Accuracy"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("f1_score", "F1-score"),
    ):
        match = re.search(
            rf"^{re.escape(label)}:\s*([0-9]*\.?[0-9]+)\s*$",
            text,
            flags=re.MULTILINE,
        )
        if not match:
            raise ValueError(f"MobileNetV2 report has no valid {label} metric.")
        scalar_values[key] = float(match.group(1))
    count_match = re.search(r"^Test image count:\s*(\d+)\s*$", text, re.MULTILINE)
    if not count_match or int(count_match.group(1)) != 628:
        raise ValueError("MobileNetV2 report does not confirm the same 628 test images.")
    expected = {
        "accuracy": 0.9522,
        "precision": 0.9383,
        "recall": 0.9429,
        "f1_score": 0.9405,
    }
    if scalar_values != expected:
        raise ValueError(
            f"MobileNetV2 report differs from the supplied comparison results: "
            f"{scalar_values}"
        )

    class_metrics = {}
    for class_name in CLASS_NAMES:
        row = re.search(
            rf"^\s*{re.escape(class_name)}\s+"
            r"([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\s+"
            r"([0-9]*\.?[0-9]+)\s+(\d+)\s*$",
            text,
            flags=re.MULTILINE,
        )
        if not row:
            raise ValueError(f"MobileNetV2 report has no row for {class_name}.")
        class_metrics[class_name] = {
            "precision": float(row.group(1)),
            "recall": float(row.group(2)),
            "f1_score": float(row.group(3)),
            "support": int(row.group(4)),
        }
    confusion_rows = []
    for class_name in CLASS_NAMES:
        row = re.search(
            rf"^{re.escape(class_name)}:\s*\[([0-9,\s]+)\]\s*$",
            text,
            flags=re.MULTILINE,
        )
        if not row:
            raise ValueError(f"MobileNetV2 report has no confusion-matrix row for {class_name}.")
        values = [int(value.strip()) for value in row.group(1).split(",")]
        if len(values) != len(CLASS_NAMES):
            raise ValueError(
                f"MobileNetV2 confusion-matrix row for {class_name} has "
                f"{len(values)} values."
            )
        confusion_rows.append(values)
    return text, scalar_values, class_metrics, confusion_rows


def make_comparison_report(
    efficient_metrics,
    efficient_report,
    efficient_metadata,
    efficient_model_path,
    efficient_parameters,
    efficient_latency,
    mobilenet_text,
    mobilenet_metrics,
    mobilenet_class_metrics,
    mobilenet_confusion_matrix,
    mobilenet_model_path,
    mobilenet_parameters,
    mobilenet_latency,
):
    comparison_path = (
        PROJECT_ROOT / "models" / "comparison" / "model_comparison_report.txt"
    )
    delta_lines = [
        (
            f"{name.title()} difference (EfficientNetB0 - MobileNetV2): "
            f"{(efficient_metrics[key] - mobilenet_metrics[key]) * 100:+.2f} percentage points"
        )
        for name, key in (
            ("accuracy", "accuracy"),
            ("macro precision", "precision"),
            ("macro recall", "recall"),
            ("macro F1-score", "f1_score"),
        )
    ]
    class_lines = []
    for class_name in CLASS_NAMES:
        eff = efficient_metrics["class_metrics"][class_name]
        mob = mobilenet_class_metrics[class_name]
        class_lines.append(
            f"{class_name}: EfficientNetB0 P/R/F1 "
            f"{eff['precision']:.4f}/{eff['recall']:.4f}/{eff['f1_score']:.4f} "
            f"(n={eff['support']}); MobileNetV2 "
            f"{mob['precision']:.4f}/{mob['recall']:.4f}/{mob['f1_score']:.4f} "
            f"(n={mob['support']})"
        )
    confusion_lines = [
        "Confusion matrices (rows: actual; columns: predicted)",
        "------------------------------------------------------",
        "EfficientNetB0",
        *[
            f"{class_name}: {row}"
            for class_name, row in zip(
                CLASS_NAMES,
                efficient_metrics["confusion_matrix"],
            )
        ],
        "MobileNetV2",
        *[
            f"{class_name}: {row}"
            for class_name, row in zip(CLASS_NAMES, mobilenet_confusion_matrix)
        ],
    ]
    lines = [
        "MAIZE DISEASE MODEL COMPARISON",
        "==============================",
        "",
        "Both models use the same existing 628-image held-out test split and class order.",
        "EfficientNetB0 checkpoint selection used validation loss only; the test set was",
        "not used for training, checkpoint selection, or hyperparameter tuning.",
        "",
        "Overall test metrics",
        "--------------------",
        (
            f"Accuracy: EfficientNetB0 {efficient_metrics['accuracy']:.4f}; "
            f"MobileNetV2 {mobilenet_metrics['accuracy']:.4f}"
        ),
        (
            f"Macro precision: EfficientNetB0 {efficient_metrics['precision']:.4f}; "
            f"MobileNetV2 {mobilenet_metrics['precision']:.4f}"
        ),
        (
            f"Macro recall: EfficientNetB0 {efficient_metrics['recall']:.4f}; "
            f"MobileNetV2 {mobilenet_metrics['recall']:.4f}"
        ),
        (
            f"Macro F1-score: EfficientNetB0 {efficient_metrics['f1_score']:.4f}; "
            f"MobileNetV2 {mobilenet_metrics['f1_score']:.4f}"
        ),
        *delta_lines,
        "",
        "Per-class test metrics (precision / recall / F1)",
        "-------------------------------------------------",
        *class_lines,
        "",
        *confusion_lines,
        "",
        "Training, size, inference, and compute",
        "--------------------------------------",
        (
            "Training time: EfficientNetB0 "
            f"{efficient_metadata['training_seconds_including_pretrained_initialization_and_saving']:.2f} s "
            "(includes pretrained initialization and model save); MobileNetV2 "
            "not recorded in its existing report."
        ),
        (
            f"Saved model size: EfficientNetB0 {efficient_model_path.stat().st_size / 1_000_000:.2f} MB "
            f"({efficient_model_path.stat().st_size} bytes, {efficient_parameters:,} parameters); "
            f"MobileNetV2 {mobilenet_model_path.stat().st_size / 1_000_000:.2f} MB "
            f"({mobilenet_model_path.stat().st_size} bytes, {mobilenet_parameters:,} parameters)."
        ),
        (
            "Average batch-1 inference latency: EfficientNetB0 "
            f"{efficient_latency * 1000:.3f} ms/image; MobileNetV2 "
            f"{mobilenet_latency * 1000:.3f} ms/image. Timed on the same 32 "
            "balanced validation images (8 per class), warmed up, excluding "
            "image decode/resize."
        ),
        (
            f"Input: both models use {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB images. "
            "Both were benchmarked on this TensorFlow CPU runtime; no GPU was available."
        ),
        (
            "Parameter count, serialized model size, and measured latency are partial "
            "computational-cost indicators. Peak memory and FLOPs were not measured."
        ),
        "",
        "Gray_Leaf_Spot focus",
        "--------------------",
        (
            f"EfficientNetB0 Gray_Leaf_Spot recall: "
            f"{efficient_metrics['class_metrics']['Gray_Leaf_Spot']['recall']:.4f}; "
            f"MobileNetV2 recall: "
            f"{mobilenet_class_metrics['Gray_Leaf_Spot']['recall']:.4f}."
        ),
        "",
        "Model details",
        "-------------",
        f"EfficientNetB0 selected phase: {efficient_metadata['selected_phase']}",
        f"EfficientNetB0 best validation loss: "
        f"{efficient_metadata['selected_validation_best_loss']:.6f}",
        "MobileNetV2 metrics and per-class report were read from the unchanged existing report.",
        "",
        "MobileNetV2 source classification report",
        "-----------------------------------------",
        mobilenet_text.split("Classification report\n---------------------", maxsplit=1)[-1]
        .split("Confusion matrix", maxsplit=1)[0]
        .strip(),
        "",
        "EfficientNetB0 classification report",
        "-------------------------------------",
        efficient_report,
        "",
        "No EfficientNetB0 result was estimated or substituted for a measured value.",
    ]
    comparison_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate():
    import tensorflow as tf
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    model_directory = PROJECT_ROOT / "models" / "efficientnetb0"
    model_path = model_directory / "model.keras"
    mapping_path = model_directory / "class_mapping.json"
    metadata_path = model_directory / "training" / "artifacts" / "metadata.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"Trained EfficientNetB0 model is missing: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"EfficientNetB0 training metadata is missing: {metadata_path}")
    class_to_index = load_class_mapping(mapping_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    dataset_directory = DATA_PROJECT_ROOT / "datasets" / "maize_disease_processed"
    split_records, errors = inspect_split_directories(dataset_directory)
    if errors:
        raise ValueError(
            "The existing dataset split failed validation:\n  - "
            + "\n  - ".join(errors)
        )
    test_records = split_records["test"]
    if len(test_records) != 628:
        raise ValueError(f"Expected 628 test images, found {len(test_records)}.")

    model = tf.keras.models.load_model(model_path, compile=False)
    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(f"Unexpected EfficientNetB0 model input: {model.input_shape}")
    if tuple(model.output_shape[1:]) != (len(CLASS_NAMES),):
        raise ValueError(f"Unexpected EfficientNetB0 model output: {model.output_shape}")

    test_dataset = make_dataset(tf, test_records, class_to_index)
    true_labels = np.asarray(
        [class_to_index[record["class_name"]] for record in test_records],
        dtype=np.int64,
    )
    probabilities = np.asarray(model.predict(test_dataset, verbose=1))
    if probabilities.shape != (len(test_records), len(CLASS_NAMES)):
        raise RuntimeError(f"Unexpected prediction shape: {probabilities.shape}")
    if not np.all(np.isfinite(probabilities)):
        raise RuntimeError("EfficientNetB0 returned non-finite prediction values.")
    predicted_labels = np.argmax(probabilities, axis=1)
    accuracy = float(accuracy_score(true_labels, predicted_labels))
    precision = float(
        precision_score(true_labels, predicted_labels, average="macro", zero_division=0)
    )
    recall = float(
        recall_score(true_labels, predicted_labels, average="macro", zero_division=0)
    )
    f1 = float(f1_score(true_labels, predicted_labels, average="macro", zero_division=0))
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
    class_report = classification_report(
        true_labels,
        predicted_labels,
        labels=list(range(len(CLASS_NAMES))),
        target_names=CLASS_NAMES,
        zero_division=0,
        digits=4,
    )

    evaluation_directory = model_directory / "evaluation"
    report_path = evaluation_directory / "test_evaluation_report.txt"
    classification_path = evaluation_directory / "classification_report.txt"
    confusion_path = evaluation_directory / "confusion_matrix.png"
    outputs = [
        report_path,
        classification_path,
        confusion_path,
        evaluation_directory / "test_predictions.csv",
        evaluation_directory / "test_metrics.json",
        evaluation_directory / "misclassified_images",
        PROJECT_ROOT / "models" / "comparison" / "model_comparison_report.txt",
    ]
    existing = [path for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(
            "EfficientNetB0 evaluation output already exists; no files were overwritten:\n  - "
            + "\n  - ".join(str(path) for path in existing)
        )

    benchmark_records = []
    for class_name in CLASS_NAMES:
        class_records = [
            record
            for record in split_records["val"]
            if record["class_name"] == class_name
        ]
        benchmark_records.extend(class_records[:8])
    benchmark_images = np.stack(
        [read_rgb_pixels(record["path"]) for record in benchmark_records]
    ).astype(np.float32)
    efficient_latency, benchmark_count = benchmark_model(
        tf, model, benchmark_images
    )
    mobilenet_path = PROJECT_ROOT / "models" / "mobilenetv2" / "model.keras"
    if not mobilenet_path.is_file():
        raise FileNotFoundError(f"Existing MobileNetV2 model is missing: {mobilenet_path}")
    mobilenet_model = tf.keras.models.load_model(mobilenet_path, compile=False)
    if tuple(mobilenet_model.input_shape[1:]) != (224, 224, 3):
        raise ValueError("Existing MobileNetV2 model does not accept 224x224 RGB.")
    mobilenet_latency, _ = benchmark_model(tf, mobilenet_model, benchmark_images)
    (
        mobilenet_text,
        mobilenet_metrics,
        mobilenet_classes,
        mobilenet_confusion_matrix,
    ) = parse_mobilenet_report(
        PROJECT_ROOT
        / "models"
        / "mobilenetv2"
        / "evaluation"
        / "test_evaluation_report.txt"
    )

    matrix_rows = [
        f"{name}: {row.tolist()}" for name, row in zip(CLASS_NAMES, matrix)
    ]
    training_seconds = metadata[
        "training_seconds_including_pretrained_initialization_and_saving"
    ]
    lines = [
        "EFFICIENTNETB0 MAIZE DISEASE TEST EVALUATION",
        "============================================",
        "",
        f"Evaluation date: {time.strftime('%Y-%m-%d %H:%M:%S %z')}",
        f"Selected model phase: {metadata['selected_phase']}",
        f"Best validation loss: {metadata['selected_validation_best_loss']:.6f}",
        f"Test image count: {len(test_records)}",
        "Test images were not used for training, checkpoint selection, or tuning.",
        "",
        "Test metrics (macro average)",
        "----------------------------",
        f"Accuracy: {accuracy:.4f}",
        f"Precision: {precision:.4f}",
        f"Recall: {recall:.4f}",
        f"F1-score: {f1:.4f}",
        f"Gray_Leaf_Spot recall: {recall_score(true_labels, predicted_labels, labels=[2], average='macro', zero_division=0):.4f}",
        "",
        "Classification report",
        "---------------------",
        report,
        "Confusion matrix (rows are actual; columns are predicted)",
        "----------------------------------------------------------",
        *matrix_rows,
        "",
        "Model and runtime",
        "-----------------",
        f"Architecture: EfficientNetB0, ImageNet pretrained, include_top=False",
        f"Input: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} RGB, raw [0,255]; internal EfficientNet rescaling",
        "Head: GlobalAveragePooling2D -> Dropout(0.3) -> Dense(4, softmax)",
        f"Training time: {training_seconds:.2f} s (including pretrained initialization and save)",
        f"Model size: {model_path.stat().st_size / 1_000_000:.2f} MB",
        f"Parameter count: {model.count_params():,}",
        (
            f"Average batch-1 inference time: {efficient_latency * 1000:.3f} ms/image "
            f"over {benchmark_count} balanced validation images; warm-up excluded, "
            "image decoding/resizing excluded."
        ),
        f"Training device: {', '.join(metadata['training_device'])}",
        "",
        "Misclassification summary",
        "-------------------------",
        f"Misclassified images: {int(np.sum(true_labels != predicted_labels))}",
        (
            "Gray_Leaf_Spot images: "
            f"{int(np.sum((true_labels == 2) & (predicted_labels != 2)))} misclassified "
            f"of {int(np.sum(true_labels == 2))}."
        ),
        "Individual misclassified test images were copied without changing source files.",
    ]

    if any(path.exists() for path in outputs):
        raise FileExistsError("An EfficientNetB0 evaluation output appeared during evaluation.")
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(confusion_path, matrix)
    classification_path.write_text(class_report + "\n", encoding="utf-8")
    misclassified_rows = write_misclassifications(
        test_records, true_labels, predicted_labels, probabilities
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    class_metrics = {}
    for index, class_name in enumerate(CLASS_NAMES):
        actual = true_labels == index
        predicted = predicted_labels == index
        tp = int(np.sum(actual & predicted))
        class_precision = tp / int(np.sum(predicted)) if np.sum(predicted) else 0.0
        class_recall = tp / int(np.sum(actual)) if np.sum(actual) else 0.0
        class_f1 = (
            2 * class_precision * class_recall / (class_precision + class_recall)
            if class_precision + class_recall
            else 0.0
        )
        class_metrics[class_name] = {
            "precision": class_precision,
            "recall": class_recall,
            "f1_score": class_f1,
            "support": int(np.sum(actual)),
        }
    metrics_data = {
        "test_image_count": len(test_records),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "class_metrics": class_metrics,
        "confusion_matrix": matrix.tolist(),
        "misclassified_count": len(misclassified_rows),
        "gray_leaf_spot_misclassified": int(
            np.sum((true_labels == 2) & (predicted_labels != 2))
        ),
    }
    (evaluation_directory / "test_metrics.json").write_text(
        json.dumps(metrics_data, indent=2) + "\n",
        encoding="utf-8",
    )

    comparison_metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "class_metrics": class_metrics,
        "confusion_matrix": matrix.tolist(),
    }
    make_comparison_report(
        comparison_metrics,
        class_report,
        metadata,
        model_path,
        model.count_params(),
        efficient_latency,
        mobilenet_text,
        mobilenet_metrics,
        mobilenet_classes,
        mobilenet_confusion_matrix,
        mobilenet_path,
        mobilenet_model.count_params(),
        mobilenet_latency,
    )
    print(f"Test accuracy: {accuracy:.4f}")
    print(f"Macro precision/recall/F1: {precision:.4f}/{recall:.4f}/{f1:.4f}")
    print(f"EfficientNetB0 average inference: {efficient_latency * 1000:.3f} ms/image")
    print(f"Benchmark images: {benchmark_count} validation images")
    print(f"Test evaluation: {report_path}")
    print(f"Classification report: {classification_path}")
    print(
        "Comparison report: "
        f"{PROJECT_ROOT / 'models' / 'comparison' / 'model_comparison_report.txt'}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the trained EfficientNetB0 on the existing test split."
    )
    parser.parse_args()
    try:
        evaluate()
    except (FileExistsError, FileNotFoundError, OSError, RuntimeError, ValueError) as error:
        print(f"EfficientNetB0 evaluation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
