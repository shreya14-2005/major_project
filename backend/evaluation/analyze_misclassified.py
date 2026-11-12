"""Find, summarize, and visualize errors on the untouched maize test set."""

import argparse
import csv
import json
import re
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
CLASS_NAMES = ("Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy")
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
CSV_NAME = "misclassified_predictions.csv"
REPORT_NAME = "misclassification_analysis.txt"
CONTACT_SHEET_NAME = "contact_sheet.png"


def load_class_mapping(mapping_path):
    """Load and validate the model's saved class-index mapping."""
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Class mapping file is missing: {mapping_path}")

    try:
        mapping_data = json.loads(mapping_path.read_text(encoding="utf-8"))
        class_to_index = mapping_data["class_to_index"]
        index_to_class = mapping_data["index_to_class"]
    except (json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid class mapping file {mapping_path}: {error}") from error

    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    if class_to_index != expected_mapping:
        raise ValueError(
            "The saved class_to_index mapping does not match the model evaluation "
            f"class order. Expected {expected_mapping}, found {class_to_index}."
        )
    expected_inverse = {
        str(index): name for name, index in expected_mapping.items()
    }
    if index_to_class != expected_inverse:
        raise ValueError(
            "The saved index_to_class mapping is inconsistent with class_to_index."
        )
    return tuple(index_to_class[str(index)] for index in range(len(CLASS_NAMES)))


def collect_test_images(test_directory, class_names):
    """Check all test images and return a stable, class-labeled file list."""
    if not test_directory.is_dir():
        raise FileNotFoundError(f"Test dataset directory is missing: {test_directory}")

    observed_classes = {
        path.name for path in test_directory.iterdir() if path.is_dir()
    }
    expected_classes = set(class_names)
    if observed_classes != expected_classes:
        raise ValueError(
            f"Test class folders differ from the class mapping. "
            f"Expected {sorted(expected_classes)}, found {sorted(observed_classes)}."
        )

    records = []
    unreadable = []
    for class_name in class_names:
        class_directory = test_directory / class_name
        image_paths = sorted(
            (path for path in class_directory.rglob("*") if path.is_file()),
            key=lambda path: str(path).casefold(),
        )
        for image_path in image_paths:
            if image_path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
                unreadable.append((image_path, "unsupported image file extension"))
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.load()
            except (OSError, ValueError, SyntaxError) as error:
                unreadable.append((image_path, str(error)))
                continue
            records.append({"path": image_path, "actual_class": class_name})

    if unreadable:
        details = "\n".join(f"  - {path}: {error}" for path, error in unreadable)
        raise ValueError(
            f"Found {len(unreadable)} unreadable or unsupported test image(s); "
            "none were silently skipped:\n" + details
        )
    if not records:
        raise ValueError(f"No test images were found under {test_directory}.")
    return records


def load_rgb_batch(records):
    """Resize test files exactly as in training and preserve RGB pixel values."""
    batch = []
    for record in records:
        with Image.open(record["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
            batch.append(np.asarray(image, dtype=np.uint8))
    return np.stack(batch, axis=0)


def predict_test_images(model, records, class_names):
    """Predict in batches using the model's embedded MobileNetV2 rescaling."""
    class_to_index = {name: index for index, name in enumerate(class_names)}
    predictions = []

    for start in range(0, len(records), BATCH_SIZE):
        batch_records = records[start : start + BATCH_SIZE]
        uint8_images = load_rgb_batch(batch_records)

        # Match training: convert uint8 to [0, 1], then back to [0, 255].
        # The saved model applies its own MobileNetV2 [-1, 1] Rescaling layer.
        float_images = (
            uint8_images.astype(np.float32) / np.float32(255.0)
        ) * np.float32(255.0)
        probabilities = np.asarray(
            model.predict(float_images, verbose=0),
            dtype=np.float64,
        )
        if probabilities.ndim != 2 or probabilities.shape != (
            len(batch_records),
            len(class_names),
        ):
            raise RuntimeError(
                "The saved model output shape does not match the batch size and "
                f"class count: got {probabilities.shape}."
            )
        if not np.all(np.isfinite(probabilities)):
            raise RuntimeError("The model returned non-finite prediction values.")

        predicted_indices = np.argmax(probabilities, axis=1)
        for record, class_index, probability_row in zip(
            batch_records,
            predicted_indices,
            probabilities,
        ):
            predictions.append(
                {
                    **record,
                    "predicted_class": class_names[int(class_index)],
                    "confidence": float(probability_row[int(class_index)]),
                    "probabilities": probability_row,
                    "actual_index": class_to_index[record["actual_class"]],
                }
            )
    return predictions


def read_original_evaluation(evaluation_directory, class_names):
    """Read the previous evaluation metrics and confusion matrix for comparison."""
    report_path = evaluation_directory / "test_evaluation_report.txt"
    matrix_path = evaluation_directory / "confusion_matrix.csv"
    previous = {"report_path": report_path, "matrix_path": matrix_path}

    if report_path.is_file():
        report_text = report_path.read_text(encoding="utf-8")
        for metric, label in (
            ("accuracy", "Accuracy"),
            ("precision", "Precision"),
            ("recall", "Recall"),
            ("f1_score", "F1-score"),
        ):
            match = re.search(
                rf"^{re.escape(label)}:\s*([0-9]*\.?[0-9]+)\s*$",
                report_text,
                flags=re.MULTILINE,
            )
            previous[metric] = float(match.group(1)) if match else None
    else:
        previous["report_missing"] = True

    previous["confusion_matrix"] = None
    if matrix_path.is_file():
        try:
            with matrix_path.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))
            if rows[0] != ["actual/predicted", *class_names]:
                raise ValueError("class headers do not match the class mapping")
            if len(rows) != len(class_names) + 1:
                raise ValueError("unexpected number of confusion-matrix rows")
            matrix = []
            for expected_class, row in zip(class_names, rows[1:]):
                if len(row) != len(class_names) + 1 or row[0] != expected_class:
                    raise ValueError(f"invalid row for class {expected_class}")
                matrix.append([int(value) for value in row[1:]])
            previous["confusion_matrix"] = np.asarray(matrix, dtype=np.int64)
        except (OSError, ValueError, IndexError) as error:
            previous["matrix_error"] = str(error)
    else:
        previous["matrix_missing"] = True
    return previous


def write_misclassification_csv(csv_path, misclassified):
    """Write all incorrect test predictions and their confidence scores."""
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "Image filename",
                "Actual class",
                "Predicted class",
                "Prediction confidence",
            ],
        )
        writer.writeheader()
        for item in misclassified:
            writer.writerow(
                {
                    "Image filename": item["path"].name,
                    "Actual class": item["actual_class"],
                    "Predicted class": item["predicted_class"],
                    "Prediction confidence": f"{item['confidence']:.8f}",
                }
            )


def create_contact_sheet(contact_sheet_path, misclassified, copied_paths):
    """Create a labeled contact sheet for every incorrect prediction."""
    columns = 4
    thumbnail_width, thumbnail_height = 240, 180
    caption_height = 70
    cell_width = thumbnail_width + 16
    cell_height = thumbnail_height + caption_height + 12
    rows = max(1, (len(misclassified) + columns - 1) // columns)
    sheet = Image.new("RGB", (columns * cell_width, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)

    if not misclassified:
        draw.text((20, 20), "No misclassified test images.", fill="black")
    else:
        for index, (item, copied_path) in enumerate(zip(misclassified, copied_paths)):
            column = index % columns
            row = index // columns
            left = column * cell_width + 8
            top = row * cell_height + 6
            with Image.open(copied_path) as source_image:
                image = ImageOps.exif_transpose(source_image).convert("RGB")
                image.thumbnail((thumbnail_width, thumbnail_height))
                x = left + (thumbnail_width - image.width) // 2
                y = top + (thumbnail_height - image.height) // 2
                sheet.paste(image, (x, y))

            draw.text(
                (left, top + thumbnail_height + 2),
                f"Actual: {item['actual_class']}",
                fill="black",
            )
            draw.text(
                (left, top + thumbnail_height + 20),
                f"Predicted: {item['predicted_class']} "
                f"({item['confidence']:.1%})",
                fill="black",
            )
            draw.text(
                (left, top + thumbnail_height + 40),
                item["path"].name[:34],
                fill="black",
            )

    sheet.save(contact_sheet_path)


def write_analysis_report(
    report_path,
    predictions,
    misclassified,
    class_names,
    previous_evaluation,
):
    """Save error counts, Blight/Gray analysis, and report comparisons."""
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )

    class_to_index = {name: index for index, name in enumerate(class_names)}
    actual = np.asarray(
        [class_to_index[item["actual_class"]] for item in predictions],
        dtype=np.int64,
    )
    predicted = np.asarray(
        [class_to_index[item["predicted_class"]] for item in predictions],
        dtype=np.int64,
    )
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=list(range(len(class_names))),
    )
    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision": float(
            precision_score(actual, predicted, average="macro", zero_division=0)
        ),
        "recall": float(
            recall_score(actual, predicted, average="macro", zero_division=0)
        ),
        "f1_score": float(
            f1_score(actual, predicted, average="macro", zero_division=0)
        ),
    }
    class_report = classification_report(
        actual,
        predicted,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
        digits=4,
    )
    confusion_counts = Counter(
        (item["actual_class"], item["predicted_class"])
        for item in misclassified
    )
    blight_to_gray = confusion_counts[("Blight", "Gray_Leaf_Spot")]
    gray_to_blight = confusion_counts[("Gray_Leaf_Spot", "Blight")]
    class_support = Counter(item["actual_class"] for item in predictions)

    report_lines = [
        "MAIZE DISEASE TEST MISCLASSIFICATION ANALYSIS",
        "=" * 47,
        "",
        f"Analysis date: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Saved model: {Path('models/maize_disease_model.keras')}",
        "Test images were read only; model weights and dataset files were not changed.",
        "No training, test-set tuning, or Grad-CAM analysis was performed.",
        f"Test images evaluated: {len(predictions)}",
        f"Incorrectly classified images: {len(misclassified)}",
        f"Correctly classified images: {len(predictions) - len(misclassified)}",
        "",
        "Metrics recomputed from these predictions",
        "------------------------------------------",
        f"Accuracy: {metrics['accuracy']:.4f}",
        f"Macro precision: {metrics['precision']:.4f}",
        f"Macro recall: {metrics['recall']:.4f}",
        f"Macro F1-score: {metrics['f1_score']:.4f}",
        "",
        "Class-wise classification report",
        "---------------------------------",
        class_report,
        "Misclassifications grouped by actual and predicted class",
        "--------------------------------------------------------",
    ]
    if confusion_counts:
        report_lines.extend(
            f"{actual_class} -> {predicted_class}: {count}"
            for (actual_class, predicted_class), count in sorted(confusion_counts.items())
        )
    else:
        report_lines.append("None")

    report_lines.extend(
        [
            "",
            "Blight and Gray_Leaf_Spot confusion",
            "------------------------------------",
            f"Blight misclassified as Gray_Leaf_Spot: {blight_to_gray}",
            f"Gray_Leaf_Spot misclassified as Blight: {gray_to_blight}",
            f"Total errors between these two classes: {blight_to_gray + gray_to_blight}",
            (
                "Blight -> Gray_Leaf_Spot rate among actual Blight images: "
                f"{blight_to_gray / class_support['Blight']:.2%}"
                if class_support["Blight"]
                else "Blight -> Gray_Leaf_Spot rate: unavailable (no Blight test images)"
            ),
            (
                "Gray_Leaf_Spot -> Blight rate among actual Gray_Leaf_Spot images: "
                f"{gray_to_blight / class_support['Gray_Leaf_Spot']:.2%}"
                if class_support["Gray_Leaf_Spot"]
                else "Gray_Leaf_Spot -> Blight rate: unavailable "
                "(no Gray_Leaf_Spot test images)"
            ),
            "",
            "Comparison with the original test evaluation",
            "---------------------------------------------",
        ]
    )

    differences = []
    if previous_evaluation.get("report_missing"):
        differences.append("Original evaluation report was not found.")
    for metric_name, display_name in (
        ("accuracy", "Accuracy"),
        ("precision", "Macro precision"),
        ("recall", "Macro recall"),
        ("f1_score", "Macro F1-score"),
    ):
        old_value = previous_evaluation.get(metric_name)
        if old_value is None:
            differences.append(f"Original report did not provide {display_name}.")
            continue
        new_value = metrics[metric_name]
        matches = round(new_value, 4) == round(old_value, 4)
        report_lines.append(
            f"{display_name}: original {old_value:.4f}, new {new_value:.4f} "
            f"({'matches to 4 decimals' if matches else 'DIFFERS'})"
        )
        if not matches:
            differences.append(
                f"{display_name} differs: original {old_value:.4f}, "
                f"new {new_value:.4f}."
            )

    original_matrix = previous_evaluation.get("confusion_matrix")
    if original_matrix is None:
        matrix_reason = previous_evaluation.get("matrix_error")
        differences.append(
            "Original confusion matrix unavailable"
            + (f": {matrix_reason}" if matrix_reason else ".")
        )
        report_lines.append("Confusion matrix: original matrix unavailable.")
    else:
        if original_matrix.shape != matrix.shape:
            matrices_match = False
            differences.append(
                "Confusion matrix shapes differ: "
                f"original {original_matrix.shape}, new {matrix.shape}."
            )
        else:
            matrices_match = bool(np.array_equal(original_matrix, matrix))
            if not matrices_match:
                changed_cells = [
                    (
                        class_names[row],
                        class_names[column],
                        int(original_matrix[row, column]),
                        int(matrix[row, column]),
                    )
                    for row in range(len(class_names))
                    for column in range(len(class_names))
                    if original_matrix[row, column] != matrix[row, column]
                ]
                differences.append(
                    "Confusion matrix changed at "
                    + "; ".join(
                        f"{row}/{column}: {old} -> {new}"
                        for row, column, old, new in changed_cells
                    )
                    + "."
                )
        report_lines.append(
            "Confusion matrix: "
            + ("identical to original." if matrices_match else "DIFFERS from original.")
        )

    report_lines.append(
        "Overall comparison: "
        + ("no differences detected to the precision available in the original report."
           if not differences
           else "differences or unavailable comparison data were found.")
    )
    if differences:
        report_lines.extend(f"- {difference}" for difference in differences)
    report_lines.extend(
        [
            "",
            "Interpretation limits",
            "---------------------",
            "This analysis reports observed prediction errors only. It does not infer "
            "their visual, biological, or dataset-related causes.",
            "",
            "Confusion matrix (rows are actual; columns are predicted)",
            "----------------------------------------------------------",
        ]
    )
    report_lines.extend(
        f"{class_name}: {row.tolist()}"
        for class_name, row in zip(class_names, matrix)
    )
    report_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return matrix, metrics, confusion_counts


def analyze(project_root):
    """Load saved artifacts, predict all test files, and save the error analysis."""
    model_path = project_root / "models" / "maize_disease_model.keras"
    mapping_path = project_root / "models" / "maize_disease_classes.json"
    test_directory = project_root / "datasets" / "maize_disease_processed" / "test"
    evaluation_directory = project_root / "evaluation"
    output_directory = project_root / "outputs" / "misclassified_images"
    csv_path = evaluation_directory / CSV_NAME
    report_path = evaluation_directory / REPORT_NAME

    if not model_path.is_file():
        raise FileNotFoundError(f"Trained model file is missing: {model_path}")
    if output_directory.exists() or csv_path.exists() or report_path.exists():
        raise FileExistsError(
            "Misclassification analysis outputs already exist. Move or rename "
            "these outputs before rerunning; existing results will not be overwritten."
        )

    class_names = load_class_mapping(mapping_path)
    records = collect_test_images(test_directory, class_names)

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required to load the saved model. Run this script using "
            "the Python environment that created the .keras model."
        ) from error

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Could not load the saved model {model_path}: {error}"
        ) from error
    if model.input_shape != (None, IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise ValueError(
            f"Saved model input shape {model.input_shape} does not match "
            f"expected RGB input {(None, IMAGE_SIZE[1], IMAGE_SIZE[0], 3)}."
        )
    if model.output_shape != (None, len(class_names)):
        raise ValueError(
            f"Saved model output shape {model.output_shape} does not match "
            f"{len(class_names)} mapped classes."
        )
    if not any(
        layer.name == "mobilenetv2_preprocessing"
        for layer in model.layers
    ):
        raise ValueError(
            "The saved model does not contain its MobileNetV2 preprocessing layer; "
            "refusing to guess or double-apply input preprocessing."
        )

    predictions = predict_test_images(model, records, class_names)
    misclassified = [
        item
        for item in predictions
        if item["actual_class"] != item["predicted_class"]
    ]
    previous_evaluation = read_original_evaluation(
        evaluation_directory,
        class_names,
    )

    evaluation_directory.mkdir(parents=True, exist_ok=True)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".misclassification_staging_",
        dir=output_directory.parent,
    ) as temporary_parent:
        staging_parent = Path(temporary_parent)
        staging_images = staging_parent / "misclassified_images"
        staging_images.mkdir()
        staging_csv = staging_parent / CSV_NAME
        staging_report = staging_parent / REPORT_NAME

        copied_paths = []
        for item in misclassified:
            class_directory = staging_images / item["actual_class"]
            relative_path = item["path"].relative_to(
                test_directory / item["actual_class"]
            )
            copied_path = class_directory / relative_path
            copied_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item["path"], copied_path)
            copied_paths.append(copied_path)

        create_contact_sheet(
            staging_images / CONTACT_SHEET_NAME,
            misclassified,
            copied_paths,
        )
        write_misclassification_csv(staging_csv, misclassified)
        matrix, metrics, confusion_counts = write_analysis_report(
            staging_report,
            predictions,
            misclassified,
            class_names,
            previous_evaluation,
        )

        staging_images.rename(output_directory)
        staging_csv.replace(csv_path)
        staging_report.replace(report_path)

    return predictions, misclassified, matrix, metrics, confusion_counts, {
        "csv": csv_path,
        "report": report_path,
        "images": output_directory,
    }


def main():
    """Run test-set misclassification analysis without training."""
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Analyze saved-model predictions on the untouched maize test set."
    )
    parser.parse_args()

    try:
        predictions, misclassified, _, metrics, confusion_counts, outputs = analyze(
            project_root
        )
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"Misclassification analysis failed: {error}", file=sys.stderr)
        return 1

    print(f"Test images evaluated: {len(predictions)}")
    print(f"Misclassified images: {len(misclassified)}")
    print("Misclassifications by actual -> predicted class:")
    if confusion_counts:
        for (actual_class, predicted_class), count in sorted(confusion_counts.items()):
            print(f"  {actual_class} -> {predicted_class}: {count}")
    else:
        print("  None")
    print(
        "Blight -> Gray_Leaf_Spot: "
        f"{confusion_counts[('Blight', 'Gray_Leaf_Spot')]}"
    )
    print(
        "Gray_Leaf_Spot -> Blight: "
        f"{confusion_counts[('Gray_Leaf_Spot', 'Blight')]}"
    )
    print(
        "Recomputed accuracy: "
        f"{metrics['accuracy']:.4f} "
        "(not a new model evaluation or training result)"
    )
    print(f"CSV: {outputs['csv']}")
    print(f"Report: {outputs['report']}")
    print(f"Misclassified image copies and contact sheet: {outputs['images']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
