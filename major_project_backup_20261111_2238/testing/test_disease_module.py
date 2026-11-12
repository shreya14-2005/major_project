"""Run disease-module checks and produce prediction, quality, and CAM reports.

From the project root:
    python testing/test_disease_module.py
    python -m pytest -q testing/test_disease_module.py
"""

import csv
import os
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTING_DIRECTORY = PROJECT_ROOT / "testing"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(TESTING_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(TESTING_DIRECTORY))

import cv2
import numpy as np
import pytest
from PIL import Image, ImageEnhance

from disease_inference import (
    CLASS_MAPPING_PATH,
    CLASS_NAMES,
    CONFIG_PATH,
    MODEL_PATH,
    REAL_WORLD_DIRECTORY,
    PROJECT_ROOT,
    build_explanation_context,
    confidence_decision,
    generate_explanations,
    image_quality_metrics,
    load_class_mapping,
    load_config,
    load_model_bundle,
    make_overlay,
    predict_image,
    read_rgb_input,
    validate_model_architecture,
    validate_probabilities,
)


TEST_DATA_DIRECTORY = PROJECT_ROOT / "datasets" / "maize_disease_processed" / "test"
MODEL_DIRECTORY = PROJECT_ROOT / "models" / "efficientnetb0"
EVALUATION_DIRECTORY = MODEL_DIRECTORY / "evaluation"
OUTPUT_DIRECTORY = (
    MODEL_DIRECTORY / "explainability" / "outputs" / "disease_testing"
)
PREDICTION_CSV = EVALUATION_DIRECTORY / "app_prediction_testing.csv"
MODULE_REPORT = EVALUATION_DIRECTORY / "app_disease_module_testing_report.txt"
REAL_WORLD_REPORT = EVALUATION_DIRECTORY / "app_real_world_testing_report.txt"
QUALITY_REPORT = EVALUATION_DIRECTORY / "app_image_quality_testing_report.txt"
CSV_COLUMNS = [
    "Image filename",
    "Actual class",
    "Predicted class",
    "Prediction confidence",
    "Prediction status",
    "Confidence status",
    "Error",
]


@pytest.fixture(scope="session")
def model_bundle():
    return load_model_bundle()


def first_test_image():
    for class_name in CLASS_NAMES:
        class_directory = TEST_DATA_DIRECTORY / class_name
        if class_directory.is_dir():
            images = sorted(
                path
                for path in class_directory.rglob("*")
                if path.is_file()
                and path.suffix.casefold()
                in {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
            )
            if images:
                return images[0]
    raise FileNotFoundError(f"No supported test images found in {TEST_DATA_DIRECTORY}")


def test_class_mapping_contains_four_expected_classes():
    assert load_class_mapping(CLASS_MAPPING_PATH) == CLASS_NAMES


def test_model_loads_and_has_training_preprocessing(model_bundle):
    assert MODEL_PATH.is_file()
    assert MODEL_PATH == MODEL_DIRECTORY / "model.keras"
    assert model_bundle.model.name == "maize_efficientnetb0"
    assert validate_model_architecture(model_bundle.model, model_bundle.class_names)
    assert model_bundle.class_names == CLASS_NAMES


def test_rgb_conversion_and_resize_match_training_preprocessing(tmp_path):
    from models.efficientnetb0.training.common import read_rgb_pixels

    image_path = tmp_path / "rgba_leaf.png"
    Image.new("RGBA", (337, 119), (190, 45, 28, 120)).save(image_path)
    original, pixels = read_rgb_input(image_path)

    training_pixels = read_rgb_pixels(image_path)
    assert original.mode == "RGB"
    assert original.size == (337, 119)
    assert pixels.shape == (224, 224, 3)
    assert pixels.dtype == np.float32
    assert np.array_equal(pixels.astype(np.uint8), training_pixels)


def test_model_predictions_have_four_valid_probabilities(model_bundle):
    _, pixels = read_rgb_input(first_test_image())
    output = model_bundle.model(pixels[np.newaxis, ...], training=False).numpy()
    assert output.shape == (1, 4)
    probabilities = validate_probabilities(output[0], len(CLASS_NAMES))
    assert np.isclose(float(probabilities.sum()), 1.0, atol=1e-3)


def test_confidence_threshold_marks_uncertain_without_discarding_score():
    assert confidence_decision(0.69, 0.70) == "needs_review"
    assert confidence_decision(0.70, 0.70) == "predicted"
    assert validate_probabilities([0.1, 0.2, 0.3, 0.4], 4).shape == (4,)
    with pytest.raises(ValueError):
        confidence_decision(0.5, 1.1)
    with pytest.raises(ValueError):
        validate_probabilities([0.5, float("nan"), 0.0, 0.0], 4)


def test_quality_warnings_detect_blur_brightness_and_small_dimensions():
    original, _ = read_rgb_input(first_test_image())
    rgb = np.asarray(original)
    config = load_config()
    blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=20)
    dark = np.asarray(ImageEnhance.Brightness(original).enhance(0.02))
    bright = np.asarray(ImageEnhance.Brightness(original).enhance(8.0))
    small = cv2.resize(rgb, (48, 36), interpolation=cv2.INTER_AREA)

    assert "possibly_blurry" in image_quality_metrics(blurred, config)["warnings"]
    assert "very_dark" in image_quality_metrics(dark, config)["warnings"]
    assert "very_bright" in image_quality_metrics(bright, config)["warnings"]
    assert "unusual_dimensions" in image_quality_metrics(small, config)["warnings"]


def test_invalid_empty_unsupported_and_corrupt_paths_are_rejected(model_bundle, tmp_path):
    result = predict_image("", bundle=model_bundle)
    assert result["status"] == "rejected"
    assert result["predicted_class"] is None

    unsupported = tmp_path / "leaf.txt"
    unsupported.write_text("not an image", encoding="utf-8")
    assert predict_image(unsupported, bundle=model_bundle)["status"] == "rejected"

    corrupted = tmp_path / "corrupted.jpg"
    corrupted.write_bytes(b"not a valid image")
    assert predict_image(corrupted, bundle=model_bundle)["status"] == "rejected"


def test_complete_inference_returns_review_and_ood_fields(model_bundle):
    context = build_explanation_context(model_bundle.model)
    result = predict_image(
        first_test_image(),
        bundle=model_bundle,
        explanation_context=context,
    )
    assert result["status"] == "success"
    assert result["predicted_class"] in CLASS_NAMES
    assert 0.0 <= result["confidence"] <= 1.0
    _, pixels = read_rgb_input(first_test_image())
    probabilities = model_bundle.model(
        pixels[np.newaxis, ...],
        training=False,
    ).numpy()[0]
    predicted_index = int(np.argmax(probabilities))
    assert result["predicted_class"] == CLASS_NAMES[predicted_index]
    assert result["confidence"] == pytest.approx(
        float(probabilities[predicted_index]),
        abs=1e-6,
    )
    assert result["class_probabilities"] == pytest.approx(
        {
            name: float(probabilities[index])
            for index, name in enumerate(CLASS_NAMES)
        },
        abs=1e-6,
    )
    assert result["confidence_status"] in {"predicted", "needs_review"}
    assert result["out_of_distribution_status"] == "not_assessed"
    assert result["error"] is None
    assert "Grad-CAM" in result["explanations"]
    assert "Grad-CAM++" in result["explanations"]


def test_prediction_matches_saved_efficientnet_evaluation(model_bundle):
    predictions_path = MODEL_DIRECTORY / "evaluation" / "test_predictions.csv"
    with predictions_path.open(newline="", encoding="utf-8") as csv_file:
        row = next(csv.DictReader(csv_file))
    image_path = PROJECT_ROOT / Path(row["Source test image"].replace("\\", "/"))

    result = predict_image(image_path, bundle=model_bundle)

    assert result["status"] == "success"
    assert result["predicted_class"] == row["Predicted class"]
    assert result["confidence"] == pytest.approx(
        float(row["Prediction confidence"]),
        abs=1e-5,
    )


def test_gradcam_and_gradcam_plus_plus_target_prediction(model_bundle):
    image_path = first_test_image()
    original, pixels = read_rgb_input(image_path)
    output = model_bundle.model(pixels[np.newaxis, ...], training=False).numpy()[0]
    predicted_index = int(np.argmax(validate_probabilities(output, len(CLASS_NAMES))))
    context = build_explanation_context(model_bundle.model)
    results = generate_explanations(pixels, context, predicted_index)

    assert "Grad-CAM" in results
    assert results["Grad-CAM"]["predicted_index"] == predicted_index
    for item in results.values():
        heatmap = item["heatmap"]
        assert heatmap.ndim == 2
        assert heatmap.size > 0
        assert np.all(np.isfinite(heatmap))
        assert float(np.max(heatmap)) > 0
        _, _, overlay = make_overlay(original, heatmap)
        assert overlay.shape[:2] == (original.height, original.width)
    assert "Grad-CAM++" in results


def test_real_world_image_folder_is_separate_from_dataset():
    assert REAL_WORLD_DIRECTORY.is_dir()
    assert REAL_WORLD_DIRECTORY.resolve() != TEST_DATA_DIRECTORY.resolve()


def dataset_records():
    """List test-set files only; never modify or relocate them."""
    records = []
    errors = []
    for class_name in CLASS_NAMES:
        class_directory = TEST_DATA_DIRECTORY / class_name
        if not class_directory.is_dir():
            errors.append(f"Missing test class directory: {class_directory}")
            continue
        paths = sorted(
            (path for path in class_directory.rglob("*") if path.is_file()),
            key=lambda path: str(path).casefold(),
        )
        for path in paths:
            records.append((class_name, path))
    return records, errors


def prediction_row(image_path, actual_class, predicted_class="", confidence=None,
                   prediction_status="failed", confidence_status="not_evaluated",
                   error=""):
    return {
        "Image filename": str(image_path.relative_to(TEST_DATA_DIRECTORY)),
        "Actual class": actual_class,
        "Predicted class": predicted_class,
        "Prediction confidence": (
            f"{confidence:.8f}" if confidence is not None else ""
        ),
        "Prediction status": prediction_status,
        "Confidence status": confidence_status,
        "Error": error,
    }


def run_dataset_predictions(bundle, config):
    """Predict all supplied test images in batches and write a per-image CSV."""
    records, scan_errors = dataset_records()
    rows = []
    batch_size = 32
    for offset in range(0, len(records), batch_size):
        batch_records = records[offset:offset + batch_size]
        valid_records = []
        batch_pixels = []
        for actual_class, path in batch_records:
            try:
                _, pixels = read_rgb_input(path)
            except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
                rows.append(prediction_row(path, actual_class, error=str(error)))
                continue
            valid_records.append((actual_class, path))
            batch_pixels.append(pixels)

        if batch_pixels:
            outputs = bundle.model(
                np.stack(batch_pixels, axis=0),
                training=False,
            ).numpy()
            if outputs.shape != (len(valid_records), len(CLASS_NAMES)):
                for actual_class, path in valid_records:
                    rows.append(
                        prediction_row(
                            path,
                            actual_class,
                            error=f"Unexpected model output shape: {outputs.shape}",
                        )
                    )
                continue
            for (actual_class, path), output in zip(valid_records, outputs):
                try:
                    probabilities = validate_probabilities(output, len(CLASS_NAMES))
                except ValueError as error:
                    rows.append(prediction_row(path, actual_class, error=str(error)))
                    continue
                predicted_class = bundle.class_names[int(np.argmax(probabilities))]
                confidence = float(np.max(probabilities))
                rows.append(
                    prediction_row(
                        path,
                        actual_class,
                        predicted_class,
                        confidence,
                        "correct" if actual_class == predicted_class else "incorrect",
                        confidence_decision(
                            confidence,
                            config["confidence_threshold"],
                        ),
                    )
                )
        print(
            f"Test-set inference: {min(offset + batch_size, len(records))}"
            f"/{len(records)} images checked"
        )

    rows.sort(key=lambda row: row["Image filename"].casefold())
    EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    with PREDICTION_CSV.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return rows, scan_errors


def choose_explanation_records(rows):
    """Select up to one correct image per class and errors in both major directions."""
    selected = []
    for class_name in CLASS_NAMES:
        candidates = [
            row for row in rows
            if row["Actual class"] == class_name
            and row["Prediction status"] == "correct"
        ]
        if candidates:
            selected.append(candidates[0])
    for actual, predicted in (
        ("Blight", "Gray_Leaf_Spot"),
        ("Gray_Leaf_Spot", "Blight"),
    ):
        candidates = [
            row for row in rows
            if row["Actual class"] == actual
            and row["Predicted class"] == predicted
            and row["Prediction status"] == "incorrect"
        ]
        if candidates:
            selected.append(candidates[0])
    return selected


def run_explainability_checks(bundle, rows):
    """Test both existing CAM implementations and save selected overlays."""
    selected = choose_explanation_records(rows)
    report_lines = [
        "EXPLAINABILITY TEST RESULTS",
        "===========================",
        "Grad-CAM uses the model's predicted class; this is not evidence that "
        "the highlighted region is a valid disease symptom.",
    ]
    results = []
    if not selected:
        return ["Skipped: no suitable correctly or incorrectly classified images."], 0, 0

    try:
        context = build_explanation_context(bundle.model)
    except (ImportError, OSError, ValueError, RuntimeError) as error:
        return [f"Failed to initialize explanation methods: {error}"], 0, len(selected)

    gradcam_plus_plus_available = context["gradcam_plus_plus"] is not None
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    explanation_directory = OUTPUT_DIRECTORY / "explanations"
    explanation_directory.mkdir(parents=True, exist_ok=True)
    passed = 0
    failed = 0
    for row in selected:
        image_path = TEST_DATA_DIRECTORY / row["Image filename"]
        record_label = (
            f"{row['Image filename']} ({row['Actual class']} -> "
            f"{row['Predicted class']})"
        )
        try:
            original, pixels = read_rgb_input(image_path)
            predicted_index = bundle.class_to_index[row["Predicted class"]]
            explanations = generate_explanations(pixels, context, predicted_index)
            expected_methods = ["Grad-CAM"]
            if gradcam_plus_plus_available:
                expected_methods.append("Grad-CAM++")
            if not set(expected_methods).issubset(explanations):
                raise ValueError("An available explanation method did not return a heatmap.")
            for method, item in explanations.items():
                heatmap = item["heatmap"]
                if (
                    heatmap.ndim != 2
                    or heatmap.size == 0
                    or not np.all(np.isfinite(heatmap))
                    or float(np.max(heatmap)) <= 0.0
                    or item["predicted_index"] != predicted_index
                ):
                    raise ValueError(f"{method} returned an invalid or mis-targeted heatmap.")
                _, _, overlay = make_overlay(original, heatmap)
                if overlay.shape[:2] != (original.height, original.width):
                    raise ValueError(f"{method} overlay has incorrect dimensions.")
                image_panel = original.resize((224, 224), Image.Resampling.BILINEAR)
                overlay_panel = Image.fromarray(overlay).resize(
                    (224, 224),
                    Image.Resampling.BILINEAR,
                )
                combined = Image.new("RGB", (448, 224))
                combined.paste(image_panel, (0, 0))
                combined.paste(overlay_panel, (224, 0))
                output_path = explanation_directory / (
                    f"{image_path.stem}_{method.lower().replace('+', 'plus')}.png"
                )
                combined.save(output_path)
            passed += len(explanations)
            results.append(f"PASS: {record_label}: {', '.join(explanations)}")
        except (OSError, ValueError, RuntimeError, cv2.error) as error:
            failed += 1
            results.append(f"FAIL: {record_label}: {error}")

    report_lines.extend([
        f"Grad-CAM++ implementation available: {'yes' if gradcam_plus_plus_available else 'no (skipped)'}",
        f"Selected images: {len(selected)}",
        *results,
        f"Explanation method checks passed: {passed}",
        f"Explanation image checks failed: {failed}",
        "Interpretation limitation: heatmap generation/appearance does not validate "
        "whether the model attends to biologically correct symptoms.",
    ])
    return report_lines, passed, failed


def run_quality_checks(config):
    """Exercise valid and deliberately degraded copies without writing to datasets."""
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    checks = []

    def record(name, passed, detail):
        checks.append((name, "PASSED" if passed else "FAILED", str(detail)))

    try:
        source_path = first_test_image()
        original, _ = read_rgb_input(source_path)
        quality = image_quality_metrics(original, config)
        record(
            "Valid maize test image",
            original.mode == "RGB" and original.size[0] > 0 and quality["width"] > 0,
            f"decoded RGB, original size={original.size}, warnings={quality['warnings']}",
        )
        rgb = np.asarray(original)

        blurred = cv2.GaussianBlur(rgb, (0, 0), sigmaX=20)
        blur_quality = image_quality_metrics(blurred, config)
        record(
            "Blurry image warning",
            "possibly_blurry" in blur_quality["warnings"],
            f"Laplacian variance={blur_quality['laplacian_variance']:.2f}",
        )

        dark = np.asarray(ImageEnhance.Brightness(original).enhance(0.02))
        dark_quality = image_quality_metrics(dark, config)
        record(
            "Very dark image warning",
            "very_dark" in dark_quality["warnings"],
            f"mean grayscale={dark_quality['mean_grayscale']:.2f}",
        )

        bright = np.asarray(ImageEnhance.Brightness(original).enhance(8.0))
        bright_quality = image_quality_metrics(bright, config)
        record(
            "Very bright image warning",
            "very_bright" in bright_quality["warnings"],
            f"mean grayscale={bright_quality['mean_grayscale']:.2f}",
        )

        small = cv2.resize(rgb, (48, 36), interpolation=cv2.INTER_AREA)
        dimensions = image_quality_metrics(small, config)
        record(
            "Unusual dimensions warning",
            "unusual_dimensions" in dimensions["warnings"],
            f"dimensions={dimensions['width']}x{dimensions['height']}",
        )

        with tempfile.TemporaryDirectory(prefix="planthealthai_image_checks_") as folder:
            temp_path = Path(folder)
            unsupported_path = temp_path / "unsupported.txt"
            unsupported_path.write_text("not an image", encoding="utf-8")
            unsupported = predict_image(unsupported_path)
            record(
                "Unsupported file format",
                unsupported["status"] == "rejected",
                unsupported["error"],
            )

            corrupt_path = temp_path / "corrupted.jpg"
            corrupt_path.write_bytes(b"not an image")
            corrupt = predict_image(corrupt_path)
            record(
                "Corrupted image file",
                corrupt["status"] == "rejected",
                corrupt["error"],
            )

            empty = predict_image("")
            record(
                "Empty input path",
                empty["status"] == "rejected",
                empty["error"],
            )
            invalid = predict_image(temp_path / "missing.jpg")
            record(
                "Invalid input path",
                invalid["status"] == "rejected",
                invalid["error"],
            )
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        record("Image quality test setup", False, error)

    passed = sum(status == "PASSED" for _, status, _ in checks)
    failed = sum(status == "FAILED" for _, status, _ in checks)
    lines = [
        "IMAGE QUALITY AND INPUT VALIDATION TEST RESULTS",
        "===============================================",
        f"Image source for valid/degraded cases: {TEST_DATA_DIRECTORY}",
        "Blur, brightness, and dimension cases are temporary in-memory "
        "transformations of one existing test image; no dataset files were changed.",
        "Quality limits are configurable heuristics, not medically or scientifically "
        "validated rejection criteria.",
        "",
        f"Passed: {passed}",
        f"Failed: {failed}",
        "",
    ]
    lines.extend(f"{status}: {name} - {detail}" for name, status, detail in checks)
    return lines, passed, failed


def run_real_world_tests(bundle, config):
    """Evaluate only images supplied to the separate real-world folder."""
    REAL_WORLD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    paths = sorted(
        (
            path for path in REAL_WORLD_DIRECTORY.rglob("*")
            if path.is_file() and path.name != ".gitkeep"
        ),
        key=lambda path: str(path).casefold(),
    )
    lines = [
        "REAL-WORLD IMAGE TESTING",
        "========================",
        f"Input folder: {REAL_WORLD_DIRECTORY}",
        "This folder is separate from the Kaggle train/validation/test data. "
        "No supplied files are copied or modified.",
        "",
    ]
    if not paths:
        lines.extend([
            "NOT PERFORMED: no real-world images were supplied.",
            "Add independent maize leaf images here (varied lighting, backgrounds, "
            "camera angles, partial damage, and mobile-camera photos), then rerun "
            "python testing/test_disease_module.py.",
            "No real-world prediction results are claimed.",
        ])
        return lines, 0, 0, 0

    passed = 0
    failed = 0
    for path in paths:
        result = predict_image(path, bundle=bundle, config=config)
        if result["status"] != "success":
            failed += 1
            lines.append(f"FAILED: {path.name}: {result['error']}")
            continue
        passed += 1
        quality_warnings = ", ".join(result["quality"]["warnings"]) or "none"
        lines.append(
            f"TESTED (no ground-truth label supplied): {path.name}; "
            f"predicted={result['predicted_class']}; "
            f"confidence={result['confidence']:.4f}; "
            f"review={result['confidence_status']}; "
            f"quality warnings={quality_warnings}; "
            "non-maize/OOD status=not assessed"
        )
    lines.extend([
        "",
        f"Images tested: {len(paths)}",
        f"Successfully processed: {passed}",
        f"Rejected/failed inputs: {failed}",
        "Without independent labels, these predictions do not measure real-world "
        "accuracy or establish that the image contains maize.",
    ])
    return lines, passed, failed, len(paths)


def write_lines(path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_unit_tests():
    """Run this file's pytest cases and return the actual collected outcomes."""
    class SummaryPlugin:
        counts = {"passed": 0, "failed": 0, "skipped": 0}

        def pytest_terminal_summary(self, terminalreporter):
            for name in self.counts:
                self.counts[name] = len(terminalreporter.stats.get(name, []))

    plugin = SummaryPlugin()
    result = pytest.main(["-q", str(Path(__file__).resolve())], plugins=[plugin])
    return plugin.counts, int(result)


def main():
    EVALUATION_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    REAL_WORLD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    config = load_config()
    run_started = datetime.now().astimezone().isoformat(timespec="seconds")

    try:
        unit_counts, pytest_exit = run_unit_tests()
        unit_error = ""
    except (ImportError, OSError, RuntimeError) as error:
        unit_counts = {"passed": 0, "failed": 0, "skipped": 0}
        pytest_exit = 1
        unit_error = f"Automated pytest suite could not complete: {error}"

    try:
        bundle = load_model_bundle()
        prediction_rows, scan_errors = run_dataset_predictions(bundle, config)
        prediction_counts = Counter(row["Prediction status"] for row in prediction_rows)
        review_count = sum(
            row["Confidence status"] == "needs_review" for row in prediction_rows
        )
        reviewed_error_count = sum(
            row["Prediction status"] == "incorrect"
            and row["Confidence status"] == "needs_review"
            for row in prediction_rows
        )
        above_threshold_error_count = sum(
            row["Prediction status"] == "incorrect"
            and row["Confidence status"] == "predicted"
            for row in prediction_rows
        )
        explanation_lines, explanation_passed, explanation_failed = (
            run_explainability_checks(bundle, prediction_rows)
        )
        real_lines, real_passed, real_failed, real_count = run_real_world_tests(
            bundle,
            config,
        )
        workflow_error = ""
    except Exception as error:
        bundle = None
        prediction_rows = []
        scan_errors = [f"Model/prediction workflow failed: {type(error).__name__}: {error}"]
        prediction_counts = Counter()
        review_count = 0
        reviewed_error_count = 0
        above_threshold_error_count = 0
        explanation_lines = ["FAILED: explanation workflow was not run because model inference failed."]
        explanation_passed = 0
        explanation_failed = 1
        real_lines, real_passed, real_failed, real_count = (
            run_real_world_tests(None, config)
            if REAL_WORLD_DIRECTORY.is_dir() and any(
                path.is_file() and path.name != ".gitkeep"
                for path in REAL_WORLD_DIRECTORY.rglob("*")
            )
            else (
                [
                    "REAL-WORLD IMAGE TESTING",
                    "========================",
                    "NOT PERFORMED: no real-world images were supplied.",
                ],
                0,
                0,
                0,
            )
        )
        workflow_error = f"{type(error).__name__}: {error}"

    quality_lines, quality_passed, quality_failed = run_quality_checks(config)
    write_lines(QUALITY_REPORT, quality_lines)
    write_lines(REAL_WORLD_REPORT, real_lines)

    total_passed = unit_counts["passed"] + quality_passed + explanation_passed
    total_failed = (
        unit_counts["failed"]
        + quality_failed
        + explanation_failed
        + prediction_counts["failed"]
        + real_failed
    )
    skipped_count = unit_counts["skipped"] + (1 if real_count == 0 else 0)
    lines = [
        "PLANTHEALTHAI MAIZE DISEASE MODULE TEST REPORT",
        "==============================================",
        f"Run date: {run_started}",
        f"Model: {MODEL_PATH}",
        f"Class mapping: {CLASS_MAPPING_PATH}",
        f"Test dataset: {TEST_DATA_DIRECTORY}",
        "Model weights, original test dataset, and train/validation/test split "
        "were not changed. No training or test-set threshold tuning was performed.",
        "",
        "Automated pytest checks",
        "-----------------------",
        f"Passed: {unit_counts['passed']}",
        f"Failed: {unit_counts['failed']}",
        f"Skipped: {unit_counts['skipped']}",
        f"Pytest exit code: {pytest_exit}",
        *([unit_error] if unit_error else []),
        "",
        "Dataset prediction checks",
        "--------------------------",
        f"Image records: {len(prediction_rows)}",
        f"Correct predictions: {prediction_counts['correct']}",
        f"Incorrect predictions: {prediction_counts['incorrect']}",
        f"Failed/unreadable inputs: {prediction_counts['failed']}",
        f"Needs review at configured threshold: {review_count}",
        f"Observed incorrect predictions marked needs_review: {reviewed_error_count}",
        f"Observed incorrect predictions at/above threshold: {above_threshold_error_count}",
        "These are descriptive test-set counts only; the threshold was not selected "
        "or validated using this test set.",
        f"Prediction CSV: {PREDICTION_CSV}",
        *[f"Dataset scan issue: {error}" for error in scan_errors],
        "",
        "Existing evaluation reference (not a threshold-calibration set)",
        "---------------------------------------------------------------",
        "Reported test accuracy: 96.18%; macro precision: 94.73%; "
        "macro recall: 95.75%; macro F1: 95.18%.",
        "Current per-image predictions are recorded in the CSV; compare those "
        "results with models/efficientnetb0/evaluation/test_evaluation_report.txt.",
        "",
        "Confidence and unsupported-image handling",
        "------------------------------------------",
        f"Configured confidence threshold: {config['confidence_threshold']:.3f}",
        f"Threshold marked validation-tested: {config['threshold_validated']}",
        "Low-confidence predictions retain the predicted class and confidence and "
        "are marked needs_review. The threshold is not scientifically validated.",
        "OOD/non-maize status: NOT ASSESSED. The confidence threshold is not an "
        "OOD detector and the classifier may assign a class to unrelated images.",
        "Recommended OOD work (not implemented or tested): "
        + config["ood_guidance"],
        "Image-quality checks are heuristic warnings only; they do not prove an "
        "input is maize or a disease is present.",
        "",
        *explanation_lines,
        "",
        "Real-world testing",
        "------------------",
        f"Real-world images supplied: {real_count}",
        f"Successfully processed: {real_passed}",
        f"Rejected/failed: {real_failed}",
        "See models/efficientnetb0/evaluation/app_real_world_testing_report.txt "
        "for details.",
        "",
        "Image quality and input validation",
        "----------------------------------",
        f"Checks passed: {quality_passed}",
        f"Checks failed: {quality_failed}",
        "See models/efficientnetb0/evaluation/app_image_quality_testing_report.txt "
        "for details.",
        "",
        "Overall executed check counts",
        "------------------------------",
        f"Passed: {total_passed}",
        f"Failed: {total_failed}",
        f"Skipped: {skipped_count}",
        f"Pytest checks skipped: {unit_counts['skipped']}",
        f"Real-world testing skipped: {'yes (no images supplied)' if real_count == 0 else 'no'}",
        "",
        "Limitations/errors",
        "------------------",
        workflow_error or "No model/prediction workflow error was raised.",
        "Grad-CAM/Grad-CAM++ can generate visual explanations but cannot establish "
        "that highlighted regions are biologically meaningful.",
        "Run `python -m pytest -q testing/test_disease_module.py` to execute just "
        "the automated test suite.",
    ]
    write_lines(MODULE_REPORT, lines)
    print(f"\nDisease testing report: {MODULE_REPORT}")
    print(f"Prediction CSV: {PREDICTION_CSV}")
    print(
        f"Automated tests: {unit_counts['passed']} passed, "
        f"{unit_counts['failed']} failed, {unit_counts['skipped']} skipped"
    )
    print(
        f"Quality checks: {quality_passed} passed, {quality_failed} failed; "
        f"real-world images supplied: {real_count}"
    )
    if unit_error or workflow_error or pytest_exit != 0:
        return 1
    if unit_counts["failed"] or quality_failed or explanation_failed or real_failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
