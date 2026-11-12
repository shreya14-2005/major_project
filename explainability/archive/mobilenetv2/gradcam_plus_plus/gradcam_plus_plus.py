"""Compare Grad-CAM++ with the existing Grad-CAM on the same test images."""

import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from explainability.archive.mobilenetv2.gradcam.gradcam import (
    CLASS_NAMES,
    IMAGE_SIZE,
    build_gradcam_model,
    generate_gradcam,
    read_model_input,
    resize_heatmap,
)


MODEL_DIRECTORY = PROJECT_ROOT / "models" / "mobilenetv2"
EXPECTED_SELECTIONS = {
    "correct_predictions": 4,
    "blight_to_gray_leaf_spot": 13,
    "gray_leaf_spot_to_blight": 7,
}
SELECTION_CSV = (
    PROJECT_ROOT
    / "explainability"
    / "archive"
    / "mobilenetv2"
    / "gradcam"
    / "outputs"
    / "gradcam_report.csv"
)
OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "explainability"
    / "archive"
    / "mobilenetv2"
    / "gradcam_plus_plus"
    / "outputs"
)
REPORT_PATH = (
    PROJECT_ROOT
    / "explainability"
    / "archive"
    / "mobilenetv2"
    / "reports"
    / "gradcam_comparison_report.txt"
)
RESULTS_CSV_NAME = "gradcam_plus_plus_report.csv"
HEATMAP_ALPHA = 0.45


def load_class_mapping(mapping_path):
    """Load the class mapping and verify the trained classifier order."""
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Class mapping is missing: {mapping_path}")
    try:
        mapping_data = json.loads(mapping_path.read_text(encoding="utf-8"))
        class_to_index = mapping_data["class_to_index"]
        index_to_class = mapping_data["index_to_class"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid class mapping {mapping_path}: {error}") from error

    expected_mapping = {
        class_name: index for index, class_name in enumerate(CLASS_NAMES)
    }
    expected_inverse = {
        str(index): class_name for index, class_name in enumerate(CLASS_NAMES)
    }
    if class_to_index != expected_mapping or index_to_class != expected_inverse:
        raise ValueError(
            "The saved class mapping does not match the model output order. "
            f"Expected {expected_mapping} and {expected_inverse}."
        )
    return expected_mapping


def load_selected_images(selection_csv, test_directory):
    """Reuse and validate the exact selection recorded by the Grad-CAM run."""
    if not selection_csv.is_file():
        raise FileNotFoundError(
            f"Existing Grad-CAM selection CSV is missing: {selection_csv}"
        )

    required_columns = {
        "Image filename",
        "Actual class",
        "Predicted class",
        "Prediction confidence",
        "Grad-CAM output filename",
        "Heatmap generation status",
        "Selection group",
    }
    rows = []
    counts = Counter()
    seen_paths = set()

    with selection_csv.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Grad-CAM selection CSV is missing columns: "
                f"{sorted(required_columns - set(reader.fieldnames or []))}"
            )
        for row in reader:
            group_name = row["Selection group"].strip()
            if group_name not in EXPECTED_SELECTIONS:
                raise ValueError(
                    f"Unexpected Grad-CAM selection group: {group_name!r}"
                )
            actual_class = row["Actual class"].strip()
            predicted_class = row["Predicted class"].strip()
            filename = row["Image filename"].strip()
            if actual_class not in CLASS_NAMES or predicted_class not in CLASS_NAMES:
                raise ValueError(
                    f"Unknown class label in Grad-CAM selection CSV for {filename}."
                )
            if row["Heatmap generation status"].strip() != "success":
                raise ValueError(
                    f"Grad-CAM previously failed for {filename}; refusing to claim "
                    "the same images were compared successfully."
                )

            expected_correct = group_name == "correct_predictions"
            if expected_correct != (actual_class == predicted_class):
                raise ValueError(
                    f"Selection group and saved Grad-CAM labels disagree for {filename}."
                )
            if group_name == "blight_to_gray_leaf_spot" and (
                actual_class,
                predicted_class,
            ) != ("Blight", "Gray_Leaf_Spot"):
                raise ValueError(
                    f"Unexpected labels for selected image {filename}: "
                    f"{actual_class} -> {predicted_class}."
                )
            if group_name == "gray_leaf_spot_to_blight" and (
                actual_class,
                predicted_class,
            ) != ("Gray_Leaf_Spot", "Blight"):
                raise ValueError(
                    f"Unexpected labels for selected image {filename}: "
                    f"{actual_class} -> {predicted_class}."
                )

            try:
                previous_confidence = float(row["Prediction confidence"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid saved confidence for {filename}."
                ) from error
            if not np.isfinite(previous_confidence) or not 0 <= previous_confidence <= 1:
                raise ValueError(f"Confidence is outside [0, 1] for {filename}.")

            image_path = test_directory / actual_class / filename
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Selected Grad-CAM test image is missing: {image_path}"
                )
            previous_gradcam_image = (
                PROJECT_ROOT
                / "models"
                / "mobilenetv2"
                / "explainability"
                / "outputs"
                / "gradcam"
                / Path(row["Grad-CAM output filename"].strip())
            )
            if not previous_gradcam_image.is_file():
                raise FileNotFoundError(
                    f"Saved Grad-CAM visualization is missing: "
                    f"{previous_gradcam_image}"
                )

            resolved_path = image_path.resolve()
            if resolved_path in seen_paths:
                raise ValueError(f"Grad-CAM selection lists {filename} more than once.")
            seen_paths.add(resolved_path)
            rows.append(
                {
                    "path": image_path,
                    "filename": filename,
                    "actual_class": actual_class,
                    "previous_predicted_class": predicted_class,
                    "previous_confidence": previous_confidence,
                    "previous_gradcam_image": previous_gradcam_image,
                    "selection_group": group_name,
                }
            )
            counts[group_name] += 1

    if len(rows) != sum(EXPECTED_SELECTIONS.values()):
        raise ValueError(
            f"The existing Grad-CAM selection has {len(rows)} rows; "
            f"expected {sum(EXPECTED_SELECTIONS.values())}."
        )
    for group_name, expected_count in EXPECTED_SELECTIONS.items():
        if counts[group_name] != expected_count:
            raise ValueError(
                f"Grad-CAM selection has {counts[group_name]} "
                f"{group_name} rows; expected {expected_count}."
            )
    return rows


def gradcam_plus_plus_heatmap(tf, gradcam_model, image_pixels):
    """Calculate Grad-CAM++ using its gradient-power alpha weighting."""
    input_batch = tf.convert_to_tensor(
        image_pixels[np.newaxis, ...],
        dtype=tf.float32,
    )
    with tf.GradientTape() as tape:
        activations, probabilities = gradcam_model(input_batch, training=False)
        predicted_index = int(tf.argmax(probabilities[0]).numpy())
        class_score = probabilities[:, predicted_index]

    gradients = tape.gradient(class_score, activations)
    if gradients is None:
        raise ValueError("Grad-CAM++ gradients are unavailable.")
    if not bool(tf.reduce_all(tf.math.is_finite(gradients)).numpy()):
        raise ValueError("Grad-CAM++ gradients contain NaN or infinite values.")

    feature_maps = activations[0]
    first_gradients = gradients[0]
    # This closed-form Grad-CAM++ weighting uses squared/cubed gradients,
    # unlike Grad-CAM's spatial average of first-order gradients.
    squared_gradients = tf.square(first_gradients)
    cubed_gradients = squared_gradients * first_gradients
    third_order_term = tf.reduce_sum(
        feature_maps * cubed_gradients,
        axis=(0, 1),
    )
    denominator = 2.0 * squared_gradients + third_order_term[tf.newaxis, tf.newaxis, :]
    epsilon = tf.constant(1e-8, dtype=denominator.dtype)
    alphas = squared_gradients / tf.maximum(denominator, epsilon)
    alphas = tf.where(
        tf.math.is_finite(alphas),
        alphas,
        tf.zeros_like(alphas),
    )

    channel_weights = tf.reduce_sum(
        alphas * tf.nn.relu(first_gradients),
        axis=(0, 1),
    )
    heatmap = tf.reduce_sum(
        feature_maps * channel_weights[tf.newaxis, tf.newaxis, :],
        axis=-1,
    )
    heatmap = tf.nn.relu(heatmap)
    maximum = float(tf.reduce_max(heatmap).numpy())
    if not np.isfinite(maximum) or maximum <= 1e-12:
        raise ValueError("Grad-CAM++ heatmap is empty or has no positive activation.")

    heatmap_array = (heatmap / maximum).numpy().astype(np.float32)
    if heatmap_array.ndim != 2 or not np.all(np.isfinite(heatmap_array)):
        raise ValueError("Normalized Grad-CAM++ heatmap is invalid.")
    confidence = float(probabilities[0, predicted_index].numpy())
    if not np.isfinite(confidence):
        raise ValueError("Grad-CAM++ prediction confidence is invalid.")
    return heatmap_array, predicted_index, confidence


def compare_heatmaps(gradcam_heatmap, gradcam_plus_plus_map):
    """Return descriptive spatial similarity values for two normalized maps."""
    first = np.asarray(gradcam_heatmap, dtype=np.float64).ravel()
    second = np.asarray(gradcam_plus_plus_map, dtype=np.float64).ravel()
    if first.shape != second.shape:
        raise ValueError("Grad-CAM maps have different shapes and cannot be compared.")
    mean_absolute_difference = float(np.mean(np.abs(first - second)))
    first_std = float(np.std(first))
    second_std = float(np.std(second))
    if first_std <= 1e-12 or second_std <= 1e-12:
        correlation = None
    else:
        correlation = float(np.corrcoef(first, second)[0, 1])
        if not np.isfinite(correlation):
            correlation = None
    return correlation, mean_absolute_difference


def blend_heatmap(original_rgb, colored_heatmap):
    """Overlay a JET-colored heatmap on an original RGB image."""
    return cv2.addWeighted(
        np.asarray(original_rgb),
        1.0 - HEATMAP_ALPHA,
        colored_heatmap,
        HEATMAP_ALPHA,
        0,
    )


def save_method_visualization(
    output_path,
    original_rgb,
    heatmap_rgb,
    actual_class,
    predicted_class,
    confidence,
):
    """Save a three-panel view for Grad-CAM++ with image labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original image")
    axes[1].imshow(heatmap_rgb)
    axes[1].set_title("Grad-CAM++ heatmap")
    axes[2].imshow(blend_heatmap(original_rgb, heatmap_rgb))
    axes[2].set_title("Grad-CAM++ overlay")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"Actual: {actual_class}  |  Predicted: {predicted_class} "
        f"({confidence:.1%})",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def save_comparison_visualization(
    output_path,
    original_rgb,
    gradcam_rgb,
    gradcam_plus_plus_rgb,
    actual_class,
    predicted_class,
    confidence,
):
    """Save five comparable panels using the same color map and overlay alpha."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 5, figsize=(23, 5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original image")
    axes[1].imshow(gradcam_rgb)
    axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(blend_heatmap(original_rgb, gradcam_rgb))
    axes[2].set_title("Grad-CAM overlay")
    axes[3].imshow(gradcam_plus_plus_rgb)
    axes[3].set_title("Grad-CAM++ heatmap")
    axes[4].imshow(blend_heatmap(original_rgb, gradcam_plus_plus_rgb))
    axes[4].set_title("Grad-CAM++ overlay")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"Actual: {actual_class}  |  Predicted: {predicted_class} "
        f"({confidence:.1%})",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def output_group_directory(group_name):
    """Map the Grad-CAM selection group to the requested output subfolder."""
    if group_name not in EXPECTED_SELECTIONS:
        raise ValueError(f"Unsupported selection group: {group_name}")
    return group_name


def make_output_filename(row, suffix):
    """Create stable visualization names without changing source filenames."""
    return f"{row['actual_class']}__{Path(row['filename']).stem}_{suffix}.png"


def generate_outputs(tf, gradcam_model, selected_rows, staging_directory):
    """Generate both explanation methods for each exact selected image."""
    report_rows = []
    comparisons = []

    for row in selected_rows:
        group_name = output_group_directory(row["selection_group"])
        gradcam_plus_plus_name = make_output_filename(row, "gradcam_plus_plus")
        comparison_name = make_output_filename(row, "comparison")
        plus_plus_path = staging_directory / group_name / gradcam_plus_plus_name
        comparison_path = staging_directory / "comparison" / comparison_name
        plus_plus_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.parent.mkdir(parents=True, exist_ok=True)

        status = "success"
        gradcam_status = "success"
        plus_plus_status = "success"
        actual_class = row["actual_class"]
        predicted_class = ""
        confidence = None
        notes = ""
        correlation = None
        map_difference = None
        try:
            original_rgb, pixels = read_model_input(row["path"])
            gradcam_map, gradcam_index, gradcam_confidence = generate_gradcam(
                tf,
                gradcam_model,
                pixels,
            )
            plus_plus_map, plus_plus_index, plus_plus_confidence = (
                gradcam_plus_plus_heatmap(tf, gradcam_model, pixels)
            )
            if gradcam_index != plus_plus_index:
                raise ValueError(
                    "Grad-CAM and Grad-CAM++ selected different predicted classes."
                )
            predicted_class = CLASS_NAMES[gradcam_index]
            confidence = gradcam_confidence
            if predicted_class != row["previous_predicted_class"]:
                raise ValueError(
                    f"Current prediction {predicted_class} differs from the saved "
                    f"Grad-CAM result {row['previous_predicted_class']}."
                )
            if abs(gradcam_confidence - row["previous_confidence"]) > 1e-4:
                raise ValueError(
                    "Current prediction confidence differs from the saved Grad-CAM "
                    f"result by more than 0.0001 "
                    f"({gradcam_confidence:.8f} vs {row['previous_confidence']:.8f})."
                )

            gradcam_resized, gradcam_rgb = resize_heatmap(
                gradcam_map,
                original_rgb.width,
                original_rgb.height,
            )
            plus_plus_resized, plus_plus_rgb = resize_heatmap(
                plus_plus_map,
                original_rgb.width,
                original_rgb.height,
            )
            if (
                gradcam_resized.shape != plus_plus_resized.shape
                or gradcam_resized.shape != (original_rgb.height, original_rgb.width)
            ):
                raise ValueError("Heatmap dimensions do not match the original image.")

            save_method_visualization(
                plus_plus_path,
                original_rgb,
                plus_plus_rgb,
                actual_class,
                predicted_class,
                plus_plus_confidence,
            )
            save_comparison_visualization(
                comparison_path,
                original_rgb,
                gradcam_rgb,
                plus_plus_rgb,
                actual_class,
                predicted_class,
                gradcam_confidence,
            )
            correlation, map_difference = compare_heatmaps(
                gradcam_resized,
                plus_plus_resized,
            )
            if (
                plus_plus_index != gradcam_index
                or abs(plus_plus_confidence - gradcam_confidence) > 1e-5
            ):
                notes = "Prediction confidence differs slightly between graph calls."
        except (OSError, ValueError, RuntimeError, cv2.error) as error:
            status = f"failed: {error}"
            gradcam_status = "failed or not comparable"
            plus_plus_status = "failed or not comparable"
            notes = str(error)

        report_rows.append(
            {
                "Image filename": row["filename"],
                "Actual class": actual_class,
                "Predicted class": predicted_class,
                "Prediction confidence": (
                    f"{confidence:.8f}" if confidence is not None else ""
                ),
                "Grad-CAM output filename": row["previous_gradcam_image"].relative_to(
                    PROJECT_ROOT
                ).as_posix(),
                "Grad-CAM++ output filename": (
                    (
                        Path("explainability")
                        / "archive"
                        / "mobilenetv2"
                        / OUTPUT_DIRECTORY.name
                        / group_name
                        / gradcam_plus_plus_name
                    )
                    .as_posix()
                    if status == "success"
                    else ""
                ),
                "Comparison output filename": (
                    (
                        Path("explainability")
                        / "archive"
                        / "mobilenetv2"
                        / OUTPUT_DIRECTORY.name
                        / "comparison"
                        / comparison_name
                    )
                    .as_posix()
                    if status == "success"
                    else ""
                ),
                "Selection group": group_name,
                "Grad-CAM status": gradcam_status,
                "Grad-CAM++ status": plus_plus_status,
                "Grad-CAM/Grad-CAM++ correlation": (
                    f"{correlation:.6f}" if correlation is not None else ""
                ),
                "Heatmap mean absolute difference": (
                    f"{map_difference:.6f}" if map_difference is not None else ""
                ),
                "Status": status,
                "Notes": notes,
            }
        )
        if status == "success":
            comparisons.append(
                {
                    "filename": row["filename"],
                    "selection_group": group_name,
                    "correlation": correlation,
                    "mean_absolute_difference": map_difference,
                }
            )
    return report_rows, comparisons


def write_results_csv(csv_path, rows):
    """Write per-image predictions, outputs, and explanation status."""
    columns = [
        "Image filename",
        "Actual class",
        "Predicted class",
        "Prediction confidence",
        "Grad-CAM output filename",
        "Grad-CAM++ output filename",
        "Comparison output filename",
        "Selection group",
        "Grad-CAM status",
        "Grad-CAM++ status",
        "Grad-CAM/Grad-CAM++ correlation",
        "Heatmap mean absolute difference",
        "Status",
        "Notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_comparison_report(
    report_path,
    rows,
    comparisons,
    conv_layer_name,
    activation_layer_name,
):
    """Report success counts, observed map differences, and limitations."""
    successes = [row for row in rows if row["Status"] == "success"]
    failures = [row for row in rows if row["Status"] != "success"]
    group_counts = Counter(row["Selection group"] for row in rows)
    valid_correlations = [
        item["correlation"]
        for item in comparisons
        if item["correlation"] is not None
    ]
    differences = [
        item["mean_absolute_difference"]
        for item in comparisons
        if item["mean_absolute_difference"] is not None
    ]

    lines = [
        "MAIZE DISEASE GRAD-CAM / GRAD-CAM++ COMPARISON",
        "=" * 49,
        "",
        f"Analysis date: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "Model: models/mobilenetv2/model.keras (inference only; weights unchanged)",
        f"Input: RGB, {IMAGE_SIZE[0]} x {IMAGE_SIZE[1]}, using the existing "
        "Grad-CAM training-matched preprocessing and embedded MobileNetV2 rescaling.",
        f"Target final convolution: {conv_layer_name}; activation map: "
        f"{activation_layer_name}.",
        f"Image selection source: {SELECTION_CSV.relative_to(PROJECT_ROOT).as_posix()}",
        "",
        "Image and heatmap counts",
        "------------------------",
        f"Images selected from existing Grad-CAM run: {len(rows)}",
        f"Images successfully processed by both methods: {len(successes)}",
        f"Images with failed or non-comparable heatmaps: {len(failures)}",
    ]
    for group_name in EXPECTED_SELECTIONS:
        success_count = sum(
            row["Selection group"] == group_name and row["Status"] == "success"
            for row in rows
        )
        lines.append(
            f"{group_name}: {group_counts[group_name]} selected, "
            f"{success_count} successfully compared"
        )
    if failures:
        lines.extend(["", "Failed images", "--------------"])
        lines.extend(
            f"- {row['Image filename']} ({row['Selection group']}): {row['Status']}"
            for row in failures
        )
    else:
        lines.append("Failed images: none.")

    lines.extend(["", "Observed map differences", "------------------------"])
    if valid_correlations and differences:
        lines.append(
            f"Mean Pearson correlation between normalized resized maps: "
            f"{np.mean(valid_correlations):.4f} "
            f"(n={len(valid_correlations)}; constant-map cases excluded)"
        )
        lines.append(
            f"Mean absolute difference between normalized resized maps: "
            f"{np.mean(differences):.4f} (n={len(differences)})"
        )
        lines.append(
            "These values summarize spatial differences only; they do not measure "
            "explanation correctness or establish that either method is superior."
        )
    else:
        lines.append(
            "No comparable heatmap pairs were available for aggregate spatial statistics."
        )

    lines.extend(
        [
            "",
            "Method details",
            "--------------",
            "Grad-CAM uses the spatial average of first-order class-score gradients "
            "as channel weights.",
            "Grad-CAM++ uses gradient-power alpha weights (squared gradients in the "
            "numerator and squared/cubed-gradient terms in the denominator), then "
            "combines positive gradients with feature-map activations.",
            "Both methods target the model's predicted class and use the same normalized "
            "7x7 activation maps, OpenCV JET color scale, original image, and 0.45 "
            "overlay opacity.",
            "",
            "Limitations",
            "-----------",
            "Both methods are approximate class-activation explanations. Heatmaps do "
            "not prove a prediction is correct, identify a disease cause, or establish "
            "causal feature importance.",
            "The final feature map is only 7x7 before resizing, so displayed regions "
            "have limited spatial detail.",
            "Grad-CAM++'s gradient-power weighting is a local attribution heuristic; "
            "map detail or concentration alone is not evidence of better explanations.",
            "This comparison reuses the same model and the exact image filenames from "
            "the existing Grad-CAM report; it does not retrain or alter the test split.",
            "",
            "Output folders",
            "---------------",
            "explainability/archive/mobilenetv2/gradcam_plus_plus/outputs/correct_predictions/",
            "explainability/archive/mobilenetv2/gradcam_plus_plus/outputs/blight_to_gray_leaf_spot/",
            "explainability/archive/mobilenetv2/gradcam_plus_plus/outputs/gray_leaf_spot_to_blight/",
            "explainability/archive/mobilenetv2/gradcam_plus_plus/outputs/comparison/",
            f"Per-image report: explainability/archive/mobilenetv2/gradcam_plus_plus/outputs/{RESULTS_CSV_NAME}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_analysis(overwrite=False):
    """Load existing files and generate Grad-CAM++ comparisons only."""
    model_path = MODEL_DIRECTORY / "model.keras"
    mapping_path = MODEL_DIRECTORY / "class_mapping.json"
    test_directory = (
        PROJECT_ROOT / "datasets" / "maize_disease_processed" / "test"
    )
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model is missing: {model_path}")
    if not test_directory.is_dir():
        raise FileNotFoundError(f"Test dataset is missing: {test_directory}")
    if (OUTPUT_DIRECTORY.exists() or REPORT_PATH.exists()) and not overwrite:
        raise FileExistsError(
            "Grad-CAM++ output already exists and will not be overwritten. "
            "Use --overwrite to replace it."
        )

    load_class_mapping(mapping_path)
    selected_rows = load_selected_images(SELECTION_CSV, test_directory)

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required. Run with the Python environment used by the "
            "saved .keras model."
        ) from error
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(f"Could not load saved model {model_path}: {error}") from error

    gradcam_model, conv_layer_name, activation_layer_name = build_gradcam_model(
        tf,
        model,
    )
    output_directory_parent = OUTPUT_DIRECTORY.parent
    output_directory_parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix=".gradcam_plus_plus_staging_",
        dir=output_directory_parent,
    ) as temporary_parent:
        temporary_parent = Path(temporary_parent)
        staging_output = temporary_parent / OUTPUT_DIRECTORY.name
        staging_output.mkdir()
        staging_report = temporary_parent / REPORT_PATH.name

        rows, comparisons = generate_outputs(
            tf,
            gradcam_model,
            selected_rows,
            staging_output,
        )
        write_results_csv(staging_output / RESULTS_CSV_NAME, rows)
        write_comparison_report(
            staging_report,
            rows,
            comparisons,
            conv_layer_name,
            activation_layer_name,
        )

        output_backup = temporary_parent / "previous_gradcam_plus_plus"
        report_backup = temporary_parent / "previous_gradcam_comparison_report.txt"
        output_backed_up = False
        report_backed_up = False
        try:
            if OUTPUT_DIRECTORY.exists():
                OUTPUT_DIRECTORY.rename(output_backup)
                output_backed_up = True
            if REPORT_PATH.exists():
                REPORT_PATH.rename(report_backup)
                report_backed_up = True
            staging_output.rename(OUTPUT_DIRECTORY)
            staging_report.rename(REPORT_PATH)
        except OSError:
            if OUTPUT_DIRECTORY.exists():
                shutil.rmtree(OUTPUT_DIRECTORY)
            if REPORT_PATH.exists():
                REPORT_PATH.unlink()
            if output_backed_up:
                output_backup.rename(OUTPUT_DIRECTORY)
            if report_backed_up:
                report_backup.rename(REPORT_PATH)
            raise

    return rows, OUTPUT_DIRECTORY, REPORT_PATH


def main():
    """Run the Grad-CAM++ comparison from the project root."""
    parser = argparse.ArgumentParser(
        description="Compare Grad-CAM++ and Grad-CAM on saved test selections."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace the model-specific Grad-CAM++ outputs and comparison report.",
    )
    arguments = parser.parse_args()
    try:
        rows, output_directory, report_path = run_analysis(
            overwrite=arguments.overwrite
        )
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"Grad-CAM++ analysis failed: {error}", file=sys.stderr)
        return 1

    success_count = sum(row["Status"] == "success" for row in rows)
    print(f"Selected images: {len(rows)}")
    print(f"Successfully compared heatmaps: {success_count}")
    print(f"Grad-CAM++ outputs: {output_directory.resolve()}")
    print(f"Comparison report: {report_path.resolve()}")
    print("No model training or weight changes were performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
