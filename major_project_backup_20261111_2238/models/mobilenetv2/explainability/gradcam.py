"""Generate Grad-CAM explanations for selected maize disease test images."""

import argparse
import csv
import json
import shutil
import sys
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps


IMAGE_SIZE = (224, 224)
CLASS_NAMES = ("Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy")
EXPECTED_CONFUSIONS = {
    ("Blight", "Gray_Leaf_Spot"): 13,
    ("Gray_Leaf_Spot", "Blight"): 7,
}
MISCLASSIFICATION_CSV = "misclassified_predictions.csv"
OUTPUT_DIRECTORY_NAME = "gradcam"
REPORT_NAME = "gradcam_analysis.txt"
CSV_NAME = "gradcam_report.csv"


def load_class_names(mapping_path):
    """Load and validate class order used by the saved model."""
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Class mapping is missing: {mapping_path}")
    try:
        mapping_data = json.loads(mapping_path.read_text(encoding="utf-8"))
        class_to_index = mapping_data["class_to_index"]
        index_to_class = mapping_data["index_to_class"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Could not read class mapping {mapping_path}: {error}") from error

    expected_mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    expected_inverse = {
        str(index): name for index, name in enumerate(CLASS_NAMES)
    }
    if class_to_index != expected_mapping or index_to_class != expected_inverse:
        raise ValueError(
            "The saved class mapping differs from the training class order. "
            f"Expected class_to_index={expected_mapping} and "
            f"index_to_class={expected_inverse}."
        )
    return tuple(index_to_class[str(index)] for index in range(len(CLASS_NAMES)))


def load_misclassification_rows(csv_path, test_directory, class_names):
    """Reuse the saved misclassification CSV for the required confused classes."""
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Required misclassification CSV is missing: {csv_path}"
        )

    selected_rows = []
    counts = Counter()
    required_columns = {
        "Image filename",
        "Actual class",
        "Predicted class",
        "Prediction confidence",
    }
    with csv_path.open(newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(
                f"Misclassification CSV must contain columns: "
                f"{sorted(required_columns)}"
            )
        for row in reader:
            actual_class = row["Actual class"].strip()
            predicted_class = row["Predicted class"].strip()
            pair = (actual_class, predicted_class)
            if pair not in EXPECTED_CONFUSIONS:
                continue
            if actual_class not in class_names or predicted_class not in class_names:
                raise ValueError(f"Unexpected class names in misclassification CSV: {pair}")
            try:
                confidence = float(row["Prediction confidence"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid confidence value in misclassification CSV for "
                    f"{row['Image filename']!r}."
                ) from error
            if not np.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError(
                    f"Confidence is outside [0, 1] for {row['Image filename']!r}."
                )

            image_path = test_directory / actual_class / row["Image filename"].strip()
            if not image_path.is_file():
                raise FileNotFoundError(
                    f"Misclassification CSV image does not exist in the test set: "
                    f"{image_path}"
                )
            selected_rows.append(
                {
                    "path": image_path,
                    "actual_class": actual_class,
                    "csv_predicted_class": predicted_class,
                    "csv_confidence": confidence,
                    "selection_group": (
                        "blight_to_gray_leaf_spot"
                        if pair == ("Blight", "Gray_Leaf_Spot")
                        else "gray_leaf_spot_to_blight"
                    ),
                }
            )
            counts[pair] += 1

    for pair, expected_count in EXPECTED_CONFUSIONS.items():
        if counts[pair] != expected_count:
            raise ValueError(
                f"The existing misclassification CSV contains {counts[pair]} "
                f"{pair[0]} -> {pair[1]} records; expected {expected_count}. "
                "No test images or predictions were changed."
            )

    unique_paths = [row["path"].resolve() for row in selected_rows]
    if len(unique_paths) != len(set(unique_paths)):
        raise ValueError("The misclassification CSV lists a required image more than once.")
    return selected_rows, counts


def list_test_images(test_directory, class_names, excluded_paths):
    """List readable test images, excluding known errors for correct examples."""
    test_records = {}
    errors = []
    for class_name in class_names:
        class_directory = test_directory / class_name
        if not class_directory.is_dir():
            errors.append(f"Missing test class directory: {class_directory}")
            continue
        candidates = sorted(
            (path for path in class_directory.rglob("*") if path.is_file()),
            key=lambda path: str(path).casefold(),
        )
        readable = []
        for image_path in candidates:
            if image_path.resolve() in excluded_paths:
                continue
            try:
                with Image.open(image_path) as image:
                    image.verify()
                with Image.open(image_path) as image:
                    image.load()
            except (OSError, ValueError, SyntaxError) as error:
                errors.append(f"Unreadable test image {image_path}: {error}")
                continue
            readable.append(
                {
                    "path": image_path,
                    "actual_class": class_name,
                    "csv_predicted_class": "",
                    "csv_confidence": None,
                    "selection_group": "correct_predictions",
                }
            )
        test_records[class_name] = readable
    if errors:
        raise ValueError(
            "Test-set validation found invalid inputs; no files were skipped:\n  - "
            + "\n  - ".join(errors)
        )
    return test_records


def read_model_input(image_path):
    """Match training input: EXIF-corrected RGB at 224x224, raw [0, 255] values."""
    with Image.open(image_path) as source:
        original_rgb = ImageOps.exif_transpose(source).convert("RGB")
        resized_rgb = original_rgb.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        pixels = np.asarray(resized_rgb, dtype=np.float32)
    return original_rgb, pixels


def build_gradcam_model(tf, model):
    """Expose MobileNetV2's final convolutional activation and model scores."""
    try:
        base_model = model.get_layer("mobilenetv2_1.00_224")
        preprocessing_layer = model.get_layer("mobilenetv2_preprocessing")
        pooling_layer = model.get_layer("global_average_pooling")
        dropout_layer = model.get_layer("head_dropout")
        classifier_layer = model.get_layer("class_probabilities")
        last_conv_layer = base_model.get_layer("Conv_1")
        activation_layer = base_model.get_layer("out_relu")
    except ValueError as error:
        raise ValueError(
            "The saved model does not have the expected MobileNetV2 layers "
            "(Conv_1 -> Conv_1_bn -> out_relu and the saved preprocessing/head)."
        ) from error

    if tuple(model.input_shape[1:]) != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise ValueError(
            f"Expected model input (224, 224, 3), found {model.input_shape}."
        )
    if tuple(model.output_shape[1:]) != (len(CLASS_NAMES),):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} model outputs, found {model.output_shape}."
        )
    if len(activation_layer.output.shape) != 4:
        raise ValueError(
            f"Grad-CAM layer {activation_layer.name} is not a spatial feature map."
        )

    base_feature_model = tf.keras.Model(
        inputs=base_model.input,
        outputs=[activation_layer.output, base_model.output],
        name="mobilenetv2_gradcam_features",
    )
    preprocessed_input = tf.keras.Model(
        inputs=model.inputs,
        outputs=preprocessing_layer.output,
        name="saved_model_input_preprocessing",
    )(model.inputs)
    conv_activations, base_output = base_feature_model(
        preprocessed_input,
        training=False,
    )
    head_output = pooling_layer(base_output)
    head_output = dropout_layer(head_output, training=False)
    predictions = classifier_layer(head_output)
    gradcam_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[conv_activations, predictions],
        name="maize_mobilenetv2_gradcam",
    )
    return gradcam_model, last_conv_layer.name, activation_layer.name


def generate_gradcam(tf, gradcam_model, image_pixels):
    """Compute a normalized class-specific Grad-CAM heatmap for one image."""
    input_batch = tf.convert_to_tensor(image_pixels[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        activations, probabilities = gradcam_model(input_batch, training=False)
        predicted_index = int(tf.argmax(probabilities[0]).numpy())
        class_score = probabilities[:, predicted_index]

    gradients = tape.gradient(class_score, activations)
    if gradients is None:
        raise ValueError("Gradients are unavailable for the selected activation map.")
    if not bool(tf.reduce_all(tf.math.is_finite(gradients)).numpy()):
        raise ValueError("Gradients contain NaN or infinite values.")

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))
    weighted_activations = activations[0] * channel_weights[0][tf.newaxis, tf.newaxis, :]
    heatmap = tf.reduce_sum(weighted_activations, axis=-1)
    heatmap = tf.nn.relu(heatmap)
    maximum = float(tf.reduce_max(heatmap).numpy())
    if not np.isfinite(maximum) or maximum <= 1e-12:
        raise ValueError("The Grad-CAM heatmap is empty or has no positive activation.")

    heatmap = heatmap / maximum
    heatmap_array = heatmap.numpy().astype(np.float32)
    if heatmap_array.ndim != 2 or not np.all(np.isfinite(heatmap_array)):
        raise ValueError("The normalized Grad-CAM heatmap is invalid.")

    return (
        heatmap_array,
        predicted_index,
        float(probabilities[0, predicted_index].numpy()),
    )


def resize_heatmap(heatmap, width, height):
    """Resize and colorize a Grad-CAM heatmap using OpenCV."""
    resized = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = np.clip(resized, 0.0, 1.0)
    colored_bgr = cv2.applyColorMap(
        np.uint8(resized * 255),
        cv2.COLORMAP_JET,
    )
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB)
    return resized, colored_rgb


def save_visualization(
    output_path,
    original_rgb,
    colored_heatmap,
    actual_class,
    predicted_class,
    confidence,
    status,
):
    """Save original, heatmap, and overlay panels with readable labels."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    original_array = np.asarray(original_rgb)
    axes[0].imshow(original_array)
    axes[0].set_title("Original image")
    axes[1].imshow(colored_heatmap)
    axes[1].set_title("Grad-CAM heatmap")
    overlay = cv2.addWeighted(original_array, 0.55, colored_heatmap, 0.45, 0)
    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap overlay")

    for axis in axes:
        axis.axis("off")

    figure.suptitle(
        f"Actual: {actual_class}  |  Predicted: {predicted_class} "
        f"({confidence:.1%})\nGrad-CAM status: {status}",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def output_subdirectory(selection_group):
    """Map each requested selection type to its output folder."""
    if selection_group == "correct_predictions":
        return "correct_predictions"
    if selection_group == "blight_to_gray_leaf_spot":
        return "blight_to_gray_leaf_spot"
    if selection_group == "gray_leaf_spot_to_blight":
        return "gray_leaf_spot_to_blight"
    raise ValueError(f"Unknown Grad-CAM selection group: {selection_group}")


def predict_and_render(tf, gradcam_model, records, class_names, staging_directory):
    """Run selected images, save visualizations, and collect report rows."""
    output_rows = []

    for record in records:
        image_path = record["path"]
        group_name = output_subdirectory(record["selection_group"])
        output_filename = f"{record['actual_class']}__{image_path.stem}.png"
        relative_output = Path(group_name) / output_filename
        visualization_path = staging_directory / relative_output
        visualization_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            original_rgb, image_pixels = read_model_input(image_path)
            heatmap, predicted_index, confidence = generate_gradcam(
                tf,
                gradcam_model,
                image_pixels,
            )
            predicted_class = class_names[predicted_index]
            actual_index = class_names.index(record["actual_class"])
            if record["selection_group"] == "correct_predictions":
                status = "success"
                notes = (
                    ""
                    if predicted_class == record["actual_class"]
                    else "Selected candidate was not classified as its actual class "
                    "during Grad-CAM inference."
                )
            else:
                csv_prediction = record["csv_predicted_class"]
                if predicted_class != csv_prediction:
                    notes = (
                        f"Model prediction {predicted_class} differs from existing "
                        f"misclassification CSV ({csv_prediction})."
                    )
                else:
                    notes = ""
                if predicted_index == actual_index:
                    notes = (notes + " " if notes else "") + (
                        "Model predicted the actual class; CSV marks it misclassified."
                    )

                status = "success"

            resized_heatmap, colored_heatmap = resize_heatmap(
                heatmap,
                original_rgb.width,
                original_rgb.height,
            )
            if resized_heatmap.shape != (original_rgb.height, original_rgb.width):
                raise ValueError("Resized heatmap does not match the original image size.")
            save_visualization(
                visualization_path,
                original_rgb,
                colored_heatmap,
                record["actual_class"],
                predicted_class,
                confidence,
                status,
            )
        except (OSError, ValueError, RuntimeError, cv2.error) as error:
            original_rgb = None
            predicted_class = ""
            confidence = None
            status = f"failed: {error}"
            notes = str(error)

        output_rows.append(
            {
                "Image filename": image_path.name,
                "Actual class": record["actual_class"],
                "Predicted class": predicted_class,
                "Prediction confidence": (
                    f"{confidence:.8f}" if confidence is not None else ""
                ),
                "Grad-CAM output filename": str(relative_output),
                "Heatmap generation status": status,
                "Selection group": record["selection_group"],
                "Existing CSV predicted class": record["csv_predicted_class"],
                "Existing CSV confidence": (
                    f"{record['csv_confidence']:.8f}"
                    if record["csv_confidence"] is not None
                    else ""
                ),
                "Notes": notes,
            }
        )
    return output_rows


def write_results_csv(csv_path, rows):
    """Save prediction details and heatmap status."""
    fieldnames = [
        "Image filename",
        "Actual class",
        "Predicted class",
        "Prediction confidence",
        "Grad-CAM output filename",
        "Heatmap generation status",
        "Selection group",
        "Existing CSV predicted class",
        "Existing CSV confidence",
        "Notes",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_report(
    report_path,
    rows,
    selected_misclassifications,
    expected_csv_confusions,
    conv_layer_name,
    activation_layer_name,
):
    """Write success counts, target confusion details, and comparison notes."""
    statuses = Counter(
        "success" if row["Heatmap generation status"] == "success" else "failed"
        for row in rows
    )
    correct_by_class = Counter(
        row["Actual class"]
        for row in rows
        if row["Selection group"] == "correct_predictions"
        and row["Heatmap generation status"] == "success"
        and row["Predicted class"] == row["Actual class"]
    )
    observed_confusions = Counter(
        (
            row["Actual class"],
            row["Existing CSV predicted class"],
        )
        for row in selected_misclassifications
    )
    prediction_differences = [
        row for row in rows
        if row["Existing CSV predicted class"]
        and row["Predicted class"] != row["Existing CSV predicted class"]
    ]
    recovered_errors = [
        row for row in rows
        if row["Existing CSV predicted class"]
        and row["Predicted class"] == row["Actual class"]
    ]
    confidence_differences = []
    for row in selected_misclassifications:
        if not row["Prediction confidence"] or not row["Existing CSV confidence"]:
            continue
        current_confidence = float(row["Prediction confidence"])
        csv_confidence = float(row["Existing CSV confidence"])
        if abs(current_confidence - csv_confidence) > 1e-4:
            confidence_differences.append(
                (row, current_confidence, csv_confidence)
            )
    failed_rows = [
        row for row in rows if row["Heatmap generation status"] != "success"
    ]

    lines = [
        "MAIZE DISEASE GRAD-CAM ANALYSIS",
        "=" * 31,
        "",
        f"Analysis date: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        f"Saved model: models/mobilenetv2/model.keras",
        "The trained model was loaded for inference only; no training or weight "
        "updates were performed.",
        "Test images were read without modification; the test split was unchanged.",
        "Image preprocessing: EXIF orientation correction, RGB conversion, resize to "
        "224x224, float32 [0, 255], then the model's embedded "
        "mobilenetv2_preprocessing Rescaling layer.",
        f"Grad-CAM final Conv2D layer: {conv_layer_name}",
        f"Grad-CAM activation tensor: {activation_layer_name}",
        "Grad-CAM target score: current model's predicted class probability.",
        "",
        "Generated visualizations",
        "------------------------",
        f"Selected images: {len(rows)}",
        f"Successful heatmaps: {statuses['success']}",
        f"Failed heatmaps: {statuses['failed']}",
        "Correct examples successfully generated per class:",
    ]
    for class_name in CLASS_NAMES:
        lines.append(f"  {class_name}: {correct_by_class[class_name]}")
    lines.extend(["", "Requested misclassification groups", "----------------------------------"])
    for pair, expected_count in EXPECTED_CONFUSIONS.items():
        observed_count = observed_confusions[pair]
        csv_count = expected_csv_confusions[pair]
        rendered_count = sum(
            1
            for row in rows
            if row["Selection group"] == (
                "blight_to_gray_leaf_spot"
                if pair == ("Blight", "Gray_Leaf_Spot")
                else "gray_leaf_spot_to_blight"
            )
            and row["Heatmap generation status"] == "success"
        )
        lines.append(
            f"{pair[0]} -> {pair[1]}: {observed_count} selected from existing CSV "
            f"(CSV count {csv_count}; expected {expected_count}), "
            f"{rendered_count} visualizations succeeded"
        )
    lines.extend(
        [
            "",
            "Comparison with existing misclassification CSV",
            "------------------------------------------------",
            f"Model predictions differing from CSV: {len(prediction_differences)}",
            f"CSV-listed errors now predicted as actual class: {len(recovered_errors)}",
            f"Prediction confidence differences greater than 0.0001: "
            f"{len(confidence_differences)}",
        ]
    )
    if prediction_differences:
        lines.append("Prediction differences:")
        lines.extend(
            f"  - {row['Image filename']}: CSV={row['Existing CSV predicted class']}, "
            f"model={row['Predicted class'] or 'unavailable'}"
            for row in prediction_differences
        )
    if recovered_errors:
        lines.append("CSV-listed images predicted correctly on this run:")
        lines.extend(
            f"  - {row['Image filename']}: actual={row['Actual class']}, "
            f"model={row['Predicted class']}"
            for row in recovered_errors
        )
    if confidence_differences:
        lines.append("Prediction confidence differences:")
        lines.extend(
            f"  - {row['Image filename']}: CSV={old:.8f}, model={new:.8f}"
            for row, new, old in confidence_differences
        )
    if failed_rows:
        lines.append("Failed image explanations:")
        lines.extend(
            f"  - {row['Image filename']}: {row['Heatmap generation status']}"
            for row in failed_rows
        )
    else:
        lines.append(
            "All requested examples were explained successfully; no unreadable "
            "images or invalid heatmaps were encountered."
        )
    lines.extend(
        [
            "",
            "Interpretation limits",
            "---------------------",
            "Grad-CAM highlights image regions that contributed to the chosen class "
            "score. It is an approximate visual explanation, not a disease detector, "
            "proof of correctness, or evidence of biological causation.",
            "A highlighted region does not establish that the predicted class is "
            "correct. Misclassified images are shown with both actual and predicted "
            "labels for comparison.",
            "This analysis makes no assumption that the folder label 'Blight' means "
            "Northern Leaf Blight.",
            "",
            "Saved output folders",
            "--------------------",
            "models/mobilenetv2/explainability/outputs/gradcam/correct_predictions/",
            "models/mobilenetv2/explainability/outputs/gradcam/blight_to_gray_leaf_spot/",
            "models/mobilenetv2/explainability/outputs/gradcam/gray_leaf_spot_to_blight/",
            f"CSV: models/mobilenetv2/explainability/outputs/gradcam/{CSV_NAME}",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_correct_examples(tf, gradcam_model, test_directory, class_names, excluded_paths):
    """Find one model-correct test image per class without using known CSV errors."""
    selected = []
    search_records = list_test_images(
        test_directory,
        class_names,
        excluded_paths,
    )
    for class_name in class_names:
        found_record = None
        for record in search_records[class_name]:
            try:
                _, pixels = read_model_input(record["path"])
                input_tensor = tf.convert_to_tensor(pixels[np.newaxis, ...], dtype=tf.float32)
                _, probabilities = gradcam_model(input_tensor, training=False)
                predicted_index = int(tf.argmax(probabilities[0]).numpy())
                if class_names[predicted_index] == class_name:
                    found_record = record
                    break
            except (OSError, ValueError, RuntimeError) as error:
                raise ValueError(
                    f"Could not find a correct example for {class_name}; failed on "
                    f"{record['path']}: {error}"
                ) from error
        if found_record is None:
            raise ValueError(
                f"No correctly classified test image was found for {class_name}."
            )
        selected.append(found_record)
    return selected


def run_analysis(project_root, overwrite=False):
    """Validate artifacts, generate selected Grad-CAM images, and write reports."""
    model_directory = project_root / "models" / "mobilenetv2"
    model_path = model_directory / "model.keras"
    mapping_path = model_directory / "class_mapping.json"
    test_directory = project_root / "datasets" / "maize_disease_processed" / "test"
    evaluation_directory = model_directory / "evaluation"
    existing_misclassification_csv = (
        evaluation_directory / MISCLASSIFICATION_CSV
    )
    output_directory = (
        model_directory / "explainability" / "outputs" / OUTPUT_DIRECTORY_NAME
    )
    report_path = evaluation_directory / REPORT_NAME
    csv_path = output_directory / CSV_NAME

    if not model_path.is_file():
        raise FileNotFoundError(f"Saved model is missing: {model_path}")
    if (output_directory.exists() or report_path.exists()) and not overwrite:
        raise FileExistsError(
            "Grad-CAM output already exists and will not be overwritten. "
            "Use --overwrite to replace the previous Grad-CAM results."
        )
    if not test_directory.is_dir():
        raise FileNotFoundError(f"Test dataset is missing: {test_directory}")

    class_names = load_class_names(mapping_path)
    misclassified_records, csv_confusions = load_misclassification_rows(
        existing_misclassification_csv,
        test_directory,
        class_names,
    )
    misclassified_paths = {
        record["path"].resolve() for record in misclassified_records
    }

    try:
        import tensorflow as tf
    except ImportError as error:
        raise RuntimeError(
            "TensorFlow is required. Run with the Python environment used to load "
            "the saved Keras model."
        ) from error

    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise RuntimeError(
            f"Could not load saved model {model_path}: {error}"
        ) from error
    gradcam_model, conv_layer_name, activation_layer_name = build_gradcam_model(
        tf,
        model,
    )

    correct_records = choose_correct_examples(
        tf,
        gradcam_model,
        test_directory,
        class_names,
        misclassified_paths,
    )
    selected_records = [*correct_records, *misclassified_records]

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    evaluation_directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".gradcam_staging_",
        dir=output_directory.parent,
    ) as temporary_parent:
        staging_directory = Path(temporary_parent) / OUTPUT_DIRECTORY_NAME
        staging_directory.mkdir()
        report_staging = Path(temporary_parent) / REPORT_NAME

        csv_rows = predict_and_render(
            tf,
            gradcam_model,
            selected_records,
            class_names,
            staging_directory,
        )
        write_results_csv(staging_directory / CSV_NAME, csv_rows)
        selected_misclassification_rows = [
            row
            for row in csv_rows
            if row["Selection group"]
            in ("blight_to_gray_leaf_spot", "gray_leaf_spot_to_blight")
        ]
        write_summary_report(
            report_staging,
            csv_rows,
            selected_misclassification_rows,
            csv_confusions,
            conv_layer_name,
            activation_layer_name,
        )

        # Keep existing results recoverable until both new outputs are ready.
        output_backup = Path(temporary_parent) / "previous_gradcam"
        report_backup = Path(temporary_parent) / "previous_gradcam_report.txt"
        output_was_backed_up = False
        report_was_backed_up = False
        try:
            if output_directory.exists():
                output_directory.rename(output_backup)
                output_was_backed_up = True
            if report_path.exists():
                report_path.rename(report_backup)
                report_was_backed_up = True

            staging_directory.rename(output_directory)
            report_staging.rename(report_path)
        except OSError:
            if output_directory.exists():
                shutil.rmtree(output_directory)
            if report_path.exists():
                report_path.unlink()
            if output_was_backed_up:
                output_backup.rename(output_directory)
            if report_was_backed_up:
                report_backup.rename(report_path)
            raise

    return csv_rows, report_path, csv_path, output_directory, csv_confusions


def main():
    """Run Grad-CAM from the project root without training or editing model files."""
    project_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(
        description="Generate Grad-CAM visualizations for selected test images."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Replace the model-specific Grad-CAM outputs and analysis report."
        ),
    )
    arguments = parser.parse_args()

    try:
        rows, report_path, csv_path, output_directory, expected_counts = run_analysis(
            project_root,
            overwrite=arguments.overwrite,
        )
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        ValueError,
    ) as error:
        print(f"Grad-CAM analysis failed: {error}", file=sys.stderr)
        return 1

    success_count = sum(
        row["Heatmap generation status"] == "success" for row in rows
    )
    print(f"Selected images: {len(rows)}")
    print(f"Successfully generated heatmaps: {success_count}")
    print("Selected misclassifications from the existing CSV:")
    for pair, count in expected_counts.items():
        print(f"  {pair[0]} -> {pair[1]}: {count}")
    print(f"Visualizations: {output_directory.resolve()}")
    print(f"CSV report: {csv_path.resolve()}")
    print(f"Summary report: {report_path.resolve()}")
    print("No model training was performed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
