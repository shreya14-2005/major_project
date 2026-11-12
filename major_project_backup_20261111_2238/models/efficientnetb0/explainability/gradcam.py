"""Generate separate EfficientNetB0 Grad-CAM visualizations from test predictions."""

import argparse
import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
from PIL import Image, ImageOps
from models.efficientnetb0.training.common import (
    CLASS_NAMES,
    IMAGE_SIZE,
    PROJECT_ROOT as DATA_PROJECT_ROOT,
    read_rgb_pixels,
)


def load_predictions(csv_path):
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"Run EfficientNetB0 evaluation before Grad-CAM: {csv_path}"
        )
    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if len(rows) != 628:
        raise ValueError(f"Expected 628 saved test predictions, found {len(rows)}.")
    for row in rows:
        if row["Actual class"] not in CLASS_NAMES or row["Predicted class"] not in CLASS_NAMES:
            raise ValueError(f"Unknown class label in predictions: {row}")
    return rows


def build_gradcam_model(tf, model):
    try:
        base_model = model.get_layer("efficientnetb0")
        conv_layer = base_model.get_layer("top_conv")
        activation_layer = base_model.get_layer("top_activation")
        augmentation_layers = (
            model.get_layer("training_horizontal_flip"),
            model.get_layer("training_rotation"),
            model.get_layer("training_zoom"),
        )
        pooling_layer = model.get_layer("global_average_pooling")
        dropout_layer = model.get_layer("head_dropout")
        classifier_layer = model.get_layer("class_probabilities")
    except ValueError as error:
        raise ValueError(
            "The EfficientNetB0 model lacks the expected top_activation/head layers."
        ) from error

    feature_model = tf.keras.Model(
        base_model.input,
        [activation_layer.output, base_model.output],
        name="efficientnetb0_gradcam_features",
    )
    features = model.input
    for layer in augmentation_layers:
        features = layer(features, training=False)
    activations, backbone_output = feature_model(features, training=False)
    head = pooling_layer(backbone_output)
    head = dropout_layer(head, training=False)
    predictions = classifier_layer(head)
    gradcam_model = tf.keras.Model(
        model.input,
        [activations, predictions],
        name="maize_efficientnetb0_gradcam",
    )
    return gradcam_model, conv_layer.name, activation_layer.name


def generate_gradcam(tf, gradcam_model, pixels):
    """Compute a normalized class-specific Grad-CAM heatmap for one image."""
    inputs = tf.convert_to_tensor(pixels[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        activations, probabilities = gradcam_model(inputs, training=False)
        class_index = int(tf.argmax(probabilities[0]).numpy())
        score = probabilities[:, class_index]
    gradients = tape.gradient(score, activations)
    if gradients is None or not bool(tf.reduce_all(tf.math.is_finite(gradients)).numpy()):
        raise ValueError("Grad-CAM gradients are missing or non-finite.")
    weights = tf.reduce_mean(gradients, axis=(1, 2))
    heatmap = tf.nn.relu(
        tf.reduce_sum(activations[0] * weights[0][tf.newaxis, tf.newaxis, :], axis=-1)
    )
    maximum = float(tf.reduce_max(heatmap).numpy())
    if not np.isfinite(maximum) or maximum <= 1e-12:
        raise ValueError("Grad-CAM produced an empty heatmap.")
    return (
        np.asarray(heatmap.numpy() / maximum, dtype=np.float32),
        class_index,
        float(probabilities[0, class_index].numpy()),
    )


def gradcam_plus_plus_heatmap(tf, gradcam_model, image_pixels):
    """Compute normalized Grad-CAM++ using gradient-power alpha weights."""
    inputs = tf.convert_to_tensor(
        image_pixels[np.newaxis, ...],
        dtype=tf.float32,
    )
    with tf.GradientTape() as tape:
        activations, probabilities = gradcam_model(inputs, training=False)
        predicted_index = int(tf.argmax(probabilities[0]).numpy())
        class_score = probabilities[:, predicted_index]

    gradients = tape.gradient(class_score, activations)
    if gradients is None:
        raise ValueError("Grad-CAM++ gradients are unavailable.")
    if not bool(tf.reduce_all(tf.math.is_finite(gradients)).numpy()):
        raise ValueError("Grad-CAM++ gradients contain NaN or infinite values.")

    feature_maps = activations[0]
    first_gradients = gradients[0]
    squared_gradients = tf.square(first_gradients)
    cubed_gradients = squared_gradients * first_gradients
    third_order_term = tf.reduce_sum(
        feature_maps * cubed_gradients,
        axis=(0, 1),
    )
    denominator = (
        2.0 * squared_gradients
        + third_order_term[tf.newaxis, tf.newaxis, :]
    )
    alphas = squared_gradients / tf.maximum(
        denominator,
        tf.constant(1e-8, dtype=denominator.dtype),
    )
    alphas = tf.where(
        tf.math.is_finite(alphas),
        alphas,
        tf.zeros_like(alphas),
    )
    channel_weights = tf.reduce_sum(
        alphas * tf.nn.relu(first_gradients),
        axis=(0, 1),
    )
    heatmap = tf.nn.relu(
        tf.reduce_sum(
            feature_maps * channel_weights[tf.newaxis, tf.newaxis, :],
            axis=-1,
        )
    )
    maximum = float(tf.reduce_max(heatmap).numpy())
    if not np.isfinite(maximum) or maximum <= 1e-12:
        raise ValueError("Grad-CAM++ produced an empty heatmap.")
    heatmap_array = np.asarray(heatmap.numpy() / maximum, dtype=np.float32)
    if heatmap_array.ndim != 2 or not np.all(np.isfinite(heatmap_array)):
        raise ValueError("Grad-CAM++ produced an invalid heatmap.")
    return (
        heatmap_array,
        predicted_index,
        float(probabilities[0, predicted_index].numpy()),
    )


def read_model_input(image_path):
    """Match EfficientNetB0 training: EXIF-corrected RGB, resized, raw [0, 255]."""
    with Image.open(image_path) as source:
        original_rgb = ImageOps.exif_transpose(source).convert("RGB")
        resized_rgb = original_rgb.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        pixels = np.asarray(resized_rgb, dtype=np.float32)
    return original_rgb, pixels


def resize_heatmap(heatmap, width, height):
    """Resize and colorize a normalized heatmap for display."""
    resized = cv2.resize(
        heatmap,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    resized = np.clip(resized, 0.0, 1.0)
    colored = cv2.applyColorMap(
        np.uint8(resized * 255),
        cv2.COLORMAP_JET,
    )
    return resized, cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def render(output_path, image_path, actual, predicted, confidence, heatmap):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with Image.open(image_path) as image:
        original = ImageOps.exif_transpose(image).convert("RGB")
    pixels = np.asarray(original)
    resized = cv2.resize(
        heatmap,
        (pixels.shape[1], pixels.shape[0]),
        interpolation=cv2.INTER_LINEAR,
    )
    colored = cv2.applyColorMap(
        np.uint8(np.clip(resized, 0, 1) * 255),
        cv2.COLORMAP_JET,
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(pixels, 0.55, colored, 0.45, 0)
    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(pixels)
    axes[0].set_title("Original image")
    axes[1].imshow(colored)
    axes[1].set_title("Grad-CAM heatmap")
    axes[2].imshow(overlay)
    axes[2].set_title("Heatmap overlay")
    for axis in axes:
        axis.axis("off")
    figure.suptitle(
        f"Actual: {actual} | Predicted: {predicted} ({confidence:.1%})"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def generate():
    import tensorflow as tf

    model_directory = PROJECT_ROOT / "models" / "efficientnetb0"
    model_path = model_directory / "model.keras"
    mapping_path = model_directory / "class_mapping.json"
    predictions_path = model_directory / "evaluation" / "test_predictions.csv"
    output_directory = model_directory / "explainability" / "outputs" / "gradcam"
    if output_directory.exists():
        raise FileExistsError(f"Grad-CAM outputs already exist: {output_directory}")
    if not mapping_path.is_file():
        raise FileNotFoundError(f"EfficientNetB0 class map is missing: {mapping_path}")
    rows = load_predictions(predictions_path)
    model = tf.keras.models.load_model(model_path, compile=False)
    gradcam_model, _, _ = build_gradcam_model(tf, model)

    selected = {}
    for class_name in CLASS_NAMES:
        first_record = next(row for row in rows if row["Actual class"] == class_name)
        selected[first_record["Source test image"]] = first_record
    gray_mistakes = [
        row
        for row in rows
        if row["Actual class"] == "Gray_Leaf_Spot"
        and row["Predicted class"] != "Gray_Leaf_Spot"
    ][:3]
    for row in gray_mistakes:
        selected[row["Source test image"]] = row

    output_directory.mkdir(parents=True)
    report_rows = []
    for index, row in enumerate(selected.values(), start=1):
        image_path = DATA_PROJECT_ROOT / Path(row["Source test image"])
        pixels = read_rgb_pixels(image_path)
        heatmap, predicted_index, confidence = generate_gradcam(
            tf, gradcam_model, pixels
        )
        predicted_class = CLASS_NAMES[predicted_index]
        if predicted_class != row["Predicted class"]:
            raise RuntimeError(
                f"Grad-CAM prediction disagrees with saved test prediction for {image_path}."
            )
        output_path = output_directory / f"{index:02d}_{image_path.stem}_gradcam.png"
        render(
            output_path,
            image_path,
            row["Actual class"],
            predicted_class,
            confidence,
            heatmap,
        )
        report_rows.append(
            {
                "Image filename": image_path.name,
                "Actual class": row["Actual class"],
                "Predicted class": predicted_class,
                "Confidence": f"{confidence:.6f}",
                "Grad-CAM output": output_path.name,
            }
        )
    with (output_directory / "gradcam_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=report_rows[0].keys())
        writer.writeheader()
        writer.writerows(report_rows)
    (model_directory / "evaluation" / "gradcam_notes.txt").write_text(
        "EfficientNetB0 Grad-CAM\n"
        "=======================\n\n"
        f"Generated explanations for {len(report_rows)} deterministic test examples.\n"
        "Grad-CAM highlights image regions influencing the model prediction. It does "
        "not represent disease severity and does not prove biological correctness.\n"
        "These visualizations are qualitative explanations, not additional accuracy "
        "measurements or evidence of causal disease features.\n",
        encoding="utf-8",
    )
    print(f"Saved {len(report_rows)} EfficientNetB0 explanations to {output_directory}")


def main():
    parser = argparse.ArgumentParser(
        description="Create separate EfficientNetB0 Grad-CAM images."
    )
    parser.parse_args()
    try:
        generate()
    except (FileNotFoundError, FileExistsError, OSError, RuntimeError, ValueError) as error:
        print(f"EfficientNetB0 Grad-CAM failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
