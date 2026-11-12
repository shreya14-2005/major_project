"""Generate Grad-CAM explanations with the standalone nutrient classifier."""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "models" / "nutrient_detection" / "nutrient_model.keras"
)
DEFAULT_CLASSES_PATH = (
    PROJECT_ROOT / "models" / "nutrient_detection" / "nutrient_classes.json"
)
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"
IMAGE_SIZE = (224, 224)
SUPPORTED_EXTENSIONS = {
    ".bmp",
    ".gif",
    ".jpeg",
    ".jpg",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
}


def load_model_and_classes(model_path=DEFAULT_MODEL_PATH, classes_path=DEFAULT_CLASSES_PATH):
    """Load only the saved nutrient model and its class mapping."""
    import tensorflow as tf

    model_path = Path(model_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Nutrient model file not found: {model_path}")

    classes_path = Path(classes_path)
    if not classes_path.is_file():
        raise FileNotFoundError(f"Nutrient classes mapping file not found: {classes_path}")

    try:
        mapping_data = json.loads(classes_path.read_text(encoding="utf-8"))
        total_classes = mapping_data["total_classes"]
        index_to_class = mapping_data["index_to_class"]
        class_names = [index_to_class[str(index)] for index in range(total_classes)]
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Could not read nutrient class mapping {classes_path}: {error}") from error

    if total_classes != 5 or len(set(class_names)) != total_classes:
        raise ValueError(
            f"Expected five unique nutrient classes in {classes_path}; found {class_names}."
        )

    try:
        model = tf.keras.models.load_model(str(model_path), compile=False)
    except Exception as error:
        raise RuntimeError(f"Could not load nutrient model {model_path}: {error}") from error

    if tuple(model.input_shape[1:]) != (224, 224, 3):
        raise ValueError(
            f"Expected nutrient model input shape (224, 224, 3), found {model.input_shape}."
        )
    if tuple(model.output_shape[1:]) != (total_classes,):
        raise ValueError(
            f"Model output shape {model.output_shape} does not match "
            f"the {total_classes}-class mapping."
        )
    return model, class_names


def build_gradcam_model(model):
    """Expose the final convolution's feature maps and the saved model scores."""
    import tensorflow as tf

    convolution_layers = [
        layer for layer in model.layers if isinstance(layer, tf.keras.layers.Conv2D)
    ]
    if not convolution_layers:
        raise ValueError("No Conv2D layer was found in the nutrient model.")

    final_conv_layer = convolution_layers[-1]
    if len(final_conv_layer.output.shape) != 4:
        raise ValueError(
            f"Final convolution layer {final_conv_layer.name} is not a spatial feature map."
        )

    gradcam_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[final_conv_layer.output, model.output],
        name="nutrient_gradcam_features",
    )
    return gradcam_model, final_conv_layer.name


def read_model_input(image_path):
    """Match nutrient inference: RGB, LANCZOS resize, float32 pixels in [0, 255]."""
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {image_path}")
    if image_path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension {image_path.suffix!r}. "
            f"Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        with Image.open(image_path) as source:
            source.verify()
        with Image.open(image_path) as source:
            original_rgb = source.convert("RGB")
            model_image = original_rgb.resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
            pixels = np.asarray(model_image, dtype=np.float32)
    except (OSError, ValueError, SyntaxError) as error:
        raise ValueError(f"Corrupted or unreadable image file {image_path}: {error}") from error

    if pixels.shape != (224, 224, 3):
        raise ValueError(f"Expected preprocessed image shape (224, 224, 3), got {pixels.shape}.")
    return original_rgb, pixels


def resolve_class_index(class_names, target_class):
    """Resolve a requested class name without depending on its capitalization."""
    for index, class_name in enumerate(class_names):
        if class_name.casefold() == target_class.casefold():
            return index
    raise ValueError(
        f"Unknown target class {target_class!r}. Choose one of: {', '.join(class_names)}."
    )


def compute_gradcam(gradcam_model, image_pixels, class_index):
    """Compute a normalized class-specific Grad-CAM from the selected score."""
    import tensorflow as tf

    input_batch = tf.convert_to_tensor(image_pixels[np.newaxis, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        activations, probabilities = gradcam_model(input_batch, training=False)
        class_score = probabilities[:, class_index]

    gradients = tape.gradient(class_score, activations)
    if gradients is None:
        raise ValueError("Gradients are unavailable for the selected nutrient class.")
    if not bool(tf.reduce_all(tf.math.is_finite(gradients)).numpy()):
        raise ValueError("Gradients contain NaN or infinite values.")

    channel_weights = tf.reduce_mean(gradients, axis=(1, 2))
    weighted_activations = activations[0] * channel_weights[0][tf.newaxis, tf.newaxis, :]
    heatmap = tf.nn.relu(tf.reduce_sum(weighted_activations, axis=-1))
    heatmap_array = heatmap.numpy().astype(np.float32)
    if heatmap_array.ndim != 2 or not np.all(np.isfinite(heatmap_array)):
        raise ValueError("The Grad-CAM heatmap is invalid or contains non-finite values.")

    maximum = float(np.max(heatmap_array))
    minimum = float(np.min(heatmap_array))
    if not np.isfinite(maximum) or maximum <= 1e-12 or maximum - minimum <= 1e-12:
        raise ValueError("The Grad-CAM heatmap is blank or constant.")

    heatmap_array /= maximum
    return heatmap_array, probabilities[0].numpy()


def save_explanation(
    image_path,
    model,
    gradcam_model,
    class_names,
    conv_layer_name,
    target_class=None,
    output_dir=DEFAULT_OUTPUT_DIR,
    display=True,
    model_path=DEFAULT_MODEL_PATH,
):
    """Create and save an original/heatmap/overlay visualization and metadata."""
    import matplotlib.pyplot as plt

    image_path = Path(image_path)
    original_rgb, image_pixels = read_model_input(image_path)
    predictions = model(
        image_pixels[np.newaxis, ...],
        training=False,
    ).numpy()[0]
    if predictions.shape != (len(class_names),) or not np.all(np.isfinite(predictions)):
        raise ValueError("The nutrient model returned invalid class scores.")

    predicted_index = int(np.argmax(predictions))
    target_index = (
        predicted_index
        if target_class is None
        else resolve_class_index(class_names, target_class)
    )
    resolved_target_class = class_names[target_index]
    heatmap, cam_predictions = compute_gradcam(gradcam_model, image_pixels, target_index)
    if not np.allclose(predictions, cam_predictions, rtol=1e-5, atol=1e-6):
        raise ValueError("Grad-CAM scores differ from the nutrient model's prediction scores.")

    heatmap_image = Image.fromarray(
        np.uint8(np.clip(plt.get_cmap("jet")(heatmap)[..., :3] * 255, 0, 255))
    ).resize(original_rgb.size, Image.Resampling.BILINEAR)
    overlay = Image.blend(original_rgb, heatmap_image, alpha=0.4)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", image_path.stem).strip("._")
    safe_parent = re.sub(r"[^A-Za-z0-9._-]+", "_", image_path.parent.name).strip("._")
    output_name = (
        f"{safe_parent}_{safe_stem}_pred-{class_names[predicted_index]}"
        f"_target-{resolved_target_class}_gradcam.png"
    )
    output_path = output_dir / output_name
    metadata_path = output_path.with_suffix(".json")

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original_rgb)
    axes[0].set_title("Original image")
    axes[1].imshow(heatmap_image)
    axes[1].set_title(f"Grad-CAM: {resolved_target_class}")
    axes[2].imshow(overlay)
    axes[2].set_title(
        f"Predicted: {class_names[predicted_index]} "
        f"({predictions[predicted_index]:.2%})\nTarget: {resolved_target_class}"
    )
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    try:
        figure.savefig(output_path, dpi=150, bbox_inches="tight")
        if display:
            plt.show()
    finally:
        plt.close(figure)

    metadata = {
        "image_path": str(image_path.resolve()),
        "model_path": str(Path(model_path).resolve()),
        "predicted_class": class_names[predicted_index],
        "prediction_confidence": float(predictions[predicted_index]),
        "target_class": resolved_target_class,
        "target_class_confidence": float(predictions[target_index]),
        "class_probabilities": {
            class_name: float(predictions[index])
            for index, class_name in enumerate(class_names)
        },
        "convolutional_layer": conv_layer_name,
        "feature_map_shape": list(heatmap.shape),
        "visualization_size": [original_rgb.width, original_rgb.height],
        "preprocessing": (
            "RGB; Pillow LANCZOS resize to 224x224; float32 values in [0,255]; "
            "the saved EfficientNetB0 model applies its own internal normalization."
        ),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "image_path": image_path,
        "output_path": output_path,
        "metadata_path": metadata_path,
        "metadata": metadata,
        "heatmap": heatmap,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Generate a Grad-CAM visualization for the nutrient EfficientNetB0 model."
    )
    parser.add_argument("--image", "-i", required=True, help="Path to a maize leaf image.")
    parser.add_argument(
        "--target-class",
        help="Class to explain (Healthy, Magnesium, Nitrogen, Phosphorus, or Potassium). "
        "Defaults to the model's predicted class.",
    )
    parser.add_argument("--model", "-m", default=str(DEFAULT_MODEL_PATH), help="Nutrient .keras model path.")
    parser.add_argument("--classes", "-c", default=str(DEFAULT_CLASSES_PATH), help="Nutrient class mapping JSON path.")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory for PNG and JSON outputs (default: {DEFAULT_OUTPUT_DIR}).",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Save output without opening the visualization window.",
    )
    args = parser.parse_args()

    try:
        model, class_names = load_model_and_classes(args.model, args.classes)
        gradcam_model, conv_layer_name = build_gradcam_model(model)
        result = save_explanation(
            args.image,
            model,
            gradcam_model,
            class_names,
            conv_layer_name,
            target_class=args.target_class,
            output_dir=args.output_dir,
            display=not args.no_display,
            model_path=args.model,
        )
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print(f"Predicted class: {result['metadata']['predicted_class']}")
    print(f"Prediction confidence: {result['metadata']['prediction_confidence']:.4f}")
    print(f"Target class: {result['metadata']['target_class']}")
    print(f"Visualization saved: {result['output_path']}")
    print(f"Metadata saved: {result['metadata_path']}")
    print(f"Grad-CAM layer: {conv_layer_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
