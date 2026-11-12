"""Reusable, inference-only helpers for maize disease module testing."""

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from explainability.gradcam import (
    CLASS_NAMES,
    IMAGE_SIZE,
    build_gradcam_model,
    generate_gradcam,
    read_model_input,
    resize_heatmap,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "maize_disease_model.keras"
CLASS_MAPPING_PATH = PROJECT_ROOT / "models" / "maize_disease_classes.json"
CONFIG_PATH = PROJECT_ROOT / "testing" / "config.json"
REAL_WORLD_DIRECTORY = PROJECT_ROOT / "testing" / "real_world_images"
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


@dataclass(frozen=True)
class ModelBundle:
    model: object
    class_names: tuple
    class_to_index: dict


def load_config(config_path=CONFIG_PATH):
    """Load configurable review and image-quality thresholds."""
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    threshold = config.get("confidence_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
        raise ValueError("confidence_threshold must be a number from 0 to 1.")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("confidence_threshold must be a number from 0 to 1.")
    quality = config.get("quality_warnings")
    if not isinstance(quality, dict):
        raise ValueError("quality_warnings must be an object.")
    return config


def load_class_mapping(mapping_path=CLASS_MAPPING_PATH):
    """Load the class labels and ensure their indices match the trained model."""
    path = Path(mapping_path)
    if not path.is_file():
        raise FileNotFoundError(f"Class mapping file is missing: {path}")
    try:
        mapping = json.loads(path.read_text(encoding="utf-8"))
        class_to_index = mapping["class_to_index"]
        index_to_class = mapping["index_to_class"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Could not load class mapping {path}: {error}") from error

    expected_forward = {name: index for index, name in enumerate(CLASS_NAMES)}
    expected_reverse = {str(index): name for index, name in enumerate(CLASS_NAMES)}
    if class_to_index != expected_forward or index_to_class != expected_reverse:
        raise ValueError(
            "Class mapping does not match the saved model's expected order: "
            f"{expected_forward}."
        )
    return tuple(index_to_class[str(index)] for index in range(len(CLASS_NAMES)))


def validate_model_architecture(model, class_names):
    """Check model input/output shapes and its training-time preprocessing layer."""
    expected_input = (IMAGE_SIZE[1], IMAGE_SIZE[0], 3)
    if tuple(model.input_shape[1:]) != expected_input:
        raise ValueError(
            f"Expected model input (224, 224, 3), found {model.input_shape}."
        )
    if tuple(model.output_shape[1:]) != (len(class_names),):
        raise ValueError(
            f"Expected {len(class_names)} model outputs, found {model.output_shape}."
        )
    try:
        preprocessing = model.get_layer("mobilenetv2_preprocessing")
    except ValueError as error:
        raise ValueError(
            "The saved model is missing its training-time MobileNetV2 rescaling layer."
        ) from error
    settings = preprocessing.get_config()
    if (
        preprocessing.__class__.__name__ != "Rescaling"
        or not math.isclose(float(settings.get("scale", 0)), 1.0 / 127.5)
        or not math.isclose(float(settings.get("offset", 0)), -1.0)
    ):
        raise ValueError(
            "Model preprocessing differs from training's 1/127.5 scale and -1 offset."
        )
    return True


@lru_cache(maxsize=1)
def load_model_bundle():
    """Load the trained model once, without compiling or changing its weights."""
    import tensorflow as tf

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Trained model file is missing: {MODEL_PATH}")
    class_names = load_class_mapping()
    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    validate_model_architecture(model, class_names)
    return ModelBundle(
        model=model,
        class_names=class_names,
        class_to_index={name: index for index, name in enumerate(class_names)},
    )


def read_rgb_input(image_path):
    """Validate a path and reuse the project's EXIF/RGB/224px preprocessing."""
    if image_path is None or not str(image_path).strip():
        raise ValueError("Input image path is empty.")
    path = Path(image_path)
    if path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image file format: {path.suffix or '(none)'}")
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    with Image.open(path) as image:
        image.verify()
    original_rgb, pixels = read_model_input(path)
    if pixels.shape != (IMAGE_SIZE[1], IMAGE_SIZE[0], 3):
        raise ValueError(
            f"Preprocessing returned {pixels.shape}; expected (224, 224, 3)."
        )
    if pixels.dtype != np.float32 or not np.all(np.isfinite(pixels)):
        raise ValueError("Preprocessed image must contain finite float32 RGB values.")
    return original_rgb, pixels


def image_quality_metrics(image_rgb, config=None):
    """Return heuristic quality measurements and warnings without rejecting input."""
    config = load_config() if config is None else config
    limits = config["quality_warnings"]
    rgb = np.asarray(image_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3 or rgb.size == 0:
        raise ValueError("Quality assessment requires a non-empty RGB image.")
    height, width = rgb.shape[:2]
    grayscale = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    brightness = float(np.mean(grayscale))
    blur_variance = float(cv2.Laplacian(grayscale, cv2.CV_64F).var())
    ratio = max(width, height) / min(width, height)
    warnings = []
    if blur_variance < float(limits["blur_laplacian_variance_below"]):
        warnings.append("possibly_blurry")
    if brightness < float(limits["dark_mean_grayscale_below"]):
        warnings.append("very_dark")
    if brightness > float(limits["bright_mean_grayscale_above"]):
        warnings.append("very_bright")
    if (
        width < int(limits["minimum_width"])
        or height < int(limits["minimum_height"])
        or max(width, height) > int(limits["maximum_dimension"])
        or ratio > float(limits["maximum_aspect_ratio"])
    ):
        warnings.append("unusual_dimensions")
    return {
        "width": width,
        "height": height,
        "mean_grayscale": brightness,
        "laplacian_variance": blur_variance,
        "warnings": warnings,
    }


def confidence_decision(confidence, threshold):
    """Mark a low-confidence prediction for review, retaining its class and score."""
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("Confidence threshold must be between 0 and 1.")
    if not np.isfinite(confidence) or not 0.0 <= float(confidence) <= 1.0:
        raise ValueError("Prediction confidence must be finite and between 0 and 1.")
    return "needs_review" if float(confidence) < float(threshold) else "predicted"


def validate_probabilities(probabilities, class_count):
    """Validate one model output row before converting it to a class prediction."""
    output = np.asarray(probabilities, dtype=np.float32)
    if output.shape != (class_count,):
        raise ValueError(
            f"Expected prediction vector shape ({class_count},), found {output.shape}."
        )
    if (
        not np.all(np.isfinite(output))
        or np.any(output < 0.0)
        or np.any(output > 1.0)
        or not np.isclose(float(np.sum(output)), 1.0, atol=1e-3)
    ):
        raise ValueError("Model returned invalid class probabilities.")
    return output


def build_explanation_context(model):
    """Build the project's Grad-CAM context and include Grad-CAM++ if present."""
    import tensorflow as tf

    gradcam_model, conv_layer, activation_layer = build_gradcam_model(tf, model)
    plus_plus = None
    plus_plus_path = PROJECT_ROOT / "explainability" / "gradcam_plus_plus.py"
    if plus_plus_path.is_file():
        from explainability.gradcam_plus_plus import gradcam_plus_plus_heatmap

        plus_plus = gradcam_plus_plus_heatmap
    return {
        "tensorflow": tf,
        "gradcam_model": gradcam_model,
        "gradcam_layer": conv_layer,
        "activation_layer": activation_layer,
        "gradcam_plus_plus": plus_plus,
    }


def generate_explanations(image_pixels, context, expected_class_index):
    """Generate available heatmaps and ensure each targets the predicted class."""
    heatmaps = {}
    heatmap, predicted_index, confidence = generate_gradcam(
        context["tensorflow"],
        context["gradcam_model"],
        image_pixels,
    )
    if predicted_index != expected_class_index:
        raise ValueError("Grad-CAM targeted a class other than the predicted class.")
    heatmaps["Grad-CAM"] = {
        "heatmap": heatmap,
        "predicted_index": predicted_index,
        "confidence": confidence,
    }
    plus_plus = context["gradcam_plus_plus"]
    if plus_plus is not None:
        heatmap, predicted_index, confidence = plus_plus(
            context["tensorflow"],
            context["gradcam_model"],
            image_pixels,
        )
        if predicted_index != expected_class_index:
            raise ValueError(
                "Grad-CAM++ targeted a class other than the predicted class."
            )
        heatmaps["Grad-CAM++"] = {
            "heatmap": heatmap,
            "predicted_index": predicted_index,
            "confidence": confidence,
        }
    return heatmaps


def predict_image(
    image_path,
    bundle=None,
    config=None,
    explanation_context=None,
):
    """Run image validation, preprocessing, prediction, review, and optional CAM."""
    config = load_config() if config is None else config
    try:
        original_rgb, image_pixels = read_rgb_input(image_path)
    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as error:
        return {
            "status": "rejected",
            "error": str(error),
            "predicted_class": None,
            "confidence": None,
            "class_probabilities": None,
            "confidence_status": "not_evaluated",
            "out_of_distribution_status": "not_assessed",
            "quality": None,
            "explanations": {},
        }

    bundle = load_model_bundle() if bundle is None else bundle
    quality = image_quality_metrics(original_rgb, config)
    probabilities = bundle.model(
        image_pixels[np.newaxis, ...],
        training=False,
    ).numpy()[0]
    probabilities = validate_probabilities(probabilities, len(bundle.class_names))
    predicted_index = int(np.argmax(probabilities))
    confidence = float(probabilities[predicted_index])
    explanations = {}
    if explanation_context is not None:
        explanations = generate_explanations(
            image_pixels,
            explanation_context,
            predicted_index,
        )
    return {
        "status": "success",
        "error": None,
        "predicted_class": bundle.class_names[predicted_index],
        "confidence": confidence,
        "class_probabilities": {
            name: float(probabilities[index])
            for index, name in enumerate(bundle.class_names)
        },
        "confidence_status": confidence_decision(
            confidence,
            config["confidence_threshold"],
        ),
        "out_of_distribution_status": "not_assessed",
        "quality": quality,
        "explanations": explanations,
    }


def make_overlay(original_rgb, heatmap):
    """Resize and blend a heatmap while preserving the original image dimensions."""
    resized, colored = resize_heatmap(
        np.asarray(heatmap, dtype=np.float32),
        original_rgb.width,
        original_rgb.height,
    )
    if resized.shape != (original_rgb.height, original_rgb.width):
        raise ValueError("Resized heatmap does not match the original image dimensions.")
    overlay = cv2.addWeighted(np.asarray(original_rgb), 0.55, colored, 0.45, 0)
    if overlay.size == 0 or not np.all(np.isfinite(overlay)):
        raise ValueError("Generated heatmap overlay is empty or invalid.")
    return resized, colored, overlay

