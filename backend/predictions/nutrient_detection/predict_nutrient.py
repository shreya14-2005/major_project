"""Standalone Maize Leaf Nutrient Deficiency Prediction Module.

Predicts one of five nutrient conditions:
- Healthy
- Magnesium deficiency
- Nitrogen deficiency
- Phosphorus deficiency
- Potassium deficiency

Applies the exact training preprocessing pipeline. Independent of the disease classification model.
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "nutrient_detection" / "nutrient_model.keras"
DEFAULT_CLASSES_PATH = PROJECT_ROOT / "models" / "nutrient_detection" / "nutrient_classes.json"

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

IMAGE_SIZE = (224, 224)

_CACHED_MODEL = None
_CACHED_CLASSES = None


def load_model_and_classes(model_path=DEFAULT_MODEL_PATH, classes_path=DEFAULT_CLASSES_PATH):
    """Load and cache the trained nutrient model and class mapping."""
    global _CACHED_MODEL, _CACHED_CLASSES

    if _CACHED_MODEL is not None and _CACHED_CLASSES is not None:
        return _CACHED_MODEL, _CACHED_CLASSES

    model_p = Path(model_path)
    if not model_p.is_file():
        raise FileNotFoundError(f"Nutrient model file not found: {model_p}")

    classes_p = Path(classes_path)
    if not classes_p.is_file():
        raise FileNotFoundError(f"Nutrient classes mapping file not found: {classes_p}")

    with open(classes_p, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)

    total_classes = mapping_data["total_classes"]
    class_names = [mapping_data["index_to_class"][str(i)] for i in range(total_classes)]

    import tensorflow as tf
    model = tf.keras.models.load_model(str(model_p))

    _CACHED_MODEL = model
    _CACHED_CLASSES = class_names
    return _CACHED_MODEL, _CACHED_CLASSES


def preprocess_image(image_input):
    """Validate, load, and resize image to model input shape (224, 224, 3) RGB."""
    if isinstance(image_input, (str, Path)):
        img_path = Path(image_input)
        if not img_path.is_file():
            raise FileNotFoundError(f"Image file does not exist: {img_path}")
        if img_path.suffix.casefold() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension '{img_path.suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

        try:
            with Image.open(img_path) as im:
                im.verify()
            with Image.open(img_path) as im:
                im_rgb = im.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                img_array = np.array(im_rgb, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Corrupted or unreadable image file: {e}") from e

    elif isinstance(image_input, Image.Image):
        im_rgb = image_input.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(im_rgb, dtype=np.float32)
    elif isinstance(image_input, np.ndarray):
        if image_input.ndim != 3 or image_input.shape[-1] != 3:
            raise ValueError(f"Expected 3-channel RGB image array, got shape {image_input.shape}")
        img_pil = Image.fromarray(image_input.astype(np.uint8)).convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
        img_array = np.array(img_pil, dtype=np.float32)
    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    return np.expand_dims(img_array, axis=0)


def predict_nutrient_deficiency(image_input, model_path=DEFAULT_MODEL_PATH, classes_path=DEFAULT_CLASSES_PATH):
    """Predict nutrient deficiency class and probabilities for a maize leaf image."""
    model, class_names = load_model_and_classes(model_path, classes_path)
    processed_tensor = preprocess_image(image_input)

    probabilities = model(processed_tensor, training=False).numpy()[0]
    top_idx = int(np.argmax(probabilities))
    predicted_class = class_names[top_idx]
    confidence = float(probabilities[top_idx])

    class_probabilities = {class_names[i]: float(probabilities[i]) for i in range(len(class_names))}

    disclaimer = (
        "DIAGNOSTIC NOTICE: Model predictions represent visual pattern associations derived from "
        "experimental field survey dataset labels. These predictions are NOT confirmed biological diagnoses "
        "and do not substitute for certified chemical foliar tissue testing (e.g., ICP-OES or Kjeldahl analysis)."
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "class_probabilities": class_probabilities,
        "disclaimer": disclaimer,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Maize Leaf Nutrient Deficiency Standalone Predictor"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        type=str,
        help="Path to maize leaf image file"
    )
    parser.add_argument(
        "--model", "-m",
        default=str(DEFAULT_MODEL_PATH),
        type=str,
        help=f"Path to saved .keras nutrient model (default: {DEFAULT_MODEL_PATH})"
    )
    parser.add_argument(
        "--classes", "-c",
        default=str(DEFAULT_CLASSES_PATH),
        type=str,
        help=f"Path to nutrient classes JSON (default: {DEFAULT_CLASSES_PATH})"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON format"
    )

    args = parser.parse_args()

    try:
        result = predict_nutrient_deficiency(
            image_input=args.image,
            model_path=args.model,
            classes_path=args.classes
        )

        if args.json:
            print(json.dumps(result, indent=2))
        else:
            print("\n=======================================================")
            print("PLANTHEALTHAI: MAIZE NUTRIENT DEFICIENCY PREDICTION")
            print("=======================================================")
            print(f"Image File:         {Path(args.image).name}")
            print(f"Predicted Class:    {result['predicted_class']}")
            print(f"Confidence:         {result['confidence'] * 100:.2f}%")
            print("\nClass Probabilities:")
            for cls_name, prob in sorted(result["class_probabilities"].items(), key=lambda x: -x[1]):
                bar = "#" * int(prob * 30)
                print(f"  {cls_name:<12} {prob*100:>6.2f}% | {bar}")

            print("\n" + "-" * 55)
            print(result["disclaimer"])
            print("=" * 55 + "\n")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
