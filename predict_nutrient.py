"""Predict a supplied dataset-label class with nutrient EfficientNetB0."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from training.nutrient_config import CLASS_ORDER, MODEL_ROOT
from training.nutrient_preprocessing import preprocess_image_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path, help="Path to a maize leaf image")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODEL_ROOT / "best_model.keras",
        help="Trained nutrient EfficientNetB0 .keras model",
    )
    args = parser.parse_args()
    if not args.image.is_file():
        parser.error(f"Image file does not exist: {args.image}")
    if not args.model.is_file():
        parser.error(f"Trained model does not exist: {args.model}")
    mapping_path = args.model.parent / "class_mapping.json"
    if not mapping_path.is_file():
        parser.error(f"Model class mapping is missing: {mapping_path}")

    try:
        with Image.open(args.image) as image:
            image.verify()
    except (OSError, ValueError) as error:
        parser.error(f"Invalid or unreadable image file: {error}")

    import tensorflow as tf

    try:
        with mapping_path.open(encoding="utf-8") as stream:
            mapping = json.load(stream)
        class_order = mapping["class_order"]
        if class_order != list(CLASS_ORDER):
            raise ValueError("Saved class order does not match the nutrient model config.")
        image_tensor = preprocess_image_path(tf.constant(str(args.image.resolve())))
        batch = tf.expand_dims(image_tensor, axis=0)
        model = tf.keras.models.load_model(args.model)
        probabilities = model(batch, training=False).numpy()[0]
    except (OSError, ValueError, KeyError, tf.errors.OpError) as error:
        parser.error(f"Could not run prediction: {error}")

    if probabilities.shape != (len(class_order),) or not np.isfinite(probabilities).all():
        parser.error("Model returned invalid class scores.")
    predicted_index = int(np.argmax(probabilities))
    print(f"Predicted class: {class_order[predicted_index]}")
    print(f"Confidence: {probabilities[predicted_index] * 100:.2f}%")
    print("Scores:")
    for label, score in zip(class_order, probabilities):
        print(f"  {label}: {score * 100:.2f}%")
    print(
        "Prediction is a learned dataset-folder label, not a scientifically "
        "verified nutrient diagnosis."
    )


if __name__ == "__main__":
    main()
