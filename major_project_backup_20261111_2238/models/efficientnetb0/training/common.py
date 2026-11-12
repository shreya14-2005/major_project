"""Shared dataset and input helpers for the independent EfficientNetB0 workflow."""

from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLASS_NAMES = ("Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy")
EXPECTED_SPLIT_COUNTS = {
    "train": {"Blight": 802, "Common_Rust": 914, "Gray_Leaf_Spot": 402, "Healthy": 814},
    "val": {"Blight": 172, "Common_Rust": 196, "Gray_Leaf_Spot": 86, "Healthy": 174},
    "test": {"Blight": 172, "Common_Rust": 196, "Gray_Leaf_Spot": 86, "Healthy": 174},
}
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
RANDOM_SEED = 42
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


def inspect_split_directories(dataset_directory):
    """Validate existing split folders and return stable, labeled image records."""
    split_records = {}
    errors = []

    for split_name, expected_counts in EXPECTED_SPLIT_COUNTS.items():
        split_directory = dataset_directory / split_name
        if not split_directory.is_dir():
            errors.append(f"Missing split folder: {split_directory}")
            continue

        observed_classes = {
            path.name for path in split_directory.iterdir() if path.is_dir()
        }
        if observed_classes != set(CLASS_NAMES):
            errors.append(
                f"{split_name}: expected class folders {sorted(CLASS_NAMES)}, "
                f"found {sorted(observed_classes)}"
            )
            continue

        unexpected_files = [
            path for path in split_directory.iterdir() if path.is_file()
        ]
        errors.extend(
            f"Unexpected file outside a class folder: {path}"
            for path in unexpected_files
        )

        records = []
        counts = Counter()
        for class_name in CLASS_NAMES:
            class_directory = split_directory / class_name
            image_paths = sorted(
                (path for path in class_directory.rglob("*") if path.is_file()),
                key=lambda path: str(path).casefold(),
            )
            for image_path in image_paths:
                if image_path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
                    errors.append(f"Unsupported image file format: {image_path}")
                    continue
                try:
                    with Image.open(image_path) as image:
                        image.verify()
                    with Image.open(image_path) as image:
                        image.load()
                except (OSError, ValueError, SyntaxError) as error:
                    errors.append(f"Unreadable image {image_path}: {error}")
                    continue

                records.append({"path": image_path, "class_name": class_name})
                counts[class_name] += 1

        if dict(counts) != expected_counts:
            errors.append(
                f"{split_name}: expected counts {expected_counts}, "
                f"found {dict(counts)}"
            )
        split_records[split_name] = records

    return split_records, errors


def load_rgb_image(path_value):
    """Read an EXIF-corrected RGB image and resize it without normalization."""
    path_bytes = path_value.numpy()
    image_path = path_bytes.decode("utf-8")
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.uint8)


def read_rgb_pixels(image_path):
    """Read one image as 224x224 RGB float pixels in the [0, 255] range."""
    with Image.open(image_path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
        return np.asarray(image, dtype=np.float32)


def make_dataset(tf, records, class_to_index, training=False):
    """Build a deterministic streaming dataset; augmentation stays in the model."""
    paths = [str(record["path"]) for record in records]
    labels = [class_to_index[record["class_name"]] for record in records]
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(records),
            seed=RANDOM_SEED,
            reshuffle_each_iteration=True,
        )

    def read_image(path, label):
        image = tf.py_function(load_rgb_image, [path], Tout=tf.uint8)
        image.set_shape((IMAGE_SIZE[1], IMAGE_SIZE[0], 3))
        image = tf.cast(image, tf.float32)
        return image, tf.one_hot(label, depth=len(CLASS_NAMES))

    dataset = dataset.map(
        read_image,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def class_mapping():
    """Return the persisted class-index mapping used by all three scripts."""
    return {
        "class_to_index": {
            name: index for index, name in enumerate(CLASS_NAMES)
        },
        "index_to_class": {
            str(index): name for index, name in enumerate(CLASS_NAMES)
        },
    }
