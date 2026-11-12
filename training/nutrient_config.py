"""Shared configuration for the separate nutrient-detection pipeline."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "datasets" / "Maize Nutrient Deficiency Dataset"
SPLITS_ROOT = PROJECT_ROOT / "datasets" / "nutrient_splits"
MODEL_ROOT = PROJECT_ROOT / "models" / "nutrient_detection" / "efficientnetb0"
EVALUATION_ROOT = PROJECT_ROOT / "evaluation" / "nutrient_model"
IMAGE_SIZE = (224, 224)
RANDOM_SEED = 42
SPLIT_PROPORTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}
PHASH_MAX_HAMMING_DISTANCE = 6
EXPECTED_TOTAL_IMAGES = 463

# Documented spelling corrections only; these do not verify scientific labels:
# Helathy -> Healthy, Magnessium -> Magnesium, Pottasium -> Potassium.
# Phosphorous -> Phosphorus is a spelling normalization; Nitrogen is unchanged.
CLASS_FOLDER_TO_LABEL = {
    "Helathy": "Healthy",
    "Magnessium": "Magnesium",
    "Nitrogen": "Nitrogen",
    "Phosphorous": "Phosphorus",
    "Pottasium": "Potassium",
}
CLASS_ORDER = (
    "Healthy",
    "Magnesium",
    "Nitrogen",
    "Phosphorus",
    "Potassium",
)
