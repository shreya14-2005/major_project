# Nutrient EfficientNetB0 Training

Only the nutrient EfficientNetB0 model was trained. No MobileNetV2 or disease model was loaded or changed.

- Best checkpoint: `C:\Users\ADMIN\Desktop\major_project\models\nutrient_detection\efficientnetb0\best_model.keras`
- Validation loss of saved checkpoint: 0.432215
- Training images: 323
- Validation images: 71
- Epochs completed (head + fine-tuning): 32
- Trainable EfficientNet base layers during fine-tuning: 23
- Batch size: 16; seed: 42
- Input: RGB 224x224 float32 in [0,255], EfficientNetB0 internal rescaling.
- Augmentation is included in the model and is active only during training.
- The test manifest was not opened or used for checkpoint selection.
- Labels reflect the supplied folder names with spelling-only mapping; biological correctness has not been independently verified.

Detailed history, training configuration, class mapping, and model summary are saved beside the checkpoint and under `evaluation/nutrient_model/`.
