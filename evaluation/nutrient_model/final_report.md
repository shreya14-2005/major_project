# Nutrient EfficientNetB0 Test Evaluation

The held-out test set was used once, in one batched prediction pass. Validation data selected the checkpoint. Latency was measured separately using a validation image.

- Test images: 69
- Accuracy: 0.8696
- Macro precision: 0.8549
- Macro recall: 0.8413
- Macro F1: 0.8456
- Model size: 27.67 MiB (29,017,877 bytes)
- Parameter count: 4,055,976
- Batch-1 latency: mean 306.90 ms, median 302.72 ms over 30 validation-image runs.

Per-class metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Healthy | 0.9000 | 0.8182 | 0.8571 | 11 |
| Magnesium | 0.9412 | 0.8889 | 0.9143 | 18 |
| Nitrogen | 0.7778 | 0.9333 | 0.8485 | 15 |
| Phosphorus | 0.9412 | 0.9412 | 0.9412 | 17 |
| Potassium | 0.7143 | 0.6250 | 0.6667 | 8 |

Labels are the supplied dataset folder labels after documented spelling normalization; scientific correctness has not been independently verified.
The model learns these supplied labels, not independently established nutrient diagnoses.
