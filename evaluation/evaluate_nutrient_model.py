"""Run one final evaluation pass on the held-out nutrient test split."""

import argparse
import csv
import json
import statistics
import time
from pathlib import Path

import numpy as np

from training.nutrient_config import (
    CLASS_ORDER,
    EVALUATION_ROOT,
    MODEL_ROOT,
    SPLITS_ROOT,
)
from training.nutrient_preprocessing import preprocess_image_path


def read_manifest(path, expected_split):
    if not path.is_file():
        raise FileNotFoundError(f"Required manifest is missing: {path}")
    with path.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"Manifest is empty: {path}")
    for row in rows:
        if row["split"] != expected_split:
            raise ValueError(f"Unexpected split {row['split']} in {path}")
        if row["label"] not in CLASS_ORDER:
            raise ValueError(f"Unknown label in {path}: {row['label']}")
        if not Path(row["image_path"]).is_file():
            raise FileNotFoundError(f"Manifest image not found: {row['image_path']}")
    return rows


def make_dataset(rows, tf, batch_size=16):
    paths = np.asarray([row["image_path"] for row in rows], dtype=str)
    labels = np.asarray([CLASS_ORDER.index(row["label"]) for row in rows], dtype=np.int32)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path, label):
        return preprocess_image_path(path), label

    return dataset.map(load, num_parallel_calls=4, deterministic=True).batch(batch_size).prefetch(1)


def measure_batch_one_latency(model, image_path, tf, repeats=30):
    image = preprocess_image_path(tf.constant(image_path))
    batch = tf.expand_dims(image, axis=0)
    for _ in range(3):
        model(batch, training=False).numpy()
    durations_ms = []
    for _ in range(repeats):
        start = time.perf_counter()
        model(batch, training=False).numpy()
        durations_ms.append((time.perf_counter() - start) * 1000)
    return statistics.mean(durations_ms), statistics.median(durations_ms), repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    metrics_path = EVALUATION_ROOT / "test_metrics.json"
    if metrics_path.exists():
        raise FileExistsError(
            f"Test evaluation already exists at {metrics_path}; refusing to evaluate "
            "the held-out set again."
        )
    model_path = MODEL_ROOT / "best_model.keras"
    mapping_path = MODEL_ROOT / "class_mapping.json"
    if not model_path.is_file() or not mapping_path.is_file():
        raise FileNotFoundError(
            "Train the nutrient EfficientNetB0 model before evaluating it."
        )

    train_rows = read_manifest(SPLITS_ROOT / "train.csv", "train")
    val_rows = read_manifest(SPLITS_ROOT / "val.csv", "val")
    test_rows = read_manifest(SPLITS_ROOT / "test.csv", "test")
    group_sets = {
        split_name: {row["related_group_id"] for row in rows}
        for split_name, rows in (
            ("train", train_rows),
            ("val", val_rows),
            ("test", test_rows),
        )
    }
    if (
        group_sets["train"] & group_sets["val"]
        or group_sets["train"] & group_sets["test"]
        or group_sets["val"] & group_sets["test"]
    ):
        raise ValueError("A related-image group crosses dataset splits.")

    with mapping_path.open(encoding="utf-8") as stream:
        mapping = json.load(stream)
    if mapping.get("class_order") != list(CLASS_ORDER):
        raise ValueError("Saved model class order differs from nutrient pipeline config.")

    import tensorflow as tf
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    model = tf.keras.models.load_model(model_path)
    test_ds = make_dataset(test_rows, tf)

    # This is the only full held-out test-set inference pass.
    probabilities = model.predict(test_ds, verbose=1)
    y_true = np.asarray([CLASS_ORDER.index(row["label"]) for row in test_rows])
    y_pred = np.argmax(probabilities, axis=1)
    accuracy = float(accuracy_score(y_true, y_pred))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_ORDER)),
        average="macro",
        zero_division=0,
    )
    per_class = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_ORDER)),
        target_names=list(CLASS_ORDER),
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_ORDER)),
    )
    validation_sample = val_rows[0]["image_path"]
    mean_latency_ms, median_latency_ms, latency_repeats = measure_batch_one_latency(
        model, validation_sample, tf
    )
    model_size_bytes = model_path.stat().st_size
    parameter_count = int(model.count_params())

    EVALUATION_ROOT.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        EVALUATION_ROOT / "confusion_matrix.csv",
        matrix,
        delimiter=",",
        fmt="%d",
        header=",".join(CLASS_ORDER),
        comments="",
    )
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER, rotation=35, ha="right")
    axis.set_yticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    axis.set_title("Nutrient EfficientNetB0 - held-out test confusion matrix")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            axis.text(column_index, row_index, str(matrix[row_index, column_index]),
                      ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(EVALUATION_ROOT / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    predictions = []
    for row, scores, predicted_index in zip(test_rows, probabilities, y_pred):
        predictions.append(
            {
                "image_path": row["image_path"],
                "actual_class": row["label"],
                "predicted_class": CLASS_ORDER[int(predicted_index)],
                "confidence": f"{float(scores[predicted_index]):.8f}",
                "correct": str(int(predicted_index) == CLASS_ORDER.index(row["label"])),
            }
        )
    with (EVALUATION_ROOT / "test_predictions.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0].keys()))
        writer.writeheader()
        writer.writerows(predictions)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(CLASS_ORDER)),
        target_names=list(CLASS_ORDER),
        zero_division=0,
    )
    (EVALUATION_ROOT / "classification_report.txt").write_text(
        report_text, encoding="utf-8"
    )
    result = {
        "model": "EfficientNetB0",
        "test_image_count": len(test_rows),
        "accuracy": accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "per_class": {
            label: {
                "precision": float(per_class[label]["precision"]),
                "recall": float(per_class[label]["recall"]),
                "f1_score": float(per_class[label]["f1-score"]),
                "support": int(per_class[label]["support"]),
            }
            for label in CLASS_ORDER
        },
        "model_file": str(model_path),
        "model_size_bytes": model_size_bytes,
        "model_size_mb": model_size_bytes / (1024 * 1024),
        "parameter_count": parameter_count,
        "batch_1_inference_latency_ms_mean": mean_latency_ms,
        "batch_1_inference_latency_ms_median": median_latency_ms,
        "latency_repeats": latency_repeats,
        "latency_sample_source": "validation split, not test split",
        "test_set_inference_passes": 1,
        "labels_are_supplied_dataset_labels_not_scientifically_verified": True,
    }
    metrics_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    lines = [
        "# Nutrient EfficientNetB0 Test Evaluation",
        "",
        "The held-out test set was used once, in one batched prediction pass. "
        "Validation data selected the checkpoint. Latency was measured separately "
        "using a validation image.",
        "",
        f"- Test images: {len(test_rows)}",
        f"- Accuracy: {accuracy:.4f}",
        f"- Macro precision: {macro_precision:.4f}",
        f"- Macro recall: {macro_recall:.4f}",
        f"- Macro F1: {macro_f1:.4f}",
        f"- Model size: {model_size_bytes / (1024 * 1024):.2f} MiB ({model_size_bytes:,} bytes)",
        f"- Parameter count: {parameter_count:,}",
        f"- Batch-1 latency: mean {mean_latency_ms:.2f} ms, median {median_latency_ms:.2f} ms "
        f"over {latency_repeats} validation-image runs.",
        "",
        "Per-class metrics:",
        "",
        "| Class | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for label in CLASS_ORDER:
        values = result["per_class"][label]
        lines.append(
            f"| {label} | {values['precision']:.4f} | {values['recall']:.4f} | "
            f"{values['f1_score']:.4f} | {values['support']} |"
        )
    lines.extend(
        [
            "",
            "Labels are the supplied dataset folder labels after documented spelling "
            "normalization; scientific correctness has not been independently verified.",
            "The model learns these supplied labels, not independently established "
            "nutrient diagnoses.",
        ]
    )
    (EVALUATION_ROOT / "final_report.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))
    print(f"Evaluation evidence: {EVALUATION_ROOT}")


if __name__ == "__main__":
    main()
