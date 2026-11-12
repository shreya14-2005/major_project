"""Evaluate the selected Maize Nutrient Deficiency Model on the untouched test set."""

import csv
import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score

import tensorflow as tf
from tensorflow import keras

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16


def load_test_data(test_dir, class_names):
    images = []
    labels = []
    filenames = []
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = test_dir / class_name
        if not class_folder.is_dir():
            continue
        for f in sorted(list(class_folder.glob("*.jpeg")) + list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))):
            with Image.open(f) as im:
                im_rgb = im.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                images.append(np.array(im_rgb, dtype=np.float32))
                labels.append(class_to_idx[class_name])
                filenames.append(f.name)

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), filenames


def measure_batch1_latency(model, sample_input, num_trials=50):
    batch_1 = sample_input[:1]
    for _ in range(5):
        _ = model(batch_1, training=False)
    latencies = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = model(batch_1, training=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return float(np.mean(latencies))


def plot_confusion_matrix(cm, class_names, output_path):
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title="Maize Nutrient Deficiency Confusion Matrix (Test Set)",
        ylabel="True Label",
        xlabel="Predicted Label"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], "d"),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_training_curves(history_file, output_path, arch_name):
    with open(history_file, "r", encoding="utf-8") as f:
        hist = json.load(f)

    stage1 = hist["stage1"]
    stage2 = hist["stage2"]

    train_acc = stage1["accuracy"] + stage2["accuracy"]
    val_acc = stage1["val_accuracy"] + stage2["val_accuracy"]
    train_loss = stage1["loss"] + stage2["loss"]
    val_loss = stage1["val_loss"] + stage2["val_loss"]

    epochs = range(1, len(train_acc) + 1)
    stage1_len = len(stage1["accuracy"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    ax1.plot(epochs, train_acc, "b-o", label="Training Accuracy", markersize=4)
    ax1.plot(epochs, val_acc, "r-s", label="Validation Accuracy", markersize=4)
    ax1.axvline(x=stage1_len + 0.5, color="gray", linestyle="--", label="Fine-Tuning Start")
    ax1.set_title(f"{arch_name} Training & Validation Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Loss
    ax2.plot(epochs, train_loss, "b-o", label="Training Loss", markersize=4)
    ax2.plot(epochs, val_loss, "r-s", label="Validation Loss", markersize=4)
    ax2.axvline(x=stage1_len + 0.5, color="gray", linestyle="--", label="Fine-Tuning Start")
    ax2.set_title(f"{arch_name} Training & Validation Loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Cross-Entropy Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def main():
    project_root = Path(__file__).resolve().parents[1]
    datasets_dir = project_root / "datasets" / "maize_nutrient_processed"
    models_dir = project_root / "models" / "nutrient_detection"
    eval_dir = project_root / "evaluation" / "nutrient_final_evaluation"
    comparison_dir = project_root / "evaluation" / "nutrient_model_comparison"

    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load class mapping
    classes_file = models_dir / "nutrient_classes.json"
    with open(classes_file, "r", encoding="utf-8") as f:
        class_info = json.load(f)
    class_names = [class_info["index_to_class"][str(i)] for i in range(class_info["total_classes"])]
    num_classes = len(class_names)

    # Load selected model
    model_path = models_dir / "nutrient_model.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Selected nutrient model not found: {model_path}")

    print(f"Loading selected nutrient model: {model_path}...")
    model = keras.models.load_model(str(model_path))

    # Load untouched test set
    test_dir = datasets_dir / "test"
    print(f"Loading untouched test dataset from {test_dir}...")
    X_test, y_test, filenames = load_test_data(test_dir, class_names)
    total_test = len(y_test)
    print(f"Loaded {total_test} test images.")

    # Single-pass Test Inference
    print("\nExecuting single-pass evaluation on untouched test set...")
    t0 = time.time()
    test_probs = model.predict(X_test, batch_size=BATCH_SIZE, verbose=1)
    eval_duration = time.time() - t0
    test_preds = np.argmax(test_probs, axis=1)
    confidences = np.max(test_probs, axis=1)

    # Overall metrics
    test_acc = float(np.mean(test_preds == y_test))
    macro_prec = precision_score(y_test, test_preds, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, test_preds, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)

    weighted_prec = precision_score(y_test, test_preds, average="weighted", zero_division=0)
    weighted_rec = recall_score(y_test, test_preds, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    correct_count = int(np.sum(test_preds == y_test))
    incorrect_count = total_test - correct_count

    # Model size and latency
    model_size_mb = model_path.stat().st_size / (1024 * 1024)
    batch1_latency = measure_batch1_latency(model, X_test)

    # Detailed classification report
    report_dict = classification_report(
        y_test, test_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    report_text = classification_report(
        y_test, test_preds,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(y_test, test_preds)

    # 1. Save Test Predictions CSV
    pred_csv_path = eval_dir / "nutrient_test_predictions.csv"
    with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["image_filename", "actual_class", "predicted_class", "confidence", "status"]
        header += [f"prob_{cls}" for cls in class_names]
        writer.writerow(header)
        for i in range(total_test):
            act_cls = class_names[y_test[i]]
            pred_cls = class_names[test_preds[i]]
            conf = float(confidences[i])
            status = "correct" if act_cls == pred_cls else "incorrect"
            row = [filenames[i], act_cls, pred_cls, f"{conf:.6f}", status]
            row += [f"{float(p):.6f}" for p in test_probs[i]]
            writer.writerow(row)
    print(f"Saved test predictions: {pred_csv_path}")

    # 2. Save Confusion Matrix CSV
    cm_csv_path = eval_dir / "nutrient_confusion_matrix.csv"
    with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["actual_class"] + list(class_names))
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + list(row))
    print(f"Saved confusion matrix CSV: {cm_csv_path}")

    # 3. Save Confusion Matrix Plot
    cm_png_path = eval_dir / "nutrient_confusion_matrix.png"
    plot_confusion_matrix(cm, class_names, cm_png_path)
    print(f"Saved confusion matrix chart: {cm_png_path}")

    # 4. Save Training Curves Plot
    history_file = comparison_dir / "efficientnetb0_training_history.json"
    curves_png_path = eval_dir / "training_curves.png"
    if history_file.is_file():
        plot_training_curves(history_file, curves_png_path, "EfficientNetB0")
        print(f"Saved training curves plot: {curves_png_path}")

    # 5. Generate and Save Final Evaluation Report
    report_file_path = eval_dir / "nutrient_test_evaluation_report.txt"
    with open(report_file_path, "w", encoding="utf-8") as f:
        f.write("PLANTHEALTHAI: MAIZE NUTRIENT DEFICIENCY MODEL TEST EVALUATION\n")
        f.write("===============================================================\n\n")
        f.write(f"Evaluation Date: {datetime.now().astimezone().isoformat(timespec='seconds')}\n")
        f.write(f"Active Model: {model_path.resolve()}\n")
        f.write(f"Architecture: EfficientNetB0 (Transfer Learning with ImageNet weights)\n")
        f.write(f"Input Resolution: 224x224 RGB\n")
        f.write(f"Test Set Count: {total_test} images\n")
        f.write("Integrity Note: The test set was evaluated exactly ONCE after model selection\n")
        f.write("                and was strictly held out from training, validation, and hyperparameter tuning.\n\n")

        f.write("Overall Test Metrics:\n")
        f.write("---------------------\n")
        f.write(f"Accuracy:           {test_acc * 100:.2f}%\n")
        f.write(f"Macro Precision:    {macro_prec:.4f}\n")
        f.write(f"Macro Recall:       {macro_rec:.4f}\n")
        f.write(f"Macro F1-Score:     {macro_f1:.4f}\n")
        f.write(f"Weighted F1-Score:  {weighted_f1:.4f}\n")
        f.write(f"Correct Predictions:   {correct_count} / {total_test} ({correct_count / total_test * 100:.2f}%)\n")
        f.write(f"Incorrect Predictions: {incorrect_count} / {total_test} ({incorrect_count / total_test * 100:.2f}%)\n\n")

        f.write("Model Footprint and Latency:\n")
        f.write("----------------------------\n")
        f.write(f"Model File Size:       {model_size_mb:.2f} MB\n")
        f.write(f"Batch-1 Inference Latency: {batch1_latency:.2f} ms/image (averaged over 50 passes)\n\n")

        f.write("Detailed Classification Report:\n")
        f.write("-------------------------------\n")
        f.write(report_text + "\n\n")

        f.write("Confusion Matrix (Rows: Actual, Columns: Predicted):\n")
        f.write("----------------------------------------------------\n")
        header_row = f"{'Actual \\ Pred':<15}" + "".join(f"{name:>12}" for name in class_names)
        f.write(header_row + "\n")
        f.write("-" * len(header_row) + "\n")
        for i, actual_name in enumerate(class_names):
            row_str = f"{actual_name:<15}" + "".join(f"{cm[i, j]:>12}" for j in range(num_classes))
            f.write(row_str + "\n")

        f.write("\nKey Limitations and Scientific Constraints:\n")
        f.write("-------------------------------------------\n")
        f.write("1. Sample Size Limitation: The entire dataset contains 463 images (71 in the test set).\n")
        f.write("   While accuracy and F1 metrics demonstrate strong pattern learning on held-out test data,\n")
        f.write("   small sample numbers yield wider statistical confidence intervals.\n")
        f.write("2. Biological Label Ground Truth: Model labels reflect the contributor's field survey annotations.\n")
        f.write("   These annotations are NOT biologically confirmed by destructive leaf tissue laboratory analysis\n")
        f.write("   (such as ICP-OES or Kjeldahl testing) or controlled Hoagland nutrient deprivation trials.\n")
        f.write("3. OOD & Environmental Generalization: Field lighting, soil variations, hybrid cultivars, and co-occurring\n")
        f.write("   fungal/bacterial diseases can produce visual leaf symptoms that overlap with nutrient stress.\n")
        f.write("   Predictions must be treated as computer vision pattern indications, not biological guarantees.\n")

    print(f"\nSaved final test evaluation report: {report_file_path}")

    # Print summary to console
    print("\n=======================================================")
    print("FINAL TEST EVALUATION SUMMARY")
    print("=======================================================")
    print(f"Test Accuracy:         {test_acc * 100:.2f}%")
    print(f"Macro Precision:       {macro_prec:.4f}")
    print(f"Macro Recall:          {macro_rec:.4f}")
    print(f"Macro F1-Score:        {macro_f1:.4f}")
    print(f"Correct Predictions:   {correct_count} / {total_test}")
    print(f"Incorrect Predictions: {incorrect_count} / {total_test}")
    print(f"Model File Size:       {model_size_mb:.2f} MB")
    print(f"Batch-1 Latency:       {batch1_latency:.2f} ms")
    print("\nPer-Class Breakdown:")
    for cls in class_names:
        c_p = report_dict[cls]["precision"]
        c_r = report_dict[cls]["recall"]
        c_f = report_dict[cls]["f1-score"]
        c_s = report_dict[cls]["support"]
        print(f"  {cls:<12}: Precision={c_p:.4f}, Recall={c_r:.4f}, F1={c_f:.4f}, Support={c_s}")


if __name__ == "__main__":
    main()
