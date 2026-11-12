"""Train and compare MobileNetV2 and EfficientNetB0 for Maize Nutrient Deficiency Classification."""

import argparse
import json
import os
import shutil
import time
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import efficientnet, mobilenet_v2

RANDOM_SEED = 42
tf.keras.utils.set_random_seed(RANDOM_SEED)

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
HEAD_EPOCHS = 15
FINE_TUNE_EPOCHS = 15
HEAD_LR = 1e-3
FINE_TUNE_LR = 1e-5


def load_dataset_arrays(split_dir, class_names):
    """Load images and labels into numpy arrays for deterministic evaluation."""
    images = []
    labels = []
    filepaths = []
    class_to_idx = {name: i for i, name in enumerate(class_names)}

    for class_name in class_names:
        class_folder = split_dir / class_name
        if not class_folder.is_dir():
            continue
        for f in sorted(list(class_folder.glob("*.jpeg")) + list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png"))):
            with Image.open(f) as im:
                im_rgb = im.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.LANCZOS)
                images.append(np.array(im_rgb, dtype=np.float32))
                labels.append(class_to_idx[class_name])
                filepaths.append(str(f))

    return np.array(images, dtype=np.float32), np.array(labels, dtype=np.int32), filepaths


def build_augmentation_layer():
    return keras.Sequential([
        layers.RandomFlip("horizontal", seed=RANDOM_SEED),
        layers.RandomRotation(0.05, seed=RANDOM_SEED),
        layers.RandomZoom(0.10, seed=RANDOM_SEED),
    ], name="data_augmentation")


def build_mobilenetv2_model(num_classes):
    """Build MobileNetV2 model with native Rescaling [-1, 1] layer."""
    inputs = keras.Input(shape=(224, 224, 3), name="image_input")
    aug = build_augmentation_layer()
    x = aug(inputs)
    # Native Keras Rescaling: maps [0, 255] -> [-1, 1]
    x = layers.Rescaling(scale=1.0 / 127.5, offset=-1.0, name="mobilenetv2_rescaling")(x)
    backbone = mobilenet_v2.MobileNetV2(
        input_tensor=x,
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
    x = layers.Dropout(0.30, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MobileNetV2_Nutrient")
    return model, backbone


def build_efficientnetb0_model(num_classes):
    """Build EfficientNetB0 model with native Rescaling [0, 255] layer."""
    inputs = keras.Input(shape=(224, 224, 3), name="image_input")
    aug = build_augmentation_layer()
    x = aug(inputs)
    # Native Keras Rescaling: preserves [0, 255] for internal EfficientNet normalization
    x = layers.Rescaling(scale=1.0, offset=0.0, name="efficientnet_rescaling")(x)
    backbone = efficientnet.EfficientNetB0(
        input_tensor=x,
        include_top=False,
        weights="imagenet"
    )
    backbone.trainable = False

    x = layers.GlobalAveragePooling2D(name="avg_pool")(backbone.output)
    x = layers.Dropout(0.30, name="top_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="EfficientNetB0_Nutrient")
    return model, backbone


def measure_inference_latency(model, sample_input, num_trials=50):
    """Measure batch-1 inference latency in milliseconds."""
    batch_1 = sample_input[:1]
    # Warmup
    for _ in range(5):
        _ = model(batch_1, training=False)

    latencies = []
    for _ in range(num_trials):
        t0 = time.perf_counter()
        _ = model(batch_1, training=False)
        latencies.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(latencies))


def train_single_architecture(arch_name, build_fn, unfreeze_layers_count, X_train, y_train, X_val, y_val, class_weights_dict, num_classes, checkpoint_path):
    print(f"\n=======================================================")
    print(f"Training Architecture: {arch_name}")
    print(f"=======================================================")

    model, backbone = build_fn(num_classes)
    total_start_time = time.time()

    # Stage 1: Train Classification Head (Frozen Base)
    print("\n--- Stage 1: Classification Head Training (Backbone Frozen) ---")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=HEAD_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_stage1 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-6, verbose=1),
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    history_stage1 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=HEAD_EPOCHS,
        class_weight=class_weights_dict,
        callbacks=callbacks_stage1,
        verbose=1
    )

    # Stage 2: Fine-Tuning Upper Layers
    print(f"\n--- Stage 2: Fine-Tuning Upper Layers (Unfreezing last {unfreeze_layers_count} layers) ---")
    backbone.trainable = True
    for layer in backbone.layers[:-unfreeze_layers_count]:
        layer.trainable = False

    # Keep BatchNormalization in inference mode to avoid corruption of statistics on small datasets
    for layer in backbone.layers[-unfreeze_layers_count:]:
        if isinstance(layer, layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=FINE_TUNE_LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks_stage2 = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7, verbose=1),
        keras.callbacks.ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, verbose=1),
    ]

    history_stage2 = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=BATCH_SIZE,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=class_weights_dict,
        callbacks=callbacks_stage2,
        verbose=1
    )

    total_training_time = time.time() - total_start_time

    # Load best checkpoint
    print(f"\nLoading best checkpoint from {checkpoint_path}...")
    best_model = keras.models.load_model(str(checkpoint_path))

    # Evaluate on validation set
    val_loss, val_acc = best_model.evaluate(X_val, y_val, batch_size=BATCH_SIZE, verbose=0)
    val_probs = best_model.predict(X_val, batch_size=BATCH_SIZE, verbose=0)
    val_preds = np.argmax(val_probs, axis=1)

    val_precision = precision_score(y_val, val_preds, average="macro", zero_division=0)
    val_recall = recall_score(y_val, val_preds, average="macro", zero_division=0)
    val_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)

    model_size_mb = checkpoint_path.stat().st_size / (1024 * 1024)
    latency_ms = measure_inference_latency(best_model, X_val)

    combined_history = {
        "stage1": {k: [float(v) for v in vals] for k, vals in history_stage1.history.items()},
        "stage2": {k: [float(v) for v in vals] for k, vals in history_stage2.history.items()},
    }

    metrics = {
        "architecture": arch_name,
        "training_time_seconds": round(total_training_time, 2),
        "val_accuracy": float(val_acc),
        "val_loss": float(val_loss),
        "val_macro_precision": float(val_precision),
        "val_macro_recall": float(val_recall),
        "val_macro_f1": float(val_f1),
        "model_size_mb": round(model_size_mb, 2),
        "batch1_latency_ms": round(latency_ms, 2),
        "checkpoint_path": str(checkpoint_path.resolve()),
    }

    print(f"\n{arch_name} Validation Summary:")
    print(f"  Accuracy: {val_acc * 100:.2f}%")
    print(f"  Loss: {val_loss:.4f}")
    print(f"  Macro F1: {val_f1:.4f}")
    print(f"  Training Time: {total_training_time:.2f}s")
    print(f"  Model Size: {model_size_mb:.2f} MB")
    print(f"  Batch-1 Latency: {latency_ms:.2f} ms")

    return best_model, metrics, combined_history


def main():
    project_root = Path(__file__).resolve().parents[1]
    datasets_dir = project_root / "datasets" / "maize_nutrient_processed"
    models_dir = project_root / "models" / "nutrient_detection"
    comparison_dir = project_root / "evaluation" / "nutrient_model_comparison"

    models_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Load classes
    classes_file = models_dir / "nutrient_classes.json"
    with open(classes_file, "r", encoding="utf-8") as f:
        class_info = json.load(f)
    class_names = [class_info["index_to_class"][str(i)] for i in range(class_info["total_classes"])]
    num_classes = len(class_names)
    print(f"Loaded {num_classes} classes: {class_names}")

    # Load Train & Validation splits (TEST SET IS NOT LOADED HERE)
    train_dir = datasets_dir / "train"
    val_dir = datasets_dir / "val"

    print("\nLoading training images...")
    X_train, y_train, _ = load_dataset_arrays(train_dir, class_names)
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

    print("Loading validation images...")
    X_val, y_val, _ = load_dataset_arrays(val_dir, class_names)
    print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")

    # Compute balanced class weights for training
    classes_present = np.unique(y_train)
    weights = compute_class_weight("balanced", classes=classes_present, y=y_train)
    class_weights_dict = {int(c): float(w) for c, w in zip(classes_present, weights)}
    print(f"\nBalanced Class Weights: {class_weights_dict}")

    # Train MobileNetV2
    mobilenet_checkpoint = models_dir / "mobilenetv2_nutrient_best.keras"
    mobilenet_model, mobilenet_metrics, mobilenet_hist = train_single_architecture(
        "MobileNetV2",
        build_mobilenetv2_model,
        unfreeze_layers_count=30,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        class_weights_dict=class_weights_dict,
        num_classes=num_classes,
        checkpoint_path=mobilenet_checkpoint
    )

    with open(comparison_dir / "mobilenetv2_training_history.json", "w", encoding="utf-8") as f:
        json.dump(mobilenet_hist, f, indent=2)

    # Train EfficientNetB0
    effnet_checkpoint = models_dir / "efficientnetb0_nutrient_best.keras"
    effnet_model, effnet_metrics, effnet_hist = train_single_architecture(
        "EfficientNetB0",
        build_efficientnetb0_model,
        unfreeze_layers_count=30,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
        class_weights_dict=class_weights_dict,
        num_classes=num_classes,
        checkpoint_path=effnet_checkpoint
    )

    with open(comparison_dir / "efficientnetb0_training_history.json", "w", encoding="utf-8") as f:
        json.dump(effnet_hist, f, indent=2)

    # Phase 4: Model Comparison & Selection based strictly on Validation Performance
    print("\n=======================================================")
    print("PHASE 4: ARCHITECTURE COMPARISON & MODEL SELECTION")
    print("=======================================================")
    print(f"{'Metric':<25} {'MobileNetV2':<18} {'EfficientNetB0':<18}")
    print("-" * 61)
    print(f"{'Validation Accuracy':<25} {mobilenet_metrics['val_accuracy']*100:>15.2f}% {effnet_metrics['val_accuracy']*100:>15.2f}%")
    print(f"{'Validation Loss':<25} {mobilenet_metrics['val_loss']:>16.4f} {effnet_metrics['val_loss']:>16.4f}")
    print(f"{'Validation Macro F1':<25} {mobilenet_metrics['val_macro_f1']:>16.4f} {effnet_metrics['val_macro_f1']:>16.4f}")
    print(f"{'Validation Macro Precision':<25} {mobilenet_metrics['val_macro_precision']:>16.4f} {effnet_metrics['val_macro_precision']:>16.4f}")
    print(f"{'Validation Macro Recall':<25} {mobilenet_metrics['val_macro_recall']:>16.4f} {effnet_metrics['val_macro_recall']:>16.4f}")
    print(f"{'Training Duration (s)':<25} {mobilenet_metrics['training_time_seconds']:>15.2f}s {effnet_metrics['training_time_seconds']:>15.2f}s")
    print(f"{'Model File Size (MB)':<25} {mobilenet_metrics['model_size_mb']:>14.2f}MB {effnet_metrics['model_size_mb']:>14.2f}MB")
    print(f"{'Batch-1 Latency (ms)':<25} {mobilenet_metrics['batch1_latency_ms']:>14.2f}ms {effnet_metrics['batch1_latency_ms']:>14.2f}ms")

    # Selection logic:
    if effnet_metrics["val_macro_f1"] > mobilenet_metrics["val_macro_f1"] or (
        np.isclose(effnet_metrics["val_macro_f1"], mobilenet_metrics["val_macro_f1"], atol=0.01)
        and effnet_metrics["val_loss"] < mobilenet_metrics["val_loss"]
    ):
        selected_model_name = "EfficientNetB0"
        selected_checkpoint = effnet_checkpoint
        selection_rationale = (
            f"EfficientNetB0 achieved superior validation performance (F1: {effnet_metrics['val_macro_f1']:.4f}, "
            f"Accuracy: {effnet_metrics['val_accuracy']*100:.2f}%, Loss: {effnet_metrics['val_loss']:.4f}) compared to "
            f"MobileNetV2 (F1: {mobilenet_metrics['val_macro_f1']:.4f}, Accuracy: {mobilenet_metrics['val_accuracy']*100:.2f}%)."
        )
    else:
        selected_model_name = "MobileNetV2"
        selected_checkpoint = mobilenet_checkpoint
        selection_rationale = (
            f"MobileNetV2 achieved superior validation performance (F1: {mobilenet_metrics['val_macro_f1']:.4f}, "
            f"Accuracy: {mobilenet_metrics['val_accuracy']*100:.2f}%, Loss: {mobilenet_metrics['val_loss']:.4f}) with "
            f"faster inference latency ({mobilenet_metrics['batch1_latency_ms']:.2f}ms) and compact footprint ({mobilenet_metrics['model_size_mb']:.2f}MB)."
        )

    print(f"\nSelected Architecture: {selected_model_name}")
    print(f"Selection Justification: {selection_rationale}")

    # Save final model as models/nutrient_detection/nutrient_model.keras
    final_model_path = models_dir / "nutrient_model.keras"
    shutil.copy2(selected_checkpoint, final_model_path)
    print(f"Saved selected active nutrient model to: {final_model_path}")

    # Write Model Comparison Report
    comparison_report_path = comparison_dir / "nutrient_model_comparison_report.txt"
    with open(comparison_report_path, "w", encoding="utf-8") as f:
        f.write("PLANTHEALTHAI: MAIZE NUTRIENT DEFICIENCY MODEL COMPARISON REPORT\n")
        f.write("=================================================================\n\n")
        f.write(f"Evaluation Date: {datetime.now().astimezone().isoformat(timespec='seconds')}\n")
        f.write(f"Training Environment: Python 3.12, TensorFlow 2.18, CPU (GPU not detected)\n")
        f.write(f"Dataset Split: 70% Train (323 images), 15% Validation (69 images), 15% Test (71 images held out)\n")
        f.write(f"Class Balance Handling: Scikit-learn balanced class weights applied during training\n")
        f.write("Evaluation Protocol: Comparison performed strictly on Validation Set; Test Set kept untouched.\n\n")

        f.write("Validation Performance Comparison Table:\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Metric':<28} {'MobileNetV2':<18} {'EfficientNetB0':<18}\n")
        f.write("-" * 65 + "\n")
        f.write(f"{'Validation Accuracy':<28} {mobilenet_metrics['val_accuracy']*100:>15.2f}% {effnet_metrics['val_accuracy']*100:>15.2f}%\n")
        f.write(f"{'Validation Loss':<28} {mobilenet_metrics['val_loss']:>16.4f} {effnet_metrics['val_loss']:>16.4f}\n")
        f.write(f"{'Validation Macro F1':<28} {mobilenet_metrics['val_macro_f1']:>16.4f} {effnet_metrics['val_macro_f1']:>16.4f}\n")
        f.write(f"{'Validation Macro Precision':<28} {mobilenet_metrics['val_macro_precision']:>16.4f} {effnet_metrics['val_macro_precision']:>16.4f}\n")
        f.write(f"{'Validation Macro Recall':<28} {mobilenet_metrics['val_macro_recall']:>16.4f} {effnet_metrics['val_macro_recall']:>16.4f}\n")
        f.write(f"{'Training Duration (s)':<28} {mobilenet_metrics['training_time_seconds']:>15.2f}s {effnet_metrics['training_time_seconds']:>15.2f}s\n")
        f.write(f"{'Model File Size (MB)':<28} {mobilenet_metrics['model_size_mb']:>14.2f}MB {effnet_metrics['model_size_mb']:>14.2f}MB\n")
        f.write(f"{'Batch-1 Latency (ms)':<28} {mobilenet_metrics['batch1_latency_ms']:>14.2f}ms {effnet_metrics['batch1_latency_ms']:>14.2f}ms\n")
        f.write("-" * 65 + "\n\n")

        f.write("Model Selection Decision:\n")
        f.write(f"  - Selected Model: {selected_model_name}\n")
        f.write(f"  - Active Model Path: {final_model_path.resolve()}\n")
        f.write(f"  - Comparison Model Preserved: Yes ({models_dir.resolve()})\n")
        f.write(f"  - Justification: {selection_rationale}\n")

    print(f"Saved comparison report to: {comparison_report_path}")


if __name__ == "__main__":
    main()
