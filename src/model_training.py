"""Crop disease detection model training with MobileNetV2 transfer learning.

This module implements:
- Model creation (MobileNetV2 base + custom head)
- Two-phase transfer learning (freeze base, then fine-tune)
- Data augmentation via dataset_prep
- Training with callbacks (early stopping, checkpoint, LR scheduler)
- Validation and evaluation
- Accuracy/loss plotting
- Confusion matrix generation
- Model saving (SavedModel format) + label persistence
- Standalone prediction function: predict_disease(image_path)
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import IMAGE_SIZE, IMAGE_CHANNELS, LABELS_PATH, MODEL_PATH
from src.dataset_prep import (
    prepare_datasets,
    save_label_map,
    load_label_map,
    decode_labels,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path("models/disease_model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CONFIG = {
    "batch_size": 32,
    "val_split": 0.15,
    "test_split": 0.15,
    "epochs_head": 15,        # Phase 1: train head only (base frozen)
    "epochs_fine": 10,        # Phase 2: fine-tune top layers
    "learning_rate_head": 1e-3,
    "learning_rate_fine": 1e-5,
    "fine_tune_at": 100,      # Unfreeze layers from this index onward
    "early_stopping_patience": 5,
    "reduce_lr_patience": 3,
    "reduce_lr_factor": 0.2,
    "seed": 42,
}

AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "vertical_flip": False,
    "rotation_factor": 0.15,
    "zoom_factor": 0.15,
    "height_factor": 0.1,
    "width_factor": 0.1,
    "brightness_factor": 0.2,
    "contrast_factor": 0.1,
    "fill_mode": "reflect",
    "seed": 42,
}

# Severity mapping per class (used by prediction function)
SEVERITY_MAP = {
    "Healthy": "None",
    "Early_Blight": "Moderate",
    "Late_Blight": "High",
}


# ---------------------------------------------------------------------------
# 1. Model creation
# ---------------------------------------------------------------------------

def create_model(
    num_classes: int,
    input_shape: tuple[int, int, int] = (*IMAGE_SIZE, IMAGE_CHANNELS),
    dropout_rate: float = 0.2,
) -> tf.keras.Model:
    """Build MobileNetV2 transfer learning model.

    Architecture:
    - MobileNetV2 base (ImageNet weights, no top)
    - GlobalAveragePooling2D
    - Dropout
    - Dense(num_classes, softmax)
    """
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # frozen initially

    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = tf.keras.Model(inputs, outputs, name="crop_disease_mobilenetv2")
    return model, base_model


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    """Compile model with Adam optimizer and sparse categorical crossentropy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


# ---------------------------------------------------------------------------
# 2. Callbacks
# ---------------------------------------------------------------------------

def get_callbacks(model_dir: Path, patience: int = 5) -> list[tf.keras.callbacks.Callback]:
    """Standard training callbacks."""
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_dir / "best_model.keras"),
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=3,
            factor=0.2,
            min_lr=1e-7,
            verbose=1,
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir=str(model_dir / "logs"),
            histogram_freq=1,
        ),
    ]


# ---------------------------------------------------------------------------
# 3. Training phases
# ---------------------------------------------------------------------------

def train_head(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    learning_rate: float,
    callbacks: list[tf.keras.callbacks.Callback],
) -> tf.keras.callbacks.History:
    """Phase 1: Train only the classification head (base frozen)."""
    base_model.trainable = False
    compile_model(model, learning_rate)
    print(f"\n{'='*60}")
    print(f"PHASE 1: Training head only (base frozen)")
    print(f"Epochs: {epochs}, LR: {learning_rate}")
    print(f"{'='*60}\n")
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )


def fine_tune(
    model: tf.keras.Model,
    base_model: tf.keras.Model,
    train_ds: tf.data.Dataset,
    val_ds: tf.data.Dataset,
    epochs: int,
    learning_rate: float,
    fine_tune_at: int,
    callbacks: list[tf.keras.callbacks.Callback],
) -> tf.keras.callbacks.History:
    """Phase 2: Fine-tune top layers of base model."""
    base_model.trainable = True
    # Freeze layers before fine_tune_at
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base_model.layers[fine_tune_at:]:
        layer.trainable = True

    compile_model(model, learning_rate)
    print(f"\n{'='*60}")
    print(f"PHASE 2: Fine-tuning (layers {fine_tune_at}+ unfrozen)")
    print(f"Epochs: {epochs}, LR: {learning_rate}")
    print(f"Trainable params: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
    print(f"{'='*60}\n")
    return model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )


# ---------------------------------------------------------------------------
# 4. Evaluation & visualization
# ---------------------------------------------------------------------------

def evaluate_model(
    model: tf.keras.Model,
    test_ds: tf.data.Dataset,
    class_names: list[str],
    model_dir: Path,
) -> dict[str, Any]:
    """Evaluate on test set, generate plots and confusion matrix."""
    # Predictions
    y_true = []
    y_pred = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Metrics
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    print(f"\nTest Accuracy: {test_acc:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save metrics
    metrics = {
        "test_accuracy": float(test_acc),
        "test_loss": float(test_loss),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    (model_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Plot confusion matrix
    plot_confusion_matrix(cm, class_names, model_dir / "confusion_matrix.png")

    return metrics


def plot_training_history(history_head: tf.keras.callbacks.History,
                          history_fine: tf.keras.callbacks.History | None,
                          model_dir: Path) -> None:
    """Plot accuracy and loss curves for both training phases."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy
    ax = axes[0]
    ax.plot(history_head.history["accuracy"], label="Train (head)", color="#2e7d32")
    ax.plot(history_head.history["val_accuracy"], label="Val (head)", color="#66bb6a")
    if history_fine:
        offset = len(history_head.history["accuracy"])
        ax.axvline(x=offset - 0.5, color="gray", linestyle="--", alpha=0.5)
        ax.plot(range(offset, offset + len(history_fine.history["accuracy"])),
                history_fine.history["accuracy"], label="Train (fine-tune)", color="#1b5e20")
        ax.plot(range(offset, offset + len(history_fine.history["val_accuracy"])),
                history_fine.history["val_accuracy"], label="Val (fine-tune)", color="#4caf50")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Model Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[1]
    ax.plot(history_head.history["loss"], label="Train (head)", color="#c62828")
    ax.plot(history_head.history["val_loss"], label="Val (head)", color="#ef5350")
    if history_fine:
        offset = len(history_head.history["loss"])
        ax.axvline(x=offset - 0.5, color="gray", linestyle="--", alpha=0.5)
        ax.plot(range(offset, offset + len(history_fine.history["loss"])),
                history_fine.history["loss"], label="Train (fine-tune)", color="#b71c1c")
        ax.plot(range(offset, offset + len(history_fine.history["val_loss"])),
                history_fine.history["val_loss"], label="Val (fine-tune)", color="#e53935")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Model Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], save_path: Path) -> None:
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# 5. Model persistence
# ---------------------------------------------------------------------------

def save_model(model: tf.keras.Model, class_names: list[str], model_dir: Path) -> None:
    """Save model in SavedModel format and persist labels."""
    # SavedModel format (directory)
    tf.saved_model.save(model, str(model_dir / "saved_model"))
    # Also save as .keras for easy loading
    model.save(model_dir / "model.keras")
    # Save labels
    save_label_map(class_names, model_dir / "labels.json")
    # Copy to config location for UI
    save_label_map(class_names, LABELS_PATH)
    # Copy model to config location
    model.save(MODEL_PATH)
    print(f"Model saved to {model_dir}")
    print(f"Labels saved to {model_dir / 'labels.json'} and {LABELS_PATH}")


# ---------------------------------------------------------------------------
# 6. Prediction function (standalone, for UI integration)
# ---------------------------------------------------------------------------

def load_trained_model(model_dir: Path | str = None) -> tuple[tf.keras.Model, list[str]]:
    """Load trained model and class labels."""
    if model_dir is None:
        model_dir = MODEL_DIR
    model_dir = Path(model_dir)

    model = tf.keras.models.load_model(model_dir / "model.keras")
    class_names = load_label_map(model_dir / "labels.json")
    # Convert to list in correct order
    inv_map = {v: k for k, v in class_names.items()}
    class_names = [inv_map[i] for i in range(len(inv_map))]
    return model, class_names


def preprocess_single_image(image_path: str | Path, target_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """Load and preprocess a single image for prediction."""
    img = tf.io.read_file(str(image_path))
    img = tf.image.decode_image(img, channels=IMAGE_CHANNELS, expand_animations=False)
    img = tf.image.resize(img, target_size, method="bilinear")
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    img = tf.expand_dims(img, axis=0)  # add batch dimension
    return img.numpy()


def predict_disease(
    image_path: str | Path,
    model: tf.keras.Model | None = None,
    class_names: list[str] | None = None,
    model_dir: Path | str = None,
) -> dict[str, Any]:
    """Predict disease from a leaf image.

    Args:
        image_path: Path to the leaf image file.
        model: Optional pre-loaded model. If not provided, loads from model_dir.
        class_names: Optional pre-loaded class names. If not provided, loads from model_dir.
        model_dir: Directory containing saved model and labels.

    Returns:
        Dict with keys: disease (str), confidence (float), severity (str).
    """
    # Load model/labels if not provided
    if model is None or class_names is None:
        model, class_names = load_trained_model(model_dir)

    # Preprocess
    img_batch = preprocess_single_image(image_path)

    # Predict
    preds = model.predict(img_batch, verbose=0)[0]
    pred_idx = int(np.argmax(preds))
    confidence = float(preds[pred_idx])
    disease = class_names[pred_idx]
    severity = SEVERITY_MAP.get(disease, "Unknown")

    return {
        "disease": disease,
        "confidence": confidence,
        "severity": severity,
    }


# ---------------------------------------------------------------------------
# 7. Main training pipeline
# ---------------------------------------------------------------------------

def run_training(data_dir: str | Path, **overrides: Any) -> dict[str, Any]:
    """Complete training pipeline from data directory to saved model.

    Args:
        data_dir: Path to class-subfolder image directory.
        **overrides: Any TRAIN_CONFIG keys to override.

    Returns:
        Dict with model, history, metrics, and paths.
    """
    config = {**TRAIN_CONFIG, **overrides}

    # 1. Prepare datasets
    print(f"Preparing datasets from {data_dir}...")
    data = prepare_datasets(
        data_dir,
        batch_size=config["batch_size"],
        val_split=config["val_split"],
        test_split=config["test_split"],
        augment_train=True,
        augmentation_config=AUGMENTATION_CONFIG,
        seed=config["seed"],
    )

    class_names = data["class_names"]
    num_classes = data["num_classes"]
    print(f"Classes: {class_names} ({num_classes})")
    print(f"Train/Val/Test: {data['counts']['train']}/{data['counts']['val']}/{data['counts']['test']}")

    # 2. Create model
    model, base_model = create_model(num_classes)

    # 3. Callbacks
    callbacks = get_callbacks(MODEL_DIR, config["early_stopping_patience"])

    # 4. Phase 1: Train head
    history_head = train_head(
        model, base_model,
        data["train_ds"], data["val_ds"],
        epochs=config["epochs_head"],
        learning_rate=config["learning_rate_head"],
        callbacks=callbacks,
    )

    # 5. Phase 2: Fine-tune
    history_fine = fine_tune(
        model, base_model,
        data["train_ds"], data["val_ds"],
        epochs=config["epochs_fine"],
        learning_rate=config["learning_rate_fine"],
        fine_tune_at=config["fine_tune_at"],
        callbacks=callbacks,
    )

    # 6. Evaluate on test set
    print("\nEvaluating on test set...")
    metrics = evaluate_model(model, data["test_ds"], class_names, MODEL_DIR)

    # 7. Plot training curves
    plot_training_history(history_head, history_fine, MODEL_DIR)

    # 8. Save model
    save_model(model, class_names, MODEL_DIR)

    return {
        "model": model,
        "class_names": class_names,
        "history_head": history_head,
        "history_fine": history_fine,
        "metrics": metrics,
        "model_dir": MODEL_DIR,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train crop disease detection model")
    parser.add_argument("--data-dir", default="data/samples", help="Path to class-subfolder image directory")
    parser.add_argument("--epochs-head", type=int, default=TRAIN_CONFIG["epochs_head"])
    parser.add_argument("--epochs-fine", type=int, default=TRAIN_CONFIG["epochs_fine"])
    parser.add_argument("--batch-size", type=int, default=TRAIN_CONFIG["batch_size"])
    parser.add_argument("--no-fine-tune", action="store_true", help="Skip fine-tuning phase")
    args = parser.parse_args()

    data_path = Path(args.data_dir)
    if not data_path.exists():
        print(f"Data directory not found: {data_path}")
        print("Expected structure: data/samples/<class_name>/*.jpg")
        print("Run with --data-dir pointing to your dataset, or place images in data/samples/")
        exit(1)

    overrides = {
        "epochs_head": args.epochs_head,
        "epochs_fine": 0 if args.no_fine_tune else args.epochs_fine,
        "batch_size": args.batch_size,
    }

    result = run_training(data_path, **overrides)
    print("\n=== Training Complete ===")
    print(f"Test Accuracy: {result['metrics']['test_accuracy']:.4f}")
    print(f"Model saved to: {result['model_dir']}")