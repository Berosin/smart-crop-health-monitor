"""Leaf disease image dataset preparation for deep learning.

This module provides reusable TensorFlow/Keras utilities for:
- Loading images from a directory structure (class subfolders)
- Resizing and normalizing images
- Label encoding (class names <-> integers)
- Train / validation / test splits (stratified)
- Data augmentation (random flip, rotation, zoom, brightness, etc.)
- Creating tf.data.Dataset pipelines with prefetching

Expected directory structure:
    data/samples/
        Tomato___Early_Blight/
            img1.jpg
            img2.jpg
            ...
        Tomato___Late_Blight/
            ...
        Healthy/
            ...

Or any flat structure where a CSV maps filenames to labels.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from config import IMAGE_SIZE, IMAGE_CHANNELS


# ---------------------------------------------------------------------------
# 1. Label encoding utilities
# ---------------------------------------------------------------------------

def get_class_names_from_dir(data_dir: str | Path) -> list[str]:
    """Infer class names from subdirectory names (sorted for determinism)."""
    p = Path(data_dir)
    classes = sorted([d.name for d in p.iterdir() if d.is_dir()])
    if not classes:
        raise ValueError(f"No class subdirectories found in {data_dir}")
    return classes


def save_label_map(class_names: list[str], path: str | Path = None) -> Path:
    """Persist class name <-> integer mapping to JSON."""
    if path is None:
        from config import LABELS_PATH
        path = LABELS_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    label_map = {name: i for i, name in enumerate(class_names)}
    path.write_text(json.dumps(label_map, indent=2))
    return path


def load_label_map(path: str | Path = None) -> dict[str, int]:
    """Load class name <-> integer mapping from JSON."""
    if path is None:
        from config import LABELS_PATH
        path = LABELS_PATH
    return json.loads(Path(path).read_text())


def encode_labels(labels: list[str], label_map: dict[str, int]) -> np.ndarray:
    """Convert list of class names to integer labels."""
    return np.array([label_map[l] for l in labels], dtype=np.int32)


def decode_labels(encoded: np.ndarray, label_map: dict[str, int]) -> list[str]:
    """Convert integer labels back to class names."""
    inv_map = {v: k for k, v in label_map.items()}
    return [inv_map[i] for i in encoded]


# ---------------------------------------------------------------------------
# 2. Image loading from directory (class subfolders)
# ---------------------------------------------------------------------------

def load_image_paths_and_labels(
    data_dir: str | Path,
    class_names: list[str] | None = None,
    extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff"),
) -> tuple[list[str], list[str]]:
    """Walk class subdirectories and collect (image_path, class_name) pairs.

    Returns:
        (paths, labels) where labels are class names (strings).
    """
    p = Path(data_dir)
    if class_names is None:
        class_names = get_class_names_from_dir(p)

    paths, labels = [], []
    for class_name in class_names:
        class_dir = p / class_name
        if not class_dir.is_dir():
            continue
        for ext in extensions:
            for img_path in class_dir.glob(f"*{ext}"):
                paths.append(str(img_path))
                labels.append(class_name)
    if not paths:
        raise ValueError(f"No images found in {data_dir} with extensions {extensions}")
    return paths, labels


# ---------------------------------------------------------------------------
# 3. tf.data pipeline: preprocessing + augmentation
# ---------------------------------------------------------------------------

AUGMENTATION_DEFAULTS = {
    "horizontal_flip": True,
    "vertical_flip": False,
    "rotation_factor": 0.15,        # fraction of 2pi
    "zoom_factor": 0.15,            # fraction of size
    "height_factor": 0.1,
    "width_factor": 0.1,
    "brightness_factor": 0.2,       # fraction of max pixel value
    "contrast_factor": 0.1,
    "fill_mode": "reflect",
    "seed": 42,
}


def build_augmentation_layer(config: dict[str, Any] | None = None) -> tf.keras.Sequential:
    """Construct a Keras preprocessing Sequential model for data augmentation.

    All layers are stateless and run on GPU/TPU during training.
    """
    cfg = {**AUGMENTATION_DEFAULTS, **(config or {})}

    layers = []
    if cfg.get("horizontal_flip"):
        layers.append(tf.keras.layers.RandomFlip("horizontal", seed=cfg["seed"]))
    if cfg.get("vertical_flip"):
        layers.append(tf.keras.layers.RandomFlip("vertical", seed=cfg["seed"]))
    if cfg.get("rotation_factor"):
        layers.append(tf.keras.layers.RandomRotation(
            cfg["rotation_factor"], fill_mode=cfg["fill_mode"], seed=cfg["seed"]))
    if cfg.get("zoom_factor"):
        layers.append(tf.keras.layers.RandomZoom(
            cfg["zoom_factor"], fill_mode=cfg["fill_mode"], seed=cfg["seed"]))
    if cfg.get("height_factor") or cfg.get("width_factor"):
        layers.append(tf.keras.layers.RandomTranslation(
            height_factor=cfg.get("height_factor", 0),
            width_factor=cfg.get("width_factor", 0),
            fill_mode=cfg["fill_mode"],
            seed=cfg["seed"],
        ))
    if cfg.get("brightness_factor"):
        layers.append(tf.keras.layers.RandomBrightness(
            cfg["brightness_factor"], value_range=(0, 1), seed=cfg["seed"]))
    if cfg.get("contrast_factor"):
        layers.append(tf.keras.layers.RandomContrast(
            cfg["contrast_factor"], seed=cfg["seed"]))

    return tf.keras.Sequential(layers, name="augmentation")


def preprocess_image(
    image_path: tf.Tensor,
    label: tf.Tensor,
    target_size: tuple[int, int] = IMAGE_SIZE,
    channels: int = IMAGE_CHANNELS,
    augment: bool = False,
    augmentation_layer: tf.keras.Sequential | None = None,
) -> tuple[tf.Tensor, tf.Tensor]:
    """Load, decode, resize, normalize a single image.

    Normalization: pixel values scaled to [0, 1] (float32).
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=channels, expand_animations=False)
    img = tf.image.resize(img, target_size, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((*target_size, channels))

    if augment and augmentation_layer is not None:
        img = augmentation_layer(img, training=True)

    return img, label


def create_dataset(
    image_paths: list[str],
    labels: np.ndarray,
    batch_size: int = 32,
    shuffle: bool = True,
    augment: bool = False,
    augmentation_config: dict[str, Any] | None = None,
    target_size: tuple[int, int] = IMAGE_SIZE,
    channels: int = IMAGE_CHANNELS,
    seed: int = 42,
) -> tf.data.Dataset:
    """Build a tf.data.Dataset pipeline from image paths and integer labels.

    Pipeline: list -> dataset -> map(preprocess) -> (shuffle) -> batch -> prefetch.
    """
    aug_layer = build_augmentation_layer(augmentation_config) if augment else None

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: preprocess_image(p, y, target_size, channels, augment, aug_layer),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


# ---------------------------------------------------------------------------
# 4. High-level: stratified split + dataset creation
# ---------------------------------------------------------------------------

def prepare_datasets(
    data_dir: str | Path,
    batch_size: int = 32,
    val_split: float = 0.15,
    test_split: float = 0.15,
    augment_train: bool = True,
    augmentation_config: dict[str, Any] | None = None,
    target_size: tuple[int, int] = IMAGE_SIZE,
    channels: int = IMAGE_CHANNELS,
    seed: int = 42,
    class_names: list[str] | None = None,
    save_labels: bool = True,
) -> dict[str, Any]:
    """End-to-end dataset preparation from a class-subfolder directory.

    Returns a dict with:
        - "train_ds", "val_ds", "test_ds": tf.data.Dataset objects
        - "class_names": list of class names (str)
        - "num_classes": int
        - "label_map": dict[str, int]
        - "steps_per_epoch": dict with train/val/test steps
    """
    # 1. Load paths + string labels
    paths, str_labels = load_image_paths_and_labels(data_dir, class_names)
    if class_names is None:
        class_names = sorted(set(str_labels))

    # 2. Build / load label map
    label_map = {name: i for i, name in enumerate(class_names)}
    if save_labels:
        save_label_map(class_names)

    # 3. Encode labels to integers
    int_labels = encode_labels(str_labels, label_map)

    # 4. Stratified split: first split off test, then val from remainder
    #    (train = 1 - val_split - test_split)
    test_size = test_split
    val_size = val_split / (1 - test_split)

    paths_train, paths_test, y_train, y_test = train_test_split(
        paths, int_labels, test_size=test_size, stratify=int_labels, random_state=seed
    )
    paths_train, paths_val, y_train, y_val = train_test_split(
        paths_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )

    # 5. Create tf.data pipelines
    train_ds = create_dataset(
        paths_train, y_train, batch_size=batch_size, shuffle=True,
        augment=augment_train, augmentation_config=augmentation_config,
        target_size=target_size, channels=channels, seed=seed
    )
    val_ds = create_dataset(
        paths_val, y_val, batch_size=batch_size, shuffle=False,
        augment=False, target_size=target_size, channels=channels
    )
    test_ds = create_dataset(
        paths_test, y_test, batch_size=batch_size, shuffle=False,
        augment=False, target_size=target_size, channels=channels
    )

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "class_names": class_names,
        "num_classes": len(class_names),
        "label_map": label_map,
        "steps_per_epoch": {
            "train": int(np.ceil(len(paths_train) / batch_size)),
            "val": int(np.ceil(len(paths_val) / batch_size)),
            "test": int(np.ceil(len(paths_test) / batch_size)),
        },
        "counts": {
            "train": len(paths_train),
            "val": len(paths_val),
            "test": len(paths_test),
        },
    }


# ---------------------------------------------------------------------------
# 5. Convenience: load from CSV (filename, label) instead of subfolders
# ---------------------------------------------------------------------------

def prepare_datasets_from_csv(
    csv_path: str | Path,
    image_root: str | Path,
    filename_col: str = "filename",
    label_col: str = "label",
    **kwargs: Any,
) -> dict[str, Any]:
    """Prepare datasets from a CSV mapping filenames to class labels.

    The CSV should have columns for the image filename (relative to image_root)
    and the class label (string). All other arguments are passed to
    `prepare_datasets` after constructing the path/label lists.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    paths = [str(Path(image_root) / fn) for fn in df[filename_col].astype(str)]
    str_labels = df[label_col].astype(str).tolist()
    class_names = sorted(set(str_labels))

    label_map = {name: i for i, name in enumerate(class_names)}
    int_labels = encode_labels(str_labels, label_map)

    # Delegate to the core split logic (copy of prepare_datasets internals)
    test_split = kwargs.pop("test_split", 0.15)
    val_split = kwargs.pop("val_split", 0.15)
    batch_size = kwargs.pop("batch_size", 32)
    seed = kwargs.pop("seed", 42)

    val_size = val_split / (1 - test_split)
    paths_train, paths_test, y_train, y_test = train_test_split(
        paths, int_labels, test_size=test_split, stratify=int_labels, random_state=seed
    )
    paths_train, paths_val, y_train, y_val = train_test_split(
        paths_train, y_train, test_size=val_size, stratify=y_train, random_state=seed
    )

    train_ds = create_dataset(paths_train, y_train, batch_size=batch_size, shuffle=True, **kwargs)
    val_ds = create_dataset(paths_val, y_val, batch_size=batch_size, shuffle=False, **kwargs)
    test_ds = create_dataset(paths_test, y_test, batch_size=batch_size, shuffle=False, **kwargs)

    return {
        "train_ds": train_ds,
        "val_ds": val_ds,
        "test_ds": test_ds,
        "class_names": class_names,
        "num_classes": len(class_names),
        "label_map": label_map,
        "steps_per_epoch": {
            "train": int(np.ceil(len(paths_train) / batch_size)),
            "val": int(np.ceil(len(paths_val) / batch_size)),
            "test": int(np.ceil(len(paths_test) / batch_size)),
        },
        "counts": {
            "train": len(paths_train),
            "val": len(paths_val),
            "test": len(paths_test),
        },
    }


# ---------------------------------------------------------------------------
# 6. Visualization helper (optional, for debugging)
# ---------------------------------------------------------------------------

def visualize_batch(dataset: tf.data.Dataset, class_names: list[str], num_images: int = 9) -> None:
    """Plot a grid of images from a dataset batch (requires matplotlib)."""
    import matplotlib.pyplot as plt
    images, labels = next(iter(dataset.take(1)))
    images = images.numpy()
    labels = labels.numpy()

    cols = min(3, num_images)
    rows = int(np.ceil(num_images / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = axes.ravel() if num_images > 1 else [axes]

    for i in range(min(num_images, len(images))):
        ax = axes[i]
        ax.imshow(images[i])
        ax.set_title(class_names[labels[i]])
        ax.axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Quick self-test with synthetic data (no real images needed)
    import tempfile
    import shutil

    print("=== Self-test: dataset_prep.py ===")

    # Create a temporary directory with dummy class folders and tiny images
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        classes = ["Healthy", "Early_Blight", "Late_Blight"]
        for c in classes:
            (tmp / c).mkdir()

        # Create 10 tiny dummy images per class (1x1 pixel PNGs)
        for c in classes:
            for i in range(10):
                img = tf.zeros((1, 1, 3), dtype=tf.uint8)
                encoded = tf.io.encode_png(img)
                tf.io.write_file(str(tmp / c / f"img_{i}.png"), encoded)

        # Run the full pipeline
        result = prepare_datasets(
            tmp,
            batch_size=4,
            val_split=0.2,
            test_split=0.2,
            augment_train=True,
            seed=42,
        )

        print(f"Classes: {result['class_names']}")
        print(f"Num classes: {result['num_classes']}")
        print(f"Label map: {result['label_map']}")
        print(f"Counts: {result['counts']}")
        print(f"Steps/epoch: {result['steps_per_epoch']}")

        # Inspect one batch
        batch = next(iter(result["train_ds"].take(1)))
        imgs, lbls = batch
        print(f"Batch image shape: {imgs.shape}, dtype: {imgs.dtype}")
        print(f"Batch label shape: {lbls.shape}, dtype: {lbls.dtype}")
        print(f"Pixel range: [{imgs.numpy().min():.3f}, {imgs.numpy().max():.3f}]")

        # Verify label mapping round-trip
        decoded = decode_labels(lbls.numpy(), result["label_map"])
        print(f"Decoded labels (first 4): {decoded}")

    print("\n=== Self-test PASSED ===")