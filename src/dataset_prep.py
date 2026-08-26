"""Leaf disease image dataset preparation for deep learning."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from config import IMAGE_SIZE, IMAGE_CHANNELS

def get_class_names_from_dir(data_dir: str | Path) -> list[str]:
    p = Path(data_dir)
    return sorted([d.name for d in p.iterdir() if d.is_dir()])

def save_label_map(class_names: list[str], path: str | Path = None) -> Path:
    if path is None:
        from config import LABELS_PATH
        path = LABELS_PATH
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    label_map = {name: i for i, name in enumerate(class_names)}
    path.write_text(json.dumps(label_map, indent=2))
    return path

# ADDED BACK: Fixes the ImportError in model_training.py
def load_label_map(path: str | Path = None) -> dict[str, int]:
    if path is None:
        from config import LABELS_PATH
        path = LABELS_PATH
    return json.loads(Path(path).read_text())

def load_class_names(path: str | Path = None) -> list[str]:
    """Load a labels.json (name -> index map) and return names ordered by index.

    Shared by every page that needs a trained model's class list (Disease
    Detection, Health Analysis, About) so the name->index->name conversion
    only lives in one place.
    """
    label_map = load_label_map(path)
    inv_map = {v: k for k, v in label_map.items()}
    return [inv_map[i] for i in range(len(inv_map))]

def encode_labels(labels: list[str], label_map: dict[str, int]) -> np.ndarray:
    return np.array([label_map[l] for l in labels], dtype=np.int32)

def decode_labels(encoded: np.ndarray, label_map: dict[str, int]) -> list[str]:
    inv_map = {v: k for k, v in label_map.items()}
    return [inv_map[i] for i in encoded]

def load_image_paths_and_labels(data_dir: str | Path, class_names: list[str] | None = None, extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")):
    p = Path(data_dir)
    if class_names is None:
        class_names = get_class_names_from_dir(p)
    paths, labels = [], []
    for class_name in class_names:
        class_dir = p / class_name
        if not class_dir.is_dir():
            continue
        for img_path in class_dir.iterdir():
            if img_path.is_file() and img_path.suffix.lower() in extensions:
                paths.append(str(img_path))
                labels.append(class_name)
    return paths, labels

def build_augmentation_layer(config: dict[str, Any] | None = None) -> tf.keras.Sequential:
    cfg = {"horizontal_flip": True, "rotation_factor": 0.05, "zoom_factor": 0.05, "seed": 42, **(config or {})}
    layers = [
        tf.keras.layers.RandomFlip("horizontal", seed=cfg["seed"]),
        tf.keras.layers.RandomRotation(cfg["rotation_factor"], fill_mode="reflect", seed=cfg["seed"]),
        tf.keras.layers.RandomZoom(cfg["zoom_factor"], fill_mode="reflect", seed=cfg["seed"])
    ]
    return tf.keras.Sequential(layers, name="augmentation")

def preprocess_image(image_path: tf.Tensor, label: tf.Tensor, target_size: tuple[int, int] = IMAGE_SIZE, channels: int = IMAGE_CHANNELS, augment: bool = False, augmentation_layer: tf.keras.Sequential | None = None):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=channels, expand_animations=False)
    img = tf.image.resize(img, target_size, method="bilinear")
    
    # Keeps pixels at 0-255 float range so MobileNetV2's internal layer works beautifully
    img = tf.cast(img, tf.float32)
    img.set_shape((*target_size, channels))
    if augment and augmentation_layer is not None:
        img = augmentation_layer(img, training=True)
    return img, label

def create_dataset(image_paths: list[str], labels: np.ndarray, batch_size: int = 32, shuffle: bool = True, augment: bool = False, augmentation_config: dict[str, Any] | None = None, target_size: tuple[int, int] = IMAGE_SIZE, channels: int = IMAGE_CHANNELS, seed: int = 42):
    aug_layer = build_augmentation_layer(augmentation_config) if augment else None
    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda p, y: preprocess_image(p, y, target_size, channels, augment, aug_layer), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

def prepare_datasets(data_dir: str | Path, batch_size: int = 32, val_split: float = 0.15, test_split: float = 0.15, augment_train: bool = True, augmentation_config: dict[str, Any] | None = None, target_size: tuple[int, int] = IMAGE_SIZE, channels: int = IMAGE_CHANNELS, seed: int = 42, class_names: list[str] | None = None, save_labels: bool = True):
    paths, str_labels = load_image_paths_and_labels(data_dir, class_names)
    if class_names is None:
        class_names = sorted(set(str_labels))
    label_map = {name: i for i, name in enumerate(class_names)}
    if save_labels:
        save_label_map(class_names)
    int_labels = encode_labels(str_labels, label_map)
    
    val_size = val_split / (1 - test_split)
    paths_train, paths_test, y_train, y_test = train_test_split(paths, int_labels, test_size=test_split, random_state=seed)
    paths_train, paths_val, y_train, y_val = train_test_split(paths_train, y_train, test_size=val_size, random_state=seed)

    return {
        "train_ds": create_dataset(paths_train, y_train, batch_size=batch_size, shuffle=True, augment=augment_train, augmentation_config=augmentation_config, target_size=target_size, channels=channels, seed=seed),
        "val_ds": create_dataset(paths_val, y_val, batch_size=batch_size, shuffle=False, augment=False, target_size=target_size, channels=channels),
        "test_ds": create_dataset(paths_test, y_test, batch_size=batch_size, shuffle=False, augment=False, target_size=target_size, channels=channels),
        "class_names": class_names, "num_classes": len(class_names), "label_map": label_map,
        "steps_per_epoch": {"train": int(np.ceil(len(paths_train)/batch_size)), "val": int(np.ceil(len(paths_val)/batch_size)), "test": int(np.ceil(len(paths_test)/batch_size))},
        "counts": {"train": len(paths_train), "val": len(paths_val), "test": len(paths_test)}
    }