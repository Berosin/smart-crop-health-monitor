"""Colab script: export per-class embedding statistics for feature-space
("Mahalanobis distance") out-of-distribution detection.

Run this ONCE PER CROP, in Colab, after that crop's model is already
trained. It does NOT retrain or modify the model — it's a read-only pass:
load the trained model, run every training image through it, record the
penultimate-layer embedding (the 1280-dim vector right before the final
classification layer, i.e. MobileNetV2's GlobalAveragePooling2D output),
and save per-class statistics computed from those embeddings.

Why this fixes the "confidently wrong on a hand photo" gap
-------------------------------------------------------------
src/ood_detection.py flags predictions the model is *unsure* about
(low softmax confidence / flat distribution). It cannot catch the opposite
failure — a network can be arbitrarily confident on an input far outside
anything it was trained on. This script produces the data a DIFFERENT,
complementary check needs: instead of asking "how confident is the
softmax", it asks "does this image's internal representation actually
resemble what the model saw during training for its predicted class" —
robust to confidently-wrong predictions, and (unlike a raw-pixel-color
heuristic) immune to the skin-tone / dried-leaf color overlap problem,
since it operates in the model's own learned feature space, not raw RGB.

Technique: Lee et al., "A Simple Unified Framework for Detecting
Out-of-Distribution Samples and Adversarial Attacks" (NeurIPS 2018) —
class-conditional Gaussians in feature space with a shared (tied)
covariance, i.e. Mahalanobis distance to the nearest class centroid.

Practical adaptation for a small-ish PlantVillage-scale dataset
-------------------------------------------------------------------
A raw 1280-dim MobileNetV2 embedding needs thousands of samples per class
for its covariance matrix to be well-conditioned — more than most
per-crop datasets here likely have. This script first reduces embeddings
to PCA_COMPONENTS dimensions (default 128) via PCA fit on this crop's own
training embeddings, then computes the shared covariance in that reduced
space. This is a standard, well-established adaptation (PCA-whitened
Mahalanobis distance), not a shortcut — and keeps the exported stats file
small (a few hundred KB, not tens of MB).

Usage in Colab
----------------
1. Mount whichever Drive has this crop's dataset (Corn's is a separate
   Drive from the others per Berosin's setup — mount/switch as needed
   between runs; this script only touches one crop per run).
2. Edit the four variables in the CONFIGURE block below.
3. Run all cells. Output: embedding_stats.npz, saved next to that crop's
   model.keras (same convention config.DISEASE_MODELS already expects for
   model.keras/labels.json), e.g.:
       models/disease_model_corn/embedding_stats.npz
4. Download that one file (Colab's file browser -> right-click -> Download,
   or copy it back to the app's Drive-synced folder) and place it in the
   matching models/disease_model_<crop>/ folder in the deployed app repo.
5. Repeat for each crop.

The app (src/ood_feature_detector.py) automatically picks up
embedding_stats.npz for any crop where it's present, and simply skips the
feature-space check (falling back to the existing softmax-only check) for
any crop where it isn't yet — so you can roll this out one crop at a time
without breaking the others.
"""

# =============================================================================
# CONFIGURE — edit these four lines for the crop you're processing this run
# =============================================================================
CROP_NAME = "Corn"                                          # must match config.DISEASE_MODELS' key
DATA_DIR = "/content/drive/MyDrive/datasets/corn_dataset"    # this crop's training folder (one subfolder per class)
MODEL_DIR = "/content/drive/MyDrive/smart-crop-health-monitor/models/disease_model_corn"  # wherever model.keras + labels.json for this crop live in Colab
PCA_COMPONENTS = 128                                         # reduced embedding dimensionality; 128 is a safe default for a few hundred-thousand images/class

# =============================================================================
# Implementation — shouldn't need edits below this line
# =============================================================================
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.decomposition import PCA


IMAGE_SIZE = (224, 224)   # must match config.IMAGE_SIZE — kept as a literal here so
IMAGE_CHANNELS = 3        # this script has zero dependency on the app's own package layout


def find_embedding_tensor(model: tf.keras.Model):
    """Same robust lookup src/gradcam.py uses for the conv feature map, but
    taking the pooling layer's OUTPUT (the flat embedding vector) instead
    of its input (the conv feature map) — see that module for why this is
    the reliable way to reach into a nested-MobileNetV2 architecture.
    """
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.GlobalAveragePooling2D,
                               tf.keras.layers.GlobalMaxPooling2D)):
            return layer.output
    raise ValueError(
        "Could not find a global-pooling layer in this model — "
        "embedding extraction expects the same architecture "
        "src/model_training.py's create_model() produces."
    )


def load_class_names(labels_path: Path) -> list[str]:
    label_map = json.loads(labels_path.read_text())
    inv_map = {v: k for k, v in label_map.items()}
    return [inv_map[i] for i in range(len(inv_map))]


def load_and_preprocess(path: str) -> np.ndarray:
    """Exactly matches src/dataset_prep.py's preprocess_image / the
    corrected src/image_preprocessing.py pipeline: resize, cast to
    float32, keep [0, 255] — the model's own first layer does the
    MobileNetV2 [-1, 1] rescale internally. Do NOT call
    tf.keras.applications.mobilenet_v2.preprocess_input here — that would
    double-normalize (see image_preprocessing.py's _to_float32 docstring
    for the exact bug this app already hit and fixed once before).
    """
    img = tf.io.read_file(path)
    img = tf.image.decode_image(img, channels=IMAGE_CHANNELS, expand_animations=False)
    img = tf.image.resize(img, IMAGE_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32)
    return img.numpy()


def main():
    model_dir = Path(MODEL_DIR)
    data_dir = Path(DATA_DIR)

    print(f"Loading model for {CROP_NAME} from {model_dir / 'model.keras'} ...")
    model = tf.keras.models.load_model(model_dir / "model.keras")
    class_names = load_class_names(model_dir / "labels.json")
    print("Classes (in model output order):", class_names)

    embedding_tensor = find_embedding_tensor(model)
    embed_model = tf.keras.Model(inputs=model.inputs, outputs=embedding_tensor)
    print(f"Embedding dimensionality (pre-PCA): {embedding_tensor.shape[-1]}")

    # -------------------------------------------------------------------
    # Pass 1: run every training image through the model, collect raw
    # embeddings + their true class index.
    # -------------------------------------------------------------------
    all_embeddings = []
    all_labels = []
    batch_size = 32

    for class_idx, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        if not class_dir.is_dir():
            print(f"  WARNING: no folder found for class '{class_name}' at {class_dir} — skipping.")
            continue

        image_paths = [
            p for p in class_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp", ".tiff")
        ]
        print(f"  {class_name}: {len(image_paths)} images")

        batch = []
        for p in image_paths:
            try:
                batch.append(load_and_preprocess(str(p)))
            except Exception as e:
                print(f"    skipping unreadable file {p.name}: {e}")
                continue
            if len(batch) == batch_size:
                embeds = embed_model.predict(np.stack(batch), verbose=0)
                all_embeddings.append(embeds)
                all_labels.extend([class_idx] * len(batch))
                batch = []
        if batch:
            embeds = embed_model.predict(np.stack(batch), verbose=0)
            all_embeddings.append(embeds)
            all_labels.extend([class_idx] * len(batch))

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = np.array(all_labels)
    print(f"\nTotal embeddings collected: {embeddings.shape[0]}, raw dim: {embeddings.shape[1]}")

    if embeddings.shape[0] < PCA_COMPONENTS * 3:
        print(
            f"  WARNING: only {embeddings.shape[0]} samples for {PCA_COMPONENTS} PCA "
            f"components — consider lowering PCA_COMPONENTS for a more stable fit."
        )

    # -------------------------------------------------------------------
    # PCA reduction (fit on this crop's own training embeddings)
    # -------------------------------------------------------------------
    n_components = min(PCA_COMPONENTS, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    reduced = pca.fit_transform(embeddings)
    print(f"PCA: {embeddings.shape[1]} -> {n_components} dims, "
          f"explained variance: {pca.explained_variance_ratio_.sum():.3f}")

    # -------------------------------------------------------------------
    # Per-class means + shared (tied) covariance, in PCA space
    # (Lee et al. 2018's class-conditional Gaussian / Mahalanobis method)
    # -------------------------------------------------------------------
    num_classes = len(class_names)
    class_means = np.zeros((num_classes, n_components), dtype=np.float64)
    centered = np.zeros_like(reduced, dtype=np.float64)

    for class_idx in range(num_classes):
        mask = labels == class_idx
        if mask.sum() == 0:
            continue
        class_means[class_idx] = reduced[mask].mean(axis=0)
        centered[mask] = reduced[mask] - class_means[class_idx]

    shared_cov = (centered.T @ centered) / len(reduced)
    # Small ridge term for numerical stability — standard shrinkage,
    # negligible effect on well-conditioned dimensions, prevents the
    # inverse from blowing up on near-zero-variance ones.
    ridge = 1e-3 * np.trace(shared_cov) / n_components
    shared_cov_reg = shared_cov + ridge * np.eye(n_components)
    inv_cov = np.linalg.inv(shared_cov_reg)

    # -------------------------------------------------------------------
    # Data-driven distance threshold: how far does this crop's OWN
    # training data typically sit from its class centroid? Use a high
    # percentile (not the max) so a few genuine outliers in the training
    # set don't set an overly lenient threshold.
    # -------------------------------------------------------------------
    def mahalanobis_to_own_class(x, class_idx):
        diff = x - class_means[class_idx]
        return float(np.sqrt(diff @ inv_cov @ diff))

    train_distances = np.array([
        mahalanobis_to_own_class(reduced[i], labels[i]) for i in range(len(reduced))
    ])
    threshold = float(np.percentile(train_distances, 99))
    print(f"\nTraining-set self-distance: mean={train_distances.mean():.2f}, "
          f"p99={threshold:.2f}, max={train_distances.max():.2f}")

    # -------------------------------------------------------------------
    # Save
    # -------------------------------------------------------------------
    out_path = model_dir / "embedding_stats.npz"
    np.savez(
        out_path,
        class_names=np.array(class_names),
        pca_mean=pca.mean_.astype(np.float32),
        pca_components=pca.components_.astype(np.float32),
        class_means=class_means.astype(np.float32),
        inv_cov=inv_cov.astype(np.float32),
        distance_threshold=np.float32(threshold),
    )
    print(f"\nSaved: {out_path} ({out_path.stat().st_size / 1024:.0f} KB)")
    print(f"Download this file and place it at "
          f"models/disease_model_{CROP_NAME.lower()}/embedding_stats.npz "
          f"in the app repo.")


if __name__ == "__main__":
    main()