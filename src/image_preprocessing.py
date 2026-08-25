"""OpenCV-based leaf image preprocessing pipeline.

Turns raw uploaded image bytes into the tensor the trained disease model
expects (224x224x3, float32, raw [0, 255] range) without touching the
model itself. Only *how* we get there changes: validation, decoding, and
every transform go through OpenCV instead of tf.io.

Note on normalization: the model's own first layer applies MobileNetV2's
preprocess_input (scaling [0, 255] -> [-1, 1]) internally — see
src/model_training.py's create_model(). This pipeline must hand off raw
[0, 255] values, NOT pre-scaled ones, or that scaling gets applied twice
and destroys almost all of the image's information (see _to_float32).

Pipeline
--------
1. Validate      — reject empty/oversized files and unsupported formats
                    before attempting to decode anything.
2. Decode         — cv2.imdecode; a corrupted file fails here and is
                    reported as such (distinct from "unsupported format").
3. Dimension check — reject decoded images that are absurdly large or too
                    small to be a useful leaf photo.
4. RGB conversion — OpenCV decodes to BGR; convert to RGB explicitly.
5. Resize         — cv2.resize to the model's expected input size.
6. [Optional] Noise reduction   — cv2.fastNlMeansDenoisingColored.
7. [Optional] Background handling — flatten non-leaf-colored background
                    toward neutral gray via an HSV color-threshold mask,
                    so the model focuses on the leaf.
8. Scale to float32 — keep the [0, 255] range (the model normalizes
                    internally), add the batch dimension.

Usage
-----
    from src.image_preprocessing import preprocess_leaf_image

    image_batch = preprocess_leaf_image(file_bytes, target_size=(224, 224))
    # image_batch.shape == (1, 224, 224, 3), dtype float32, range [0, 255]

    # with the optional steps enabled:
    image_batch = preprocess_leaf_image(
        file_bytes, denoise=True, remove_background=True,
    )
"""

from __future__ import annotations

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Limits — tune here, not scattered through the pipeline.
# ---------------------------------------------------------------------------
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

MIN_DIMENSION = 32          # px, per side — below this isn't a usable photo
MAX_DIMENSION = 8000        # px, per side — guards against decompression-bomb-style images
MAX_MEGAPIXELS = 40         # width * height upper bound, independent of file size

# Signature (magic-byte) sniffing so we can say "unsupported format" instead
# of a generic "corrupted" error for something like a PDF or SVG upload.
_SIGNATURES: dict[bytes, str] = {
    b"\xff\xd8\xff":                       "JPEG",
    b"\x89PNG\r\n\x1a\n":                  "PNG",
    b"BM":                                 "BMP",
    b"RIFF":                               "WEBP",  # confirmed further below (RIFF....WEBP)
}
SUPPORTED_FORMATS = {"JPEG", "PNG", "BMP", "WEBP"}


class ImageValidationError(ValueError):
    """Raised for any rejected upload (unsupported format, corrupted,
    too large, too small). Subclasses ValueError so existing
    `except ValueError` call sites keep working unchanged.
    """


# ---------------------------------------------------------------------------
# 1. Validation
# ---------------------------------------------------------------------------
def _sniff_format(file_bytes: bytes) -> str | None:
    for magic, fmt in _SIGNATURES.items():
        if file_bytes.startswith(magic):
            if fmt == "WEBP":
                # RIFF is a container; confirm the WEBP fourCC at offset 8.
                if len(file_bytes) >= 12 and file_bytes[8:12] == b"WEBP":
                    return "WEBP"
                continue
            return fmt
    return None


def validate_image_bytes(file_bytes: bytes) -> str:
    """Validate raw upload bytes before any decoding is attempted.

    Returns the sniffed format name on success. Raises
    ImageValidationError with a specific, user-facing reason otherwise.
    """
    if not file_bytes:
        raise ImageValidationError("The uploaded file is empty.")

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        size_mb = len(file_bytes) / (1024 * 1024)
        raise ImageValidationError(
            f"Image is too large ({size_mb:.1f} MB). Maximum allowed is "
            f"{MAX_FILE_SIZE_MB} MB."
        )

    fmt = _sniff_format(file_bytes)
    if fmt is None:
        raise ImageValidationError(
            "Unsupported file format. Please upload a JPEG, PNG, BMP, or WEBP image."
        )
    return fmt


def _validate_dimensions(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    if h < MIN_DIMENSION or w < MIN_DIMENSION:
        raise ImageValidationError(
            f"Image is too small ({w}x{h}px). Minimum is "
            f"{MIN_DIMENSION}x{MIN_DIMENSION}px."
        )
    if h > MAX_DIMENSION or w > MAX_DIMENSION:
        raise ImageValidationError(
            f"Image dimensions are too large ({w}x{h}px). Maximum is "
            f"{MAX_DIMENSION}x{MAX_DIMENSION}px per side."
        )
    megapixels = (h * w) / 1_000_000
    if megapixels > MAX_MEGAPIXELS:
        raise ImageValidationError(
            f"Image resolution is too large ({megapixels:.1f} MP). "
            f"Maximum is {MAX_MEGAPIXELS} MP."
        )


# ---------------------------------------------------------------------------
# 2-3. Decode + dimension check
# ---------------------------------------------------------------------------
def _decode(file_bytes: bytes) -> np.ndarray:
    buffer = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)  # -> BGR, uint8
    if img is None:
        raise ImageValidationError(
            "Could not decode the image — the file appears to be corrupted."
        )
    _validate_dimensions(img)
    return img


# ---------------------------------------------------------------------------
# 6. Optional noise reduction
# ---------------------------------------------------------------------------
def _denoise(img_rgb: np.ndarray) -> np.ndarray:
    """Non-local-means color denoising. Applied post-resize so it stays
    fast regardless of the original upload's resolution.
    """
    return cv2.fastNlMeansDenoisingColored(img_rgb, None, h=7, hColor=7,
                                           templateWindowSize=7, searchWindowSize=21)


# ---------------------------------------------------------------------------
# 7. Optional background handling
# ---------------------------------------------------------------------------
def _flatten_background(img_rgb: np.ndarray) -> np.ndarray:
    """Softly suppress non-leaf-colored background via an HSV threshold
    mask, so the model's attention isn't split by clutter behind the leaf.

    This is a lightweight color-based heuristic, not semantic
    segmentation: pixels outside a broad green/yellow/brown "plant" hue
    range are blended toward neutral gray. The leaf itself is left
    untouched, and the blend is soft (not a hard cutout), so a busy
    background gets suppressed without introducing harsh edges the CNN
    hasn't seen in training.
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Broad hue range covering healthy green through senescent yellow/brown
    # leaf tones (OpenCV hue range is 0-179).
    lower = np.array([15, 25, 25], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    # Clean up small holes/specks in the mask.
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # If the mask is basically empty or basically everything, the
    # heuristic isn't confident about what's background — skip it rather
    # than risk erasing the actual leaf.
    coverage = mask.mean() / 255.0
    if coverage < 0.03 or coverage > 0.97:
        return img_rgb

    mask_3ch = cv2.merge([mask, mask, mask]).astype(np.float32) / 255.0
    neutral_gray = np.full_like(img_rgb, 127, dtype=np.uint8)

    blended = (img_rgb.astype(np.float32) * mask_3ch
               + neutral_gray.astype(np.float32) * (1 - mask_3ch))
    return blended.astype(np.uint8)


# ---------------------------------------------------------------------------
# 8. Scale to float32 (model owns MobileNetV2 normalization internally)
# ---------------------------------------------------------------------------
def _to_float32(img_rgb: np.ndarray) -> np.ndarray:
    """Cast uint8 [0, 255] pixels to float32, keeping the same [0, 255] range.

    IMPORTANT: this does NOT rescale to [-1, 1]. The trained model's first
    layer already calls tf.keras.applications.mobilenet_v2.preprocess_input
    internally (see src/model_training.py's create_model()), which expects
    raw [0, 255] input and does the /127.5 - 1.0 scaling itself. If this
    function also rescaled to [-1, 1], the model would apply that scaling
    a second time on already-normalized data, collapsing every pixel into
    a narrow band near -1 and destroying almost all image information —
    this previously caused a train/serve mismatch: training (via
    src/dataset_prep.py, which also just casts to float32 without
    rescaling) was fine, but real predictions through this module were not.
    """
    return img_rgb.astype(np.float32)


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------
def preprocess_leaf_image(
    file_bytes: bytes,
    target_size: tuple[int, int] = (224, 224),
    *,
    denoise: bool = False,
    remove_background: bool = False,
) -> np.ndarray:
    """Validate and preprocess raw image bytes into a model-ready batch.

    Args:
        file_bytes: raw bytes of the uploaded file.
        target_size: (height, width) the model expects.
        denoise: optional non-local-means noise reduction.
        remove_background: optional HSV-based background flattening.

    Returns:
        np.ndarray of shape (1, height, width, 3), dtype float32, values
        in [0, 255] — a drop-in input for model.predict(). The model's
        first layer applies MobileNetV2's own normalization internally,
        so this pipeline must NOT rescale to [-1, 1] itself (see
        _to_float32's docstring for why that would double-normalize).

    Raises:
        ImageValidationError (a ValueError subclass): unsupported format,
        corrupted file, or the image is too large/small.
    """
    validate_image_bytes(file_bytes)          # format + file-size checks
    img_bgr = _decode(file_bytes)              # decode + dimension checks

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # RGB conversion
    img_rgb = cv2.resize(img_rgb, (target_size[1], target_size[0]),
                         interpolation=cv2.INTER_AREA)    # resize

    if denoise:
        img_rgb = _denoise(img_rgb)
    if remove_background:
        img_rgb = _flatten_background(img_rgb)

    scaled = _to_float32(img_rgb)                          # float32, still [0, 255]
    return np.expand_dims(scaled, axis=0)                   # batch dimension


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def _make_test_jpeg(w: int, h: int, solid: bool = False) -> bytes:
        if solid:
            img = np.full((h, w, 3), 120, dtype=np.uint8)  # low-entropy, compresses small
        else:
            img = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        ok, buf = cv2.imencode(".jpg", img)
        assert ok
        return buf.tobytes()

    print("--- Valid image ---")
    good = _make_test_jpeg(640, 480)
    batch = preprocess_leaf_image(good, target_size=(224, 224))
    print("shape:", batch.shape, "dtype:", batch.dtype,
          "range:", (batch.min(), batch.max()))

    print("\n--- With denoise + background handling ---")
    batch2 = preprocess_leaf_image(good, denoise=True, remove_background=True)
    print("shape:", batch2.shape, "range:", (batch2.min(), batch2.max()))

    print("\n--- Rejections ---")
    for label, data in [
        ("empty file", b""),
        ("unsupported format", b"%PDF-1.4 not an image"),
        ("corrupted JPEG signature", b"\xff\xd8\xff" + b"\x00" * 50),
        ("oversized file", b"\xff\xd8\xff" + b"0" * (MAX_FILE_SIZE_BYTES + 1)),
    ]:
        try:
            preprocess_leaf_image(data)
            print(f"{label}: NOT REJECTED (unexpected)")
        except ImageValidationError as e:
            print(f"{label}: rejected — {e}")

    print("\n--- Oversized decoded image (small file size, huge pixel dimensions) ---")
    huge = _make_test_jpeg(9000, 9000, solid=True)  # low-entropy -> compresses small
    print(f"encoded size: {len(huge)/1024:.1f} KB")
    try:
        preprocess_leaf_image(huge)
        print("NOT REJECTED (unexpected)")
    except ImageValidationError as e:
        print(f"rejected — {e}")