"""Grad-CAM (Gradient-weighted Class Activation Mapping) for disease models.

Explainability module: given a trained MobileNetV2 disease-detection model
and a preprocessed image batch, produces a heatmap over the *last
convolutional feature map* showing which regions of the leaf most drove the
predicted class, then overlays that heatmap on the image.

Why this matters for this project: `.predict()` alone gives a class and a
confidence score but says nothing about *why* the model reached that
verdict. Grad-CAM answers that by backpropagating the predicted class
score to the last conv layer's activations, using the resulting gradients
to weight each channel's importance, and summing the weighted channels
into a single-channel "attention" map — the same idea a saliency map or
attention visualization uses in other architectures.

Reference: Selvaraju et al., "Grad-CAM: Visual Explanations from Deep
Networks via Gradient-based Localization" (2017).

Handling the nested MobileNetV2 backbone
-----------------------------------------
src/model_training.py's create_model() builds the network as:

    inputs -> preprocess_input -> base_model(x) -> GlobalAveragePooling2D
           -> Dropout -> Dense(num_classes, softmax)

`base_model` (MobileNetV2 with include_top=False) is itself a full Keras
Functional model, called as a single nested layer of the outer model. That
nesting makes the usual "grab `model.get_layer(name).output`" recipe
ambiguous: base_model's internal layers were traced once when base_model
was built standalone, and traced *again* when called inside the outer
model, so `.output`/`.input` on an internal layer of base_model is not
guaranteed to point at the outer graph's tensor.

The reliable fix used here: instead of reaching inside base_model, grab the
**input tensor of the outer model's own GlobalAveragePooling2D layer**.
That tensor is exactly base_model's output as seen by the outer graph
(it's what GAP pools over), and since GAP is a top-level layer of `model`
(called exactly once), `.input` is unambiguous. This makes the whole
module architecture-agnostic beyond "some global-pooling layer sits
between the conv backbone and the classifier head" — true for every crop
model trained by src/model_training.py, and robust if the backbone itself
is ever swapped for something other than MobileNetV2.

Usage
-----
    from src.gradcam import generate_gradcam

    result = generate_gradcam(model, image_batch, pred_index=pred_idx)
    # result["heatmap"]  -> 2D float32 array in [0, 1], conv-layer resolution
    # result["overlay"]  -> uint8 RGB image, same size as the input batch
    # result["pred_index"] -> the class index the heatmap explains
"""

from __future__ import annotations

import numpy as np
import cv2
import tensorflow as tf

from src.errors import GradCAMError, logger

# ---------------------------------------------------------------------------
# grad_model cache — building the auxiliary [conv_output, predictions] model
# involves a graph-construction pass, which is unnecessary to repeat on
# every single prediction. Keyed by id() of the loaded Keras model, which
# is stable for the lifetime of that model object (the app loads each
# crop's model once via st.cache_resource and reuses it thereafter).
# ---------------------------------------------------------------------------
_grad_model_cache: dict[int, tf.keras.Model] = {}


def _find_conv_feature_tensor(model: tf.keras.Model):
    """Locate the tensor feeding into the model's global-pooling layer.

    That tensor is the last convolutional feature map — exactly what
    Grad-CAM needs to weight and sum. Falls back to the last 4D-output
    layer in the graph if no pooling layer is found, so this keeps working
    even if the head architecture changes slightly.
    """
    for layer in model.layers:
        if isinstance(layer, (tf.keras.layers.GlobalAveragePooling2D,
                               tf.keras.layers.GlobalMaxPooling2D)):
            return layer.input

    for layer in reversed(model.layers):
        shape = getattr(layer, "output_shape", None)
        if isinstance(shape, tuple) and len(shape) == 4:
            return layer.output

    raise GradCAMError(
        "Could not locate a convolutional feature layer for Grad-CAM — "
        "this model's architecture isn't supported for explainability."
    )


def _build_grad_model(model: tf.keras.Model) -> tf.keras.Model:
    conv_tensor = _find_conv_feature_tensor(model)
    return tf.keras.Model(inputs=model.inputs, outputs=[conv_tensor, model.output])


def _get_grad_model(model: tf.keras.Model) -> tf.keras.Model:
    key = id(model)
    grad_model = _grad_model_cache.get(key)
    if grad_model is None:
        grad_model = _build_grad_model(model)
        _grad_model_cache[key] = grad_model
    return grad_model


# ---------------------------------------------------------------------------
# Core algorithm
# ---------------------------------------------------------------------------
def make_gradcam_heatmap(
    model: tf.keras.Model,
    img_batch: np.ndarray,
    pred_index: int | None = None,
) -> tuple[np.ndarray, int]:
    """Compute a Grad-CAM heatmap for one prediction.

    Args:
        model: the trained disease-detection Keras model (already loaded).
        img_batch: shape (1, H, W, 3) float32 — the exact tensor that was
            (or would be) passed to model.predict(). Must match the
            preprocessing the model expects (raw [0, 255] here, since this
            model normalizes internally — see src/image_preprocessing.py).
        pred_index: which class to explain. Defaults to the model's own
            top prediction (i.e. "why did it predict what it predicted").

    Returns:
        (heatmap, pred_index) — heatmap is a 2D float32 array in [0, 1] at
        the last conv layer's spatial resolution (e.g. 7x7 for MobileNetV2
        at 224x224 input); pred_index is the class index actually explained.
    """
    grad_model = _get_grad_model(model)
    img_tensor = tf.convert_to_tensor(img_batch)

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_tensor, training=False)
        if pred_index is None:
            pred_index = int(tf.argmax(predictions[0]))
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    if grads is None:
        raise GradCAMError(
            "Grad-CAM could not compute gradients for this model — the "
            "conv layer may be disconnected from the prediction output."
        )

    # Mean gradient per channel = that channel's importance to this class.
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]

    # Weighted sum of feature-map channels by their importance, then ReLU
    # (Grad-CAM only cares about features with a *positive* influence on
    # the target class) and normalize to [0, 1] for display.
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0)
    max_val = tf.math.reduce_max(heatmap)
    heatmap = heatmap / (max_val + 1e-8)

    return heatmap.numpy().astype(np.float32), int(pred_index)


def overlay_heatmap(
    heatmap: np.ndarray,
    base_image_rgb: np.ndarray,
    alpha: float = 0.4,
    colormap: int = cv2.COLORMAP_JET,
) -> np.ndarray:
    """Blend a Grad-CAM heatmap onto an RGB image.

    Args:
        heatmap: 2D float array in [0, 1] (any resolution — resized here).
        base_image_rgb: uint8 RGB image, shape (H, W, 3), to overlay onto.
        alpha: heatmap opacity, 0 (invisible) to 1 (heatmap only).
        colormap: an OpenCV colormap constant (default: jet — blue = low
            contribution, red = high contribution).

    Returns:
        uint8 RGB image, same (H, W) as base_image_rgb.
    """
    h, w = base_image_rgb.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)
    heatmap_uint8 = np.uint8(np.clip(heatmap_resized, 0, 1) * 255)

    heatmap_color_bgr = cv2.applyColorMap(heatmap_uint8, colormap)
    heatmap_color_rgb = cv2.cvtColor(heatmap_color_bgr, cv2.COLOR_BGR2RGB)

    blended = (
        heatmap_color_rgb.astype(np.float32) * alpha
        + base_image_rgb.astype(np.float32) * (1 - alpha)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Convenience wrapper — what pages/disease.py actually calls
# ---------------------------------------------------------------------------
def generate_gradcam(
    model: tf.keras.Model,
    img_batch: np.ndarray,
    pred_index: int | None = None,
    alpha: float = 0.4,
) -> dict:
    """Run Grad-CAM end-to-end and return heatmap + overlay + base image.

    `img_batch` doubles as the overlay's base image (cast to uint8 RGB) —
    it's exactly what the model saw, so heatmap and image line up pixel
    for pixel with no separate resizing needed.

    Raises:
        GradCAMError: on any failure (unsupported architecture, gradient
        computation failure, ...). Callers should catch this and degrade
        gracefully (skip the heatmap, keep showing the core prediction).
    """
    try:
        base_image_rgb = np.clip(img_batch[0], 0, 255).astype(np.uint8)
        heatmap, used_index = make_gradcam_heatmap(model, img_batch, pred_index=pred_index)
        overlay = overlay_heatmap(heatmap, base_image_rgb, alpha=alpha)
        return {
            "heatmap": heatmap,
            "base_image": base_image_rgb,
            "overlay": overlay,
            "pred_index": used_index,
        }
    except GradCAMError:
        raise
    except Exception as e:
        logger.exception("Grad-CAM generation failed")
        raise GradCAMError(
            "Couldn't generate the explainability heatmap for this "
            "prediction. The core prediction above is unaffected."
        ) from e


# ---------------------------------------------------------------------------
# Self-test / demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from config import IMAGE_SIZE, IMAGE_CHANNELS

    print("--- Building a throwaway model with the same architecture ---")
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, IMAGE_CHANNELS))
    x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, IMAGE_CHANNELS), include_top=False, weights=None,
    )
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(3, activation="softmax", name="predictions")(x)
    test_model = tf.keras.Model(inputs, outputs)

    dummy_batch = np.random.uniform(0, 255, (1, *IMAGE_SIZE, IMAGE_CHANNELS)).astype(np.float32)

    print("--- Running Grad-CAM ---")
    result = generate_gradcam(test_model, dummy_batch)
    print("heatmap shape:", result["heatmap"].shape,
          "range:", (result["heatmap"].min(), result["heatmap"].max()))
    print("overlay shape:", result["overlay"].shape, "dtype:", result["overlay"].dtype)
    print("explained class index:", result["pred_index"])