"""Out-of-distribution (OOD) detection — "is this even a leaf?"

Softmax always sums to 1, no matter what image goes in. Feed a trained
crop-disease model a photo of a hand, a dog, or the wrong crop's leaf, and
it will still confidently output *some* class — there's no built-in "I
don't know" option. This module adds one: a lightweight, well-established
robustness check that flags when a prediction is probably not trustworthy,
so the app can say "this doesn't look like a confident leaf match" instead
of quietly presenting a wrong answer as if it were a real diagnosis.

Technique
---------
Two classic signals from the softmax output itself, no retraining or extra
model needed (Hendrycks & Gimpel, "A Baseline for Detecting Misclassified
and Out-of-Distribution Examples in Neural Networks", 2017):

1. **Max softmax probability (MSP)** — the top class's confidence. A
   genuinely in-distribution, well-trained-on example usually produces a
   strong peak (often >0.85 for the small 3-4 class heads this app trains).
   A meaningfully lower peak means the model itself isn't sure.

2. **Normalized predictive entropy** — how *flat* the whole distribution
   is, not just the top value. Shannon entropy H = -sum(p_i * log(p_i)),
   normalized by log(num_classes) so it's comparable across crops with a
   different number of classes (0 = fully confident one-hot, 1 = perfectly
   uniform "no idea"). This catches a case MSP alone can miss: a top
   probability that looks moderately confident purely because it's the
   *largest* of several near-equal, all-low values.

Either signal tripping is treated as suspicious — this errs toward
flagging more, not less, since the cost of an unnecessary "try a clearer
photo" is low and the cost of a confidently-wrong diagnosis in a live demo
(or a real field decision) is high.

What this is not
------------------
Not a trained OOD classifier, not a second model, not guaranteed to catch
every out-of-distribution image (a photo of a *different* healthy-looking
green leaf can still land in a confident wrong bucket) — it's a cheap,
real signal from the classifier's own uncertainty, layered on top of the
prediction it already made. It doesn't replace or hide that prediction;
callers show both, so the person can judge for themselves.
"""

from __future__ import annotations

import numpy as np

# Thresholds are intentionally simple, fixed constants (same
# banded-rules-over-black-box philosophy as src/health_engine.py and
# src/outbreak_detection.py) rather than a second learned model. For the
# small 3-4 class heads trained here, chance-level max-prob is already
# ~0.25-0.33, so 0.45 sits meaningfully above "the model is just guessing"
# while still catching genuinely weak top predictions.
MAX_PROB_THRESHOLD = 0.45
NORMALIZED_ENTROPY_THRESHOLD = 0.75


def compute_ood_signal(probs: np.ndarray) -> dict:
    """Uncertainty signal for one softmax output vector.

    Args:
        probs: 1D array of per-class probabilities for one prediction
            (e.g. model.predict(batch)[0]). Assumed to already sum to ~1
            (a softmax output) — this function does not renormalize.

    Returns:
        {
            "max_prob": float,               # top class's confidence
            "entropy": float,                # raw Shannon entropy, nats
            "normalized_entropy": float,      # entropy / log(num_classes), in [0, 1]
            "is_likely_ood": bool,            # either signal tripped
            "reason": str,                    # which signal(s), in plain language
        }
    """
    probs = np.asarray(probs, dtype=np.float64)
    num_classes = len(probs)

    max_prob = float(np.max(probs))

    # Avoid log(0): probabilities exactly at 0 contribute 0 to entropy by
    # convention (the standard 0*log(0) := 0 limit), not undefined/NaN.
    safe_probs = probs[probs > 0]
    entropy = float(-np.sum(safe_probs * np.log(safe_probs)))
    max_entropy = float(np.log(num_classes)) if num_classes > 1 else 1.0
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

    low_confidence = max_prob < MAX_PROB_THRESHOLD
    flat_distribution = normalized_entropy > NORMALIZED_ENTROPY_THRESHOLD
    is_likely_ood = low_confidence or flat_distribution

    if low_confidence and flat_distribution:
        reason = (
            f"Top match is only {max_prob * 100:.0f}% confident, and the model's "
            f"confidence is spread fairly evenly across all classes — signs this "
            f"may not be a clear photo of a leaf this model was trained on."
        )
    elif low_confidence:
        reason = (
            f"Top match is only {max_prob * 100:.0f}% confident — lower than "
            f"expected for a clear, in-distribution photo."
        )
    elif flat_distribution:
        reason = (
            "The model's confidence is spread fairly evenly across all "
            "possible classes rather than settling on one — a sign of "
            "genuine uncertainty, even though one class scored highest."
        )
    else:
        reason = f"Top match is {max_prob * 100:.0f}% confident, with a clear peak."

    return {
        "max_prob": max_prob,
        "entropy": entropy,
        "normalized_entropy": round(normalized_entropy, 3),
        "is_likely_ood": is_likely_ood,
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# Self-test — confident in-distribution case, uniform garbage-input case,
# and a genuinely-ambiguous-but-real-class borderline case.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("--- Confident, clearly in-distribution (e.g. a textbook Late Blight photo) ---")
    confident = np.array([0.02, 0.03, 0.95])
    r1 = compute_ood_signal(confident)
    print(r1)
    assert not r1["is_likely_ood"]

    print("\n--- Uniform / garbage input (e.g. a photo of a hand) ---")
    uniform = np.array([0.34, 0.33, 0.33])
    r2 = compute_ood_signal(uniform)
    print(r2)
    assert r2["is_likely_ood"]
    assert r2["normalized_entropy"] > 0.99

    print("\n--- Genuinely ambiguous between two real, visually-similar classes ---")
    ambiguous = np.array([0.05, 0.48, 0.47])
    r3 = compute_ood_signal(ambiguous)
    print(r3)
    assert r3["is_likely_ood"], "Should still be flagged — the model itself isn't confident either way"

    print("\n--- Clear-ish top pick, but with meaningful competition ---")
    moderate = np.array([0.05, 0.15, 0.80])
    r4 = compute_ood_signal(moderate)
    print(r4)
    assert not r4["is_likely_ood"], "0.80 max prob with a low-entropy tail should NOT be flagged"

    print("\nOK — all checks passed.")