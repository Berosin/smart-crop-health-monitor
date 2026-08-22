"""Environmental crop risk prediction — scikit-learn classifiers.

Predicts a crop's environmental risk level (Optimal / Low / Moderate / High
/ Critical) from crop type + four sensor readings (temperature, humidity,
soil moisture, rainfall).

Pipeline:
- generate_synthetic_dataset(): builds a labeled dataset from the same
  ideal-range knowledge already encoded in config.ENV_CROP_RANGES, with
  sensor noise and boundary jitter so the classification task isn't
  trivial (mirrors how a rule-based label would be produced in the field,
  then measured with imperfect sensors).
- train_and_compare(): trains a DecisionTreeClassifier and a
  RandomForestClassifier, evaluates both on a held-out test set, and
  keeps the better one (by macro F1).
- The winning model is saved to models/environment_model/.
- predict_environmental_risk(data): loads the saved model and returns
  risk level, probability, a plain-language explanation, and a
  recommendation for a single reading.

Run directly to train:
    python -m src.environment_model --samples-per-crop 2000
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

from config import ENV_CROP_RANGES, ENV_RANGES, ENV_RISK_THRESHOLDS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL_DIR = Path("models/environment_model")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODEL_DIR / "model.joblib"
METADATA_PATH = MODEL_DIR / "metadata.json"

FEATURE_COLUMNS = ["crop", "temperature", "humidity", "soil_moisture", "rainfall"]
NUMERIC_COLUMNS = ["temperature", "humidity", "soil_moisture", "rainfall"]
CROPS = list(ENV_CROP_RANGES.keys())

# Risk labels in increasing severity order, taken from the same thresholds
# the rule-based UI already uses (config.ENV_RISK_THRESHOLDS).
RISK_LABELS = [ENV_RISK_THRESHOLDS[k][0] for k in sorted(ENV_RISK_THRESHOLDS)]

TRAIN_CONFIG = {
    "samples_per_crop": 2000,
    "test_size": 0.2,
    "seed": 42,
    "sensor_noise_pct": 0.02,   # +/- 2% of each factor's full range, as measurement noise
    "extreme_share": 0.15,      # share of "out-of-range" factors pushed fully outside [low, high]
}

# Plain-language advice per factor + direction, reused by the explanation
# and recommendation builders. Kept self-contained here (src/ should not
# import from pages/) even though pages/environment.py has a similar
# rule-based table for its own standalone display.
TIPS = {
    "temperature": {
        "Low":  "Consider row covers or warming the soil — cold slows growth of {crop}.",
        "High": "Provide shade cloth or irrigate in the cool evening to reduce heat stress on {crop}.",
        "Extreme": "Temperature is outside the safe band — protect {crop} or delay sensitive field operations.",
    },
    "humidity": {
        "Low":  "Humidity is low; mulch around {crop} to reduce moisture loss and leaf curl.",
        "High": "High humidity raises fungal-disease risk for {crop}; improve airflow and avoid overhead watering.",
        "Extreme": "Humidity level is extreme — monitor {crop} closely for disease or wilting.",
    },
    "soil_moisture": {
        "Low":  "Soil is dry for {crop}; schedule irrigation and check root zones.",
        "High": "Soil is waterlogged for {crop}; improve drainage and hold off irrigation.",
        "Extreme": "Soil moisture is off-scale for {crop}; correct drainage or watering immediately.",
    },
    "rainfall": {
        "Low":  "Rainfall is light; supplement {crop} with irrigation.",
        "High": "Heavy rainfall expected; watch for runoff and leaching around {crop}; ensure field drainage.",
        "Extreme": "Extreme rainfall for {crop}; protect low-lying areas and check for standing water.",
    },
}

FACTOR_LABELS = {
    "temperature": "Temperature",
    "humidity": "Humidity",
    "soil_moisture": "Soil moisture",
    "rainfall": "Rainfall",
}


# ---------------------------------------------------------------------------
# Ground-truth labeling rule (shared by dataset generation & explanations)
# ---------------------------------------------------------------------------
def _factor_status(crop: str, key: str, value: float) -> str:
    """Return 'Optimal' | 'Low' | 'High' | 'Extreme' for one reading."""
    low, opt_min, opt_max, high = ENV_CROP_RANGES[crop][key]
    if opt_min <= value <= opt_max:
        return "Optimal"
    if low <= value < opt_min:
        return "Low"
    if opt_max < value <= high:
        return "High"
    return "Extreme"  # below `low` or above `high`


def _label_row(crop: str, row: dict) -> tuple[str, int, dict[str, str]]:
    """Score all four factors for one reading -> (risk_level, suboptimal_count, statuses)."""
    statuses = {key: _factor_status(crop, key, row[key]) for key in NUMERIC_COLUMNS}
    suboptimal = sum(1 for s in statuses.values() if s != "Optimal")
    risk_level = ENV_RISK_THRESHOLDS[min(suboptimal, 4)][0]
    return risk_level, suboptimal, statuses


# ---------------------------------------------------------------------------
# 1. Synthetic dataset generation
# ---------------------------------------------------------------------------
def generate_synthetic_dataset(
    samples_per_crop: int = TRAIN_CONFIG["samples_per_crop"],
    seed: int = TRAIN_CONFIG["seed"],
) -> pd.DataFrame:
    """Build a labeled (crop, 4 readings) -> risk_level dataset.

    For each crop, samples are drawn targeting each count of suboptimal
    factors (0..4) in roughly equal shares so all five risk levels are
    represented, then perturbed with sensor noise. The final label is
    always recomputed from the exact rule on the *noisy* values, so noise
    near a boundary can genuinely flip the true label — this is what makes
    the classification task non-trivial and gives Random Forest a real
    chance to outperform a single Decision Tree.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    for crop in CROPS:
        ranges = ENV_CROP_RANGES[crop]
        for _ in range(samples_per_crop):
            # Choose how many of the 4 factors should start out-of-range.
            target_suboptimal = int(rng.integers(0, 5))  # 0..4 inclusive
            out_keys = set(rng.choice(NUMERIC_COLUMNS, size=target_suboptimal, replace=False))

            row = {"crop": crop}
            for key in NUMERIC_COLUMNS:
                low, opt_min, opt_max, high = ranges[key]
                gmin, gmax = ENV_RANGES[key]["min"], ENV_RANGES[key]["max"]

                if key in out_keys:
                    push_extreme = rng.random() < TRAIN_CONFIG["extreme_share"]
                    go_high = rng.random() < 0.5
                    if push_extreme:
                        value = (rng.uniform(gmin, low) if not go_high
                                 else rng.uniform(high, gmax))
                    else:
                        value = (rng.uniform(low, opt_min) if not go_high
                                 else rng.uniform(opt_max, high))
                else:
                    value = rng.uniform(opt_min, opt_max)

                # Sensor measurement noise.
                span = gmax - gmin
                value += rng.normal(0, TRAIN_CONFIG["sensor_noise_pct"] * span)
                row[key] = float(np.clip(value, gmin, gmax))

            risk_level, suboptimal, _ = _label_row(crop, row)
            row["suboptimal_count"] = suboptimal
            row["risk_level"] = risk_level
            rows.append(row)

    df = pd.DataFrame(rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


# ---------------------------------------------------------------------------
# 2. Model pipelines
# ---------------------------------------------------------------------------
def _build_pipeline(estimator) -> Pipeline:
    preprocess = ColumnTransformer(
        transformers=[("crop_ohe", OneHotEncoder(handle_unknown="ignore"), ["crop"])],
        remainder="passthrough",  # numeric columns pass through untouched
    )
    return Pipeline(steps=[("preprocess", preprocess), ("model", estimator)])


def _make_candidates(seed: int) -> dict[str, Pipeline]:
    return {
        "DecisionTree": _build_pipeline(
            DecisionTreeClassifier(max_depth=16, min_samples_leaf=4, random_state=seed)
        ),
        "RandomForest": _build_pipeline(
            RandomForestClassifier(
                n_estimators=200, max_depth=12, min_samples_leaf=3,
                random_state=seed, n_jobs=-1,
            )
        ),
    }


# ---------------------------------------------------------------------------
# 3. Evaluation
# ---------------------------------------------------------------------------
def _evaluate(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_pred = pipeline.predict(X_test)
    return {
        "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
        "macro_f1": round(float(f1_score(y_test, y_pred, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, y_pred, average="weighted")), 4),
        "classification_report": classification_report(
            y_test, y_pred, labels=RISK_LABELS, zero_division=0, output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred, labels=RISK_LABELS).tolist(),
    }


# ---------------------------------------------------------------------------
# 4. Train, compare, select, save
# ---------------------------------------------------------------------------
def train_and_compare(
    df: pd.DataFrame | None = None,
    save: bool = True,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train Decision Tree + Random Forest, evaluate both, keep the better one."""
    config = TRAIN_CONFIG
    if df is None:
        df = generate_synthetic_dataset(config["samples_per_crop"], config["seed"])

    X = df[FEATURE_COLUMNS]
    y = df["risk_level"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config["test_size"], random_state=config["seed"], stratify=y,
    )

    results: dict[str, dict] = {}
    fitted: dict[str, Pipeline] = {}
    for name, pipeline in _make_candidates(config["seed"]).items():
        pipeline.fit(X_train, y_train)
        metrics = _evaluate(pipeline, X_test, y_test)
        results[name] = metrics
        fitted[name] = pipeline
        if verbose:
            print(f"\n{name}")
            print(f"  accuracy    : {metrics['accuracy']:.4f}")
            print(f"  macro F1    : {metrics['macro_f1']:.4f}")
            print(f"  weighted F1 : {metrics['weighted_f1']:.4f}")

    # Select the better model: macro F1 first (fairer under class imbalance),
    # tie-broken by accuracy, then by preferring the ensemble (Random Forest)
    # since it generalizes better under the sensor-noise jitter we injected.
    def _rank(name: str) -> tuple[float, float, int]:
        m = results[name]
        return (m["macro_f1"], m["accuracy"], 1 if name == "RandomForest" else 0)

    best_name = max(results, key=_rank)
    best_pipeline = fitted[best_name]
    best_metrics = results[best_name]

    if verbose:
        print(f"\nSelected model: {best_name} "
              f"(macro F1 {best_metrics['macro_f1']:.4f}, accuracy {best_metrics['accuracy']:.4f})")

    if save:
        _save_artifacts(best_pipeline, best_name, results, len(df))

    return {
        "selected_model": best_name,
        "pipeline": best_pipeline,
        "all_results": results,
        "n_samples": len(df),
    }


def _save_artifacts(pipeline: Pipeline, selected_model: str,
                     all_results: dict, n_samples: int) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "selected_model": selected_model,
        "feature_columns": FEATURE_COLUMNS,
        "numeric_columns": NUMERIC_COLUMNS,
        "crops": CROPS,
        "risk_labels": RISK_LABELS,
        "n_samples": n_samples,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "metrics": {
            name: {k: v for k, v in m.items() if k != "confusion_matrix"}
            for name, m in all_results.items()
        },
        "confusion_matrices": {name: m["confusion_matrix"] for name, m in all_results.items()},
    }
    METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    print(f"\nSaved model      -> {MODEL_PATH}")
    print(f"Saved metadata   -> {METADATA_PATH}")


# ---------------------------------------------------------------------------
# 5. Loading + inference
# ---------------------------------------------------------------------------
_MODEL_CACHE: dict[str, Any] = {}


def load_environment_model(force_reload: bool = False) -> tuple[Pipeline, dict]:
    """Load the saved pipeline + metadata (cached in-process after first call)."""
    if not force_reload and "pipeline" in _MODEL_CACHE:
        return _MODEL_CACHE["pipeline"], _MODEL_CACHE["metadata"]

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Environmental risk model not found at {MODEL_PATH}. "
            "Train it first with: python -m src.environment_model"
        )

    pipeline = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text()) if METADATA_PATH.exists() else {}
    _MODEL_CACHE["pipeline"] = pipeline
    _MODEL_CACHE["metadata"] = metadata
    return pipeline, metadata


def _validate_input(data: dict) -> None:
    missing = [k for k in FEATURE_COLUMNS if k not in data]
    if missing:
        raise ValueError(f"Missing required field(s): {', '.join(missing)}")
    if data["crop"] not in CROPS:
        raise ValueError(f"Unknown crop '{data['crop']}'. Expected one of: {', '.join(CROPS)}")
    for key in NUMERIC_COLUMNS:
        try:
            float(data[key])
        except (TypeError, ValueError):
            raise ValueError(f"'{key}' must be a number, got {data[key]!r}")


def _build_explanation(crop: str, statuses: dict[str, str], data: dict,
                        risk_level: str, probability: float) -> str:
    ranges = ENV_CROP_RANGES[crop]
    off_range = [k for k, s in statuses.items() if s != "Optimal"]

    if not off_range:
        return (
            f"Predicted risk: {risk_level} ({probability*100:.0f}% confidence). "
            f"All four readings fall inside {crop}'s ideal range, so the model "
            "found no stress signal in this reading."
        )

    parts = []
    for key in off_range:
        opt_min, opt_max = ranges[key][1], ranges[key][2]
        unit = ENV_RANGES[key]["unit"]
        parts.append(
            f"{FACTOR_LABELS[key]} ({data[key]:.1f}{unit}, ideal {opt_min}-{opt_max}{unit})"
        )
    factor_text = "; ".join(parts)
    return (
        f"Predicted risk: {risk_level} ({probability*100:.0f}% confidence). "
        f"{len(off_range)} of 4 reading(s) fall outside {crop}'s ideal range — "
        f"{factor_text}. These are the factors driving the risk score."
    )


def _build_recommendation(crop: str, statuses: dict[str, str]) -> str:
    tips = []
    for key, status in statuses.items():
        if status != "Optimal" and status in TIPS[key]:
            tips.append(TIPS[key][status].format(crop=crop.lower()))
    if not tips:
        return (f"All environmental factors are within the ideal range for {crop}. "
                "Maintain current practices and keep monitoring.")
    return " ".join(tips)


def predict_environmental_risk(data: dict) -> dict[str, Any]:
    """Predict environmental crop risk for one reading.

    Args:
        data: {"crop": str, "temperature": float, "humidity": float,
               "soil_moisture": float, "rainfall": float}

    Returns:
        {
          "risk_level": str,                    # e.g. "Moderate"
          "probability": float,                 # confidence in the predicted class, 0..1
          "probabilities": {label: float, ...},  # full class distribution
          "explanation": str,                   # plain-language reasoning
          "recommendation": str,                # actionable advice
          "model_used": str,                    # "DecisionTree" or "RandomForest"
        }
    """
    _validate_input(data)
    pipeline, metadata = load_environment_model()

    crop = data["crop"]
    row = {k: float(data[k]) for k in NUMERIC_COLUMNS}
    row["crop"] = crop
    X = pd.DataFrame([row])[FEATURE_COLUMNS]

    predicted = pipeline.predict(X)[0]
    proba = pipeline.predict_proba(X)[0]
    classes = list(pipeline.named_steps["model"].classes_)
    probabilities = {cls: round(float(p), 4) for cls, p in zip(classes, proba)}
    probability = probabilities[predicted]

    _, _, statuses = _label_row(crop, row)

    return {
        "risk_level": predicted,
        "probability": probability,
        "probabilities": dict(sorted(probabilities.items(), key=lambda kv: -kv[1])),
        "explanation": _build_explanation(crop, statuses, row, predicted, probability),
        "recommendation": _build_recommendation(crop, statuses),
        "model_used": metadata.get("selected_model", "unknown"),
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train the environmental crop risk model")
    parser.add_argument("--samples-per-crop", type=int, default=TRAIN_CONFIG["samples_per_crop"])
    parser.add_argument("--seed", type=int, default=TRAIN_CONFIG["seed"])
    args = parser.parse_args()

    TRAIN_CONFIG["samples_per_crop"] = args.samples_per_crop
    TRAIN_CONFIG["seed"] = args.seed

    print(f"Generating synthetic dataset ({args.samples_per_crop} samples/crop x {len(CROPS)} crops)...")
    dataset = generate_synthetic_dataset(args.samples_per_crop, args.seed)
    print(f"Dataset shape: {dataset.shape}")
    print(dataset["risk_level"].value_counts())

    print("\nTraining Decision Tree and Random Forest...")
    result = train_and_compare(dataset)

    print("\n--- Example predictions ---")
    examples = [
        {"crop": "Tomato", "temperature": 24, "humidity": 60, "soil_moisture": 55, "rainfall": 30},
        {"crop": "Rice", "temperature": 38, "humidity": 30, "soil_moisture": 20, "rainfall": 5},
    ]
    for ex in examples:
        pred = predict_environmental_risk(ex)
        print(f"\nInput: {ex}")
        print(f"  risk_level     : {pred['risk_level']}")
        print(f"  probability    : {pred['probability']}")
        print(f"  model_used     : {pred['model_used']}")
        print(f"  explanation    : {pred['explanation']}")
        print(f"  recommendation : {pred['recommendation']}")