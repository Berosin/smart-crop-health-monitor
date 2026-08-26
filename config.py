"""Central configuration for the Crop Health Monitoring System.

Keeping all constants here makes it easy to tune paths, model settings,
and UI text in one place without hunting through the codebase.
"""

# ---------------------------------------------------------------------------
# Application metadata (used by the Streamlit UI)
# ---------------------------------------------------------------------------
APP_CONFIG = {
    "title": "Smart Crop Health Monitoring & Disease Detection",
    "subtitle": "AI-powered crop disease detection and health scoring",
    # Semantic key resolved through utils.icons (Tabler Icons), not an emoji.
    "page_icon": "leaf",
}

# ---------------------------------------------------------------------------
# File / directory paths
# ---------------------------------------------------------------------------
DB_PATH = "database/crop_health.db"

# ---------------------------------------------------------------------------
# Disease detection models — one trained MobileNetV2 model per crop
# ---------------------------------------------------------------------------
# Each crop gets its own folder under models/ holding model.keras + labels.json.
# To add a new crop: train it (src/model_training.py --crop <Name>), then add
# an entry here — Disease Detection, Health Analysis, and About all read from
# this single registry, so nothing else needs to change.
DISEASE_MODELS = {
    "Tomato": {
        "model_path": "models/disease_model_tomato/model.keras",
        "labels_path": "models/disease_model_tomato/labels.json",
    },
    "Potato": {
        "model_path": "models/disease_model_potato/model.keras",
        "labels_path": "models/disease_model_potato/labels.json",
    },
}
DEFAULT_DISEASE_CROP = "Tomato"

# Backward-compatible aliases for any code referencing a single model.
MODEL_PATH = DISEASE_MODELS[DEFAULT_DISEASE_CROP]["model_path"]
LABELS_PATH = DISEASE_MODELS[DEFAULT_DISEASE_CROP]["labels_path"]


def get_trained_crops() -> list[str]:
    """Return DISEASE_MODELS crops whose model file is actually present on disk.

    Pages use this instead of hardcoding a crop list, so a newly trained
    crop appears in every dropdown as soon as its model file exists —
    no other code changes needed.
    """
    import os
    return [crop for crop, paths in DISEASE_MODELS.items() if os.path.exists(paths["model_path"])]


ENV_MODEL_DIR = "models/environment_model"
ENV_MODEL_PATH = "models/environment_model/model.joblib"
ENV_MODEL_METADATA_PATH = "models/environment_model/metadata.json"

UPLOAD_DIR = "data/uploads"
SAMPLE_DIR = "data/samples"

# ---------------------------------------------------------------------------
# Model / image settings (placeholders for the AI stage)
# ---------------------------------------------------------------------------
IMAGE_SIZE = (224, 224)          # target resolution fed to the CNN
IMAGE_CHANNELS = 3              # RGB
CONFIDENCE_THRESHOLD = 0.70     # below this the model is treated as "unsure"

# ---------------------------------------------------------------------------
# Environmental condition ranges
# ---------------------------------------------------------------------------
# Single source of truth for the four environmental factors used throughout
# the app: sensor bounds (min/max/unit), display metadata (icon key, label),
# and the widget step size (nudge) for number_input sliders. Previously
# duplicated as separate dicts in pages/environment.py and pages/health.py.
ENV_RANGES = {
    "temperature": {
        "min": -10, "max": 50, "unit": "°C",
        "icon": "temperature", "label": "Temperature", "nudge": 2.0,
    },
    "humidity": {
        "min": 0, "max": 100, "unit": "%",
        "icon": "humidity", "label": "Humidity", "nudge": 5.0,
    },
    "soil_moisture": {
        "min": 0, "max": 100, "unit": "%",
        "icon": "soil", "label": "Soil moisture", "nudge": 5.0,
    },
    "rainfall": {
        "min": 0, "max": 500, "unit": "mm",
        "icon": "rainfall", "label": "Rainfall", "nudge": 2.0,
    },
}

# ---------------------------------------------------------------------------
# Per-crop ideal environmental ranges used by the Environmental Analysis page
# ---------------------------------------------------------------------------
# Each tuple is (low, optimal_min, optimal_max, high) — values below `low` or
# above `high` are suboptimal; between optimal_min and optimal_max is ideal.
ENV_CROP_RANGES = {
    "Tomato": {
        "temperature":   (10, 18, 30, 38),   # °C
        "humidity":      (40, 55, 70, 90),   # %
        "soil_moisture": (35, 50, 65, 85),   # %
        "rainfall":      (0,  20, 80, 150),  # mm/day
    },
    "Corn": {
        "temperature":   (10, 18, 32, 40),
        "humidity":      (45, 55, 75, 90),
        "soil_moisture": (40, 50, 70, 85),
        "rainfall":      (0,  25, 100, 160),
    },
    "Potato": {
        "temperature":   (7,  15, 25, 32),
        "humidity":      (50, 60, 80, 95),
        "soil_moisture": (45, 55, 70, 85),
        "rainfall":      (0,  20, 90, 140),
    },
    "Rice": {
        "temperature":   (15, 22, 33, 42),
        "humidity":      (55, 65, 85, 98),
        "soil_moisture": (60, 75, 95, 100),
        "rainfall":      (0,  40, 150, 250),
    },
    "Wheat": {
        "temperature":   (5,  12, 25, 35),
        "humidity":      (40, 50, 70, 88),
        "soil_moisture": (35, 45, 65, 80),
        "rainfall":      (0,  15, 70, 120),
    },
}

ENV_RISK_THRESHOLDS = {
    # (count of suboptimal factors) -> (risk level, label color)
    0: ("Optimal",   "#2F6D46"),
    1: ("Low",       "#7FA687"),
    2: ("Moderate",  "#C97A3B"),
    3: ("High",      "#B5564B"),
    4: ("Critical",  "#7C3730"),
}