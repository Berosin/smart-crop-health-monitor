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

MODEL_PATH = "models/disease_model/model.keras"
LABELS_PATH = "models/disease_labels.json"

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
ENV_RANGES = {
    "temperature": {"min": -10, "max": 50, "unit": "°C"},    # air temperature
    "humidity":    {"min": 0,   "max": 100, "unit": "%"},    # relative humidity
    "soil_moisture": {"min": 0, "max": 100, "unit": "%"},    # soil water content
    "rainfall":    {"min": 0,   "max": 500, "unit": "mm"},   # daily rainfall
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

