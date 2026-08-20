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
    "page_icon": "🌿",
}

# ---------------------------------------------------------------------------
# File / directory paths
# ---------------------------------------------------------------------------
DB_PATH = "database/crop_health.db"

MODEL_PATH = "models/crop_disease_model.h5"
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
