# Smart Crop Health Monitoring & Disease Detection System

**AI3021 — IT in Agriculture, Mini Project**

An AI-powered Streamlit application that helps farmers and agronomists
detect crop diseases from leaf photos, assess environmental growing
conditions, and combine both into a single explainable crop health score
with practical, actionable recommendations — all backed by trained
machine learning models and persistent SQLite storage.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [Features](#4-features)
5. [System Architecture](#5-system-architecture)
6. [Modules](#6-modules)
7. [Technology Stack](#7-technology-stack)
8. [Dataset](#8-dataset)
9. [AI/ML Methodology](#9-aiml-methodology)
10. [Disease Detection Workflow](#10-disease-detection-workflow)
11. [Environmental Analysis](#11-environmental-analysis)
12. [Health Score Methodology](#12-health-score-methodology)
13. [Database Design](#13-database-design)
14. [Project Structure](#14-project-structure)
15. [Installation](#15-installation)
16. [How to Run](#16-how-to-run)
17. [Sample Usage](#17-sample-usage)
18. [Results](#18-results)
19. [Limitations](#19-limitations)
20. [Future Enhancements](#20-future-enhancements)
21. [Conclusion](#21-conclusion)

---

## 1. Project Overview

Smart Crop Health Monitor is a full-stack, AI-driven web application built
with Streamlit that brings together two independent machine learning
models — a **deep learning image classifier** for leaf disease detection
and a **classical ML classifier** for environmental risk — and fuses their
outputs through a custom, explainable scoring engine into one overall crop
health assessment. Completed Crop Health Analysis results can be saved to a local
SQLite database, browsed later in a filterable history, and summarized on
a live statistics dashboard.

The project was built to demonstrate an end-to-end applied-AI workflow in
an agricultural context: data preparation → model training → model
comparison and selection → inference → rule-based decision support →
persistence → visualization, all wrapped in a polished, demo-ready web UI.

## 2. Problem Statement

Smallholder and even commercial farmers often lack easy, immediate access
to expert diagnosis when a crop shows signs of disease or stress. By the
time visible symptoms are correctly identified and understood alongside
the day's environmental conditions (temperature, humidity, soil moisture,
rainfall), a treatable problem can have already spread. There is a need
for an accessible tool that:

- Identifies common leaf diseases from a simple photo, without requiring
  a plant pathologist.
- Interprets environmental readings against crop-specific ideal ranges,
  since the same reading (e.g. 90% humidity) can be fine for rice and
  dangerous for wheat.
- Combines both signals into a single, easy-to-understand health status
  rather than forcing the user to mentally reconcile two separate reports.
- Gives specific, actionable next steps — not just a diagnosis.
- Keeps a historical record so trends can be tracked over time.

## 3. Objectives

1. Train and deploy a convolutional neural network (CNN) that classifies
   crop leaf images as healthy or diseased, with severity.
2. Train and compare multiple classical ML models to classify
   environmental risk from sensor-style readings, and automatically select
   the better-performing one.
3. Design a modular, explainable engine that combines both signals into a
   single 0–100 health score with a clear textual rationale.
4. Build a rule-based recommendation engine (no external APIs or LLMs)
   that produces specific, traceable agricultural advice.
5. Persist completed crop health assessments to a relational database and provide a
   dashboard and searchable history.
6. Deliver all of the above through a single, cohesive, professionally
   styled Streamlit interface suitable for a live demonstration.

## 4. Features

- **Disease Detection** — upload a leaf photo; a trained MobileNetV2 CNN
  predicts the disease (or healthy), with confidence and severity.
- **Image validation pipeline** — OpenCV-based checks reject corrupted
  files, unsupported formats, and oversized images before inference,
  with optional denoising and background flattening.
- **Environmental Analysis** — enter temperature, humidity, soil
  moisture, and rainfall for a chosen crop; a trained classifier predicts
  a 5-level risk category with confidence and a per-factor breakdown.
- **Crop Health Analysis** — combines the disease result and
  environmental risk into one 0–100 health score, classified into four
  bands, with a plain-language explanation of how the score was reached.
- **Rule-Based Recommendations** — a deterministic engine (no ML, no
  external API) producing prioritized, explainable action items.
- **Persistent Storage** — save completed Crop Health Analysis results to SQLite with one
  click; duplicate-save protection across Streamlit reruns.
- **Dashboard** — live KPIs (total analyses, healthy/diseased counts,
  average health score, high-risk cases) and four Plotly charts, all
  computed from real saved data.
- **Analysis History** — review saved Crop Health Analysis results; filter by crop/disease/date, sort by any column,
  inspect full detail per record, and delete records (two-step confirm).
- **Centralized validation & error handling** — every user input is
  validated with clear messages; unexpected errors are logged server-side
  and never shown to the user as a raw stack trace.
- **Cohesive visual design** — a custom "field journal & spectral scan"
  theme (dedicated color palette, typography, and Tabler Icons) applied
  consistently across every page.

## 5. System Architecture

The application follows a layered architecture that keeps the core logic
completely independent of the UI framework:

```
┌─────────────────────────────────────────────────────────────────┐
│                         Streamlit UI Layer                       │
│   pages/  (disease · environment · health · dashboard · history) │
│              utils/ (design system, icons, shared widgets)       │
└───────────────────────────────┬───────────────────────────────────┘
                                 │  calls
┌───────────────────────────────▼───────────────────────────────────┐
│                          Core Logic Layer (src/)                   │
│                                                                     │
│  image_preprocessing.py   →  OpenCV validation & preprocessing     │
│  model_training.py        →  Disease CNN (MobileNetV2)             │
│  environment_model.py     →  Env. risk (Decision Tree / RF)        │
│  health_engine.py         →  Combines both into a health score     │
│  recommendation_engine.py →  Rule-based advice generator           │
│  validation.py            →  Shared input validation               │
│  errors.py                →  Shared error boundary / logging       │
│  db.py                    →  SQLite persistence                    │
└───────────────────────────────┬───────────────────────────────────┘
                                 │  reads/writes
┌───────────────────────────────▼───────────────────────────────────┐
│         models/disease_model/  ·  models/environment_model/        │
│                    database/crop_health.db (SQLite)                 │
└─────────────────────────────────────────────────────────────────┘
```

Every module under `src/` is plain Python with no Streamlit dependency,
so each one can be run, tested, or imported independently of the web app
(each has a `python -m src.<module>` self-test/CLI entry point).

## 6. Modules

| Module | Responsibility |
|---|---|
| `src/dataset_prep.py` | Loads and splits the leaf-image dataset for training (directory-per-class or CSV-labeled layouts), builds `tf.data` pipelines with augmentation. |
| `src/model_training.py` | Builds, trains, evaluates, and saves the MobileNetV2 disease-detection CNN; two-phase transfer learning (frozen head, then fine-tuning). |
| `src/image_preprocessing.py` | OpenCV pipeline: format/corruption/size validation, RGB conversion, resize, optional denoising and background flattening, MobileNetV2-compatible normalization. |
| `src/environment_model.py` | Generates a synthetic labeled dataset from known crop ideal ranges, trains and compares a Decision Tree and a Random Forest, saves the better one, and exposes `predict_environmental_risk()`. |
| `src/health_engine.py` | Modular, explainable scoring: `compute_disease_risk_score()`, `compute_environmental_risk_score()`, `compute_health_score()`, `classify_health_status()`. |
| `src/recommendation_engine.py` | Pure rule-based recommendation generator — disease-, environment-, and health-score-driven rules, each returning its own textual "reason". |
| `src/validation.py` | Centralized input validation (crop, environmental readings, confidence, severity, health score) — one consistent error message format app-wide. |
| `src/errors.py` | Shared exception hierarchy (`DatabaseError`, `ModelNotFoundError`, `PredictionError`) and `safe_action()`, a context manager that logs unexpected errors server-side and shows only a clean message to the user. |
| `src/db.py` | SQLite schema, connection handling, and CRUD functions — framework-independent, unit-testable on its own. |
| `utils/ui.py` | The shared design system: CSS injection, page header/sidebar/footer, metric tiles, risk badges, health-score card, Plotly chart theme. |
| `utils/icons.py` | Renders Tabler Icons (via `pytablericons`) as cached base64 images for use throughout the UI. |
| `pages/*.py` | One module per page — each owns its own layout and calls straight into the `src/` layer above. |

## 7. Technology Stack

| Layer | Technology |
|---|---|
| Web UI / App framework | Streamlit |
| Deep Learning | TensorFlow / Keras (MobileNetV2 transfer learning) |
| Classical ML | scikit-learn (Decision Tree, Random Forest) |
| Image Processing | OpenCV (`opencv-python-headless`) |
| Data handling | pandas, NumPy |
| Visualization | Plotly (interactive charts), Matplotlib + Seaborn (training plots) |
| Database | SQLite (Python standard library `sqlite3`) |
| Model persistence | Keras native format (`.keras`) for the CNN, `joblib` for the scikit-learn pipeline |
| Icons | Tabler Icons via `pytablericons` |
| Language | Python 3 |

## 8. Dataset

**Disease detection (image data).** The CNN is trained on a
directory-per-class image dataset (`data/samples/<ClassName>/*.jpg`) —
compatible with public leaf-disease datasets such as PlantVillage. The
model as shipped/documented targets three classes: `Healthy`,
`Early_Blight`, and `Late_Blight`; `src/dataset_prep.py` infers class
names automatically from subdirectory names, so retraining on a different
class set (e.g. adding more diseases or crops) requires no code changes —
just a differently organized `data/` folder. A CSV-labeled layout is also
supported via `prepare_datasets_from_csv()` for datasets that ship labels
as a spreadsheet rather than folders.

**Environmental risk (tabular data).** Rather than depending on a scarce
labeled sensor dataset, `src/environment_model.py` generates a realistic
**synthetic dataset** from domain knowledge already encoded in the app:
each of 5 crops (Tomato, Corn, Potato, Rice, Wheat) has an expert-defined
ideal range for temperature, humidity, soil moisture, and rainfall
(`config.ENV_CROP_RANGES`). Samples are drawn to cover all five risk
levels evenly, then perturbed with realistic sensor measurement noise so
that some readings land ambiguously near a boundary — this is what makes
the resulting classification problem non-trivial (a simple lookup table
would not need machine learning at all). 10,000 samples (2,000 per crop)
were used for the trained model shipped with this project.

## 9. AI/ML Methodology

Two independent models are trained and evaluated using standard
supervised-learning practice:

**Disease detection — deep learning.**
- Transfer learning on **MobileNetV2** (ImageNet-pretrained weights, top
  layers removed), with a custom head: `GlobalAveragePooling2D → Dropout
  → Dense(num_classes, softmax)`.
- Two-phase training: (1) freeze the MobileNetV2 base and train only the
  new head, (2) unfreeze and fine-tune at a lower learning rate.
- Callbacks: early stopping, model checkpointing, learning-rate
  scheduling, and TensorBoard logging.
- Evaluated with accuracy, a classification report, and a confusion
  matrix on a held-out test split.

**Environmental risk — classical ML, trained and compared.**
- Features: crop type (one-hot encoded) + 4 numeric readings.
- Two candidate models trained on an identical 80/20 stratified
  train/test split: `DecisionTreeClassifier` and `RandomForestClassifier`.
- Both evaluated on accuracy, macro F1, weighted F1, a full
  classification report, and a confusion matrix.
- The better model is selected automatically by macro F1 (tie-broken by
  accuracy, then by preferring the Random Forest for its better
  generalization) and persisted — no manual model choice required.

On the shipped 10,000-sample dataset, the comparison was:

| Model | Accuracy | Macro F1 |
|---|---|---|
| Decision Tree | 69.9% | 0.713 |
| **Random Forest (selected)** | **79.4%** | **0.804** |

## 10. Disease Detection Workflow

1. **Upload** — the user uploads a leaf photo (JPEG/PNG/BMP/WEBP) on the
   Disease Detection page.
2. **Validate** (`src/image_preprocessing.py`) — reject empty files,
   unsupported formats (magic-byte sniffed), corrupted images
   (`cv2.imdecode` failure), and images that are too large (>10 MB, or
   decoded dimensions >8000 px/side or >40 MP) or too small (<32 px/side).
3. **Preprocess** — decode with OpenCV, convert BGR→RGB, resize to
   224×224, optionally denoise (`fastNlMeansDenoisingColored`) and/or
   flatten the background (HSV color-threshold masking), then normalize
   to `[-1, 1]` exactly as MobileNetV2 expects.
4. **Predict** — the cached, trained Keras model returns per-class
   probabilities; the top class, its confidence, and a severity label are
   derived, with a "low confidence" flag if below the configured
   threshold (default 70%).
5. **Recommend** — a disease-specific recommendation is looked up and
   shown alongside the prediction breakdown (a bar chart of all class
   probabilities).
6. **Save (optional)** — the result can be persisted to SQLite.

## 11. Environmental Analysis

1. The user selects a **crop** and enters **temperature, humidity, soil
   moisture, and rainfall**.
2. Each reading is compared against that crop's ideal range
   (`config.ENV_CROP_RANGES`) to produce a per-factor status (Optimal /
   Low / High / Extreme) — shown in a table and a radar chart.
3. The same four readings plus crop are passed to the **trained
   environment model** (`predict_environmental_risk()`), which returns:
   - `risk_level` — one of Optimal / Low / Moderate / High / Critical
   - `probability` — the model's confidence in that class
   - `probabilities` — the full probability distribution over all classes
   - `explanation` — a plain-language description of which readings are
     driving the risk
   - `recommendation` — actionable advice for any out-of-range factor
4. A continuous 0–100 "environmental health score" is derived from the
   *probability-weighted* average of each risk class's representative
   score, rather than snapping to one bucket — a more honest estimate
   near classification boundaries.

## 12. Health Score Methodology

`src/health_engine.py` combines the disease and environmental signals
through a small set of pure, independently testable functions:

1. **Disease score** (0–100, 100 = no risk): a confident "healthy" call
   scores near 100; a diseased call is penalized by both severity
   (`None`→0.0, `Mild`→0.3, `Moderate`→0.6, `High`→0.9 weight) and model
   confidence.
2. **Environmental score** (0–100, 100 = no risk): the environment
   model's risk class mapped through representative anchor scores
   (Optimal=95 … Critical=10), probability-weighted when the full
   distribution is available.
3. **Overall health score**: a weighted blend —
   **55% disease + 45% environment** — rounded to an integer 0–100.
4. **Classification bands**:

   | Score | Status |
   |---|---|
   | 80 – 100 | **Healthy** |
   | 60 – 79 | **Moderate** |
   | 40 – 59 | **At Risk** |
   | 0 – 39 | **Critical** |

5. **Explanation**: every step above returns its own plain-language
   reasoning string; `analyze_crop_health()` rolls all of them into one
   final explanation shown directly in the UI — the score is never a
   black box.

## 13. Database Design

A single SQLite table, `analyses`, stores every completed analysis:

```sql
CREATE TABLE IF NOT EXISTS analyses (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    crop_name          TEXT    NOT NULL,
    image_path         TEXT,
    disease            TEXT,
    confidence         REAL,
    severity           TEXT,
    temperature        REAL,
    humidity           REAL,
    soil_moisture      REAL,
    rainfall           REAL,
    health_score       INTEGER,
    disease_risk       TEXT,
    environmental_risk TEXT,
    recommendation     TEXT,
    created_at         TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ','now'))
);
```

- `recommendation` stores the full rule-based recommendation output as
  JSON (summary, prioritized action list, and per-item reasons), so the
  History page can re-render the same rich, explainable view later.
- `created_at` is filled in automatically (UTC, ISO-8601) if not supplied.
- `src/db.py` is entirely framework-independent (no Streamlit import), so
  it can be exercised directly from a Python shell or unit tests.
- Every database operation is wrapped so a raw `sqlite3.Error` is never
  surfaced to the UI — it becomes a clean `DatabaseError` with an
  actionable message instead.

## 14. Project Structure

```text
smart-crop-health-monitor/
├── app.py                       # Streamlit entry point (home page + navigation)
├── config.py                    # Central configuration & constants
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── .streamlit/config.toml       # App theme (colors, fonts)
├── database/                    # SQLite database file (created on first save)
├── models/
│   ├── disease_model/            # Trained MobileNetV2 Keras model + labels
│   └── environment_model/         # Trained scikit-learn model + metadata
├── data/                          # Training/sample images (not committed)
├── src/                             # Core logic, independent of Streamlit
│   ├── model_training.py             # Disease CNN: build, train, evaluate
│   ├── dataset_prep.py                # Dataset loading/splitting utilities
│   ├── environment_model.py            # Env. risk model: train, compare, predict
│   ├── health_engine.py                 # Combines disease + env into a health score
│   ├── recommendation_engine.py          # Rule-based agricultural recommendations
│   ├── image_preprocessing.py             # OpenCV validation/preprocessing pipeline
│   ├── validation.py                       # Shared input validation
│   ├── errors.py                            # Shared exceptions + safe error handling
│   └── db.py                                 # SQLite persistence layer
├── utils/                            # Streamlit-specific shared helpers
│   ├── ui.py                          # Design system, page chrome, UI components
│   └── icons.py                        # Tabler Icons rendering
└── pages/                            # One module per app page
    ├── disease.py                     # Disease Detection (image upload)
    ├── environment.py                  # Environmental Analysis
    ├── health.py                        # Crop Health Analysis (+ Save Analysis)
    ├── dashboard.py                      # Statistics & charts
    ├── history.py                         # Saved analyses: filter/sort/delete
    └── about.py                            # Project info
```

## 15. Installation

**1. Clone/copy the project and create a virtual environment (recommended):**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

**2. Install dependencies:**

```bash
pip install -r requirements.txt
```

**3. (Optional) Train the models.** The app runs without this step — any
page that needs a missing model shows a clear message with the exact
command to train it — but predictions won't work until the models exist.

```bash
# Disease detection CNN (needs a labeled image dataset, one folder per class)
python -m src.model_training --data-dir data/samples

# Environmental risk model (trains on an auto-generated synthetic dataset)
python -m src.environment_model
```

## 16. How to Run

```bash
streamlit run app.py
```

The app opens automatically in your browser at `http://localhost:8501`.
Use the sidebar to navigate between Home, Dashboard, Disease Detection,
Environmental Analysis, Crop Health Analysis, Analysis History, and About.

## 17. Sample Usage

**Scenario: checking on a tomato plant.**

1. Open **Environmental Analysis**, select **Tomato**, and enter today's
   readings (e.g. 32°C, 88% humidity, 75% soil moisture, 90 mm rainfall).
   Click **Assess environment** — the trained model returns something
   like *"Critical (79% confidence)"* with a factor-by-factor breakdown
   showing which readings are out of range.
2. Open **Disease Detection**, upload a photo of an affected leaf, and
   click **Analyze** — the CNN returns a prediction (e.g. *"Late_Blight,
  91% confidence, High severity"*) with a recommendation. This page does
  not save records to the database.
3. Open **Crop Health Analysis**, select the same crop/disease/severity/
   readings, and click **Calculate crop health** — the engine blends both
   signals into one score (e.g. **13/100 — Critical**) with a full
   explanation and a prioritized, rule-based action list.
4. Click **Save Analysis** on the Crop Health Analysis page to persist the
  complete record.
5. Open **Analysis History** to filter, sort, inspect, or delete saved
   records, or **Dashboard** to see it reflected in the live statistics.

## 18. Results

- The disease-detection CNN follows a standard MobileNetV2 transfer-
  learning recipe and is evaluated with accuracy, a full classification
  report, and a confusion matrix on a held-out test split (see
  `models/disease_model/metrics.json` and `confusion_matrix.png` after
  training).
- The environmental risk model was trained and compared on 10,000
  synthetic samples: the selected **Random Forest** reached **79.4%
  accuracy** and a **0.804 macro F1**, clearly outperforming the
  **Decision Tree** (69.9% accuracy, 0.713 macro F1) — a genuine,
  measured improvement from the ensemble, not an assumed one.
- The health-scoring engine was validated against hand-checked scenarios
  (e.g. a confident healthy prediction in optimal conditions scores in
  the high 90s; a high-severity, high-confidence disease detection in
  poor conditions scores in the low teens), confirming the weighting and
  classification bands behave as intended across the full range.
- All pages and the full save → view → filter → delete lifecycle were
  verified end-to-end using Streamlit's `AppTest` framework, exercising
  the real application code path (not just a syntax check), with zero
  unhandled exceptions across all seven pages in both empty-database and
  populated-database states.

## 19. Limitations

- The disease-detection model is trained on a limited class set
  (`Healthy`, `Early_Blight`, `Late_Blight`) and a single MobileNetV2
  input resolution (224×224); diseases outside this set will be
  misclassified into the nearest known class rather than flagged as
  "unknown."
- The environmental risk model is trained on **synthetic data** generated
  from expert-defined ideal ranges rather than real historical sensor
  readings — it captures the encoded domain knowledge well, but has not
  been validated against real-world field sensor data.
- The health-score weighting (55% disease / 45% environment) and the
  disease-severity weights are reasonable, explainable defaults chosen
  for this project, not values fitted to real outcome data.
- The recommendation engine is intentionally rule-based (no LLM/external
  API), so its advice is limited to the scenarios its rule tables cover;
  it will not generalize to novel situations the way a language model
  might.
- The application uses SQLite, which is well suited to a single-user
  demonstration but not to concurrent multi-user production traffic.
- There is no user authentication or multi-farm/multi-user data
  separation — all saved analyses share one database.

## 20. Future Enhancements

- Expand the disease-detection dataset to more crops and diseases, and
  add an explicit "unknown/other" class for out-of-distribution images.
- Replace the synthetic environmental training data with real historical
  sensor/weather data once available, and re-validate model performance.
- Support live IoT sensor integration for automatic environmental
  readings instead of manual entry.
- Add user accounts and per-farm data separation.
- Migrate persistence to a production-grade database (e.g. PostgreSQL)
  for multi-user deployment.
- Add satellite/drone imagery support for field-scale (not just
  single-leaf) health monitoring.
- Introduce localization/multi-language support for broader farmer
  accessibility.

## 21. Conclusion

This project demonstrates a complete, working application of AI in
agriculture: two independently trained and evaluated machine learning
models (a MobileNetV2 CNN for image-based disease detection and a
compared, automatically-selected Decision Tree/Random Forest for
environmental risk), unified by a custom explainable scoring engine and a
deterministic rule-based recommendation system, all backed by persistent
storage and presented through a cohesive, professional Streamlit
interface. Beyond the modeling itself, the project also demonstrates good
software-engineering practice for an applied-AI system — centralized
configuration, input validation, layered error handling that never leaks
internals to the user, and a codebase kept free of dead code and
duplication — making it stable and ready for live demonstration.

## License

MIT