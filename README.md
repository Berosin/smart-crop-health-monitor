# AI-Based Smart Crop Health Monitoring and Early Disease Detection System 🌿

A **software-only** AI system that monitors crop health and detects plant
diseases early by combining deep-learning image analysis with environmental
data. No hardware is required — there are no Arduinos, ESP32s, Raspberry Pis,
sensors, cameras, or IoT devices involved.

## Features

- 📷 Accepts crop leaf images uploaded by the user
- 🦠 Detects crop diseases using a deep-learning model (TensorFlow/Keras)
- 📊 Displays disease confidence and severity
- 🌡️ Accepts environmental conditions: temperature, humidity, soil moisture, rainfall
- 🧮 Calculates an overall crop health score
- 🌾 Provides actionable agricultural recommendations
- 💾 Stores every analysis in SQLite for future reference
- 📈 Surfaces statistics and trends via an interactive Plotly dashboard

## Tech Stack

| Layer              | Technology          |
|--------------------|---------------------|
| Frontend / UI      | Streamlit           |
| Programming        | Python              |
| AI / Deep Learning | TensorFlow / Keras  |
| Machine Learning   | Scikit-learn        |
| Image Processing   | OpenCV              |
| Data Processing    | Pandas + NumPy      |
| Database           | SQLite              |
| Visualization      | Plotly              |

> **Hardware:** None. The entire system runs in software.

## Project Structure

```text
Agriculture/
├── app.py                 # Streamlit entry point (home page)
├── config.py              # Central configuration & constants
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── database/              # SQLite database file(s)
├── models/                # Saved AI model weights & labels
├── data/                  # Uploaded images & sample data
├── src/                   # Core logic (detection, scoring, recommendations)
├── utils/                 # Shared helpers
└── pages/                 # Streamlit multipage modules
```

## Getting Started

### 1. Create a virtual environment (recommended)

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

The app opens in your browser at `http://localhost:8501`.

## Status

This repository currently contains the **project structure and a basic
Streamlit home page only**. AI model training, database logic, and advanced UI
modules are intentionally not implemented yet.

## License

MIT
