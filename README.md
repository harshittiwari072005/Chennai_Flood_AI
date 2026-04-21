# 🌊 Chennai Flood Risk AI AND RESOURCE ALLOCAT

> AI-driven flood prediction and emergency resource allocation system for Chennai, Tamil Nadu — built using real ERA5 meteorological data, ensemble machine learning, and LSTM time-series modeling.

<img width="1919" height="849" alt="Screenshot 2026-03-31 162105" src="https://github.com/user-attachments/assets/73d3c193-90e7-4d28-b82d-b7bfe2547c5c" />


---

##  What This System Does

- **Predicts flood probability** for Chennai using today's rainfall, river levels, and seasonal patterns
- **Compares 4 ML models** — Random Forest, XGBoost, LightGBM, and LSTM
- **LSTM selected as final model** with 85.7% recall — catching 6 out of 7 real flood events in test set
- **Estimates emergency resources** — displaced people, food, boats, and medical kits
- **Ranks all 15 Chennai zones** by flood risk using composite scoring

---

##  Model Performance

| Model | Accuracy | Recall | F1 Score |
|---|---|---|---|
| Random Forest | 99.09% | 0.57 | 0.44 |
| XGBoost | 94.98% | 0.57 | 0.13 |
| LightGBM | 98.36% | 0.57 | 0.31 |
| **LSTM ✓ Selected** | **96.99%** | **0.857** | **0.27** |

> **Why LSTM?** It looks at the last 30 days as a sequence — capturing the trend of rising river levels and accumulating rainfall that precede a flood. Random Forest sees only today's snapshot. LSTM sees the buildup.

> **Why Recall over Accuracy?** Missing a real flood = lives lost. A false alarm = wasted resources. For disaster management, catching every real flood matters more than overall accuracy.

---

## 📊 Dataset

- **Source:** Open-Meteo ERA5 Reanalysis (real ECMWF data)
- **Coverage:** 8 Chennai stations · 2000–2023 · 24 years
- **Size:** 1,683,072 hourly rows → 131,490 daily ML-ready rows
- **Features:** 76 features including rainfall rolling windows, river discharge, soil moisture, cyclical season encoding
- **Flood Events:** 12 real documented events from TNSDMA reports (2005–2023)
- **Key event:** December 2015 Chennai flood — 490mm/24hr, 94 deaths, 500,000 displaced

---

## 🗺️ Chennai Zones Covered

| Zone | Risk Class |
|---|---|
| Velachery | 🔴 Critical (83.3) |
| Adyar Riverbank | 🔴 Critical (77.0) |
| Marina North, Mudichur, Saidapet | 🟠 High |
| Porur, Anna Nagar, Ambattur | 🟡 Medium |
| Nungambakkam | 🟢 Low |

---

## 🚀 Setup & Run

### Backend (FastAPI)
```bash
cd backend
pip install fastapi uvicorn tensorflow lightgbm xgboost scikit-learn joblib pandas numpy
uvicorn main:app --reload --port 8000
```
API docs available at: `http://127.0.0.1:8000/docs`

### Frontend (React)
```bash
cd frontend
npm install
npm start
```
Dashboard at: `http://localhost:3000`

### Model Files
Unzip `output/chennai_flood_models.zip` into `backend/saved_models/`

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/predict-risk` | Flood probability from 4 inputs |
| POST | `/api/estimate-resources` | Resource requirements |
| GET | `/api/zones` | All 15 Chennai zones with risk scores |

---

## 🧠 How Prediction Works
```
User inputs 4 values:
  → Rainfall today (mm)
  → 7-day accumulated rainfall (mm)
  → Adyar river level (metres)
  → Month

LSTM model processes 30-day sequence pattern
  → Outputs flood probability (0–100%)
  → Classifies: LOW / MODERATE / HIGH / CRITICAL

Resource allocation engine:
  → Displaced = Population × Flood Probability
  → Food = Displaced × 2kg/day (NDMA guideline)
  → Boats = Displaced ÷ 3000 (NDMA norm)
  → Medical kits = Displaced × 0.025 (WHO standard)
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Models | LSTM · Random Forest · XGBoost · LightGBM |
| Data | ERA5 Reanalysis · TNSDMA Reports · CWC Chennai |
| Backend | FastAPI · Uvicorn · Python |
| Frontend | React.js · Axios |
| Training | Google Colab · TensorFlow · Scikit-learn |
| Imbalance Handling | SMOTE (imbalanced-learn) |

---

## 📁 Project Structure
```
chennai_flood_ai/
├── backend/
│   ├── main.py              ← FastAPI entry point
│   ├── model_loader.py      ← Loads all 4 models
│   ├── routers/
│   │   ├── predict.py       ← /predict-risk endpoint
│   │   ├── resources.py     ← /estimate-resources endpoint
│   │   └── zones.py         ← /zones endpoint
│   └── models/              ← Saved model files
├── frontend/
│   └── src/App.js           ← React dashboard
├── notebooks/
│   └── Minor_project.ipynb  ← Full training notebook
├── scripts/                 ← Data collection scripts
└── output/                  ← Dataset + model zip
```

---

## 👨‍💻 Research Contribution

This project proposes a hybrid framework combining:
1. **Ensemble ML** (RF + XGBoost + LightGBM) for tabular flood prediction
2. **LSTM time-series modeling** for capturing 30-day rainfall trends
3. **Risk-weighted resource allocation** based on NDMA guidelines

Validated against real 2021 and 2023 Chennai flood events.

---

*Built for Minor Project — AI for Disaster Risk Management*

