# Chennai Flood Risk AI

AI-powered flood prediction and emergency resource allocation for Chennai 
using LSTM, Random Forest, XGBoost, and LightGBM ensemble.

## Setup

### Backend
```bash
cd backend
pip install fastapi uvicorn tensorflow lightgbm xgboost scikit-learn joblib
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm start
```

## Model Files
Unzip `output/chennai_flood_models.zip` into `backend/saved_models/`

## Tech Stack
- **ML Models:** LSTM · Random Forest · XGBoost · LightGBM
- **Backend:** FastAPI (Python)
- **Frontend:** React.js
- **Data Sources:** IMD · CWC Chennai · NDMA · APDM
```

