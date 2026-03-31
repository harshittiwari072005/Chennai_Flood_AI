import zipfile, joblib, json, os
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from tensorflow.keras.models import load_model

MODELS_ZIP  = os.path.join(os.path.dirname(__file__), "models", "chennai_flood_models.zip")
EXTRACT_DIR = os.path.join(os.path.dirname(__file__), "models", "extracted")

# Global model variables
rf_model     = None
xgb_model    = None
lgb_model    = None
lstm_model   = None
scaler       = None
feature_cols = None
config       = None

def load_all_models():
    global rf_model, xgb_model, lgb_model
    global lstm_model, scaler, feature_cols, config

    # Unzip if not already extracted
    if not os.path.exists(EXTRACT_DIR):
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with zipfile.ZipFile(MODELS_ZIP, "r") as z:
            z.extractall(EXTRACT_DIR)
        print("Models unzipped!")

    # Load all models
    rf_model   = joblib.load(f"{EXTRACT_DIR}/random_forest.pkl")
    lstm_model = load_model(f"{EXTRACT_DIR}/lstm_model.keras")
    scaler     = joblib.load(f"{EXTRACT_DIR}/scaler.pkl")

    xgb_model = xgb.XGBClassifier()
    xgb_model.load_model(f"{EXTRACT_DIR}/xgboost_model.json")

    lgb_model = lgb.Booster(model_file=f"{EXTRACT_DIR}/lightgbm_model.txt")

    with open(f"{EXTRACT_DIR}/feature_cols.json") as f:
        feature_cols = json.load(f)

    with open(f"{EXTRACT_DIR}/model_config.json") as f:
        config = json.load(f)

    print("All models loaded successfully!")
    return True


def get_models():
    # Load models on first call
    if rf_model is None:
        load_all_models()
    return {
        "rf"      : rf_model,
        "xgb"     : xgb_model,
        "lgb"     : lgb_model,
        "lstm"    : lstm_model,
        "scaler"  : scaler,
        "features": feature_cols,
        "config"  : config,
    }