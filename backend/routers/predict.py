from fastapi import APIRouter
from pydantic import BaseModel
import numpy as np
from model_loader import get_models

router = APIRouter()

class PredictInput(BaseModel):
    rainfall_today: float
    rainfall_7days: float
    adyar_level:    float
    month:          int

@router.post("/predict-risk")
def predict_risk(data: PredictInput):
    m = get_models()

    feature_row = {f: 0.0 for f in m["features"]}
    feature_row.update({
        "rainfall_mm_daily"   : data.rainfall_today,
        "rainfall_rolling_7d" : data.rainfall_7days,
        "rainfall_rolling_3d" : data.rainfall_today * 2.5,
        "rainfall_rolling_14d": data.rainfall_7days * 1.8,
        "rainfall_rolling_30d": data.rainfall_7days * 3.5,
        "rainfall_lag_1d"     : data.rainfall_today * 0.6,
        "rainfall_lag_2d"     : data.rainfall_today * 0.3,
        "adyar_water_level_m" : data.adyar_level,
        "adyar_pct_of_danger" : (data.adyar_level/2.70)*100,
        "adyar_flood_alert"   : int(data.adyar_level >= 2.10),
        "cooum_water_level_m" : data.adyar_level * 0.7,
        "cooum_pct_of_danger" : (data.adyar_level*0.7/2.40)*100,
        "kosas_water_level_m" : data.adyar_level * 0.9,
        "kosas_pct_of_danger" : (data.adyar_level*0.9/3.10)*100,
        "month"               : data.month,
        "month_sin"           : np.sin(2*np.pi*data.month/12),
        "month_cos"           : np.cos(2*np.pi*data.month/12),
        "is_northeast_monsoon": int(data.month in [10,11,12]),
        "is_southwest_monsoon": int(data.month in [6,7,8,9]),
        "is_pre_monsoon"      : int(data.month in [3,4,5]),
        "months_from_ne_peak" : abs(data.month - 11),
        "antecedent_rain_index": data.rainfall_7days/7*0.3,
        "soil_moisture_0_7cm" : min(0.35, data.rainfall_today/600),
        "humidity_mean_pct"   : 60 + (data.rainfall_today/10),
        "temperature_mean_c"  : 32 - (data.rainfall_today/50),
    })

    X    = np.array([[feature_row[f] for f in m["features"]]])
    X_sc = m["scaler"].transform(X)

    # LSTM prediction
    X_seq    = np.tile(X_sc,(30,1)).reshape(1,30,-1)
    lstm_raw = float(m["lstm"].predict(X_seq, verbose=0)[0][0])

    rain_norm  = min(data.rainfall_today/250.0, 1.0)
    river_norm = min(data.adyar_level/2.70,     1.0)
    week_norm  = min(data.rainfall_7days/800.0, 1.0)
    month_mult = 1.0 if data.month in [10,11,12] else 0.35
    physical   = (rain_norm*0.35 + river_norm*0.40 + week_norm*0.25)

    prob = float(np.clip(
        lstm_raw*0.5 + physical*month_mult*0.5, 0.0, 1.0))

    if data.rainfall_today < 20 and data.adyar_level < 1.0:
        prob = min(prob, 0.05)
    if data.rainfall_today > 200 and data.adyar_level > 2.40:
        prob = max(prob, 0.75)

    severity = (
        "CRITICAL" if prob >= 0.65 else
        "HIGH"     if prob >= 0.45 else
        "MODERATE" if prob >= 0.25 else
        "LOW"
    )

    return {
        "flood_probability": round(prob*100, 1),
        "severity":          severity,
        "lstm_raw":          round(lstm_raw*100, 1),
        "physical_score":    round(physical*100, 1),
        "adyar_danger_pct":  round((data.adyar_level/2.70)*100, 1),
        "is_monsoon_season": bool(data.month in [10,11,12]),
    }