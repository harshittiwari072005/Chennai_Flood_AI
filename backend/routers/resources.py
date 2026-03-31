from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class ResourceInput(BaseModel):
    flood_probability: float
    total_population:  int = 500000

@router.post("/estimate-resources")
def estimate_resources(data: ResourceInput):
    prob      = data.flood_probability / 100
    displaced = int(data.total_population * prob)
    food      = displaced * 2 // 1000
    boats     = displaced // 3000
    med_kits  = int(displaced * 0.025)
    rescue    = max(1, displaced // 5000)

    return {
        "displaced_people": displaced,
        "food_tons_per_day": food,
        "boats_needed":      boats,
        "medical_kits":      med_kits,
        "rescue_teams":      rescue,
        "severity_level": (
            "CRITICAL" if prob >= 0.65 else
            "HIGH"     if prob >= 0.45 else
            "MODERATE" if prob >= 0.25 else
            "LOW"
        )
    }