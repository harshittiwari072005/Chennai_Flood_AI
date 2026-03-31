import pandas as pd
import numpy as np
import os

print("="*55)
print("STEP 3: Building Zone Features + Flood Labels")
print("="*55)

PROC_DIR   = r"C:\Users\harsh\Desktop\chennai_flood_ai\data\processed"
RAW_DIR    = r"C:\Users\harsh\Desktop\chennai_flood_ai\data\raw"
OUTPUT_DIR = r"C:\Users\harsh\Desktop\chennai_flood_ai\output"
os.makedirs(PROC_DIR,   exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 15 Chennai Zones ─────────────────────────────────────
ZONES = [
    ("Z01","Velachery",        12.9788,80.2209,14.2, 320000,3.2, 0.5,0.8, 1.2,78,6,1,1,0,1),
    ("Z02","Adyar_Riverbank",  13.0012,80.2565,8.5,  185000,2.1, 0.2,0.3, 3.5,82,5,1,2,0,0),
    ("Z03","Tambaram",         12.9249,80.1000,22.4, 415000,4.8, 1.2,2.1, 0.8,65,4,2,2,0,0),
    ("Z04","Mudichur",         12.8938,80.0641,18.6, 145000,3.5, 0.8,1.5, 0.5,55,3,1,1,0,1),
    ("Z05","Saidapet",         13.0211,80.2244,6.8,  210000,3.8, 1.1,0.5, 2.8,88,4,2,2,0,0),
    ("Z06","Porur",            13.0358,80.1568,12.1, 190000,5.2, 2.0,2.8, 0.4,62,3,2,3,0,1),
    ("Z07","Kolathur",         13.1165,80.2090,10.3, 255000,6.1, 2.5,1.8, 0.6,71,3,2,2,0,1),
    ("Z08","Perambur",         13.1128,80.2455,9.2,  298000,5.5, 2.2,1.2, 1.4,85,3,2,2,0,0),
    ("Z09","Anna_Nagar",       13.0878,80.2101,11.4, 320000,7.2, 3.5,3.4, 2.1,90,2,2,3,0,0),
    ("Z10","T_Nagar",          13.0418,80.2341,7.6,  280000,6.8, 3.8,1.8, 2.9,92,2,2,2,0,0),
    ("Z11","Ambattur",         13.1142,80.1544,18.8, 310000,6.5, 3.0,3.5, 0.8,58,2,2,2,0,1),
    ("Z12","Sholinganallur",   12.9010,80.2279,21.2, 180000,4.1, 1.5,1.4, 0.6,48,2,1,2,0,1),
    ("Z13","Nungambakkam",     13.0580,80.2474,8.2,  195000,10.2,6.0,2.8, 4.2,88,1,3,3,0,0),
    ("Z14","Alwarpet",         13.0320,80.2568,5.4,  125000,8.5, 5.5,1.2, 3.8,91,1,3,3,0,0),
    ("Z15","Marina_North",     13.0827,80.2800,12.5, 220000,2.8, 0.8,0.2, 8.5,72,1,1,3,1,0),
]

COLS = [
    "zone_id","zone_name","lat","lon","area_km2","population_2023",
    "elevation_mean_m","elevation_min_m","dist_adyar_river_km",
    "dist_nearest_lake_km","impervious_surface_pct",
    "historical_flood_events","drainage_quality_score",
    "soil_type_code","is_coastal","is_marsh_adjacent",
]

# ── Historical flood events (TNSDMA reports) ─────────────
EVENTS = [
    ("E001","2005-10-26","2005-10-28",240,3.2,3,"Z01,Z02,Z03,Z05",85000,12),
    ("E002","2008-11-01","2008-11-03",180,2.6,2,"Z01,Z02,Z05,Z09",45000,4),
    ("E003","2010-11-07","2010-11-10",220,2.95,2,"Z01,Z02,Z03,Z04,Z05",62000,6),
    ("E004","2011-10-31","2011-11-02",195,2.75,2,"Z01,Z05,Z07,Z08",38000,3),
    ("E005","2015-11-08","2015-11-10",345,3.85,3,"Z01,Z02,Z03,Z04,Z05,Z06",190000,28),
    ("E006","2015-12-01","2015-12-02",490,4.70,3,"Z01,Z02,Z03,Z04,Z05,Z06,Z07,Z08,Z11,Z12",500000,94),
    ("E007","2016-11-05","2016-11-06",160,2.45,1,"Z01,Z02,Z05",22000,1),
    ("E008","2017-11-01","2017-11-03",175,2.58,2,"Z01,Z02,Z03,Z05,Z12",35000,2),
    ("E009","2019-11-11","2019-11-13",210,2.88,2,"Z01,Z02,Z05,Z06,Z07",58000,5),
    ("E010","2021-11-06","2021-11-09",280,3.40,3,"Z01,Z02,Z03,Z04,Z05,Z06,Z12",125000,11),
    ("E011","2021-11-11","2021-11-11",195,2.90,2,"Z01,Z02,Z05,Z07",42000,3),
    ("E012","2023-12-04","2023-12-05",260,3.10,3,"Z01,Z02,Z03,Z05,Z07,Z09",95000,8),
]

ECOLS = [
    "event_id","date_start","date_end","peak_rainfall_mm",
    "max_adyar_level_m","severity_label","zones_affected",
    "estimated_displaced","deaths",
]

# ── Build zone dataframe ──────────────────────────────────
print("\nBuilding zone static features...")
zones_df = pd.DataFrame(ZONES, columns=COLS)
zones_df["pop_density_per_km2"] = (zones_df["population_2023"] / zones_df["area_km2"]).round(0).astype(int)

soil_map  = {1:"clay", 2:"sandy_clay", 3:"sandy", 4:"rock"}
drain_map = {1:"poor", 2:"moderate",   3:"good"}
runoff_map= {1:0.68,   2:0.58,         3:0.45,   4:0.35}

zones_df["soil_type"]        = zones_df["soil_type_code"].map(soil_map)
zones_df["drainage_quality"] = zones_df["drainage_quality_score"].map(drain_map)
zones_df["runoff_coeff_C"]   = zones_df["soil_type_code"].map(runoff_map)

# Composite risk score
z = zones_df.copy()
def norm_inv(col): return 1-(z[col]-z[col].min())/(z[col].max()-z[col].min()+1e-9)
def norm(col):     return   (z[col]-z[col].min())/(z[col].max()-z[col].min()+1e-9)

zones_df["static_risk_score"] = (
    norm_inv("elevation_mean_m")         * 0.22 +
    norm_inv("drainage_quality_score")   * 0.18 +
    norm_inv("dist_adyar_river_km")      * 0.18 +
    norm_inv("dist_nearest_lake_km")     * 0.10 +
    norm("impervious_surface_pct")       * 0.12 +
    norm("population_2023")              * 0.08 +
    zones_df["is_marsh_adjacent"].astype(float) * 0.06 +
    zones_df["is_coastal"].astype(float)        * 0.03 +
    norm("historical_flood_events")      * 0.03
) * 100

zones_df["static_risk_score"] = zones_df["static_risk_score"].round(1)
zones_df["static_risk_class"] = zones_df["static_risk_score"].apply(
    lambda s: "critical" if s>=70 else "high" if s>=50 else "medium" if s>=30 else "low"
)

out1 = os.path.join(PROC_DIR, "zone_static_features.csv")
zones_df.to_csv(out1, index=False)
print(f"  Saved {len(zones_df)} zones → {out1}")
print(zones_df[["zone_id","zone_name","static_risk_score","static_risk_class"]].to_string(index=False))

# ── Build flood events dataframe ──────────────────────────
print("\nBuilding flood events table...")
events_df = pd.DataFrame(EVENTS, columns=ECOLS)
events_df["date_start"] = pd.to_datetime(events_df["date_start"])
events_df["date_end"]   = pd.to_datetime(events_df["date_end"])
events_df["duration_days"] = (events_df["date_end"] - events_df["date_start"]).dt.days + 1

out2 = os.path.join(PROC_DIR, "flood_events_historical.csv")
events_df.to_csv(out2, index=False)
print(f"  Saved {len(events_df)} events → {out2}")
print(f"  Severe events (label=3): {(events_df['severity_label']==3).sum()}")
print(f"  Total displaced (worst): {events_df['estimated_displaced'].max():,}")

print("\n" + "="*55)
print("STEP 3 DONE!")
print(f"  zone_static_features.csv    → {len(zones_df)} zones")
print(f"  flood_events_historical.csv → {len(events_df)} events")
print("="*55)
print("\nReady for Step 4 — Feature Engineering & Final Merge!")