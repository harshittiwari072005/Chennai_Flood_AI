import pandas as pd
import numpy as np
import os

print("="*55)
print("STEP 4: Feature Engineering + Final Dataset Build")
print("="*55)

RAW_DIR    = r"C:\Users\harsh\Desktop\chennai_flood_ai\data\raw"
PROC_DIR   = r"C:\Users\harsh\Desktop\chennai_flood_ai\data\processed"
OUTPUT_DIR = r"C:\Users\harsh\Desktop\chennai_flood_ai\output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Load all data ─────────────────────────────────────────
print("\nLoading all data files...")
rain      = pd.read_csv(f"{RAW_DIR}\\rainfall_hourly.csv",        parse_dates=["datetime"])
discharge = pd.read_csv(f"{RAW_DIR}\\river_discharge_daily.csv",  parse_dates=["date"])
zones     = pd.read_csv(f"{PROC_DIR}\\zone_static_features.csv")
events    = pd.read_csv(f"{PROC_DIR}\\flood_events_historical.csv", parse_dates=["date_start","date_end"])
print(f"  Rainfall    : {len(rain):,} rows")
print(f"  Discharge   : {len(discharge):,} rows")
print(f"  Zones       : {len(zones)}")
print(f"  Flood events: {len(events)}")

# ── Step 4a: Daily rainfall features per station ──────────
print("\nBuilding daily rainfall features...")
rain["date"] = rain["datetime"].dt.date
daily = rain.groupby(["station","date"]).agg(
    rainfall_mm_daily      =("precipitation","sum"),
    max_hourly_intensity   =("precipitation","max"),
    rain_hours_nonzero     =("precipitation",lambda x:(x>0.1).sum()),
    temperature_mean_c     =("temperature_2m","mean"),
    humidity_mean_pct      =("relative_humidity_2m","mean"),
    wind_speed_mean_kmh    =("wind_speed_10m","mean"),
    pressure_mean_hpa      =("surface_pressure","mean"),
    soil_moisture_0_7cm    =("soil_moisture_0_to_7cm","mean"),
    soil_moisture_7_28cm   =("soil_moisture_7_to_28cm","mean"),
).reset_index()
daily["date"] = pd.to_datetime(daily["date"])
daily.sort_values(["station","date"], inplace=True)

# Rolling windows
for w, s in [(3,"3d"),(7,"7d"),(14,"14d"),(30,"30d")]:
    daily[f"rainfall_rolling_{s}"] = (
        daily.groupby("station")["rainfall_mm_daily"]
             .transform(lambda x: x.rolling(w, min_periods=1).sum()))

# Antecedent moisture index
daily["antecedent_rain_index"] = (
    daily.groupby("station")["rainfall_mm_daily"]
         .transform(lambda x: x.ewm(span=5, min_periods=1).mean()))

# Lag features
daily["rainfall_lag_1d"] = (
    daily.groupby("station")["rainfall_mm_daily"]
         .transform(lambda x: x.shift(1)))
daily["rainfall_lag_2d"] = (
    daily.groupby("station")["rainfall_mm_daily"]
         .transform(lambda x: x.shift(2)))

print(f"  Daily station rows: {len(daily):,}")

# ── Step 4b: Wide discharge ───────────────────────────────
print("Building discharge features...")
disc_wide_parts = []
for river in discharge["river"].unique():
    r = discharge[discharge["river"]==river].copy().sort_values("date")
    p = river.lower()[:5]
    for w, s in [(3,"r3d"),(7,"r7d")]:
        r[f"{p}_discharge_{s}"] = r["discharge_m3s"].rolling(w, min_periods=1).mean()
    r[f"{p}_discharge_delta"] = r["discharge_m3s"].diff()
    r[f"{p}_wl_delta"]        = r["water_level_m"].diff()
    r = r.rename(columns={
        "discharge_m3s":  f"{p}_discharge_m3s",
        "water_level_m":  f"{p}_water_level_m",
        "pct_of_danger":  f"{p}_pct_of_danger",
        "flood_alert":    f"{p}_flood_alert",
    })
    keep = ["date"] + [c for c in r.columns if c.startswith(p)]
    disc_wide_parts.append(r[keep])

disc_wide = disc_wide_parts[0]
for part in disc_wide_parts[1:]:
    disc_wide = disc_wide.merge(part, on="date", how="outer")
print(f"  Discharge wide cols: {len(disc_wide.columns)}")

# ── Zone to station mapping ───────────────────────────────
ZONE_STATIONS = {
    "Z01":["Velachery","Adyar","Nungambakkam"],
    "Z02":["Adyar","Nungambakkam"],
    "Z03":["Tambaram","Meenambakkam"],
    "Z04":["Tambaram","Meenambakkam"],
    "Z05":["Nungambakkam","Adyar"],
    "Z06":["Porur","Nungambakkam"],
    "Z07":["Kolathur","Nungambakkam"],
    "Z08":["Kolathur","Nungambakkam"],
    "Z09":["Nungambakkam","Kolathur"],
    "Z10":["Nungambakkam","Adyar"],
    "Z11":["Porur","Kolathur"],
    "Z12":["Sholinganallur","Velachery"],
    "Z13":["Nungambakkam"],
    "Z14":["Nungambakkam","Adyar"],
    "Z15":["Nungambakkam"],
}

STATIC_COLS = [
    "zone_id","zone_name","lat","lon","area_km2","population_2023",
    "pop_density_per_km2","elevation_mean_m","elevation_min_m",
    "dist_adyar_river_km","dist_nearest_lake_km","impervious_surface_pct",
    "runoff_coeff_C","drainage_quality_score","soil_type_code",
    "soil_type","drainage_quality","is_coastal","is_marsh_adjacent",
    "static_risk_score","static_risk_class","historical_flood_events",
]

# ── Step 4c: Build per-zone dataframes ────────────────────
print("Merging zones with rainfall + discharge...")
all_frames = []

for _, zone in zones.iterrows():
    zid   = zone["zone_id"]
    stns  = ZONE_STATIONS.get(zid, ["Nungambakkam"])

    # Average rainfall across zone stations
    zrain = (daily[daily["station"].isin(stns)]
             .groupby("date").mean(numeric_only=True)
             .reset_index())
    zrain["date"] = pd.to_datetime(zrain["date"])

    # Merge discharge
    df = zrain.merge(disc_wide, on="date", how="left")

    # Temporal features
    dt = pd.to_datetime(df["date"])
    df["year"]         = dt.dt.year
    df["month"]        = dt.dt.month
    df["day_of_year"]  = dt.dt.dayofyear
    df["month_sin"]    = np.sin(2*np.pi*df["month"]/12)
    df["month_cos"]    = np.cos(2*np.pi*df["month"]/12)
    df["doy_sin"]      = np.sin(2*np.pi*df["day_of_year"]/365)
    df["doy_cos"]      = np.cos(2*np.pi*df["day_of_year"]/365)
    df["is_northeast_monsoon"] = dt.dt.month.isin([10,11,12]).astype(int)
    df["is_southwest_monsoon"] = dt.dt.month.isin([6,7,8,9]).astype(int)
    df["is_pre_monsoon"]       = dt.dt.month.isin([3,4,5]).astype(int)
    df["months_from_ne_peak"]  = ((df["month"]-11).abs()).clip(0,6)

    # Static zone features
    for col in STATIC_COLS:
        df[col] = zone[col]

    # Flood labels
    df["flood_label"] = 0
    for _, ev in events.iterrows():
        zones_hit = [z.strip() for z in str(ev["zones_affected"]).split(",")]
        if zid in zones_hit:
            mask = (df["date"]>=ev["date_start"]) & (df["date"]<=ev["date_end"])
            df.loc[mask,"flood_label"] = int(ev["severity_label"])

    label_map = {0:"no_flood",1:"minor",2:"moderate",3:"severe"}
    df["flood_label_name"] = df["flood_label"].map(label_map)

    all_frames.append(df)

print(f"  Built {len(all_frames)} zone dataframes")

# ── Step 4d: Combine + Clean ──────────────────────────────
print("Combining and cleaning...")
final = pd.concat(all_frames, ignore_index=True)
final["date"] = pd.to_datetime(final["date"])
final = final[(final["date"].dt.year>=2000) & (final["date"].dt.year<=2023)]
final.sort_values(["zone_id","date"], inplace=True)
final.reset_index(drop=True, inplace=True)

# Clean negatives
final["rainfall_mm_daily"] = final["rainfall_mm_daily"].clip(0, 600)

# Fill missing discharge with median
for col in [c for c in final.columns if "discharge" in c or "water_level" in c or "delta" in c]:
    if final[col].isna().sum() > 0:
        final[col] = final.groupby("zone_id")[col].transform(lambda x: x.fillna(x.median()))

# Fill met cols
for col in ["temperature_mean_c","humidity_mean_pct","wind_speed_mean_kmh",
            "pressure_mean_hpa","soil_moisture_0_7cm","soil_moisture_7_28cm"]:
    if col in final.columns:
        final[col] = final.groupby("zone_id")[col].transform(lambda x: x.ffill().bfill())

# Fill any remaining
final.fillna(final.median(numeric_only=True), inplace=True)

# Round floats
float_cols = final.select_dtypes(include=[float]).columns
final[float_cols] = final[float_cols].round(4)

# ── Validation ────────────────────────────────────────────
print("\nValidating...")
assert final["flood_label"].between(0,3).all(),     "flood_label out of range!"
assert not final.duplicated(["zone_id","date"]).any(), "Duplicate rows found!"
assert final.isnull().sum().sum() == 0,              "NaN values remain!"
assert final["zone_id"].nunique() == 15,             "Missing zones!"
print("  All checks passed!")

# ── Save ──────────────────────────────────────────────────
out_csv    = f"{OUTPUT_DIR}\\chennai_flood_FINAL.csv"
out_sample = f"{OUTPUT_DIR}\\sample_50rows.csv"
final.to_csv(out_csv, index=False)
final.head(50).to_csv(out_sample, index=False)

# ── Summary ───────────────────────────────────────────────
print("\n" + "="*55)
print("FINAL DATASET SUMMARY")
print("="*55)
print(f"Total rows    : {len(final):,}")
print(f"Total columns : {len(final.columns)}")
print(f"Zones         : {final['zone_id'].nunique()}")
print(f"Date range    : {final['date'].min().date()} to {final['date'].max().date()}")
print(f"Missing values: {final.isnull().sum().sum()}")
print(f"\nFlood label distribution:")
vc = final["flood_label_name"].value_counts()
for lbl, cnt in vc.items():
    bar = "█" * int(cnt/len(final)*40)
    print(f"  {lbl:<12} {cnt:>7,} ({cnt/len(final)*100:.2f}%)  {bar}")
print(f"\nRisk classes:")
rc = final.groupby("static_risk_class")["zone_id"].nunique()
for cls, n in rc.items():
    print(f"  {cls:<10} {n} zones")
print(f"\nSaved to: {out_csv}")
print(f"Sample  : {out_sample}")
print("="*55)
print("\nDATASET READY FOR ML TRAINING!")