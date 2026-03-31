import pandas as pd
import numpy as np
import os

print("="*55)
print("STEP 2: Building River Discharge Data")
print("Source: Rational Method on Real ERA5 Rainfall")
print("="*55)

# ── Paths ────────────────────────────────────────────────
RAINFALL_PATH  = r"C:\Users\harsh\Desktop\chennai_flood_ai\data\raw\rainfall_hourly.csv"
OUTPUT_PATH    = r"C:\Users\harsh\Desktop\chennai_flood_ai\data\raw\river_discharge_daily.csv"

# ── River configs (CWC published values) ─────────────────
RIVERS = {
    "Adyar": {
        "stations":   ["Velachery", "Adyar", "Nungambakkam"],
        "C":          0.65,    # runoff coefficient
        "A":          860,     # catchment area km²
        "bf":         5.0,     # baseflow m³/s
        "k":          0.72,    # recession constant
        "danger_m":   2.70,    # CWC danger level (m)
        "warning_m":  2.10,    # CWC warning level (m)
        "a_rc":       0.08,    # rating curve coefficient
        "b_rc":       0.45,    # rating curve exponent
    },
    "Cooum": {
        "stations":   ["Porur", "Nungambakkam", "Kolathur"],
        "C":          0.60,
        "A":          285,
        "bf":         2.0,
        "k":          0.70,
        "danger_m":   2.40,
        "warning_m":  1.80,
        "a_rc":       0.06,
        "b_rc":       0.48,
    },
    "Kosasthalaiyar": {
        "stations":   ["Kolathur", "Sholinganallur"],
        "C":          0.55,
        "A":          1714,
        "bf":         8.0,
        "k":          0.74,
        "danger_m":   3.10,
        "warning_m":  2.50,
        "a_rc":       0.06,
        "b_rc":       0.48,
    },
}

# ── Load real rainfall ────────────────────────────────────
print("\nLoading real rainfall data...")
rain = pd.read_csv(RAINFALL_PATH, parse_dates=["datetime"])
rain["date"] = rain["datetime"].dt.date

daily = (
    rain.groupby(["station", "date"])["precipitation"]
    .sum()
    .reset_index()
)
daily["date"] = pd.to_datetime(daily["date"])
print(f"  Loaded {len(rain):,} hourly rows")
print(f"  Aggregated to {len(daily):,} daily rows")
print(f"  Stations: {daily['station'].unique()}")

# ── Compute discharge for each river ─────────────────────
print("\nComputing river discharge...")
all_rows = []

for river, cfg in RIVERS.items():
    print(f"\n  Processing {river} river...")

    # Average rainfall across the river's catchment stations
    catchment_rain = (
        daily[daily["station"].isin(cfg["stations"])]
        .groupby("date")["precipitation"]
        .mean()
        .reset_index()
        .sort_values("date")
        .reset_index(drop=True)
    )

    rain_vals = catchment_rain["precipitation"].values
    n         = len(rain_vals)
    Q         = np.zeros(n)
    WL        = np.zeros(n)

    # Rational method + recession curve
    for i, rain_mm in enumerate(rain_vals):
        Q_in = cfg["C"] * rain_mm * cfg["A"] / (3.6 * 24)
        if i == 0:
            Q[i] = cfg["bf"] + Q_in
        else:
            Q[i] = cfg["k"] * Q[i-1] + (1 - cfg["k"]) * cfg["bf"] + Q_in
        Q[i]  = float(np.clip(Q[i], cfg["bf"], 8000.0))
        WL[i] = cfg["a_rc"] * (Q[i] ** cfg["b_rc"])

    # Build dataframe
    river_df = catchment_rain.copy()
    river_df["river"]            = river
    river_df["discharge_m3s"]    = np.round(Q,  2)
    river_df["water_level_m"]    = np.round(WL, 3)
    river_df["danger_level_m"]   = cfg["danger_m"]
    river_df["warning_level_m"]  = cfg["warning_m"]
    river_df["pct_of_danger"]    = np.round(WL / cfg["danger_m"] * 100, 1)
    river_df["flood_alert"]      = (WL >= cfg["warning_m"]).astype(int)
    river_df["data_source"]      = "rational_method_on_real_era5_rainfall"
    river_df.drop(columns=["precipitation"], inplace=True)

    all_rows.append(river_df)

    # Stats
    flood_days  = river_df["flood_alert"].sum()
    max_q       = river_df["discharge_m3s"].max()
    max_wl      = river_df["water_level_m"].max()
    max_date    = river_df.loc[river_df["discharge_m3s"].idxmax(), "date"]
    print(f"    Rows        : {len(river_df):,}")
    print(f"    Max discharge: {max_q:.1f} m³/s on {max_date.date()}")
    print(f"    Max level    : {max_wl:.2f}m (danger={cfg['danger_m']}m)")
    print(f"    Flood alerts : {flood_days} days")

# ── Combine & save ────────────────────────────────────────
print("\nCombining rivers...")
final = pd.concat(all_rows, ignore_index=True)
final.sort_values(["river", "date"], inplace=True)
final.reset_index(drop=True, inplace=True)

final.to_csv(OUTPUT_PATH, index=False)

print("\n" + "="*55)
print("DONE!")
print(f"Total rows : {len(final):,}")
print(f"Rivers     : {final['river'].nunique()} (Adyar, Cooum, Kosasthalaiyar)")
print(f"Date range : {final['date'].min().date()} to {final['date'].max().date()}")
print(f"Saved to   : {OUTPUT_PATH}")
print("="*55)

# ── Verify 2015 flood ─────────────────────────────────────
print("\n2015 FLOOD VERIFICATION (Dec 1-2, 2015):")
flood_2015 = final[
    (final["date"] >= "2015-11-01") &
    (final["date"] <= "2015-12-10") &
    (final["river"] == "Adyar") &
    (final["flood_alert"] == 1)
]
if len(flood_2015) > 0:
    print(flood_2015[["date","river","discharge_m3s","water_level_m","pct_of_danger"]].to_string())
else:
    print("  No flood alerts for Adyar in Nov-Dec 2015")
    print("  (Check Nov 2015 peak manually below)")
    peak = final[
        (final["date"] >= "2015-11-01") &
        (final["date"] <= "2015-12-10") &
        (final["river"] == "Adyar")
    ].nlargest(5, "discharge_m3s")
    print(peak[["date","discharge_m3s","water_level_m","pct_of_danger"]].to_string())