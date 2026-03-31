import openmeteo_requests
import requests_cache
from retry_requests import retry
import pandas as pd
import time
import os

print("Starting Chennai rainfall data fetch...")
print("This will take 8-10 minutes due to API rate limits. Do not close the window.\n")

STATIONS = {
    "Nungambakkam":   {"lat": 13.0827, "lon": 80.2707},
    "Meenambakkam":   {"lat": 12.9941, "lon": 80.1709},
    "Tambaram":       {"lat": 12.9249, "lon": 80.1000},
    "Velachery":      {"lat": 12.9788, "lon": 80.2209},
    "Adyar":          {"lat": 13.0012, "lon": 80.2565},
    "Kolathur":       {"lat": 13.1165, "lon": 80.2090},
    "Porur":          {"lat": 13.0358, "lon": 80.1568},
    "Sholinganallur": {"lat": 12.9010, "lon": 80.2279},
}

OUTPUT_PATH = r"C:\Users\harsh\Desktop\chennai_flood_ai\data\raw\rainfall_hourly.csv"

cache_session = requests_cache.CachedSession(".openmeteo_cache", expire_after=-1)
retry_session = retry(cache_session, retries=5, backoff_factor=0.3)
om = openmeteo_requests.Client(session=retry_session)

all_frames = []
total = len(STATIONS)

for i, (name, coords) in enumerate(STATIONS.items(), 1):
    print(f"[{i}/{total}] Fetching {name} ({coords['lat']}, {coords['lon']})...")

    params = {
        "latitude":   coords["lat"],
        "longitude":  coords["lon"],
        "start_date": "2000-01-01",
        "end_date":   "2023-12-31",
        "hourly": [
            "precipitation",
            "temperature_2m",
            "relative_humidity_2m",
            "wind_speed_10m",
            "surface_pressure",
            "soil_moisture_0_to_7cm",
            "soil_moisture_7_to_28cm",
        ],
        "timezone": "Asia/Kolkata",
    }

    try:
        responses = om.weather_api(
            "https://archive-api.open-meteo.com/v1/archive",
            params=params
        )
        response = responses[0]
        hourly   = response.Hourly()

        df = pd.DataFrame({
            "datetime": pd.date_range(
                start     = pd.to_datetime(hourly.Time(),    unit="s", utc=True).tz_convert("Asia/Kolkata"),
                end       = pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True).tz_convert("Asia/Kolkata"),
                freq      = pd.Timedelta(seconds=hourly.Interval()),
                inclusive = "left"
            ),
            "precipitation":           hourly.Variables(0).ValuesAsNumpy(),
            "temperature_2m":          hourly.Variables(1).ValuesAsNumpy(),
            "relative_humidity_2m":    hourly.Variables(2).ValuesAsNumpy(),
            "wind_speed_10m":          hourly.Variables(3).ValuesAsNumpy(),
            "surface_pressure":        hourly.Variables(4).ValuesAsNumpy(),
            "soil_moisture_0_to_7cm":  hourly.Variables(5).ValuesAsNumpy(),
            "soil_moisture_7_to_28cm": hourly.Variables(6).ValuesAsNumpy(),
        })

        df["datetime"] = df["datetime"].dt.tz_localize(None)
        df["station"]  = name
        df["lat"]      = coords["lat"]
        df["lon"]      = coords["lon"]

        all_frames.append(df)
        print(f"  Done — {len(df):,} rows fetched")

        if i < total:
            print(f"  Waiting 65 seconds before next station (API rate limit)...")
            time.sleep(65)
        print()

    except Exception as e:
        print(f"  ERROR fetching {name}: {e}")
        print(f"  Skipping and waiting 65 seconds...\n")
        time.sleep(65)

if not all_frames:
    print("ERROR: No data fetched at all. Check your internet connection.")
else:
    print("Combining all stations...")
    combined = pd.concat(all_frames, ignore_index=True)
    combined.sort_values(["station", "datetime"], inplace=True)
    combined.reset_index(drop=True, inplace=True)

    combined.to_csv(OUTPUT_PATH, index=False)

    print("\n" + "="*50)
    print("DONE!")
    print(f"Total rows : {len(combined):,}")
    print(f"Stations   : {combined['station'].nunique()}")
    print(f"Date range : {combined['datetime'].min()} to {combined['datetime'].max()}")
    print(f"Saved to   : {OUTPUT_PATH}")
    print("="*50)