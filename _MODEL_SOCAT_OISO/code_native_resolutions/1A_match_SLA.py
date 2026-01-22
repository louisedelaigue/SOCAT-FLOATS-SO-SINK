# === SOCAT–SLA Monthly Gridding (yearly loop) ===
# Louise Delaigue – 2025

"""
This script grids SOCATv2025 fCO₂ observations onto the Copernicus Marine Service
SLA grid (0.125°) for the Southern Ocean. For each year, it:
 - Loads the corresponding SLA subset (to manage memory efficiently)
 - Aggregates SOCAT fCO₂ observations to monthly × spatial grid cells
   using the median per SLA pixel
 - Computes monthly SLA means on the same grid
 - Merges both into a co-located monthly dataset
The outputs for all years are concatenated and saved as one CSV file.
"""

import os
import warnings
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm
import copernicusmarine

warnings.filterwarnings("ignore")

# === Paths ===
socat_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/SOCATv2025_SO_clean.csv"
output_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/SOCATv2025_SO_clean_SLA_monthly.csv"

# === Dataset configuration ===
dataset_id = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
variable = "sla"

# Southern Ocean domain
lat_min, lat_max = -90, -30
pixel_size = 0.125  # degrees (SLA native resolution)
sla_start_year = 1993
sla_end_year = 2024  # data available until ~2024-11-19

# === Load SOCAT ===
print("Loading clean SOCATv2025 Southern Ocean dataset...")
df_all = pd.read_csv(socat_path, low_memory=False)
df_all["time"] = pd.to_datetime(df_all[["year", "month", "day"]])
df_all["longitude"] = np.where(df_all["longitude"] > 180, df_all["longitude"] - 360, df_all["longitude"])

# Restrict to Southern Ocean
df_all = df_all[(df_all["latitude"] >= lat_min) & (df_all["latitude"] <= lat_max)].copy()

# Prepare list of valid years
years = sorted(df_all["year"].unique())
years = [y for y in years if sla_start_year <= y <= sla_end_year]
print(f"Processing {len(years)} years: {years[0]}–{years[-1]}")

# === Loop through years ===
all_years = []

for year in years:
    print(f"\n[INFO] Processing SOCAT–SLA gridding for {year}...")

    # Subset SOCAT for the year
    df_year = df_all[df_all["year"] == year].copy()
    if df_year.empty:
        print(f"[WARN] No SOCAT data for {year}. Skipping.")
        continue

    # Define SLA time window
    time_start = f"{year}-01-01"
    time_end = "2024-11-19" if year == sla_end_year else f"{year}-12-31"

    # --- Load SLA subset ---
    print(f"[INFO] Loading SLA data for {year}...")
    ds = copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        variables=[variable],
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        start_datetime=time_start,
        end_datetime=time_end,
    ).load()

    # Extract SLA grid
    lats = ds.latitude.values
    lons = ds.longitude.values

    # --- Assign SOCAT obs to SLA grid + monthly bins ---
    df_year["lat_center"] = (np.round((df_year["latitude"] - lats[0]) / pixel_size) * pixel_size + lats[0])
    df_year["lon_center"] = (np.round((df_year["longitude"] - lons[0]) / pixel_size) * pixel_size + lons[0])
    df_year["month_center"] = (
        df_year["time"].dt.to_period("M").dt.to_timestamp(how="end").dt.normalize()
    )

    # --- Aggregate fCO2 by grid + month (median + count) ---
    df_grouped = (
        df_year.groupby(["month_center", "lat_center", "lon_center"], as_index=False)
              .agg(fco2_rec_median=("fco2_rec", "median"),
                   fco2_rec_count=("fco2_rec", "count"))
    )

    if df_grouped.empty:
        print(f"[WARN] No SOCAT observations found in {year} after grouping.")
        continue

    # --- Compute SLA monthly means ---
    ds_monthly = ds[variable].resample(time="1M").mean(skipna=True)
    sla_monthly_df = (
        ds_monthly.to_dataframe().reset_index()
        .rename(columns={
            "time": "month_center",
            "latitude": "lat_center",
            "longitude": "lon_center",
            variable: "sla_mean"
        })
    )
    sla_monthly_df["month_center"] = sla_monthly_df["month_center"].dt.normalize()

    # --- Merge SOCAT and SLA monthly data ---
    df_merged = pd.merge(
        df_grouped,
        sla_monthly_df,
        on=["month_center", "lat_center", "lon_center"],
        how="inner"
    )

    print(f"[INFO] Year {year}: {len(df_merged)} co-located grid cells.")

    all_years.append(df_merged)

# === Combine all years and save ===
if all_years:
    df_out = pd.concat(all_years, ignore_index=True)
    df_out.to_csv(output_path, index=False)
    print(f"\nDone. Final gridded dataset: {len(df_out)} rows.")
    print(f"Saved to {output_path}")
else:
    print("\nNo data matched across all years.")
