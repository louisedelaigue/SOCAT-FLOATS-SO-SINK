# ===============================================================
# SOCAT–MLD (ARMOR3D) Monthly Regridding on 1° Grid
# Louise Delaigue – 2025
# ===============================================================

"""
Regrids SOCATv2025 fCO₂, SST, and SSS onto the ARMOR3D MLD 1° × 1° monthly grid
for the Southern Ocean (1993–2024).

Steps:
- Load ARMOR3D MLD+salinity grid (monthly, 1° resolution)
- Snap SOCAT points to same 1° grid centers and monthly timestamps
- Aggregate SOCAT data (median fCO₂, temperature, salinity per cell×month)
- Concatenate with MLD fields
"""

import os
import pandas as pd
import numpy as np

# === Paths ===
mld_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/PREDICTION_MATRIX/processing_steps/ARMOR3D_MLD_TS_SouthernOcean_1997_2024_monthly_1deg.csv"
socat_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/SOCATv2025_SO_clean.csv"
output_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/SOCATv2025_SO_clean_MLD_monthly_regridded.csv"

# === Parameters ===
lat_min, lat_max = -90, -30
grid_res = 1.0

# === Load MLD grid ===
print("Loading MLD DataFrame...")
df_mld = pd.read_csv(mld_path)
df_mld["month_center"] = pd.to_datetime(df_mld["month_center"], errors="coerce")

# Drop any missing time or coordinates
df_mld.dropna(subset=["month_center", "lat_center", "lon_center"], inplace=True)

# Restrict to Southern Ocean
df_mld = df_mld[(df_mld["lat_center"] >= lat_min) & (df_mld["lat_center"] <= lat_max)]
print(f"MLD grid: {len(df_mld):,} cells, {df_mld['month_center'].min().date()} → {df_mld['month_center'].max().date()}")

# === Load SOCAT ===
print("Loading SOCAT dataset...")
df_socat = pd.read_csv(socat_path, low_memory=False)
df_socat["time"] = pd.to_datetime(df_socat[["year", "month", "day"]], errors="coerce")
df_socat["month_center"] = df_socat["time"].dt.to_period("M").dt.to_timestamp(how="start")

# Fix longitude convention
df_socat["longitude"] = np.where(df_socat["longitude"] > 180, df_socat["longitude"] - 360, df_socat["longitude"])

# Subset to Southern Ocean
df_socat = df_socat[(df_socat["latitude"] >= lat_min) & (df_socat["latitude"] <= lat_max)].copy()
print(f"SOCAT points in Southern Ocean: {len(df_socat):,}")

# === Regrid SOCAT to 1° grid ===
print("Regridding SOCAT data to 1° MLD grid...")

# Snap to the same 1° grid with centers at ±0.5°
df_socat["lat_center"] = np.floor(df_socat["latitude"]) + 0.5
df_socat["lon_center"] = np.floor(df_socat["longitude"]) + 0.5

# Aggregate by month × grid cell
df_socat_agg = (
    df_socat.groupby(["lat_center", "lon_center", "month_center"], as_index=False)
    .agg(
        fco2_rec_median=("fco2_rec", "median"),
        SOCAT_temperature=("sst_degC", "median"),
        SOCAT_salinity=("salinity", "median"),
        fco2_rec_count=("fco2_rec", "count")
    )
)

print(f"SOCAT aggregated: {len(df_socat_agg):,} grid cells")

# === Align both datasets ===
print("Aligning timestamps and coordinates...")

df_socat_agg["month_center"] = pd.to_datetime(df_socat_agg["month_center"])
df_mld["month_center"] = pd.to_datetime(df_mld["month_center"])

df_socat_agg["lat_center"] = df_socat_agg["lat_center"].round(3)
df_socat_agg["lon_center"] = df_socat_agg["lon_center"].round(3)
df_mld["lat_center"] = df_mld["lat_center"].round(3)
df_mld["lon_center"] = df_mld["lon_center"].round(3)

# === Concatenate (merge by exact grid + month)
print("Concatenating SOCAT with MLD grid...")
df_out = pd.merge(df_mld, df_socat_agg, on=["lat_center", "lon_center", "month_center"], how="left")

# === Drop rows with missing SOCAT data
print("Filtering rows with valid SOCAT data...")
df_out = df_out.dropna(subset=["fco2_rec_median", "SOCAT_temperature", "SOCAT_salinity"])

print(f"Final matched grid: {len(df_out):,} rows")

# === Save output ===
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_out.to_csv(output_path, index=False)

print(f"\nDone. Final regridded dataset saved to:\n{output_path}")
