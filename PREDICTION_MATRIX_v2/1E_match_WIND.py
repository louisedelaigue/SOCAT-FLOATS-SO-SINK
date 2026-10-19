# ===============================================================
# PM–MLD–SLA–RRS–PAR–ERA5 Wind + MSL (Full Dataset, 1° Grid)
# Louise Delaigue – 2025
# ===============================================================
"""
Full pipeline for merging ERA5 10 m wind fields (u10, v10, wind_speed)
and mean sea level pressure (msl) onto the 1°×1° monthly
PM–MLD–SLA–RRS–PAR grid.

Pipeline consistency:
 - Coordinates aligned to 1° centers at ±0.5°
 - ERA5 file is already monthly -> no temporal resampling
 - Timestamps aligned to month-end
 - Float rounding applied before merge
"""

import os
import warnings
import pandas as pd
import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# === Paths ===
PM_par_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-SO-SINK-main/"
    "PREDICTION_MATRIX_v2/processing_steps/"
    "ARMOR3D_TSMLD_SouthernOcean_2003_2024_monthly_1deg_SLA_RRS_PAR.csv"
)
wind_nc_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-SO-SINK-main/data/wind/"
    "e4dde318f3aedf9d1177508c7662a49f.nc"
)
output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-SO-SINK-main/"
    "PREDICTION_MATRIX_v2/processing_steps/"
    "ARMOR3D_TSMLD_SouthernOcean_2003_2024_monthly_1deg_SLA_RRS_PAR_WIND.csv"
)

# === Parameters ===
lat_min, lat_max = -90, -30
target_res = 1.0
grid_center_offset = 0.5  # aligns with ±0.5°
pa_per_atm = 101325.0

# ===============================================================
# 1. Load PM–MLD–SLA–RRS–PAR grid
# ===============================================================
print("Loading PM–MLD–SLA–RRS–PAR dataset...")
df_all = pd.read_csv(PM_par_path)
df_all["month_center"] = pd.to_datetime(df_all["month_center"], errors="coerce")

# Keep only Southern Ocean
df_all = df_all[(df_all["lat_center"] >= lat_min) & (df_all["lat_center"] <= lat_max)]
print(f"Loaded {len(df_all):,} PM grid cells")

# Align timestamps to month-end
df_all["month_center"] = df_all["month_center"].dt.to_period("M").dt.to_timestamp(how="end")
df_all["year"] = df_all["month_center"].dt.year
df_all["month"] = df_all["month_center"].dt.month

# ===============================================================
# 2. Load ERA5 dataset
# ===============================================================
print(f"\nLoading ERA5 wind + MSL dataset:")
print(wind_nc_path)
ds = xr.open_dataset(wind_nc_path)

# Fix coordinate names
if "valid_time" in ds.coords and "time" not in ds.coords:
    ds = ds.rename({"valid_time": "time"})
if "lat" in ds.coords:
    ds = ds.rename({"lat": "latitude"})
if "lon" in ds.coords:
    ds = ds.rename({"lon": "longitude"})

# Fix longitude to -180..180
if ds.longitude.max() > 180:
    ds = ds.assign_coords(longitude=((ds.longitude + 180) % 360) - 180)

# ===============================================================
# 3. Compute wind_speed + msl and regrid to monthly 1° grid
#    (no temporal resampling — ERA5 file is already monthly)
# ===============================================================
print("Computing wind speed & regridding wind + MSL to 1°×1° grid...")

# Convert directly to DataFrame — one timestamp per month already
df_wind = ds[["u10", "v10", "msl"]].to_dataframe().reset_index()

# Wind speed
df_wind["wind_speed"] = np.sqrt(df_wind["u10"]**2 + df_wind["v10"]**2)
df_wind.dropna(subset=["wind_speed"], inplace=True)

# Snap coordinates to 1° grid centers at ±0.5°
df_wind["lat_center"] = np.floor(df_wind["latitude"]) + grid_center_offset
df_wind["lon_center"] = np.floor(df_wind["longitude"]) + grid_center_offset

# Spatial regrid only (native ERA5 resolution -> 1° grid), median per cell.
# "time" stays a groupby key so distinct monthly timestamps are preserved.
df_wind = (
    df_wind.groupby(["time", "lat_center", "lon_center"], as_index=False)
    .median(numeric_only=True)
    .rename(columns={"time": "month_center"})
)
df_wind["month_center"] = (
    pd.to_datetime(df_wind["month_center"]).dt.to_period("M").dt.to_timestamp(how="end")
)
df_wind["year"] = df_wind["month_center"].dt.year
df_wind["month"] = df_wind["month_center"].dt.month

# MSL in atm, for convenience alongside the raw Pa value
df_wind["msl_atm"] = df_wind["msl"] / pa_per_atm

print(f"Regridded ERA5 records: {len(df_wind):,}")

# ===============================================================
# 4. Merge with PM grid
# ===============================================================
print("Merging ERA5 wind + MSL with PM grid...")

# Ensure perfect alignments
for df in (df_all, df_wind):
    df["lat_center"] = df["lat_center"].round(3)
    df["lon_center"] = df["lon_center"].round(3)

df_merged = pd.merge(
    df_all,
    df_wind[["year", "month", "lat_center", "lon_center", "u10", "v10", "wind_speed", "msl", "msl_atm"]],
    on=["year", "month", "lat_center", "lon_center"],
    how="left"
)

# ===============================================================
# 5. Diagnostics
# ===============================================================
matched = df_merged["wind_speed"].notna().sum()
total = len(df_merged)
pct = 100 * matched / total if total > 0 else 0
print(f"\nMatched {matched:,} of {total:,} PM grid cells ({pct:.2f}%)")

matched_msl = df_merged["msl"].notna().sum()
print(f"Matched {matched_msl:,} of {total:,} PM grid cells with MSL ({100*matched_msl/total:.2f}%)")

# ===============================================================
# 6. Save output
# ===============================================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_merged.to_csv(output_path, index=False)

print("\nDone.")
print(f"Final ERA5-wind+MSL merged dataset saved to: {output_path}")
print(f"Final dataset size: {len(df_merged):,}")