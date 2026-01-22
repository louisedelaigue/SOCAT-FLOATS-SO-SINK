# ===============================================================
# SOCAT–MLD–SLA–RRS–PAR–ERA5 Wind (Full Dataset, 1° Grid)
# Louise Delaigue – 2025
# ===============================================================

"""
Full pipeline for merging ERA5 10 m wind fields (u10, v10, wind_speed)
onto the 1°×1° monthly SOCAT–MLD–SLA–RRS–PAR grid.

Pipeline consistency:
 - Coordinates aligned to 1° centers at ±0.5°
 - Monthly aggregation = median
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
socat_par_path = (
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/"
    "MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
    "SOCATv2025_SO_clean_MLD_SLA_RRS_PAR_monthly_1deg.csv"
)
wind_nc_path = (
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/wind/"
    "37f05a757edface13840bd98a5e1143b.nc"
)
output_path = (
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/"
    "MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
    "SOCATv2025_SO_clean_MLD_SLA_RRS_PAR_WIND_monthly_1deg.csv"
)

# === Parameters ===
lat_min, lat_max = -90, -30
target_res = 1.0
grid_center_offset = 0.5  # aligns with ±0.5°

# ===============================================================
# 1. Load SOCAT–MLD–SLA–RRS–PAR grid
# ===============================================================
print("Loading SOCAT–MLD–SLA–RRS–PAR dataset...")
df_all = pd.read_csv(socat_par_path)
df_all["month_center"] = pd.to_datetime(df_all["month_center"], errors="coerce")

# Keep only Southern Ocean
df_all = df_all[(df_all["lat_center"] >= lat_min) & (df_all["lat_center"] <= lat_max)]
print(f"Loaded {len(df_all):,} SOCAT grid cells")

# Align timestamps to month-end
df_all["month_center"] = df_all["month_center"].dt.to_period("M").dt.to_timestamp(how="end")
df_all["year"] = df_all["month_center"].dt.year
df_all["month"] = df_all["month_center"].dt.month

# ===============================================================
# 2. Load ERA5 dataset
# ===============================================================
print(f"\nLoading ERA5 wind dataset:")
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
# 3. Compute wind_speed and regrid to monthly 1° grid
# ===============================================================
print("Computing monthly wind speed & regridding to 1°×1° grid...")

# Monthly means
ds_monthly = ds.resample(time="1ME").mean(skipna=True)

# Convert to DataFrame
df_wind = ds_monthly[["u10", "v10"]].to_dataframe().reset_index()

# Wind speed
df_wind["wind_speed"] = np.sqrt(df_wind["u10"]**2 + df_wind["v10"]**2)
df_wind.dropna(subset=["wind_speed"], inplace=True)

# Snap coordinates to 1° grid centers at ±0.5°
df_wind["lat_center"] = np.floor(df_wind["latitude"]) + grid_center_offset
df_wind["lon_center"] = np.floor(df_wind["longitude"]) + grid_center_offset

# Monthly median per cell
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

print(f"Regridded ERA5 records: {len(df_wind):,}")

# ===============================================================
# 4. Merge with SOCAT grid
# ===============================================================
print("Merging ERA5 with SOCAT grid...")

# Ensure perfect alignments
for df in (df_all, df_wind):
    df["lat_center"] = df["lat_center"].round(3)
    df["lon_center"] = df["lon_center"].round(3)

df_merged = pd.merge(
    df_all,
    df_wind[["year", "month", "lat_center", "lon_center", "u10", "v10", "wind_speed"]],
    on=["year", "month", "lat_center", "lon_center"],
    how="left"
)

# ===============================================================
# 5. Diagnostics
# ===============================================================
matched = df_merged["wind_speed"].notna().sum()
total = len(df_merged)
pct = 100 * matched / total if total > 0 else 0

print(f"\nMatched {matched:,} of {total:,} SOCAT grid cells ({pct:.2f}%)")

# ===============================================================
# 6. Save output
# ===============================================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_merged.to_csv(output_path, index=False)

print("\nDone.")
print(f"Final ERA5-wind merged dataset saved to: {output_path}")
print(f"Final dataset size: {len(df_merged):,}")
