# ===============================================================
# ARMOR 3D Southern Ocean MLD, Temperature, and Salinity
# Monthly Regridding to 1° Grid (aligned to 0.5° centers)
# Louise Delaigue – 2025
# ===============================================================

"""
Produces a monthly 1×1° grid from ARMOR3D REP + NRT (0.25°) containing:
- MLD (mlotst)
- Temperature (to)
- Salinity (so)

Fix included:
→ After spatial coarsening, temporal duplicates are removed
  by aggregating monthly by (lat_center, lon_center).

Also produces:
- A monthly climatology (mean over all years)
"""

import os
import xarray as xr
import pandas as pd
import numpy as np

# === Paths ===
rep_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/mld/dataset-armor-3d-rep-monthly_to-mlotst-so_179.88W-179.88E_82.12S-30.12S_0.00m_1993-01-01-2022-12-01.nc"
nrt_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/mld/dataset-armor-3d-nrt-monthly_to-mlotst-so_179.88W-179.88E_82.12S-30.12S_0.00m_2023-01-01-2024-12-01.nc"

combined_nc_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/baseline_matrix/ARMOR3D_MLD_TS_combined_1997_2024_1deg.nc"
final_csv_path    = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/PREDICTION_MATRIX/processing_steps/ARMOR3D_MLD_TS_SouthernOcean_1997_2024_monthly_1deg.csv"
clim_csv_path     = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/PREDICTION_MATRIX/processing_steps/ARMOR3D_MLD_TS_monthly_climatology_1deg.csv"

# === Parameters ===
lat_min = -90
lat_max = -30
pixel_size = 1.0
grid_center_offset = 0.5

# ===============================================================
# Load & Concatenate
# ===============================================================
print("[INFO] Loading REP dataset...")
rep = xr.load_dataset(rep_path)

print("[INFO] Loading NRT dataset...")
nrt = xr.load_dataset(nrt_path)

print("[INFO] Concatenating REP and NRT datasets...")
combined = xr.concat([rep, nrt], dim="time").sortby("time")

keep_vars = [v for v in ["mlotst", "to", "so"] if v in combined.data_vars]
ds = combined[keep_vars]

# Subset
ds = ds.sel(latitude=slice(lat_min, lat_max))

# ===============================================================
# Coarsening factors
# ===============================================================
lat_step = float(ds.latitude[1] - ds.latitude[0])
lon_step = float(ds.longitude[1] - ds.longitude[0])

factor_lat = int(round(pixel_size / abs(lat_step)))
factor_lon = int(round(pixel_size / abs(lon_step)))

print("[INFO] Coarsening to 1° grid...")

# ===============================================================
# Spatial coarsening
# ===============================================================
ds_1deg = ds.coarsen(
    latitude=factor_lat,
    longitude=factor_lon,
    boundary="trim"
).median(skipna=True)

# Align centers to ±0.5°
ds_1deg = ds_1deg.assign_coords(
    latitude=np.round(ds_1deg.latitude, 0) + np.sign(ds_1deg.latitude) * grid_center_offset,
    longitude=np.round(ds_1deg.longitude, 0) + np.sign(ds_1deg.longitude) * grid_center_offset,
)

# ===============================================================
# Flatten to DataFrame
# ===============================================================
print("[INFO] Flattening to DataFrame...")
df = ds_1deg.to_dataframe().reset_index()

df.dropna(subset=keep_vars, how="all", inplace=True)

df.rename(columns={
    "latitude": "lat_center",
    "longitude": "lon_center",
    "mlotst": "MLD",
    "to": "ARMOR3D_temperature",
    "so": "ARMOR3D_salinity",
}, inplace=True)

df["month_center"] = pd.to_datetime(df["time"])
df.drop(columns=["time"], inplace=True)

# Remove duplicates
print("[INFO] Aggregating duplicates...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

df = (
    df.groupby(["lat_center", "lon_center", "month_center"], as_index=False)[numeric_cols]
      .median()
)

print(f"[INFO] Final rows (monthly 1°): {len(df):,}")

# ===============================================================
# ADD: COMPUTE CLIMATOLOGY AND SAVE SEPARATELY
# ===============================================================
print("[INFO] Computing monthly climatology...")

df["month"] = df["month_center"].dt.month

clim = (
    df.groupby(["lat_center", "lon_center", "month"])[["MLD", "ARMOR3D_temperature", "ARMOR3D_salinity"]]
      .mean()
      .reset_index()
      .rename(columns={
          "MLD": "MLD_clim",
          "ARMOR3D_temperature": "ARMOR3D_temperature_clim",
          "ARMOR3D_salinity": "ARMOR3D_salinity_clim"
      })
)

print(f"[INFO] Saving climatology → {clim_csv_path}")
clim.to_csv(clim_csv_path, index=False)

# ===============================================================
# Save original monthly 1° dataset
# ===============================================================
print("[INFO] Saving monthly 1° dataset...")
ds_1deg.to_netcdf(combined_nc_path)
df.to_csv(final_csv_path, index=False)

print("[SUCCESS] Saved:")
print(f" - 1° time series: {final_csv_path}")
print(f" - 1° climatology: {clim_csv_path}")
print("[DONE] ARMOR3D 1° monthly grid completed.")
