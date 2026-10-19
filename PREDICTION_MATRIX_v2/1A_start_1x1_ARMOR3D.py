# ===============================================================
# ARMOR 3D Southern Ocean MLD, Temperature, and Salinity
# Monthly Regridding to 1° Grid (aligned to 0.5° centers)
# Louise Delaigue – 2025
# ===============================================================

"""
Produces a monthly 1×1° grid from ARMOR3D MY (0.125°) containing:
- MLD (mlotst)
- Temperature (to)
- Salinity (so)

Fix included:
→ After spatial coarsening, temporal duplicates are removed
  by aggregating monthly by (lat_center, lon_center).
"""

import os
import xarray as xr
import pandas as pd
import numpy as np

# === Paths ===
rep_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/ARMOR-3D/cmems_obs-mob_glo_phy_my_0.125deg_P1M-m_so-to-mlotst_179.94W-179.94E_82.19S-20.06S_0.00m_2003-01-01-2024-12-01.nc"
)

final_csv_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "PREDICTION_MATRIX_v2/processing_steps/"
    "ARMOR3D_MLD_TS_SouthernOcean_2003_2024_monthly_1deg.csv"
)

# === Parameters ===
lat_min = -90
lat_max = -30
pixel_size = 1.0
grid_center_offset = 0.5

# ===============================================================
# Load dataset
# ===============================================================
print("[INFO] Loading ARMOR3D MY dataset...")
ds = xr.load_dataset(rep_path)

keep_vars = [v for v in ["mlotst", "to", "so"] if v in ds.data_vars]
ds = ds[keep_vars]

# Subset latitude range
ds = ds.sortby("latitude")
ds = ds.sel(latitude=slice(lat_min, lat_max))

# ===============================================================
# Coarsening factors
# ===============================================================
lat_step = float(ds.latitude[1] - ds.latitude[0])
lon_step = float(ds.longitude[1] - ds.longitude[0])

factor_lat = int(round(pixel_size / abs(lat_step)))
factor_lon = int(round(pixel_size / abs(lon_step)))

print("[INFO] Coarsening to 1° grid...")
print(f"[DEBUG] Native lat_step: {lat_step}, lon_step: {lon_step}")
print(f"[DEBUG] Coarsening factors -> latitude: {factor_lat}, longitude: {factor_lon}")

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
    latitude=np.floor(ds_1deg.latitude) + grid_center_offset,
    longitude=np.floor(ds_1deg.longitude) + grid_center_offset,
)

# ===============================================================
# Flatten to DataFrame
# ===============================================================
print("[INFO] Flattening to DataFrame...")
df = ds_1deg.to_dataframe().reset_index()

# Drop rows where all key variables are NaN
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

# ===============================================================
# Remove duplicates
# ===============================================================
print("[INFO] Aggregating duplicates...")
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

df = (
    df.groupby(["lat_center", "lon_center", "month_center"], as_index=False)[numeric_cols]
      .median()
)

print(f"[INFO] Final rows (monthly 1°): {len(df):,}")

# ===============================================================
# Save original monthly 1° dataset
# ===============================================================
print("[INFO] Saving monthly 1° dataset...")
os.makedirs(os.path.dirname(final_csv_path), exist_ok=True)
df.to_csv(final_csv_path, index=False)

print("[SUCCESS] Saved:")
print(f" - 1° time series: {final_csv_path}")
print("[DONE] ARMOR3D 1° monthly grid completed.")