# ===============================================================
# ARMOR3D Southern Ocean T/S/MLD Processing (Pipeline Version)
# Monthly, Surface Layer, 1° Regridding with MEDIAN aggregation
# Louise Delaigue – 2025
# ===============================================================

import os
import xarray as xr
import pandas as pd
import numpy as np

# === Paths ===
rep_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/mld/dataset-armor-3d-rep-monthly_to-mlotst-so_179.88W-179.88E_82.12S-30.12S_0.00m_1993-01-01-2022-12-01.nc"
nrt_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/mld/dataset-armor-3d-nrt-monthly_to-mlotst-so_179.88W-179.88E_82.12S-30.12S_0.00m_2023-01-01-2024-12-01.nc"

combined_nc_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/baseline_matrix/ARMOR3D_combined_1997_2024.nc"

final_csv_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/PREDICTION_MATRIX_v2/processing_steps/ARMOR3D_TSMLD_SouthernOcean_1997_2024_monthly_surface.csv"

final_1deg_csv_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/PREDICTION_MATRIX_v2/processing_steps/ARMOR3D_TSMLD_SouthernOcean_1997_2024_monthly_surface_1deg_median.csv"

# === Load datasets ===
print("[INFO] Loading REP dataset...")
rep = xr.load_dataset(rep_path)

print("[INFO] Loading NRT dataset...")
nrt = xr.load_dataset(nrt_path)

# === Combine datasets ===
print("[INFO] Concatenating REP and NRT...")
combined = xr.concat([rep, nrt], dim="time").sortby("time")

# === Rename variables ===
rename_dict = {}
if "to" in combined:
    rename_dict["to"] = "temperature"
if "so" in combined:
    rename_dict["so"] = "salinity"

if rename_dict:
    combined = combined.rename(rename_dict)

# === Save combined NC ===
combined.to_netcdf(combined_nc_path)
print(f"[OK] Combined dataset saved → {combined_nc_path}")

# ===============================================================
# 1. Extract surface T/S + MLD
# ===============================================================
print("[INFO] Extracting surface temperature, salinity, and MLD...")

ds_surface = combined.sel(depth=0, method="nearest")[["temperature", "salinity"]]

if "mlotst" in combined:
    ds_surface["MLD"] = combined["mlotst"]
    print("[INFO] MLD included.")
else:
    print("[WARNING] MLD ('mlotst') NOT FOUND!")

# ===============================================================
# 2. Convert to DataFrame
# ===============================================================
print("[INFO] Converting to DataFrame...")
df = ds_surface.to_dataframe().reset_index()

df = df.drop(columns="depth", errors="ignore")
df = df.dropna(subset=["temperature", "salinity"])

# Save 0.25° flattened file
#df.to_csv(final_csv_path, index=False)
#print(f"[OK] 0.25° surface T/S/MLD CSV saved → {final_csv_path}")

# ===============================================================
# 3. Regrid to 1° grid USING MEDIAN
# ===============================================================
print("[INFO] Regridding to 1° × 1° grid using MEDIAN aggregation...")

# SOCAT pipeline: centers are ±0.5°
df["lat_center"] = np.floor(df["latitude"]) + 0.5
df["lon_center"] = np.floor(df["longitude"]) + 0.5

# Monthly timestamp → month_center at end of the month
df["month_center"] = (
    pd.to_datetime(df["time"], errors="coerce")
    .dt.to_period("M")
    .dt.to_timestamp(how="end")
)

# Group & aggregate
agg_dict = {
    "temperature": "median",
    "salinity": "median"
}
if "MLD" in df.columns:
    agg_dict["MLD"] = "median"

df_1deg = (
    df.groupby(["lat_center", "lon_center", "month_center"], as_index=False)
      .agg(agg_dict)
)

print(f"[INFO] 1° grid created with {len(df_1deg):,} rows.")

# ===============================================================
# 4. Save final 1° product
# ===============================================================
df_1deg.to_csv(final_1deg_csv_path, index=False)
print(f"[SUCCESS] 1° monthly MEDIAN grid saved → {final_1deg_csv_path}")

print("[DONE] Processing complete.")
