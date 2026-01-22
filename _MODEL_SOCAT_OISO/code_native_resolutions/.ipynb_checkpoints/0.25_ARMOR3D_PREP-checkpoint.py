# ARMOR 3D Southern Ocean MLD: Combine, Subset, Flatten, and Export (native 0.25°)
# Louise Delaigue - 2025

"""
Combines monthly mixed layer depth (MLD, from mlotst)
fields from Copernicus Marine ARMOR 3D REP and NRT products (0.25° x 0.25°)
for the Southern Ocean (-90° to -30°) from 1997 to 2024.

Workflow:
- Load REP and NRT NetCDF files
- Concatenate along time
- Keep only MLD (mlotst)
- Subset to latitudes south of -30°
- Flatten to pandas DataFrame
- Save to CSV (native 0.25° grid)
"""

import xarray as xr
import pandas as pd

# === Paths ===
rep_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/mld/dataset-armor-3d-rep-monthly_to-mlotst-so_179.88W-179.88E_82.12S-30.12S_0.00m_1993-01-01-2022-12-01.nc"
nrt_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/mld/dataset-armor-3d-nrt-monthly_to-mlotst-so_179.88W-179.88E_82.12S-30.12S_0.00m_2023-01-01-2024-12-01.nc"
combined_nc_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/baseline_matrix/ARMOR3D_MLD_combined_1997_2024.nc"
final_csv_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/PREDICTION_MATRIX/processing_steps/ARMOR3D_MLD_SouthernOcean_1997_2024_monthly_surface.csv"

# === Load datasets ===
print("[INFO] Loading REP dataset...")
rep = xr.load_dataset(rep_path)

print("[INFO] Loading NRT dataset...")
nrt = xr.load_dataset(nrt_path)

# === Combine along time ===
print("[INFO] Concatenating REP and NRT datasets...")
combined = xr.concat([rep, nrt], dim="time").sortby("time")

# === Keep only MLD ===
if "mlotst" not in combined.data_vars:
    raise KeyError("Variable 'mlotst' (mixed layer depth) not found in dataset!")

ds_mld = combined[["mlotst"]]

# === Convert to pandas DataFrame ===
print("[INFO] Converting MLD data to DataFrame...")
df = ds_mld.to_dataframe().reset_index()

# === Clean up ===
df.dropna(subset=["mlotst"], inplace=True)

# === Save to CSV (native 0.25° grid) ===
df.to_csv(final_csv_path, index=False)
print(f"[SUCCESS] Flattened MLD data (0.25° native, lat ≤ -30°) saved to:\n{final_csv_path}")

print("[DONE] MLD-only processing completed successfully.")
