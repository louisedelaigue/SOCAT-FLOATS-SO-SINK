# ===============================================================
# BGC-Argo – ARMOR3D MLD Monthly Regridding on 1° Grid
# Pipeline-matched to SOCAT monthly workflow (median aggregation)
# Louise Delaigue – 2025
# ===============================================================

import os
import pandas as pd
import numpy as np

# === Paths ===
mld_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "PREDICTION_MATRIX_v2/processing_steps/"
    "ARMOR3D_MLD_TS_SouthernOcean_1997_2024_monthly_1deg.csv"
)

floats_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_v2/processing_steps/independent_validation/pco2_independent_validation_data.csv"
)

output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_v2/processing_steps/independent_validation/"
    "pco2_independent_validation_data_TSMLD.csv"
)

# === Parameters ===
lat_min, lat_max = -90, -30
grid_center_offset = 0.5
lat_center_max = -30.5  # enforce "30S and south" on 1° grid centers

# ===============================================================
# 1. Load ARMOR3D MLD Grid
# ===============================================================
print("Loading MLD DataFrame...")
df_mld = pd.read_csv(mld_path)

df_mld["month_center"] = pd.to_datetime(df_mld["month_center"], errors="coerce")
df_mld.dropna(subset=["month_center", "lat_center", "lon_center"], inplace=True)

# Normalize to month START (pipeline convention)
df_mld["month_center"] = df_mld["month_center"].dt.to_period("M").dt.to_timestamp(how="start")

# Restrict to Southern Ocean on centers
df_mld = df_mld[(df_mld["lat_center"] >= lat_min) & (df_mld["lat_center"] <= lat_max)]
df_mld = df_mld[df_mld["lat_center"] <= lat_center_max]

print(f"MLD grid loaded: {len(df_mld):,} rows")

# ===============================================================
# 2. Load BGC-Argo Data
# ===============================================================
print("Loading BGC-Argo dataset...")
df_float = pd.read_csv(floats_path)

# Convert datetime to pandas datetime
df_float["datetime"] = pd.to_datetime(df_float["datetime"], errors="coerce")
df_float.dropna(subset=["datetime", "latitude", "longitude"], inplace=True)

# Fix longitude convention to [-180, 180] (important for consistent snapping/merge)
df_float["longitude"] = np.where(df_float["longitude"] > 180, df_float["longitude"] - 360, df_float["longitude"])

# Restrict to Southern Ocean on raw lat first (fast pre-filter)
df_float = df_float[
    (df_float["latitude"] >= lat_min) &
    (df_float["latitude"] <= lat_max)
].copy()

print(f"Float profiles in Southern Ocean (raw lat filter): {len(df_float):,}")

# ===============================================================
# 2B. Rename temperature and salinity to SOCAT-compatible names
# ===============================================================
print("Renaming float T/S to FLOATS_temperature / FLOATS_salinity...")
df_float = df_float.rename(columns={
    "temperature": "FLOATS_temperature",
    "salinity": "FLOATS_salinity"
})

# Ensure key numeric cols are numeric (avoid string weirdness)
for c in ["pco2", "pco2_error", "FLOATS_temperature", "FLOATS_salinity"]:
    if c in df_float.columns:
        df_float[c] = pd.to_numeric(df_float[c], errors="coerce")

# ===============================================================
# 3. Assign floats to SOCAT-style monthly grid
# ===============================================================
print("Assigning floats to monthly grid (start of month)...")

# Month-start timestamps (must match MLD/SOCAT pipeline)
df_float["month_center"] = df_float["datetime"].dt.to_period("M").dt.to_timestamp(how="start")

# Snap to 1° grid centers
df_float["lat_center"] = np.floor(df_float["latitude"]) + grid_center_offset
df_float["lon_center"] = np.floor(df_float["longitude"]) + grid_center_offset

# Enforce "30S and south" on the snapped grid centers
df_float = df_float[df_float["lat_center"] <= lat_center_max].copy()

print(f"Float profiles after snapped-center filter (<= -30.5): {len(df_float):,}")

# ===============================================================
# 4. Aggregate per 1° cell × month (MEDIAN)
# ===============================================================
print("Aggregating float data (median per cell × month)...")

df_float_agg = (
    df_float.groupby(["lat_center", "lon_center", "month_center"], as_index=False)
    .agg(
        FLOATS_pco2_median=("pco2", "median"),
        FLOATS_pco2_error_median=("pco2_error", "median"),
        FLOATS_temperature_median=("FLOATS_temperature", "median"),
        FLOATS_salinity_median=("FLOATS_salinity", "median"),
        FLOATS_count=("pco2", "count")
    )
)

print(f"Aggregated float cell-month rows: {len(df_float_agg):,}")

# ===============================================================
# 5. Merge with ARMOR3D grid
# ===============================================================
print("Merging float grid with MLD grid...")

# Round to avoid float mismatch
for d in (df_float_agg, df_mld):
    d["lat_center"] = d["lat_center"].round(3)
    d["lon_center"] = d["lon_center"].round(3)

df_out = pd.merge(
    df_mld,
    df_float_agg,
    on=["lat_center", "lon_center", "month_center"],
    how="left"
)

print(f"Merged dataset size (MLD grid rows): {len(df_out):,}")

# Keep rows where floats actually exist (your intention)
df_out = df_out.dropna(
    subset=[
        "FLOATS_pco2_median",
        "FLOATS_temperature_median",
        "FLOATS_salinity_median"
    ]
).copy()

print(f"Final dataset after filtering to float-present cells: {len(df_out):,} rows")

# ===============================================================
# 6. Save output
# ===============================================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_out.to_csv(output_path, index=False)

print(f"\n[DONE] Final monthly float grid saved to:\n{output_path}")

