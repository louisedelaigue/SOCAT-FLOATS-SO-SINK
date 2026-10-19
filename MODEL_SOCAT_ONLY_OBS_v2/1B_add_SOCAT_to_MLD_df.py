# ===============================================================
# SOCAT–MLD (ARMOR3D) Monthly Regridding on 1° Grid
# Louise Delaigue – 2025
# ===============================================================

"""
Regrids SOCATv2025 fCO₂, SST, and SSS onto the ARMOR3D MLD 1° × 1° monthly grid
for the Southern Ocean (1993–2024).

Steps:
- Load ARMOR3D MLD+TS grid (monthly, 1° resolution)
- Snap SOCAT points to same 1° grid centers and monthly timestamps
- Aggregate SOCAT data (median fCO₂, temperature, salinity per cell×month)
- Merge SOCAT aggregates onto the MLD grid
"""

import os
import pandas as pd
import numpy as np

# === Paths ===
mld_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "PREDICTION_MATRIX_v2/processing_steps/"
    "ARMOR3D_MLD_TS_SouthernOcean_1997_2024_monthly_1deg.csv"
)

socat_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "MODEL_SOCAT_ONLY_OBS_v2/processing_steps/SOCATv2025_SO_clean_without_ind_expocodes.csv"
)

output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
    "SOCATv2025_SO_clean_MLD_monthly_regridded.csv"
)

# === Parameters ===
lat_min, lat_max = -90, -30
grid_center_offset = 0.5  # centers at ... -30.5, -31.5, ...

# ===============================================================
# 1) Load MLD grid (ARMOR3D-derived)
# ===============================================================
print("Loading MLD DataFrame...")
df_mld = pd.read_csv(mld_path)

df_mld["month_center"] = pd.to_datetime(df_mld["month_center"], errors="coerce")
df_mld.dropna(subset=["month_center", "lat_center", "lon_center"], inplace=True)

# Ensure month-start keys (your MLD already looks like YYYY-MM-01, but keep robust)
df_mld["month_center"] = (
    df_mld["month_center"].dt.to_period("M").dt.to_timestamp(how="start")
)

# Restrict to Southern Ocean domain (on centers)
df_mld = df_mld[(df_mld["lat_center"] >= lat_min) & (df_mld["lat_center"] <= lat_max)]
# Enforce "30S and south" in 1° grid sense: keep -30.5, -31.5, ...
df_mld = df_mld[df_mld["lat_center"] <= -30.5]

print(
    f"MLD grid: {len(df_mld):,} rows, "
    f"{df_mld['month_center'].min().date()} → {df_mld['month_center'].max().date()}"
)

# ===============================================================
# 2) Load SOCAT points
# ===============================================================
print("Loading SOCAT dataset...")
df_socat = pd.read_csv(socat_path, low_memory=False)

df_socat["time"] = pd.to_datetime(df_socat[["year", "month", "day"]], errors="coerce")
df_socat.dropna(subset=["time", "latitude", "longitude"], inplace=True)

# Month-start timestamps (matches MLD)
df_socat["month_center"] = df_socat["time"].dt.to_period("M").dt.to_timestamp(how="start")

# Fix longitude convention to [-180, 180]
df_socat["longitude"] = np.where(
    df_socat["longitude"] > 180, df_socat["longitude"] - 360, df_socat["longitude"]
)

# Subset to Southern Ocean on raw latitude first
df_socat = df_socat[(df_socat["latitude"] >= lat_min) & (df_socat["latitude"] <= lat_max)].copy()
print(f"SOCAT points in Southern Ocean (raw lat filter): {len(df_socat):,}")

# ===============================================================
# 3) Snap SOCAT to the same 1° grid (centers at 0.5°)
# ===============================================================
print("Snapping SOCAT points to 1° grid centers...")
df_socat["lat_center"] = np.floor(df_socat["latitude"]) + grid_center_offset
df_socat["lon_center"] = np.floor(df_socat["longitude"]) + grid_center_offset

# Enforce "30S and south" in 1° grid sense on snapped centers
df_socat = df_socat[df_socat["lat_center"] <= -30.5].copy()
print(f"SOCAT points after snapped-center filter (<= -30.5): {len(df_socat):,}")

# ===============================================================
# 4) Aggregate SOCAT by (cell, month)
# ===============================================================
print("Aggregating SOCAT by month × 1° cell...")
df_socat_agg = (
    df_socat.groupby(["lat_center", "lon_center", "month_center"], as_index=False)
    .agg(
        fco2_rec_median=("fco2_rec", "median"),
        SOCAT_temperature=("sst_degC", "median"),
        SOCAT_salinity=("salinity", "median"),
        fco2_rec_count=("fco2_rec", "count"),
    )
)

# Normalize month keys again (safe)
df_socat_agg["month_center"] = (
    pd.to_datetime(df_socat_agg["month_center"]).dt.to_period("M").dt.to_timestamp(how="start")
)

print(f"SOCAT aggregated: {len(df_socat_agg):,} cell-month rows")

# ===============================================================
# 5) Align merge keys (avoid float representation issues)
# ===============================================================
df_socat_agg["lat_center"] = df_socat_agg["lat_center"].round(3)
df_socat_agg["lon_center"] = df_socat_agg["lon_center"].round(3)
df_mld["lat_center"] = df_mld["lat_center"].round(3)
df_mld["lon_center"] = df_mld["lon_center"].round(3)

# ===============================================================
# 6) Merge SOCAT aggregates onto MLD grid
# ===============================================================
print("Merging SOCAT with MLD grid...")
df_out = pd.merge(
    df_mld,
    df_socat_agg,
    on=["lat_center", "lon_center", "month_center"],
    how="left",
)

# Keep only rows where SOCAT exists (as in your original intent)
print("Filtering rows with valid SOCAT data...")
df_out = df_out.dropna(subset=["fco2_rec_median", "SOCAT_temperature", "SOCAT_salinity"])

print(f"Final matched grid: {len(df_out):,} rows")

# ===============================================================
# 7) Save
# ===============================================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_out.to_csv(output_path, index=False)

print(f"\nDone. Final regridded dataset saved to:\n{output_path}")
