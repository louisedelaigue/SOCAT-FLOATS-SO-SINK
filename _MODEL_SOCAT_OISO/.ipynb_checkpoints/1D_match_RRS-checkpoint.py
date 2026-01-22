# ===============================================================
# SOCAT–MLD–SLA–RRS Monthly Merge on 1° Grid (Aligned to 0.5° Centers)
# Louise Delaigue – 2025
# ===============================================================

"""
Merges the SOCAT–MLD–SLA (1° × monthly) dataset with Copernicus Ocean Colour
Remote Reflectance (RRS) data for the Southern Ocean (-90° to -30°).

Steps:
 1. Load SOCAT–MLD–SLA 1°×1° monthly dataset
 2. Regrid Copernicus RRS bands (0.04° daily) to 1° monthly (median per cell)
 3. Merge RRS with the SOCAT–MLD–SLA grid
"""

import os
import pandas as pd
import numpy as np
import copernicusmarine
import warnings

warnings.filterwarnings("ignore")

# === Paths ===
socat_sla_path = (
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/"
    "MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
    "SOCATv2025_SO_clean_MLD_SLA_monthly_1deg.csv"
)
output_path = (
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/"
    "MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
    "SOCATv2025_SO_clean_MLD_SLA_RRS_monthly_1deg.csv"
)

# === Dataset configuration ===
dataset_id = "cmems_obs-oc_glo_bgc-reflectance_my_l3-multi-4km_P1D"
variables_all = ["RRS412", "RRS443", "RRS490", "RRS555", "RRS670"]

lat_min, lat_max = -90, -30
target_res = 1.0
grid_center_offset = 0.5  # grid centers at ±0.5°
rrs_start_year = 1997  # RRS data starts around 1997

# ===============================================================
# 1. Load SOCAT–MLD–SLA monthly grid
# ===============================================================
print("Loading SOCAT–MLD–SLA dataset...")
df_all = pd.read_csv(socat_sla_path)
df_all["month_center"] = pd.to_datetime(df_all["month_center"], errors="coerce")
df_all = df_all[(df_all["lat_center"] >= lat_min) & (df_all["lat_center"] <= lat_max)]

years = sorted(df_all["month_center"].dt.year.unique())
years = [y for y in years if y >= rrs_start_year]
print(f"Processing years: {years[0]}–{years[-1]}")

# ===============================================================
# 2. Process each RRS variable
# ===============================================================
for var in variables_all:
    print(f"\n[INFO] Starting RRS processing for {var}...")

    matched_years = []

    for year in years:
        print(f"  • Processing year {year}")

        df_year = df_all[df_all["month_center"].dt.year == year].copy()
        if df_year.empty:
            continue

        # --- Load RRS dataset ---
        ds = copernicusmarine.open_dataset(
            dataset_id=dataset_id,
            variables=[var],
            minimum_latitude=lat_min,
            maximum_latitude=lat_max,
            start_datetime=f"{year}-01-01",
            end_datetime=f"{year}-12-31",
        ).load()

        # --- Monthly median ---
        ds_monthly = ds[var].resample(time="1ME").median(skipna=True)
        df_rrs = ds_monthly.to_dataframe().reset_index()

        # --- Snap to 1° grid (±0.5° centers) ---
        df_rrs["lat_center"] = np.floor(df_rrs["latitude"]) + grid_center_offset
        df_rrs["lon_center"] = np.floor(df_rrs["longitude"]) + grid_center_offset

        # --- Aggregate to 1° monthly median ---
        df_rrs = (
            df_rrs.groupby(["time", "lat_center", "lon_center"], as_index=False)[var]
            .median()
            .rename(columns={"time": "month_center", var: var.lower()})
        )
        df_rrs["month_center"] = pd.to_datetime(df_rrs["month_center"]).dt.to_period("M").dt.to_timestamp(how="end")

        # --- Align merge keys ---
        df_year["month_center"] = pd.to_datetime(df_year["month_center"]).dt.to_period("M").dt.to_timestamp(how="end")
        df_rrs["lat_center"] = df_rrs["lat_center"].round(3)
        df_rrs["lon_center"] = df_rrs["lon_center"].round(3)
        df_year["lat_center"] = df_year["lat_center"].round(3)
        df_year["lon_center"] = df_year["lon_center"].round(3)

        # --- Merge RRS with SOCAT–MLD–SLA grid ---
        df_year = pd.merge(
            df_year,
            df_rrs,
            on=["month_center", "lat_center", "lon_center"],
            how="left"
        )

        matched = df_year[var.lower()].notna().sum()
        total = len(df_year)
        pct = 100 * matched / total if total > 0 else 0
        print(f"    Matched {matched:,} of {total:,} grid cells ({pct:.2f}%) with {var}")

        matched_years.append(df_year)

    # Combine across all years for this RRS variable
    df_all = pd.concat(matched_years, ignore_index=True)

    total_all = len(df_all)
    matched_all = df_all[var.lower()].notna().sum()
    pct_all = 100 * matched_all / total_all if total_all > 0 else 0
    print(f"\n[SUMMARY] {var}: matched {matched_all:,} / {total_all:,} total cells ({pct_all:.2f}%)")

# ===============================================================
# 3. Save final merged dataset
# ===============================================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_all.to_csv(output_path, index=False)

print("\nDone.")
print(f"Final merged dataset saved to: {output_path}")
print(f"Final dataset size: {len(df_all):,} rows")
