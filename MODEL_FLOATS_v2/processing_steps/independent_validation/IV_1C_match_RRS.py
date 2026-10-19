# ===============================================================
# FLOATS–MLD–SLA–RRS Monthly Merge on 1° Grid (Aligned to SOCAT Pipeline)
# Louise Delaigue – 2025
# ===============================================================

"""
Merges the FLOATS–MLD–SLA (1° × monthly) dataset with Copernicus Ocean Colour
Remote Reflectance (RRS) data for the Southern Ocean (30°S and south).

Steps:
 1. Load FLOATS–MLD–SLA 1°×1° monthly dataset
 2. Regrid Copernicus RRS bands (daily) to 1° monthly (median per cell)
 3. Merge RRS with the FLOATS–MLD–SLA grid
"""

import os
os.environ["CMEMS_DISABLE_CONSOLIDATED"] = "1"  # Zarr missing-chunk bug

import pandas as pd
import numpy as np
import copernicusmarine
import warnings

warnings.filterwarnings("ignore")

# === Paths ===
floats_sla_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_v2/processing_steps/independent_validation/"
    "pco2_independent_validation_data_TSMLD_SLA.csv"
)

output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_v2/processing_steps/independent_validation/"
    "pco2_independent_validation_data_TSMLD_SLA_RRS.csv"
)

# === Dataset configuration ===
dataset_id = "cmems_obs-oc_glo_bgc-reflectance_my_l3-multi-4km_P1D"
variables_all = ["RRS412", "RRS443", "RRS490", "RRS555", "RRS670"]

lat_min, lat_max = -90, -30
grid_center_offset = 0.5
lat_center_max = -30.5
rrs_start_year = 1997  # RRS data starts around 1997

# ===============================================================
# 1. Load FLOATS–MLD–SLA monthly grid
# ===============================================================
print("Loading FLOATS–MLD–SLA dataset...")
df_all = pd.read_csv(floats_sla_path)

df_all["month_center"] = pd.to_datetime(df_all["month_center"], errors="coerce")
df_all.dropna(subset=["month_center", "lat_center", "lon_center"], inplace=True)

# Normalize to month START
df_all["month_center"] = df_all["month_center"].dt.to_period("M").dt.to_timestamp(how="start")

# Enforce SO domain on centers
df_all = df_all[(df_all["lat_center"] >= lat_min) & (df_all["lat_center"] <= lat_max)]
df_all = df_all[df_all["lat_center"] <= lat_center_max]

years = sorted(df_all["month_center"].dt.year.unique())
years = [y for y in years if y >= rrs_start_year]
if not years:
    raise ValueError("No years to process after applying rrs_start_year and filters.")

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
        )

        # IMPORTANT: Wrap .load() in try/except
        try:
            ds = ds.load()
        except KeyError as e:
            print(f"[ERROR] Missing S3 chunk for {var} in year {year}: {e}. Skipping this year.")
            continue

        # --- Monthly median (month-start labels) ---
        ds_monthly = ds[var].resample(time="MS").median(skipna=True)
        df_rrs = ds_monthly.to_dataframe().reset_index()

        # --- Snap to 1° grid (±0.5° centers) ---
        df_rrs["lat_center"] = np.floor(df_rrs["latitude"]) + grid_center_offset
        df_rrs["lon_center"] = np.floor(df_rrs["longitude"]) + grid_center_offset

        # Enforce 30S+ on snapped centers
        df_rrs = df_rrs[df_rrs["lat_center"] <= lat_center_max]

        # --- Aggregate to 1° monthly median ---
        df_rrs = (
            df_rrs.groupby(["time", "lat_center", "lon_center"], as_index=False)[var]
            .median()
            .rename(columns={"time": "month_center", var: var.lower()})
        )

        # Normalize to month START
        df_rrs["month_center"] = pd.to_datetime(df_rrs["month_center"]).dt.to_period("M").dt.to_timestamp(how="start")
        df_year["month_center"] = pd.to_datetime(df_year["month_center"]).dt.to_period("M").dt.to_timestamp(how="start")

        # --- Align merge keys ---
        for d in (df_rrs, df_year):
            d["lat_center"] = d["lat_center"].round(3)
            d["lon_center"] = d["lon_center"].round(3)

        # --- Merge RRS with FLOATS–MLD–SLA grid ---
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
    if not matched_years:
        print(f"[WARN] No matched years produced for {var}. Keeping df_all unchanged.")
        continue

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
