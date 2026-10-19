# ===============================================================
# FLOATS–MLD–SLA Monthly Merge on 1° Grid (Aligned to SOCAT Pipeline)
# Louise Delaigue – 2025
# ===============================================================

"""
Merges:
    BGC-Argo (1° × monthly, medians)
    ARMOR3D MLD (1° × monthly)
    Copernicus SLA (0.125° daily → 1° monthly median)
    
Output:
    BGC-Argo–MLD–SLA dataset on the SOCAT-style 1° monthly grid.
"""

import os
import pandas as pd
import numpy as np
import copernicusmarine

# ===============================================================
# Paths
# ===============================================================
floats_mld_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_v2/processing_steps/"
    "BGC-Argo_SO_clean_MLD_monthly_regridded.csv"
)

output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_v2/processing_steps/"
    "BGC-Argo_SO_clean_MLD_monthly_regridded_SLA.csv"
)

# SLA dataset
dataset_id = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
variable = "sla"

lat_min, lat_max = -90, -30
lat_center_max = -30.5
sla_start_year = 1993
sla_end_year = 2024
grid_center_offset = 0.5  # centers at .5°

# ===============================================================
# 1. Load FLOATS–MLD dataset
# ===============================================================
print("Loading FLOATS–MLD dataset...")
df_float_mld = pd.read_csv(floats_mld_path)

df_float_mld["month_center"] = pd.to_datetime(df_float_mld["month_center"], errors="coerce")
df_float_mld.dropna(subset=["month_center", "lat_center", "lon_center"], inplace=True)

# Normalize to month START (SOCAT pipeline)
df_float_mld["month_center"] = df_float_mld["month_center"].dt.to_period("M").dt.to_timestamp(how="start")

# Enforce SO domain on centers
df_float_mld = df_float_mld[
    (df_float_mld["lat_center"] >= lat_min) &
    (df_float_mld["lat_center"] <= lat_max)
]
df_float_mld = df_float_mld[df_float_mld["lat_center"] <= lat_center_max]

print(f"Loaded {len(df_float_mld):,} float–MLD grid cells.")
print(f"Time range: {df_float_mld['month_center'].min().date()} → "
      f"{df_float_mld['month_center'].max().date()}")

years = sorted(df_float_mld["month_center"].dt.year.unique())
years = [y for y in years if sla_start_year <= y <= sla_end_year]
if not years:
    raise ValueError("No years to process for SLA after filtering float–MLD dataset.")

print(f"SLA processing years: {years[0]}–{years[-1]} ({len(years)} years)")

# ===============================================================
# 2. Load & regrid SLA by year
# ===============================================================
all_years = []

for year in years:
    print(f"\n[INFO] Downloading SLA for {year}...")

    start_time = f"{year}-01-01"
    end_time = "2024-12-31" if year == sla_end_year else f"{year}-12-31"

    ds = copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        variables=[variable],
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        start_datetime=start_time,
        end_datetime=end_time,
    ).load()

    print("[INFO] Resampling SLA to monthly (MS, month-start labels)...")
    ds_monthly = ds[variable].resample(time="MS").median(skipna=True)

    sla_df = ds_monthly.to_dataframe().reset_index()

    # Snap to 1° grid centers
    sla_df["lat_center"] = np.floor(sla_df["latitude"]) + grid_center_offset
    sla_df["lon_center"] = np.floor(sla_df["longitude"]) + grid_center_offset

    # Enforce 30S+ on snapped centers
    sla_df = sla_df[sla_df["lat_center"] <= lat_center_max]

    # Aggregate median per month × grid
    sla_monthly = (
        sla_df.groupby(["time", "lat_center", "lon_center"], as_index=False)[variable]
        .median()
        .rename(columns={"time": "month_center", variable: "sla_median"})
    )

    sla_monthly["month_center"] = (
        pd.to_datetime(sla_monthly["month_center"]).dt.to_period("M").dt.to_timestamp(how="start")
    )

    all_years.append(sla_monthly)

# ===============================================================
# 3. Concatenate all SLA years
# ===============================================================
df_sla = pd.concat(all_years, ignore_index=True)
print(f"\nTotal SLA cell-month rows: {len(df_sla):,}")

# ===============================================================
# 4. Align timestamps + coordinates
# ===============================================================
print("Aligning timestamps and coordinates...")

for d in (df_sla, df_float_mld):
    d["lat_center"] = d["lat_center"].round(3)
    d["lon_center"] = d["lon_center"].round(3)
    d["month_center"] = pd.to_datetime(d["month_center"]).dt.to_period("M").dt.to_timestamp(how="start")

# ===============================================================
# 5. Merge FLOATS–MLD with SLA
# ===============================================================
print("Merging FLOATS–MLD with SLA...")

df_merged = pd.merge(
    df_float_mld,
    df_sla,
    on=["lat_center", "lon_center", "month_center"],
    how="left"  # SLA required for model inputs
)

print(f"Merged dataset: {len(df_merged):,} rows")
print(f"Coverage: {len(df_merged) / len(df_float_mld) * 100:.2f}%")

# ===============================================================
# 6. Save
# ===============================================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_merged.to_csv(output_path, index=False)

print("\n[DONE] FLOATS–MLD–SLA monthly grid saved to:")
print(output_path)