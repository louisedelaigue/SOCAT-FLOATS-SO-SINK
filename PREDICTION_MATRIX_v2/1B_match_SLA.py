# ===============================================================
# PM–MLD–SLA Monthly Merge on 1° Grid (Aligned to SOCAT Pipeline)
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
PM_mld_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "PREDICTION_MATRIX_v2/processing_steps/"
    "ARMOR3D_MLD_TS_SouthernOcean_2003_2024_monthly_1deg.csv"
)

output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/PREDICTION_MATRIX_v2/"
    "ARMOR3D_TSMLD_SouthernOcean_2003_2024_monthly_1deg_SLA.csv"
)

# SLA dataset
dataset_id = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
variable = "sla"

lat_min, lat_max = -90, -30
sla_start_year = 2003
sla_end_year = 2024
grid_center_offset = 0.5  # centers at .5°

# ===============================================================
# 1. Load PM–MLD dataset
# ===============================================================

print("Loading PM–MLD dataset...")
df_PM_mld = pd.read_csv(PM_mld_path)

df_PM_mld["month_center"] = pd.to_datetime(df_PM_mld["month_center"], errors="coerce")
df_PM_mld = df_PM_mld[
    (df_PM_mld["lat_center"] >= lat_min) &
    (df_PM_mld["lat_center"] <= lat_max)
]

print(f"Loaded {len(df_PM_mld):,} PM–MLD grid cells.")
print(f"Time range: {df_PM_mld['month_center'].min().date()} → "
      f"{df_PM_mld['month_center'].max().date()}")

years = sorted(df_PM_mld["month_center"].dt.year.unique())
years = [y for y in years if sla_start_year <= y <= sla_end_year]

print(f"SLA processing years: {years[0]}–{years[-1]} ({len(years)} years)")

# ===============================================================
# 2. Load & regrid SLA by year
# ===============================================================

all_years = []

for year in years:
    print(f"\n[INFO] Downloading SLA for {year}...")

    start_time = f"{year}-01-01"
    end_time = f"{year}-12-31" if year < sla_end_year else "2024-12-31"

    ds = copernicusmarine.open_dataset(
        dataset_id=dataset_id,
        variables=[variable],
        minimum_latitude=lat_min,
        maximum_latitude=lat_max,
        start_datetime=start_time,
        end_datetime=end_time,
    ).load()

    print("[INFO] Resampling SLA to monthly (1ME)...")
    ds_monthly = ds[variable].resample(time="1ME").median(skipna=True)

    sla_df = ds_monthly.to_dataframe().reset_index()

    # Snap to 1° grid centers
    sla_df["lat_center"] = np.floor(sla_df["latitude"]) + grid_center_offset
    sla_df["lon_center"] = np.floor(sla_df["longitude"]) + grid_center_offset

    # Aggregate median per month × grid
    sla_monthly = (
        sla_df.groupby(["time", "lat_center", "lon_center"], as_index=False)[variable]
        .median()
        .rename(columns={"time": "month_center", variable: "sla_median"})
    )

    sla_monthly["month_center"] = pd.to_datetime(sla_monthly["month_center"])

    all_years.append(sla_monthly)

# ===============================================================
# 3. Concatenate all SLA years
# ===============================================================

df_sla = pd.concat(all_years, ignore_index=True)
print(f"\nTotal SLA grid cells: {len(df_sla):,}")

# ===============================================================
# 4. Align timestamps + coordinates
# ===============================================================

print("Aligning timestamps and coordinates...")

# round centers to avoid FP mismatch
df_sla["lat_center"] = df_sla["lat_center"].round(3)
df_sla["lon_center"] = df_sla["lon_center"].round(3)
df_PM_mld["lat_center"] = df_PM_mld["lat_center"].round(3)
df_PM_mld["lon_center"] = df_PM_mld["lon_center"].round(3)

# convert to month-end stamps (SOCAT pipeline)
df_sla["month_center"] = (
    df_sla["month_center"].dt.to_period("M").dt.to_timestamp(how="end")
)
df_PM_mld["month_center"] = (
    df_PM_mld["month_center"].dt.to_period("M").dt.to_timestamp(how="end")
)

# ===============================================================
# 5. Merge PM–MLD with SLA
# ===============================================================

print("Merging PM–MLD with SLA...")

df_merged = pd.merge(
    df_PM_mld,
    df_sla,
    on=["lat_center", "lon_center", "month_center"],
    how="inner"    # SLA required for model inputs
)

print(f"Merged dataset: {len(df_merged):,} rows")
print(f"Coverage: {len(df_merged) / len(df_PM_mld) * 100:.2f}%")

# ===============================================================
# 6. Save
# ===============================================================

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_merged.to_csv(output_path, index=False)

print("\n[DONE] PM–MLD–SLA monthly grid saved to:")
print(output_path)
