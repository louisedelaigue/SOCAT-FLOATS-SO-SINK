# ===============================================================
# SOCAT–MLD–SLA Monthly Merge on 1° Grid (Aligned to 0.5° Centers)
# Louise Delaigue – 2025
# ===============================================================

"""
Merges the SOCAT–MLD (1° × monthly) dataset with Copernicus SLA (0.125°)
for the Southern Ocean (30°S and south).

Steps:
 1. Regrid Copernicus SLA (0.125° daily) to 1° monthly (mean per month, then median per cell)
 2. Merge SLA with the SOCAT–MLD grid (keeps SOCAT–MLD rows, adds SLA where available)
"""

import os
import pandas as pd
import numpy as np
import copernicusmarine

# === Paths ===
socat_mld_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "_MODEL_SOCAT_OISO/MODEL_NO_OISO/processing_steps/"
    "SOCATv2025_SO_NO_OISO_clean_MLD_monthly_regridded.csv"
)

output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "_MODEL_SOCAT_OISO/MODEL_NO_OISO/processing_steps/"
    "SOCATv2025_SO_NO_OISO_clean_MLD_monthly_regridded_SLA.csv"
)

# === SLA dataset info ===
dataset_id = "cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D"
variable = "sla"

lat_min, lat_max = -90, -30
sla_start_year = 1993
sla_end_year = 2024

grid_center_offset = 0.5  # centers at ... -30.5, -31.5, ...
lat_center_max = -30.5    # enforce "30S and south" on the snapped grid

# ===============================================================
# 1) Load SOCAT–MLD dataset
# ===============================================================
print("Loading SOCAT–MLD monthly dataset...")
df_socat_mld = pd.read_csv(socat_mld_path)

df_socat_mld["month_center"] = pd.to_datetime(df_socat_mld["month_center"], errors="coerce")
df_socat_mld.dropna(subset=["month_center", "lat_center", "lon_center"], inplace=True)

# Normalize to month START (your SOCAT–MLD uses YYYY-MM-01)
df_socat_mld["month_center"] = (
    df_socat_mld["month_center"].dt.to_period("M").dt.to_timestamp(how="start")
)

# Enforce Southern Ocean on centers
df_socat_mld = df_socat_mld[(df_socat_mld["lat_center"] >= lat_min) & (df_socat_mld["lat_center"] <= lat_max)]
df_socat_mld = df_socat_mld[df_socat_mld["lat_center"] <= lat_center_max]

print(
    f"Loaded {len(df_socat_mld):,} SOCAT–MLD rows "
    f"from {df_socat_mld['month_center'].min().date()} to {df_socat_mld['month_center'].max().date()}"
)

years = sorted(df_socat_mld["month_center"].dt.year.unique())
years = [y for y in years if sla_start_year <= y <= sla_end_year]
if not years:
    raise ValueError("No years found in SOCAT–MLD within the SLA year range.")

print(f"Processing SLA for {len(years)} years: {years[0]}–{years[-1]}")

# ===============================================================
# 2) Load and regrid SLA by year
# ===============================================================
all_years = []

for year in years:
    print(f"\n[INFO] Loading SLA data for {year}...")

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

    print("[INFO] Resampling SLA to monthly (month-start labels)...")
    # Month-start timestamps to match your SOCAT–MLD convention
    ds_monthly = ds[variable].resample(time="MS").mean(skipna=True)

    sla_df = ds_monthly.to_dataframe().reset_index()

    # Snap SLA coordinates to 1° grid centered at .5°
    sla_df["lat_center"] = np.floor(sla_df["latitude"]) + grid_center_offset
    sla_df["lon_center"] = np.floor(sla_df["longitude"]) + grid_center_offset

    # Enforce "30S and south" on snapped centers
    sla_df = sla_df[sla_df["lat_center"] <= lat_center_max]

    # Aggregate SLA to monthly × 1° cells (median across native cells in each 1° bin)
    sla_monthly = (
        sla_df.groupby(["time", "lat_center", "lon_center"], as_index=False)[variable]
        .median()
        .rename(columns={"time": "month_center", variable: "sla_median"})
    )

    # Ensure month-start timestamps
    sla_monthly["month_center"] = (
        pd.to_datetime(sla_monthly["month_center"]).dt.to_period("M").dt.to_timestamp(how="start")
    )

    all_years.append(sla_monthly)

# ===============================================================
# 3) Concatenate and merge with SOCAT–MLD
# ===============================================================
if not all_years:
    raise ValueError("No SLA data processed for any year.")

df_sla = pd.concat(all_years, ignore_index=True)
print(f"\nTotal SLA cell-month rows after regridding: {len(df_sla):,}")

# Align coordinates and timestamps (safe)
print("Aligning keys before merge...")
for df in (df_sla, df_socat_mld):
    df["lat_center"] = df["lat_center"].round(3)
    df["lon_center"] = df["lon_center"].round(3)
    df["month_center"] = pd.to_datetime(df["month_center"]).dt.to_period("M").dt.to_timestamp(how="start")

print("SLA months:", df_sla["month_center"].min(), "→", df_sla["month_center"].max())
print("SOCAT–MLD months:", df_socat_mld["month_center"].min(), "→", df_socat_mld["month_center"].max())

# Merge SLA with SOCAT–MLD (LEFT join keeps SOCAT–MLD rows, adds SLA where available)
df_merged = pd.merge(
    df_socat_mld,
    df_sla,
    on=["lat_center", "lon_center", "month_center"],
    how="left",
)

coverage = df_merged["sla_median"].notna().mean() * 100
print(f"\nMerged SOCAT–MLD–SLA dataset: {len(df_merged):,} rows")
print(f"SLA coverage on SOCAT–MLD grid: {coverage:.2f}%")

# ===============================================================
# 4) Save final dataset
# ===============================================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_merged.to_csv(output_path, index=False)

print("\nDone.")
print(f"Final merged dataset saved to: {output_path}")
