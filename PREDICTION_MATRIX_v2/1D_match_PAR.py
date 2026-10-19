# ===============================================================
# PM–MLD–SLA–RRS–PAR Monthly Merge on 1° Grid (Aligned to 0.5° Centers)
# Louise Delaigue – 2025
# ===============================================================

"""
Merges the PM–MLD–SLA–RRS 1°×1° monthly dataset with
Copernicus Ocean Colour PAR (Photosynthetically Available Radiation) data
for the Southern Ocean (-90° to -30°).

Steps:
 1. Load PM–MLD–SLA–RRS monthly grid
 2. Load monthly PAR data from local files (multiple filename patterns)
 3. Regrid PAR to 1° grid (median per cell)
 4. Merge PAR with PM–MLD–SLA–RRS
"""

import os
import warnings
import fnmatch
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")

# === Paths ===
PM_rrs_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "PREDICTION_MATRIX_v2/processing_steps/"
    "ARMOR3D_TSMLD_SouthernOcean_2003_2024_monthly_1deg_SLA_RRS.csv"
)

par_dir = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/OCEAN_COLOR/PAR_monthly"
output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "PREDICTION_MATRIX_v2/processing_steps/"
    "ARMOR3D_TSMLD_SouthernOcean_2003_2024_monthly_1deg_SLA_RRS_PAR.csv"
)

# === Parameters ===
lat_min, lat_max = -90, -30
target_res = 1.0
grid_center_offset = 0.5  # centers at ±0.5°
varname = "PAR_mean"

# ===============================================================
# 1. Load PM–MLD–SLA–RRS dataset
# ===============================================================
print("Loading PM–MLD–SLA–RRS dataset...")
df_all = pd.read_csv(PM_rrs_path)
df_all["month_center"] = pd.to_datetime(df_all["month_center"], errors="coerce")
df_all = df_all[(df_all["lat_center"] >= lat_min) & (df_all["lat_center"] <= lat_max)]

years = sorted(df_all["month_center"].dt.year.unique())
print(f"Processing PAR data for {years[0]}–{years[-1]}")

# ===============================================================
# 2. Helper functions
# ===============================================================

def find_par_file(year, month, data_dir):
    """Find PAR NetCDF file for a given year/month across possible naming schemes."""
    month_start = f"{year}{month:02d}01"
    patterns = [
        f"L3m_{month_start}-*__GLOB_4_AV-SWF_PAR_MO_00.nc",
        f"L3m_{month_start}-*__GLOB_4_AV-MOD_PAR_MO_00.nc",
        f"L3m_{month_start}-*__GLOB_4_AVW-MODSWF_PAR_MO_00.nc",
        f"L3m_{month_start}-*__GLOB_4_AVW-MODVIR_PAR_MO_00.nc",
    ]
    for pattern in patterns:
        matches = [f for f in os.listdir(data_dir) if fnmatch.fnmatch(f, pattern)]
        if matches:
            return os.path.join(data_dir, sorted(matches)[0])
    return None


def load_par_year(year):
    """Load and regrid PAR monthly data for one year onto a 1° grid."""
    files = [find_par_file(year, m, par_dir) for m in range(1, 13)]
    files = [f for f in files if f]
    if not files:
        print(f"[WARN] No PAR files found for {year}")
        return None

    dfs = []
    for fpath in tqdm(files, desc=f"Loading PAR {year}"):
        try:
            ds = xr.open_dataset(fpath)
            if "lon" in ds.coords and ds.lon.max() > 180:
                ds = ds.assign_coords(lon=((ds.lon + 180) % 360) - 180)

            da = ds[varname]
            df_month = (
                da.to_dataframe()
                .reset_index()
                .dropna(subset=[varname])
                .rename(columns={"lat": "latitude", "lon": "longitude"})
            )

            # Extract year-month from filename
            fname = os.path.basename(fpath)
            ym_str = fname.split("_")[1][:6]
            df_month["month_center"] = pd.to_datetime(ym_str, format="%Y%m") + pd.offsets.MonthEnd(0)

            dfs.append(df_month)
        except Exception as e:
            print(f"[WARN] Skipping {fpath}: {e}")

    if not dfs:
        return None

    df_par = pd.concat(dfs, ignore_index=True)

    # === Regrid to 1° grid aligned to ±0.5° centers ===
    df_par["lat_center"] = np.floor(df_par["latitude"]) + grid_center_offset
    df_par["lon_center"] = np.floor(df_par["longitude"]) + grid_center_offset

    # Aggregate by grid + month (median)
    df_par = (
        df_par.groupby(["month_center", "lat_center", "lon_center"], as_index=False)[varname]
        .median()
    )

    # Align timestamps to month-end and round coordinates
    df_par["month_center"] = (
        pd.to_datetime(df_par["month_center"]).dt.to_period("M").dt.to_timestamp(how="end")
    )
    df_par["lat_center"] = df_par["lat_center"].round(3)
    df_par["lon_center"] = df_par["lon_center"].round(3)

    return df_par


# ===============================================================
# 3. Main processing loop
# ===============================================================
matched_years = []

for year in years:
    print(f"\n[INFO] Processing PAR for {year}")
    df_year = df_all[df_all["month_center"].dt.year == year].copy()
    df_year["month_center"] = (
        pd.to_datetime(df_year["month_center"]).dt.to_period("M").dt.to_timestamp(how="end")
    )

    df_par = load_par_year(year)
    if df_par is None:
        continue

    # Merge on month_center + grid
    df_year = pd.merge(
        df_year,
        df_par[["month_center", "lat_center", "lon_center", varname]],
        on=["month_center", "lat_center", "lon_center"],
        how="left"
    )

    matched = df_year[varname].notna().sum()
    total = len(df_year)
    pct = 100 * matched / total if total > 0 else 0
    print(f"  Matched {matched:,} of {total:,} grid cells ({pct:.2f}%) with PAR")

    matched_years.append(df_year)

# ===============================================================
# 4. Combine and save
# ===============================================================
if matched_years:
    df_final = pd.concat(matched_years, ignore_index=True)
else:
    df_final = df_all.copy()

os.makedirs(os.path.dirname(output_path), exist_ok=True)
df_final.to_csv(output_path, index=False)

matched_total = df_final[varname].notna().sum()
total_total = len(df_final)
pct_total = 100 * matched_total / total_total if total_total > 0 else 0

print("\nDone.")
print(f"Final PAR-matched dataset saved to: {output_path}")
print(f"Matched {matched_total:,} / {total_total:,} total grid cells ({pct_total:.2f}%)")
print(f"Final dataset size: {len(df_final):,} rows")
