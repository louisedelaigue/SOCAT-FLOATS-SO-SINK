# ===============================================================
# SOCAT–SLA–RRS–PAR Monthly Matchup (Handles multiple PAR file patterns)
# Louise Delaigue — 2025
# ===============================================================

import os
import warnings
import fnmatch
import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

warnings.filterwarnings("ignore")

# === Paths ===
socat_sla_rrs_path = (
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/MODEL_SOCAT_ONLY_OBS_v2/"
    "processing_steps/SOCATv2025_SO_clean_SLA_RRS_monthly.csv"
)
par_dir = (
    "/home/ldelaigue/Documents/Python/SOCA-CO2/DATA/OCEAN_COLOR/PAR_monthly"
)
output_path = (
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/MODEL_SOCAT_ONLY_OBS_v2/"
    "processing_steps/SOCATv2025_SO_clean_SLA_RRS_PAR_monthly.csv"
)

# === Parameters ===
pixel_size_sla = 0.125           # SLA grid (1/8°)
sla_offset = pixel_size_sla / 2  # grid center offset (0.0625°)

print("Loading SOCAT–SLA–RRS dataset...")
df_all = pd.read_csv(socat_sla_rrs_path)
df_all["month_center"] = pd.to_datetime(df_all["month_center"], errors="coerce")

years = sorted(df_all["month_center"].dt.year.unique())
print(f"Processing PAR for {years[0]}–{years[-1]}")

# === Function to find PAR files with multiple possible patterns ===
def find_par_file(year, month, data_dir):
    month_start = f"{year}{month:02d}01"
    possible_patterns = [
        f"L3m_{month_start}-*__GLOB_4_AV-SWF_PAR_MO_00.nc",
        f"L3m_{month_start}-*__GLOB_4_AV-MOD_PAR_MO_00.nc",
        f"L3m_{month_start}-*__GLOB_4_AVW-MODSWF_PAR_MO_00.nc",
        f"L3m_{month_start}-*__GLOB_4_AVW-MODVIR_PAR_MO_00.nc",
    ]
    for pattern in possible_patterns:
        matches = [f for f in os.listdir(data_dir) if fnmatch.fnmatch(f, pattern)]
        if matches:
            return os.path.join(data_dir, sorted(matches)[0])
    return None

# === Function to load and process all PAR data for one year ===
def load_par_year(year):
    files = []
    for month in range(1, 13):
        f = find_par_file(year, month, par_dir)
        if f:
            files.append(f)
    if not files:
        print(f"[WARN] No PAR files found for {year}")
        return None

    dfs = []
    for fpath in tqdm(files, desc=f"Loading PAR {year}"):
        try:
            ds = xr.open_dataset(fpath)
            if ds.lon.max() > 180:
                ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180))
            da = ds["PAR_mean"]

            df_month = (
                da.to_dataframe()
                .reset_index()
                .dropna(subset=["PAR_mean"])
                .rename(columns={"lat": "latitude", "lon": "longitude"})
            )

            # Extract year-month from filename (e.g., L3m_20190101-20190131__...)
            fname = os.path.basename(fpath)
            month_str = fname.split("_")[1][:6]
            df_month["month_center"] = pd.to_datetime(month_str, format="%Y%m") + pd.offsets.MonthEnd(0)

            dfs.append(df_month)
        except Exception as e:
            print(f"[WARN] Skipping {fpath}: {e}")

    if not dfs:
        return None

    df_par = pd.concat(dfs, ignore_index=True)

    # === Regrid PAR to SLA grid centers ===
    df_par["lat_center"] = (
        np.round((df_par["latitude"] - sla_offset) / pixel_size_sla) * pixel_size_sla + sla_offset
    )
    df_par["lon_center"] = (
        np.round((df_par["longitude"] - sla_offset) / pixel_size_sla) * pixel_size_sla + sla_offset
    )

    # === Aggregate to SLA grid (median of all PAR points within each cell) ===
    df_par = (
        df_par.groupby(["month_center", "lat_center", "lon_center"], as_index=False)
        .median(numeric_only=True)
    )

    return df_par

# === Main loop across all years ===
matched_years = []
for year in years:
    print(f"\n[INFO] Processing PAR for {year}")
    df_year = df_all[df_all["month_center"].dt.year == year].copy()
    df_par = load_par_year(year)

    if df_par is None:
        continue

    df_year = pd.merge(
        df_year,
        df_par[["month_center", "lat_center", "lon_center", "PAR_mean"]],
        on=["month_center", "lat_center", "lon_center"],
        how="left"
    )
    matched = df_year["PAR_mean"].notna().sum()
    print(f"  → Matched {matched} SLA grid cells with PAR")
    matched_years.append(df_year)

# === Combine all and save ===
if matched_years:
    df_final = pd.concat(matched_years, ignore_index=True)
else:
    df_final = df_all.copy()

df_final.to_csv(output_path, index=False)
print(f"\nDone. Final PAR-matched dataset saved to: {output_path}")
print(f"Final dataset size: {len(df_final)} rows")
