# ===============================================================
# Feature Engineering for PM fCO₂ Modeling (Pipeline Version)
# Louise Delaigue – 2025
# ===============================================================

"""
This script loads the fully merged PM 1°×1° monthly dataset
(PM + MLD + SLA + RRS + PAR + WIND + ATM CO₂)
and computes additional engineered features:

- decimal year (from month_center)
- day-of-year sine/cosine (seasonal cycle)
- 3D Cartesian coordinates (x, y, z)
- GEBCO bathymetry at the 1° grid point

Pipeline rules:
• Use month_center aligned to month-end
• Only use lat_center / lon_center
• Never use raw daily timestamps
"""

import pandas as pd
import numpy as np
import xarray as xr
from tqdm import tqdm

# ===============================================================
# 1. Load dataset
# ===============================================================
print("Loading dataset...")
df = pd.read_csv(
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/"
    "PREDICTION_MATRIX_v2/processing_steps/"
    "ARMOR3D_TSMLD_SouthernOcean_1997_2024_monthly_1deg_SLA_RRS_PAR_WIND_ATM_CO2.csv"
)

print(f"Loaded {len(df):,} rows.")

# ---------------------------------------------------------------
# Ensure month_center is correctly formatted
# ---------------------------------------------------------------
df["month_center"] = pd.to_datetime(df["month_center"], errors="coerce")
df["month_center"] = df["month_center"].dt.to_period("M").dt.to_timestamp(how="end")

# Extract time components
df["year"]  = df["month_center"].dt.year
df["month"] = df["month_center"].dt.month
df["doy"]   = df["month_center"].dt.dayofyear

# ===============================================================
# 2. Decimal Year (pipeline-compliant)
# ===============================================================
print("Computing decimal year...")

def compute_decimal_year(ts):
    if pd.isnull(ts):
        return np.nan
    year_start = pd.Timestamp(year=ts.year, month=1, day=1)
    next_year  = pd.Timestamp(year=ts.year + 1, month=1, day=1)
    return ts.year + (ts - year_start).days / (next_year - year_start).days

tqdm.pandas(desc="Decimal year")
df["decimal_year"] = df["month_center"].progress_apply(compute_decimal_year)

# ===============================================================
# 3. Seasonal cycle (sin/cos)
# ===============================================================
print("Computing day-of-year sine and cosine...")

df["doy_radians"] = 2 * np.pi * df["doy"] / 365.0
df["doy_sin"] = np.sin(df["doy_radians"])
df["doy_cos"] = np.cos(df["doy_radians"])

# ===============================================================
# 4. Cartesian coordinates from 1° grid centers
# ===============================================================
print("Computing Cartesian coordinates from lat/lon...")

lat_rad = np.radians(df["lat_center"])
lon_rad = np.radians(df["lon_center"])

df["x_cart"] = np.cos(lat_rad) * np.cos(lon_rad)
df["y_cart"] = np.cos(lat_rad) * np.sin(lon_rad)
df["z_cart"] = np.sin(lat_rad)

# ===============================================================
# 5. GEBCO bathymetry at grid centers
# ===============================================================
print("Loading GEBCO bathymetry...")

bathy = xr.open_dataset(
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/bathymetry/GEBCO_2024.nc"
)

# Convert centers to DataArrays for selection
lats = xr.DataArray(df["lat_center"].values, dims="points")
lons = xr.DataArray(df["lon_center"].values, dims="points")

# GEBCO lon may be 0–360; convert if needed
if bathy["lon"].max() > 180:
    bathy = bathy.assign_coords(
        lon=((bathy["lon"] + 180) % 360) - 180
    )

depth = bathy["elevation"].sel(
    lat=lats, lon=lons, method="nearest"
).values

df["bottom_depth_m"] = -depth  # GEBCO elevation: negative = ocean

# Remove grid cells where GEBCO indicates land or zero-depth
before = len(df)
df = df[df["bottom_depth_m"] > 0].copy()
after = len(df)

print(f"Dropped {before - after:,} land/invalid bathymetry points.")

# ===============================================================
# 6. Save final engineered dataset
# ===============================================================
output_path = (
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/"
    "PREDICTION_MATRIX_v2/processing_steps/"
    "ARMOR3D_TSMLD_SouthernOcean_1997_2024_monthly_1deg_feature_engineered_monthly_1deg.csv"
)

print(f"Saving engineered dataset to:\n{output_path}")
df.to_csv(output_path, index=False)

print("\nFeature engineering complete.")
print(f"Final dataset size: {len(df):,} rows")
