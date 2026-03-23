# ===============================================================
# Feature Engineering for SOCAT fCO₂ Modeling (Pipeline Version)
# Louise Delaigue – 2025
# ===============================================================

"""
Loads the fully merged SOCAT 1°×1° monthly dataset
(SOCAT + MLD + SLA + RRS + PAR + WIND + ATM CO₂)
and computes additional engineered features:

- decimal year (from month_center)
- day-of-year sine/cosine (seasonal cycle)
- 3D Cartesian coordinates (x, y, z)
- GEBCO bathymetry at the 1° grid point

Pipeline rules (current):
• month_center aligned to month-start (YYYY-MM-01)
• Only use lat_center / lon_center
"""

import pandas as pd
import numpy as np
import xarray as xr

# ===============================================================
# 1. Load dataset
# ===============================================================
print("Loading dataset...")
df = pd.read_csv("/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/independent_validation/pco2_independent_validation_data_regridded_SLA_RRS_PAR_WIND_ATM_CO2.csv")

print(f"Loaded {len(df):,} rows.")

# ---------------------------------------------------------------
# Ensure month_center is correctly formatted (month-start)
# ---------------------------------------------------------------
df["month_center"] = pd.to_datetime(df["month_center"], errors="coerce")
df.dropna(subset=["month_center"], inplace=True)
df["month_center"] = df["month_center"].dt.to_period("M").dt.to_timestamp(how="start")

# Extract time components
df["year"]  = df["month_center"].dt.year
df["month"] = df["month_center"].dt.month
df["doy"]   = df["month_center"].dt.dayofyear

# ===============================================================
# 2. Decimal Year (vectorized)
# ===============================================================
print("Computing decimal year...")

year_start = pd.to_datetime(df["year"].astype(str) + "-01-01")
next_year_start = pd.to_datetime((df["year"] + 1).astype(str) + "-01-01")

df["decimal_year"] = df["year"] + (
    (df["month_center"] - year_start).dt.days / (next_year_start - year_start).dt.days
)

# ===============================================================
# 3. Seasonal cycle (sin/cos)
# ===============================================================
print("Computing day-of-year sine and cosine...")

# Use 365.25 if you want a tiny leap-year correction; 365 is usually fine.
df["doy_radians"] = 2 * np.pi * df["doy"] / 365.0
df["doy_sin"] = np.sin(df["doy_radians"])
df["doy_cos"] = np.cos(df["doy_radians"])

# ===============================================================
# 4. Cartesian coordinates from 1° grid centers
# ===============================================================
print("Computing Cartesian coordinates from lat/lon...")

lat_rad = np.radians(df["lat_center"].astype(float))
lon_rad = np.radians(df["lon_center"].astype(float))

df["x_cart"] = np.cos(lat_rad) * np.cos(lon_rad)
df["y_cart"] = np.cos(lat_rad) * np.sin(lon_rad)
df["z_cart"] = np.sin(lat_rad)

# ===============================================================
# 5. GEBCO bathymetry at grid centers
# ===============================================================
print("Loading GEBCO bathymetry...")

bathy = xr.open_dataset(
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/bathymetry/GEBCO_2024.nc"
)

# Normalize bathy lon to [-180, 180] if needed
if float(bathy["lon"].max()) > 180:
    bathy = bathy.assign_coords(lon=((bathy["lon"] + 180) % 360) - 180)

lats = xr.DataArray(df["lat_center"].values, dims="points")
lons = xr.DataArray(df["lon_center"].values, dims="points")

depth = bathy["elevation"].sel(lat=lats, lon=lons, method="nearest").values
df["bottom_depth_m"] = -depth

before = len(df)
df = df[df["bottom_depth_m"] > 0].copy()
after = len(df)

print(f"Dropped {before - after:,} land/invalid bathymetry points.")

# ===============================================================
# 6. Save final engineered dataset
# ===============================================================
output_path = ("/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/independent_validation/pco2_independent_validation_data_regridded_SLA_RRS_PAR_WIND_ATM_CO2_feature_engineered_monthly_1deg.csv")

print(f"Saving engineered dataset to:\n{output_path}")
df.to_csv(output_path, index=False)

print("\nFeature engineering complete.")
print(f"Final dataset size: {len(df):,} rows")
