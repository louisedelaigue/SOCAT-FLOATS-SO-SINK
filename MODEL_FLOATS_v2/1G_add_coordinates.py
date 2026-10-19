# ===============================================================
# Feature Engineering for FLOATS pCO₂ Modeling (Pipeline Version)
# Louise Delaigue – 2025
# ===============================================================

"""
Loads the fully merged FLOATS 1°×1° monthly dataset
(FLOATS + MLD + SLA + RRS + PAR + WIND + ATM CO₂)
and computes engineered features:

- decimal year (from month_center)
- day-of-year sine/cosine (seasonal cycle)
- 3D Cartesian coordinates (x, y, z)
- GEBCO bathymetry at the 1° grid point

Pipeline rules:
• month_center aligned to month-start (YYYY-MM-01)
• Only use lat_center / lon_center
• atm_co2 is kept as Cape Grim dry-air xCO2 (ppm), unconverted, as in
  the prediction matrix. Conversion to atmospheric pCO2 (water-vapour
  correction, local sea-level pressure) is done only in the flux
  calculation, not for the MLP predictor.
"""

import pandas as pd
import numpy as np
import xarray as xr


# ===============================================================
# Paths
# ===============================================================
input_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "MODEL_FLOATS_v2/processing_steps/"
    "BGC-Argo_SO_clean_MLD_monthly_regridded_SLA_RRS_PAR_WIND_ATM_CO2.csv"
)

output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "MODEL_FLOATS_v2/processing_steps/"
    "BGC-Argo_SO_feature_engineered_monthly_1deg.csv"
)


# ===============================================================
# 1. Load dataset
# ===============================================================
print("Loading dataset...")

df = pd.read_csv(input_path, low_memory=False)

print(f"Loaded {len(df):,} rows.")


# ---------------------------------------------------------------
# Ensure month_center is correctly formatted (month-start)
# ---------------------------------------------------------------
df["month_center"] = pd.to_datetime(
    df["month_center"],
    errors="coerce"
)

df.dropna(subset=["month_center"], inplace=True)

df["month_center"] = (
    df["month_center"]
    .dt.to_period("M")
    .dt.to_timestamp(how="start")
)


# Extract time components
df["year"] = df["month_center"].dt.year
df["month"] = df["month_center"].dt.month
df["doy"] = df["month_center"].dt.dayofyear


# ===============================================================
# 2. Decimal Year
# ===============================================================
print("\nComputing decimal year...")

year_start = pd.to_datetime(
    df["year"].astype(str) + "-01-01"
)

next_year_start = pd.to_datetime(
    (df["year"] + 1).astype(str) + "-01-01"
)

df["decimal_year"] = df["year"] + (
    (df["month_center"] - year_start).dt.days
    / (next_year_start - year_start).dt.days
)


# ===============================================================
# 3. Seasonal cycle (sin/cos)
# ===============================================================
print("Computing day-of-year sine and cosine...")

df["doy_radians"] = (
    2 * np.pi * df["doy"] / 365.0
)

df["doy_sin"] = np.sin(df["doy_radians"])
df["doy_cos"] = np.cos(df["doy_radians"])


# ===============================================================
# 4. Cartesian coordinates from 1° grid centers
# ===============================================================
print("Computing Cartesian coordinates from lat/lon...")

lat_rad = np.radians(
    df["lat_center"].astype(float)
)

lon_rad = np.radians(
    df["lon_center"].astype(float)
)

df["x_cart"] = (
    np.cos(lat_rad) * np.cos(lon_rad)
)

df["y_cart"] = (
    np.cos(lat_rad) * np.sin(lon_rad)
)

df["z_cart"] = np.sin(lat_rad)


# ===============================================================
# 5. GEBCO bathymetry at grid centers
# ===============================================================
print("Loading GEBCO bathymetry...")

bathy = xr.open_dataset(
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "data/bathymetry/GEBCO_2024.nc"
)


# GEBCO longitude may be 0–360; convert to -180–180 if needed
if float(bathy["lon"].max()) > 180:
    bathy = bathy.assign_coords(
        lon=((bathy["lon"] + 180) % 360) - 180
    )

    bathy = bathy.sortby("lon")


# Query GEBCO at FLOAT grid centers
lats = xr.DataArray(
    df["lat_center"].values,
    dims="points"
)

lons = xr.DataArray(
    df["lon_center"].values,
    dims="points"
)

depth = (
    bathy["elevation"]
    .sel(
        lat=lats,
        lon=lons,
        method="nearest"
    )
    .values
)

# GEBCO elevation:
# negative = below sea level
df["bottom_depth_m"] = -depth


# Remove land / invalid bathymetry points
before = len(df)

df = df[
    df["bottom_depth_m"] > 0
].copy()

after = len(df)

print(
    f"Dropped {before - after:,} "
    f"land/invalid bathymetry points."
)


# ===============================================================
# 6. Save final engineered dataset
# ===============================================================
print(
    f"\nSaving engineered dataset to:\n"
    f"{output_path}"
)

df.to_csv(
    output_path,
    index=False
)


# ===============================================================
# Summary
# ===============================================================
print("\nFeature engineering complete.")
print(f"Final dataset size: {len(df):,} rows")
print("atm_co2 contains Cape Grim dry-air xCO₂ (ppm), unconverted.")