#!/usr/bin/env python3
# ============================================================
# Build monthly transects (lat x time) averaged over 140–143E
# for:
#   1) SOCAT-only model predictions
#   2) SOCAT+FLOATS model predictions (weighted)
#
# Output NetCDF variables (dims: time, lat):
#   temp, sal, wind, pco2_pred, pco2_std, flux
#
# Notes:
# - lon is averaged (140–143E), so output has no lon dim.
# - flux is bulk air-sea CO2 flux in gC m-2 s-1 (same as your function)
# - Feb 2023 is removed to match your pipeline
# ============================================================

import numpy as np
import pandas as pd
import xarray as xr
import pyseaflux


# -------------------------
# User settings
# -------------------------
LON_MIN_E = 140.0
LON_MAX_E = 143.0
LAT_MAX = -30.0  # keep lat from minimum available up to -30
LAT_RES_DEG = 1.0
LON_RES_DEG = 1.0

# Input files
SOCAT_PRED_CSV = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "PREDICTION_MATRIX_v2/matrix_predictions/pco2_prediction_matrix_SOCAT_only_MCensemble.csv"
)
COMBINED_WEIGHTED_PRED_CSV = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "PREDICTION_MATRIX_v2/matrix_predictions/pco2_prediction_matrix_SOCAT_AND_FLOATS_weighted_MCensemble.csv"
)

# Output files
OUT_SOCAT_NC = "transect_SOCAT_only_140E_143E_lat_min_to30S_monthly.nc"
OUT_COMBINED_W_NC = "transect_SOCATplusFLOATS_weighted_140E_143E_lat_min_to30S_monthly.nc"


# -------------------------
# Helpers
# -------------------------
def to_0360(lon):
    """Convert longitude to [0, 360)."""
    lon = np.asarray(lon, dtype=float)
    return np.mod(lon, 360.0)

def drop_feb_2023(df, time_col="month_center"):
    t = pd.to_datetime(df[time_col])
    return df.loc[~((t.dt.year == 2023) & (t.dt.month == 2))].copy()

def flux_from_pco2(temp, sal, pco2_sw, wind, atm_co2):
    """
    Bulk air-sea CO2 flux (gC m-2 s-1), matching your earlier code.
    """
    pres = 1013.25
    k_w = pyseaflux.gas_transfer_velocity.k_Ho06(wind, temp)
    flux_mol = pyseaflux.flux_calculations.flux_bulk(
        temp, sal, pco2_sw, atm_co2, pres, k_w
    )
    return flux_mol * 12.01  # gC m-2 s-1

def build_transect_nc(
    df,
    *,
    out_nc,
    source_name,
    time_col="month_center",
    lat_col="latitude",
    lon_col="longitude",
    temp_col="temperature",
    sal_col="salinity",
    wind_col="wind_speed",
    pco2_pred_col="ensemble_pco2_pred",
    pco2_std_col="ensemble_pco2_std",
    lon_min_e=LON_MIN_E,
    lon_max_e=LON_MAX_E,
    lat_max=LAT_MAX,
    lat_res_deg=LAT_RES_DEG,
    lon_res_deg=LON_RES_DEG,
):
    """
    Create (time, lat) monthly transect averaged over lon band.
    Saves to NetCDF and returns xarray.Dataset.
    """

    d = df.copy()

    # Monthly time stamp (month start) for clean grouping / CF-friendly time axis
    d["time"] = pd.to_datetime(d[time_col]).dt.to_period("M").dt.to_timestamp()

    # Longitudes in 0..360 so 140–143E is unambiguous
    d["_lon0360"] = to_0360(d[lon_col].values)

    # 1-degree binning (same spirit as your pipeline)
    d["lat_bin"] = np.floor(d[lat_col].astype(float).values / lat_res_deg) * lat_res_deg
    d["lon_bin"] = np.floor(d["_lon0360"].astype(float).values / lon_res_deg) * lon_res_deg

    lat_min = np.nanmin(d["lat_bin"].values)

    # Subset to lon band and lat range
    m = (
        (d["lat_bin"] >= lat_min) &
        (d["lat_bin"] <= lat_max) &
        (d["lon_bin"] >= lon_min_e) &
        (d["lon_bin"] <= lon_max_e)
    )

    needed = ["time", "lat_bin", "lon_bin", temp_col, sal_col, wind_col, pco2_pred_col, pco2_std_col, "atm_co2"]
    missing = [c for c in needed if c not in d.columns]
    if missing:
        raise ValueError(f"{source_name}: missing required columns: {missing}")

    d = d.loc[m, needed].dropna().copy()

    # Compute flux (gC m-2 s-1) at native rows, then aggregate
    d["flux"] = flux_from_pco2(
        temp=d[temp_col].astype(float).values,
        sal=d[sal_col].astype(float).values,
        pco2_sw=d[pco2_pred_col].astype(float).values,
        wind=d[wind_col].astype(float).values,
        atm_co2=d["atm_co2"].astype(float).values,
    )

    # Average within each (time,lat,lon) bin first
    g = (
        d.groupby(["time", "lat_bin", "lon_bin"], as_index=False)
         .agg(
             temp=(temp_col, "mean"),
             sal=(sal_col, "mean"),
             wind=(wind_col, "mean"),
             pco2_pred=(pco2_pred_col, "mean"),
             pco2_std=(pco2_std_col, "mean"),
             flux=("flux", "mean"),
             n=("flux", "size"),
         )
    )

    # Then average across lon bins -> (time,lat)
    t = (
        g.groupby(["time", "lat_bin"], as_index=False)
         .agg(
             temp=("temp", "mean"),
             sal=("sal", "mean"),
             wind=("wind", "mean"),
             pco2_pred=("pco2_pred", "mean"),
             pco2_std=("pco2_std", "mean"),
             flux=("flux", "mean"),
             n_cells=("n", "sum"),
         )
         .rename(columns={"lat_bin": "lat"})
         .sort_values(["time", "lat"])
    )

    # Build xarray Dataset
    ds = (
        t.set_index(["time", "lat"])
         .to_xarray()
         .transpose("time", "lat")
    )

    # Attributes
    ds["lat"].attrs.update({"standard_name": "latitude", "units": "degrees_north"})
    ds["time"].attrs.update({"standard_name": "time"})

    ds["temp"].attrs.update({"long_name": "Sea surface temperature used in flux", "units": "degC"})
    ds["sal"].attrs.update({"long_name": "Sea surface salinity used in flux", "units": "1e-3"})  # adjust if needed
    ds["wind"].attrs.update({"long_name": "Wind speed used in flux", "units": "m s-1"})
    ds["pco2_pred"].attrs.update({"long_name": "Predicted surface pCO2", "units": "uatm"})
    ds["pco2_std"].attrs.update({"long_name": "Ensemble pCO2 std (mean over lon band)", "units": "uatm"})
    ds["flux"].attrs.update({"long_name": "Air-sea CO2 flux (bulk)", "units": "gC m-2 d-1"})
    ds["n_cells"].attrs.update({"long_name": "Number of contributing grid cells in lon band"})

    ds.attrs.update({
        "source": source_name,
        "lon_band_E": f"{lon_min_e}–{lon_max_e}",
        "lat_range_degN": f"{float(ds['lat'].min())}–{float(ds['lat'].max())}",
        "note": "Lon-averaged transect (NOT area-integrated). Monthly means over 140–143E.",
    })

    # Compression
    encoding = {v: {"zlib": True, "complevel": 4} for v in ds.data_vars}
    ds.to_netcdf(out_nc, encoding=encoding)
    return ds


# ============================================================
# MAIN
# ============================================================

# ---------- Load SOCAT-only predictions ----------
print("Loading SOCAT-only predictions...")
pred_socat = pd.read_csv(SOCAT_PRED_CSV)
pred_socat["month_center"] = pd.to_datetime(pred_socat["month_center"])
pred_socat = drop_feb_2023(pred_socat, "month_center")

# SOCAT-only uses SOCAT_temperature / SOCAT_salinity per your earlier script
print("Building SOCAT-only lon-averaged transect NetCDF...")
ds_socat = build_transect_nc(
    pred_socat,
    out_nc=OUT_SOCAT_NC,
    source_name="SOCAT-only predictions",
    time_col="month_center",
    lat_col="latitude",
    lon_col="longitude",
    temp_col="SOCAT_temperature",
    sal_col="SOCAT_salinity",
    wind_col="wind_speed",
    pco2_pred_col="ensemble_pco2_pred",
    pco2_std_col="ensemble_pco2_std",
)

print(f"  wrote: {OUT_SOCAT_NC}")

# ---------- Load SOCAT+FLOATS weighted predictions ----------
print("Loading SOCAT+FLOATS (weighted) predictions...")
pred_comb_w = pd.read_csv(COMBINED_WEIGHTED_PRED_CSV)
pred_comb_w["month_center"] = pd.to_datetime(pred_comb_w["month_center"])
pred_comb_w = drop_feb_2023(pred_comb_w, "month_center")

# Combined model typically uses temperature / salinity (as in your combined flux function)
print("Building SOCAT+FLOATS weighted lon-averaged transect NetCDF...")
ds_comb_w = build_transect_nc(
    pred_comb_w,
    out_nc=OUT_COMBINED_W_NC,
    source_name="SOCAT+FLOATS predictions (weighted)",
    time_col="month_center",
    lat_col="latitude",
    lon_col="longitude",
    temp_col="temperature",
    sal_col="salinity",
    wind_col="wind_speed",
    pco2_pred_col="ensemble_pco2_pred",
    pco2_std_col="ensemble_pco2_std",
)

print(f"  wrote: {OUT_COMBINED_W_NC}")

print("Done.")