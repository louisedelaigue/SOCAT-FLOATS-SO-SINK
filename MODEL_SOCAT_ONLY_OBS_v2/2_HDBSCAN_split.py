# ===============================================================
# SOCAT 1×1° Monthly Data Splitting with HDBSCAN
# NO INDEPENDENT VALIDATION VERSION — Simple 70/15/15 Split
# Louise Delaigue – 2025
# ===============================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import hdbscan

# === Parameters ===
RANDOM_STATE = 42
BASE_PATH = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
DATA_PATH = BASE_PATH + "SOCATv2025_SO_feature_engineered_monthly_1deg_pCO2.csv"
OUTPUT_PATH = BASE_PATH
VAR_MLP = "pco2"

LAT_CENTER_MAX = -30.5  # 30S and south on 1° grid centers (… -30.5, -31.5, …)

# ===============================================================
# 1. Load Data
# ===============================================================
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.loc[:, ~df.columns.duplicated()]

df.rename(columns={"lat_center": "latitude", "lon_center": "longitude"}, inplace=True)

df["month_center"] = pd.to_datetime(df["month_center"], errors="coerce")
df.dropna(subset=["month_center"], inplace=True)

# Normalize to month-start (pipeline convention)
df["month_center"] = df["month_center"].dt.to_period("M").dt.to_timestamp(how="start")

# Southern Ocean filter on grid centers
df = df[df["latitude"] <= LAT_CENTER_MAX]

# ===============================================================
# 2. Temporal Features (only add what isn't already present)
# ===============================================================
print("Computing temporal features...")
min_date = df["month_center"].min()
df["days_since_start"] = (df["month_center"] - min_date).dt.days

# If these already exist from feature engineering, keep them
if "doy" not in df.columns:
    df["doy"] = df["month_center"].dt.dayofyear
if "decimal_year" not in df.columns:
    # consistent-ish fallback: year + fraction of year using doy
    df["decimal_year"] = df["month_center"].dt.year + (df["doy"] - 1) / 365.0
if "year" not in df.columns:
    df["year"] = df["month_center"].dt.year

# ===============================================================
# 3. Require all predictor columns
# ===============================================================
REQUIRED_COLUMNS = [
    "longitude", "latitude",
    "fco2_rec_median", "pco2", "SOCAT_salinity", "SOCAT_temperature",
    "MLD", "sla_median",
    "rrs412", "rrs443", "rrs490", "rrs555", "rrs670",
    "PAR_mean",
    "u10", "v10", "wind_speed",
    "atm_co2",
    "decimal_year", "doy", "doy_sin", "doy_cos",
    "x_cart", "y_cart", "z_cart",
    "bottom_depth_m",
    "month_center",
    "days_since_start",
]

initial_rows = len(df)
df.dropna(subset=REQUIRED_COLUMNS, inplace=True)
print(f"Dropped {initial_rows - len(df):,} rows due to missing values.")

# ===============================================================
# 4. HDBSCAN Clustering
# ===============================================================
FEATURE_NAMES = ["longitude", "latitude", "decimal_year", "doy", "days_since_start"]

print("Scaling for HDBSCAN...")
scaler = StandardScaler()
df[FEATURE_NAMES] = scaler.fit_transform(df[FEATURE_NAMES])

print("Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5, core_dist_n_jobs=1)
df["cluster"] = clusterer.fit_predict(df[FEATURE_NAMES])
print(f"Found {df['cluster'].nunique()} clusters (including noise if -1 present).")

# ===============================================================
# 5. Simple 70/15/15 Split (stratified by cluster)
# ===============================================================
print("Train/val/test splitting (stratified by cluster)...")

train_df, temp_df = train_test_split(
    df,
    train_size=0.7,
    random_state=RANDOM_STATE,
    stratify=df["cluster"],
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=temp_df["cluster"],
)

# ===============================================================
# 6. Reverse scaling
# ===============================================================
print("Reversing scaling...")
for d in [train_df, val_df, test_df]:
    d.loc[:, FEATURE_NAMES] = scaler.inverse_transform(d[FEATURE_NAMES])

# ===============================================================
# 7. Save scaler and splits
# ===============================================================
joblib.dump(scaler, OUTPUT_PATH + "split_scaler.joblib")
joblib.dump(min_date, OUTPUT_PATH + "split_min_date.joblib")

splits = {
    "train_data": train_df,
    "validation_data": val_df,
    "test_data": test_df,
}

for name, data in splits.items():
    path = f"{OUTPUT_PATH}{VAR_MLP}_{name}.csv"
    data.to_csv(path, index=False)
    print(f"Saved {name} → {path}")

print("---------------------------")
print("HDBSCAN SIMPLE SPLITTING DONE.")
