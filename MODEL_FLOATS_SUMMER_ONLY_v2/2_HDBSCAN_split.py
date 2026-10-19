# ===============================================================
# FLOATS 1×1° Monthly Data Splitting with HDBSCAN
# Cluster-stratified Train/Val/Test (no independent validation)
# Louise Delaigue – 2025
# Summer-only variant
# ===============================================================

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import hdbscan

# === Parameters ===
RANDOM_STATE = 42
BASE_PATH = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_v2/processing_steps/"
DATA_PATH = BASE_PATH + "BGC-Argo_SO_feature_engineered_monthly_1deg.csv"
OUTPUT_PATH = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_SUMMER_ONLY_v2/processing_steps/"

# Name prefix for output files
VAR_MLP = "FLOATS_pco2"  # avoids confusion with SOCAT runs

# Southern domain rule on 0.5° centers:
LAT_CENTER_MAX = -30.5

# --- Season definition ---
# Choose one: "DJF" (austral summer), "NDJFM" (extended austral summer), "JJA"
SUMMER_DEF = "DJF"

# ===============================================================
# 1. Load Data
# ===============================================================
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.loc[:, ~df.columns.duplicated()]

df.rename(columns={"lat_center": "latitude", "lon_center": "longitude"}, inplace=True)

df["month_center"] = pd.to_datetime(df["month_center"], errors="coerce")
df.dropna(subset=["month_center", "latitude", "longitude"], inplace=True)

# Normalize to month START (pipeline convention)
df["month_center"] = df["month_center"].dt.to_period("M").dt.to_timestamp(how="start")

# ===============================================================
# 1b. Seasonal filter (apply BEFORE temporal features + HDBSCAN)
# ===============================================================
month = df["month_center"].dt.month

if SUMMER_DEF == "DJF":
    summer_mask = month.isin([12, 1, 2])
elif SUMMER_DEF == "NDJFM":
    summer_mask = month.isin([11, 12, 1, 2, 3])
elif SUMMER_DEF == "JJA":
    summer_mask = month.isin([6, 7, 8])
else:
    raise ValueError(f"Unknown SUMMER_DEF: {SUMMER_DEF}")

before = len(df)
df = df[summer_mask].copy()
print(f"Rows after {SUMMER_DEF} filter: {len(df):,} (dropped {before - len(df):,})")

# Apply spatial filter on grid centers
df = df[df["latitude"] <= LAT_CENTER_MAX].copy()

print(f"Rows after spatial/time filtering: {len(df):,}")

# ===============================================================
# 2. Temporal Features (only add missing ones)
# ===============================================================
print("Computing temporal features...")
min_date = df["month_center"].min()
df["days_since_start"] = (df["month_center"] - min_date).dt.days

if "doy" not in df.columns:
    df["doy"] = df["month_center"].dt.dayofyear
if "decimal_year" not in df.columns:
    df["decimal_year"] = df["month_center"].dt.year + (df["doy"] - 1) / 365.0
if "year" not in df.columns:
    df["year"] = df["month_center"].dt.year

# ===============================================================
# 3. Require all predictor columns
# ===============================================================
REQUIRED_COLUMNS = [
    "longitude", "latitude",
    "FLOATS_pco2_median", "FLOATS_salinity_median", "FLOATS_temperature_median",
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
print(f"Dropped {initial_rows - len(df):,} rows due to missing required values.")

# ===============================================================
# 4. HDBSCAN Clustering
# ===============================================================
FEATURE_NAMES = ["longitude", "latitude", "decimal_year", "doy", "days_since_start"]

print("Scaling for HDBSCAN...")
scaler = StandardScaler()
df.loc[:, FEATURE_NAMES] = scaler.fit_transform(df[FEATURE_NAMES])

print("Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5, core_dist_n_jobs=1)
df["cluster"] = clusterer.fit_predict(df[FEATURE_NAMES])

print(f"Found {df['cluster'].nunique()} clusters (including noise if -1 present).")

# ===============================================================
# 5. 70/15/15 Split (stratified by cluster)
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
for d in (train_df, val_df, test_df):
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
print(f"HDBSCAN SIMPLE SPLITTING DONE ({SUMMER_DEF} only).")
