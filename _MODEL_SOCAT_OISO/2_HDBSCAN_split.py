# ===============================================================
# SOCAT 1×1° Monthly Data Splitting with HDBSCAN
# NO EXPOCODE VERSION — Cluster-Based Independent Validation
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
N_CLUSTERS_TOTAL = 12
BASE_PATH = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
DATA_PATH = BASE_PATH + "SOCATv2025_SO_feature_engineered_monthly_1deg_pCO2.csv"
OUTPUT_PATH = BASE_PATH
VAR_MLP = "pco2"
MAX_CLUSTER_SIZE = 120    # only use clusters smaller than this for validation
MIN_DISTANCE_KM = 800      # minimum spacing between independent clusters (centroid–centroid)

# ===============================================================
# 1. Load Data
# ===============================================================
print("Loading data...")
df = pd.read_csv(DATA_PATH, low_memory=False)
df = df.loc[:, ~df.columns.duplicated()]

# Use lat/lon grid centres
df.rename(columns={"lat_center": "latitude", "lon_center": "longitude"}, inplace=True)
df["month_center"] = pd.to_datetime(df["month_center"], errors="coerce")
df.dropna(subset=["month_center"], inplace=True)

# Restrict Southern Ocean range
df = df[df["latitude"] <= -35]

# ===============================================================
# 2. Temporal Features
# ===============================================================
print("Computing temporal features...")
min_date = df["month_center"].min()
df["days_since_start"] = (df["month_center"] - min_date).dt.days
df["doy"] = df["month_center"].dt.dayofyear
df["decimal_year"] = df["month_center"].dt.year + (df["doy"] - 1) / 365.25
df["year"] = df["month_center"].dt.year

# ===============================================================
# 3. Require all predictor columns
# ===============================================================
REQUIRED_COLUMNS = [
    # Coordinates
    "longitude", "latitude",

    # SOCAT + carbonate system
    "fco2_rec_median", "pco2", "SOCAT_salinity", "SOCAT_temperature",

    # ARMOR + SLA
    "MLD", "sla_median",

    # RRS
    "rrs412", "rrs443", "rrs490", "rrs555", "rrs670",

    # PAR
    "PAR_mean",

    # Wind
    "u10", "v10", "wind_speed",

    # Atmospheric CO2
    "atm_co2",

    # Feature engineering
    "decimal_year", "doy", "doy_sin", "doy_cos",
    "x_cart", "y_cart", "z_cart",

    # Bathymetry
    "bottom_depth_m",

    # Time
    "month_center",
]

initial_rows = len(df)
df.dropna(subset=REQUIRED_COLUMNS, inplace=True)
final_rows = len(df)
print(f"Dropped {initial_rows - final_rows} rows due to missing values.")

# ===============================================================
# 4. HDBSCAN Clustering
# ===============================================================
FEATURE_NAMES = ["longitude", "latitude", "decimal_year", "doy", "days_since_start"]

print("Scaling for HDBSCAN...")
scaler = StandardScaler()
df[FEATURE_NAMES] = scaler.fit_transform(df[FEATURE_NAMES])

print("Running HDBSCAN clustering...")
clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
df["cluster"] = clusterer.fit_predict(df[FEATURE_NAMES])
unique_clusters = np.unique(df["cluster"])
print(f"Found {len(unique_clusters)} clusters.")

# ===============================================================
# 5. Option 3: small clusters + spatial separation
# ===============================================================
print("Filtering for small clusters (Option 3)...")
cluster_sizes = df["cluster"].value_counts()

# Keep clusters smaller than threshold
valid_small_clusters = cluster_sizes[cluster_sizes < MAX_CLUSTER_SIZE].index.tolist()
df_small = df[df["cluster"].isin(valid_small_clusters)].copy()
print(f"Kept {len(valid_small_clusters)} clusters with size < {MAX_CLUSTER_SIZE}.")

# --- Assign 5-year periods ---
bins = [1994, 1999, 2004, 2009, 2014, 2019, 2024]
labels = ["1995–1999", "2000–2004", "2005–2009", "2010–2014", "2015–2019", "2020–2024"]
df_small["period"] = pd.cut(df_small["year"], bins=bins, labels=labels)

# --- Compute cluster centroids (in lat/lon) for small clusters only ---
cluster_centroids = (
    df_small.groupby("cluster")[["latitude", "longitude"]]
    .mean()
    .reset_index()
)

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two points."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(np.radians(lat1))
        * np.cos(np.radians(lat2))
        * np.sin(dlon / 2.0) ** 2
    )
    return 2 * R * np.arcsin(np.sqrt(a))

def is_far_enough(cluster_id, selected_ids):
    """Check centroid distance from all already selected clusters."""
    row = cluster_centroids[cluster_centroids["cluster"] == cluster_id].iloc[0]
    lat1, lon1 = row["latitude"], row["longitude"]

    for cid in selected_ids:
        row2 = cluster_centroids[cluster_centroids["cluster"] == cid].iloc[0]
        dist = haversine(lat1, lon1, row2["latitude"], row2["longitude"])
        if dist < MIN_DISTANCE_KM:
            return False
    return True

print("Selecting independent validation clusters with spatial separation...")
selected_clusters = []
rng = np.random.default_rng(RANDOM_STATE)

# --- 1 cluster per 5-year period, enforcing min distance ---
for period in labels:
    period_clusters = df_small[df_small["period"] == period]["cluster"].unique()
    if period_clusters.size == 0:
        continue

    candidates = rng.permutation(period_clusters)
    chosen = None

    for c in candidates:
        if is_far_enough(c, selected_clusters):
            chosen = c
            break

    # If none satisfy the distance constraint, fall back to first candidate
    if chosen is None:
        chosen = candidates[0]

    selected_clusters.append(chosen)

print(f"After period-based selection: {selected_clusters}")

# --- Extra clusters (random within small clusters, also spaced) ---
remaining_clusters = [c for c in valid_small_clusters if c not in selected_clusters]
remaining_clusters = rng.permutation(remaining_clusters)

needed_extra = N_CLUSTERS_TOTAL - len(selected_clusters)
for c in remaining_clusters:
    if is_far_enough(c, selected_clusters):
        selected_clusters.append(c)
    if len(selected_clusters) == N_CLUSTERS_TOTAL:
        break

# If still not enough (very strict spacing), top up without distance filter
if len(selected_clusters) < N_CLUSTERS_TOTAL:
    print("Not enough spatially separated clusters, topping up without distance constraint.")
    remaining_clusters = [c for c in valid_small_clusters if c not in selected_clusters]
    needed = N_CLUSTERS_TOTAL - len(selected_clusters)
    selected_clusters.extend(list(remaining_clusters[:needed]))

print(f"Final selected independent clusters: {selected_clusters}")

# ===============================================================
# 6. Build splits
# ===============================================================
indep_df = df[df["cluster"].isin(selected_clusters)].copy()
main_df = df[~df["cluster"].isin(selected_clusters)].copy()

print("Train/val/test splitting (stratified by cluster)...")
train_data, temp_data = train_test_split(
    main_df,
    train_size=0.7,
    random_state=RANDOM_STATE,
    stratify=main_df["cluster"],
)

val_data, test_data = train_test_split(
    temp_data,
    test_size=0.5,
    random_state=RANDOM_STATE,
    stratify=temp_data["cluster"],
)

# ===============================================================
# 7. Reverse scaling for saved splits
# ===============================================================
print("Reversing scaling...")
for d in [train_data, val_data, test_data, indep_df]:
    d.loc[:, FEATURE_NAMES] = scaler.inverse_transform(d[FEATURE_NAMES])

# ===============================================================
# 8. Save scaler and splits
# ===============================================================
joblib.dump(scaler, OUTPUT_PATH + "split_scaler.joblib")
joblib.dump(min_date, OUTPUT_PATH + "split_min_date.joblib")

datasets = {
    "train_data": train_data,
    "test_data": test_data,
    "validation_data": val_data,
    "independent_validation_data": indep_df,
}

for name, data in datasets.items():
    path = f"{OUTPUT_PATH}{VAR_MLP}_{name}.csv"
    data.to_csv(path, index=False)
    print(f"Saved {name} → {path}")

print("---------------")
print("HDBSCAN SPLITTING DONE.")
