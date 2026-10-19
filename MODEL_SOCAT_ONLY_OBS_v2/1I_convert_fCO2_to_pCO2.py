# ===============================================================
# Convert SOCAT fCO₂ → pCO₂ using PyCO2SYS (SOCAT-only thermodynamics)
# Louise Delaigue – 2025
# ===============================================================

import pandas as pd
import numpy as np
import PyCO2SYS as pyco2

# === Paths ===
input_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
    "SOCATv2025_SO_feature_engineered_monthly_1deg.csv"
)

output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
    "SOCATv2025_SO_feature_engineered_monthly_1deg_pCO2.csv"
)

# === Load Data ===
print("[INFO] Loading dataset...")
df = pd.read_csv(input_path, low_memory=False)
print(f"[INFO] Loaded {len(df):,} rows.")

# === Filter Southern Ocean on 1° grid centers (30S and south) ===
df = df[df["lat_center"] <= -30.5]
print(f"[INFO] Southern Ocean subset: {len(df):,} rows (lat_center <= -30.5)")

# === Required SOCAT variables ===
required_vars = [
    "fco2_rec_median",     # fCO2 (µatm)
    "SOCAT_temperature",   # temperature (°C)
    "SOCAT_salinity"       # salinity
]

print("[INFO] Dropping rows missing SOCAT thermodynamic variables...")
df = df.dropna(subset=required_vars).copy()
print(f"[INFO] Remaining rows: {len(df):,}")

# Make sure numeric
for c in required_vars:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=required_vars).copy()

# ===============================================================
#  RUN PYCO2SYS USING SOCAT VARIABLES ONLY
# ===============================================================
print("[INFO] Running PyCO2SYS...")

results = pyco2.sys(
    par1=df["fco2_rec_median"].values,
    par1_type=5,  # fCO2 input (µatm)
    temperature=df["SOCAT_temperature"].values,
    salinity=df["SOCAT_salinity"].values,
    pressure=4.79,     # SOCAT average surface sampling pressure (dbar)
    opt_pH_scale=1,
    opt_k_carbonic=10,
    opt_k_bisulfate=1,
    opt_total_borate=1,
    opt_buffers_mode=0
)

# Add pCO2 to DataFrame
df["pco2"] = results["pCO2"]

print("[INFO] Conversion complete.")
print(df[["fco2_rec_median", "pco2"]].head())

# === Save final dataset ===
df.to_csv(output_path, index=False)

print(f"[INFO] Saved output to: {output_path}")
print("[DONE]")
