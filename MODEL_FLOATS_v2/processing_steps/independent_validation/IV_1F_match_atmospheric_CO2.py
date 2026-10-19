# ===============================================================
# FLOATS–MLD–SLA–RRS–PAR–WIND–ATM CO₂ Monthly Merge
# Atmospheric CO₂ from Cape Grim (in situ) matched to FLOATS grid
# Louise Delaigue – 2025
# ===============================================================

import os
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ===============================================================
# Paths
# ===============================================================
floats_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_v2/processing_steps/independent_validation/"
    "pco2_independent_validation_data_TSMLD_SLA_RRS_PAR_WIND.csv"
)


capegrim_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/CapeGrim_CO2_clean.csv"
)

output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_v2/processing_steps/independent_validation/"
    "pco2_independent_validation_data_TSMLD_SLA_RRS_PAR_WIND_ATM_CO2.csv"
)

# ===============================================================
# 1. Load FLOATS dataset
# ===============================================================
print("Loading FLOATS monthly dataset...")
df = pd.read_csv(floats_path)

df["month_center"] = pd.to_datetime(df["month_center"], errors="coerce")
df.dropna(subset=["month_center"], inplace=True)

# Normalize to month START (pipeline convention)
df["month_center"] = df["month_center"].dt.to_period("M").dt.to_timestamp(how="start")
df["month_key"] = df["month_center"].dt.to_period("M")

print(f"FLOATS rows loaded: {len(df):,}")

# Clean old time columns if they exist
for col in ["year", "month", "day", "date"]:
    if col in df.columns:
        df.drop(columns=[col], inplace=True, errors="ignore")

# ===============================================================
# 2. Load Cape Grim CO₂
# ===============================================================
print("Loading Cape Grim CO₂...")
cg = pd.read_csv(capegrim_path)

# Rename CO₂ column for clarity
if "CO2(ppm)" in cg.columns:
    cg.rename(columns={"CO2(ppm)": "co2"}, inplace=True)

# Remove missing or invalid values
cg = cg[cg["co2"].notna()].copy()

# Build a proper datetime column from YYYY/MM/DD
cg["date"] = pd.to_datetime(
    dict(year=cg["YYYY"], month=cg["MM"], day=cg["DD"]),
    errors="coerce"
)
cg.dropna(subset=["date"], inplace=True)

# Normalize to month START
cg["month_center"] = cg["date"].dt.to_period("M").dt.to_timestamp(how="start")
cg["month_key"] = cg["month_center"].dt.to_period("M")

print(f"Cape Grim daily records loaded: {len(cg):,}")

# Monthly aggregation (choose mean or median)
cg_monthly = (
    cg.groupby("month_key", as_index=True)["co2"]
    .mean()
)

print(f"Cape Grim monthly records: {len(cg_monthly):,}")

# ===============================================================
# 3. Temporal merge
# ===============================================================
print("Matching Cape Grim CO₂ to FLOATS...")
df["atm_co2"] = df["month_key"].map(cg_monthly)

matched = df["atm_co2"].notna().sum()
print(f"Matched {matched:,} FLOATS grid rows ({matched/len(df)*100:.2f}%)")

# Remove helper key
df.drop(columns=["month_key"], inplace=True)

# ===============================================================
# 4. Save output
# ===============================================================
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print("\nDone.")
print(f"Final dataset saved to: {output_path}")
print(f"Rows: {len(df):,}")
