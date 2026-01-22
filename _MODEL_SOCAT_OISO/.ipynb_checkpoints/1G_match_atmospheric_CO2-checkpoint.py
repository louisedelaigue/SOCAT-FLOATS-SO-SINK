# ===============================================================
# SOCAT–MLD–SLA–RRS–PAR–WIND–ATM CO₂ Monthly Merge
# Atmospheric CO₂ from Cape Grim (in situ) matched to SOCAT grid
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
socat_path = (
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/"
    "MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
    "SOCATv2025_SO_clean_MLD_SLA_RRS_PAR_WIND_monthly_1deg.csv"
)

capegrim_path = (
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/data/CapeGrim_CO2_clean.csv"
)

output_path = (
    "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/"
    "MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
    "SOCATv2025_SO_clean_MLD_SLA_RRS_PAR_WIND_ATM_CO2_monthly_1deg.csv"
)

# ===============================================================
# 1. Load SOCAT dataset
# ===============================================================
print("Loading SOCAT monthly dataset...")
df = pd.read_csv(socat_path)

df["month_center"] = pd.to_datetime(df["month_center"], errors="coerce")
df["month_center"] = df["month_center"].dt.to_period("M").dt.to_timestamp(how="end")
df["month_key"] = df["month_center"].dt.to_period("M")

print(f"SOCAT rows loaded: {len(df):,}")

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
cg.rename(columns={"CO2(ppm)": "co2"}, inplace=True)

# Remove missing or invalid values
cg = cg[cg["co2"].notna()]

# Build a proper datetime column from YYYY/MM/DD
cg["date"] = pd.to_datetime(
    dict(year=cg["YYYY"], month=cg["MM"], day=cg["DD"]),
    errors="coerce"
)

# Convert to month-end (pipeline rule)
cg["month_center"] = cg["date"].dt.to_period("M").dt.to_timestamp(how="end")
cg["month_key"] = cg["month_center"].dt.to_period("M")

# Convert to (key → CO₂) Series
cg_monthly = cg.set_index("month_key")["co2"]

print(f"Cape Grim records loaded: {len(cg):,}")

# ===============================================================
# 3. Temporal merge
# ===============================================================
print("Matching Cape Grim CO₂ to SOCAT...")

df["atm_co2"] = df["month_key"].map(cg_monthly)

matched = df["atm_co2"].notna().sum()
print(f"Matched {matched:,} SOCAT grid cells ({matched/len(df)*100:.2f}%)")

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
