import pandas as pd
import numpy as np

# ===============================================================
# PATHS
# ===============================================================

BASE_SOCAT  = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
BASE_FLOATS = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_v2/processing_steps/"
BASE_OUT    = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_AND_FLOATS_v2/processing_steps/"

# ===============================================================
# GRID CELL IDENTIFIER
# ===============================================================

cell_id_cols = ["month_center", "latitude", "longitude"]

# ===============================================================
# UNCERTAINTY SETTINGS
# ===============================================================

SIGMA_SOCAT = 5.0   # µatm
SIGMA0 = 5.0        # for weight scaling

# ===============================================================
# FINAL UNIFIED INPUT FEATURES
# ===============================================================

input_columns = [
    "temperature",
    "salinity",
    "MLD",
    "sla_median",
    "rrs412", "rrs443", "rrs490", "rrs555", "rrs670",
    "PAR_mean",
    "wind_speed",
    "atm_co2",
    "doy_sin",
    "doy_cos",
    "x_cart", "y_cart", "z_cart",
    "bottom_depth_m"
]

# ===============================================================
# FUSION FUNCTION (ONE SPLIT)
# ===============================================================

def fuse_split(split_name):

    print(f"\n==================== {split_name.upper()} ====================")

    socat  = pd.read_csv(BASE_SOCAT  + f"pco2_{split_name}_data.csv")
    floats = pd.read_csv(BASE_FLOATS + f"FLOATS_pco2_{split_name}_data.csv")

    print("[INFO] SOCAT :", socat.shape)
    print("[INFO] FLOATS:", floats.shape)

    # -----------------------------------------------------------
    # Rename to unified variable names
    # -----------------------------------------------------------

    socat = socat.rename(columns={
        "SOCAT_temperature": "temperature",
        "SOCAT_salinity": "salinity"
    })

    floats = floats.rename(columns={
        "FLOATS_temperature_median": "temperature",
        "FLOATS_salinity_median": "salinity",
        "FLOATS_pco2_median": "pco2",
        "FLOATS_pco2_error_median": "uncertainty_pco2"
    })

    # SOCAT uncertainty is fixed
    socat["uncertainty_pco2"] = SIGMA_SOCAT

    # -----------------------------------------------------------
    # Keep only required columns
    # -----------------------------------------------------------

    keep_cols = cell_id_cols + input_columns + ["pco2", "uncertainty_pco2"]

    socat  = socat[keep_cols]
    floats = floats[keep_cols]

    # -----------------------------------------------------------
    # Merge on exact grid cell (month, lat, lon)
    # -----------------------------------------------------------

    merged = socat.merge(
        floats,
        on=cell_id_cols,
        how="outer",
        suffixes=("_SOCAT", "_FLOATS")
    )

    print("[INFO] After merge:", merged.shape)

    # -----------------------------------------------------------
    # FUSION LOGIC
    # -----------------------------------------------------------

    p_fused  = []
    sigma_f  = []
    src_flag = []

    for _, row in merged.iterrows():

        has_socat  = not np.isnan(row["pco2_SOCAT"])
        has_floats = not np.isnan(row["pco2_FLOATS"])

        # -------- BOTH PRESENT → FUSED --------
        if has_socat and has_floats:

            pS = row["pco2_SOCAT"]
            pF = row["pco2_FLOATS"]

            sS = row["uncertainty_pco2_SOCAT"]
            sF = row["uncertainty_pco2_FLOATS"]

            wS = 1.0 / sS**2
            wF = 1.0 / sF**2

            p = (pS * wS + pF * wF) / (wS + wF)
            s = np.sqrt(1.0 / (wS + wF))

            p_fused.append(p)
            sigma_f.append(s)
            src_flag.append(2)

        # -------- SOCAT ONLY --------
        elif has_socat:

            p_fused.append(row["pco2_SOCAT"])
            sigma_f.append(row["uncertainty_pco2_SOCAT"])
            src_flag.append(0)

        # -------- FLOATS ONLY --------
        elif has_floats:

            p_fused.append(row["pco2_FLOATS"])
            sigma_f.append(row["uncertainty_pco2_FLOATS"])
            src_flag.append(1)

        else:
            p_fused.append(np.nan)
            sigma_f.append(np.nan)
            src_flag.append(-1)

    merged["pco2"] = p_fused
    merged["uncertainty_pco2"] = sigma_f
    merged["source_flag"] = src_flag

    # -----------------------------------------------------------
    # PHYSICAL SAMPLE WEIGHT FROM FINAL SIGMA
    # -----------------------------------------------------------

    merged["sample_weight"] = 1.0 / (1.0 + merged["uncertainty_pco2"] / SIGMA0)

    # -----------------------------------------------------------
    # BUILD FINAL FEATURE TABLE
    # -----------------------------------------------------------

    final_features = []

    for col in input_columns:
        col_s = col + "_SOCAT"
        col_f = col + "_FLOATS"
        new = merged[col_s].combine_first(merged[col_f])
        final_features.append(new)

    final = pd.concat(final_features, axis=1)
    final.columns = input_columns

    final = pd.concat([
        merged[cell_id_cols],
        final,
        merged[["pco2", "uncertainty_pco2", "sample_weight", "source_flag"]]
    ], axis=1)

    # -----------------------------------------------------------
    # SAVE
    # -----------------------------------------------------------

    out_path = BASE_OUT + f"pco2_{split_name}_data.csv"
    final.to_csv(out_path, index=False)

    print(f"[INFO] Saved → {out_path}")
    print("[INFO] source_flag distribution:")
    print(final["source_flag"].value_counts())

# ===============================================================
# RUN FOR ALL SPLITS
# ===============================================================

for split in ["train", "validation", "test"]:
    fuse_split(split)

print("\nALL SPLITS FUSED SUCCESSFULLY AND SAFELY")

# ===============================================================
# BUILD AND SAVE COMBINED (TRAIN + VALIDATION + TEST)
# ===============================================================

print("\n==================== COMBINED ====================")

dfs = []
for split in ["train", "validation", "test"]:
    path = BASE_OUT + f"pco2_{split}_data.csv"
    df = pd.read_csv(path)
    df["split"] = split  # optionally label origin
    dfs.append(df)

combined = pd.concat(dfs, ignore_index=True)

out_combined = BASE_OUT + "pco2_all_splits_combined.csv"
combined.to_csv(out_combined, index=False)

print(f"[INFO] Combined dataset saved → {out_combined}")
print("[INFO] source_flag distribution (combined):")
print(combined["source_flag"].value_counts())
print("[INFO] split distribution:")
print(combined["split"].value_counts())

