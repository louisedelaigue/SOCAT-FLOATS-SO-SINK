# ===============================================================
# Extract Surface (0–10 m) BGC-Argo pCO2 + Temp + Salinity
# For CMEMS MULTIOBS BGC-ARGO files
# Louise Delaigue – 2025
# ===============================================================

import os
from tqdm import tqdm
import xarray as xr
import pandas as pd
import numpy as np


# ===============================================================
# Extract per-profile surface means (0–10 m)
# ===============================================================
def extract_surface_means(nc_file, surface_depth=10):
    ds = xr.open_dataset(nc_file)

    # Required variables
    required = [
        "PRES_ADJUSTED",
        "PCO2_CONTENT", "PCO2_CONTENT_ERROR",
        "TEMP_ADJUSTED", "PSAL_ADJUSTED",
        "JULD", "LATITUDE", "LONGITUDE",
        "PLATFORM_NUMBER", "CYCLE_NUMBER"
    ]

    if any(v not in ds.variables for v in required):
        print(f"[WARNING] Required variable missing → {nc_file}")
        return pd.DataFrame()

    pres = ds["PRES_ADJUSTED"].values
    pco2 = ds["PCO2_CONTENT"].values
    pco2_err = ds["PCO2_CONTENT_ERROR"].values

    temp = ds["TEMP_ADJUSTED"].values
    psal = ds["PSAL_ADJUSTED"].values

    time = ds["JULD"].values
    lat = ds["LATITUDE"].values
    lon = ds["LONGITUDE"].values
    platform = ds["PLATFORM_NUMBER"].values
    cycle = ds["CYCLE_NUMBER"].values

    n_prof = ds.sizes["N_PROF"]
    results = []

    for i in range(n_prof):

        timestamp = pd.to_datetime(time[i])
        if pd.isna(timestamp):
            continue

        pres_i = pres[i, :]
        mask = (pres_i <= surface_depth) & (~np.isnan(pres_i))

        if not np.any(mask):
            continue

        co2 = pco2[i, mask]
        co2err = pco2_err[i, mask]
        t = temp[i, mask]
        s = psal[i, mask]

        if np.all(np.isnan(co2)):
            continue

        results.append({
            "datetime": timestamp,
            "latitude": float(lat[i]),
            "longitude": float(lon[i]),
            "platform_number": platform[i].decode("utf-8").strip() if isinstance(platform[i], bytes) else str(platform[i]).strip(),
            "cycle_number": int(cycle[i]),
            "pco2": float(np.nanmean(co2)),
            "pco2_error": float(np.nanmean(co2err)),
            "temperature": float(np.nanmean(t)),
            "salinity": float(np.nanmean(s))
        })

    return pd.DataFrame(results)


# ===============================================================
# Main Loop
# ===============================================================

base_dir = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/BGC-ARGO/cmems_obs-mob_glo_bgc-nut-car_mynrt_irr_i_202411"

all_dfs = []
years = sorted([y for y in os.listdir(base_dir) if y.isdigit()])

for year in years:
    year_path = os.path.join(base_dir, year)

    if not os.path.isdir(year_path):
        continue

    print(f"\n=== YEAR {year} ===")

    for file in tqdm(sorted(os.listdir(year_path))):
        if file.endswith(".nc"):
            fp = os.path.join(year_path, file)

            try:
                df = extract_surface_means(fp)
                if len(df) > 0:
                    all_dfs.append(df)
            except Exception as e:
                print(f"[ERROR] {fp}: {e}")

# ===============================================================
# Combine extraction results
# ===============================================================

if len(all_dfs) > 0:
    final_df = pd.concat(all_dfs, ignore_index=True)
else:
    final_df = pd.DataFrame()

print("\n====================================================")
print("Extraction complete.")
print(f"Final dataset size: {len(final_df)} profiles")
print("====================================================")


# ===============================================================
# Define independent validation FLOAT LIST
# ===============================================================

validation_platforms = [
    "5904657",
    "5904841",
    "6902905",
    "5903717",

    "2902130",
    "5901742",
    "5906246",
    "5906487",
    "5905197",

    "3901497",
    "5901047",
    "5905142",
    "6900896",

    "2903455",
    "5904694",
    "5900422"
]


# ===============================================================
# Independent validation dataset  (FLOATS ONLY)
# ===============================================================

df_validation = final_df[ final_df["platform_number"].isin(validation_platforms) ].copy()
print(f"\nIndependent validation dataset: {df_validation.shape[0]} profiles.")

iv_output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "MODEL_FLOATS_v2/processing_steps/independent_validation/"
    "pco2_independent_validation_data.csv"
)

df_validation.to_csv(iv_output_path, index=False)
print(f"Independent validation saved to:\n{iv_output_path}")


# ===============================================================
# Training dataset  (ALL FLOATS EXCEPT IND-VAL)
# ===============================================================

df_training = final_df[ ~final_df["platform_number"].isin(validation_platforms) ].copy()
print(f"\nTraining dataset: {df_training.shape[0]} profiles.")

training_output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_v2/processing_steps/"
    "BGC-ARGO_surface_0_10m_pco2_temp_sal_all_profiles.csv"
)

df_training.to_csv(training_output_path, index=False)
print(f"Training dataset saved to:\n{training_output_path}")


# ===============================================================
# FINAL CHECK — ensure no validation floats are in training
# ===============================================================

leaks = df_training["platform_number"].isin(validation_platforms).any()

print("\n====================================================")
print("Integrity Check: Independent Validation vs Training")
print("====================================================")

if leaks:
    bad = df_training[df_training["platform_number"].isin(validation_platforms)]["platform_number"].unique()
    print("ERROR: Some validation floats remain in the training dataset!")
    print("Floats still present:", bad)
else:
    print("SUCCESS: No validation floats appear in the training dataset.")

print("====================================================")


