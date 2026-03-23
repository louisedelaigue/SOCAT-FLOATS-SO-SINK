import pandas as pd

print("Loading SOCATv2025 data...")
df = pd.read_csv(
    '/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/data/SOCATv2025_SO.tsv',
    low_memory=False,
    skiprows=1499,
    sep='\t'
)
print(f"Original dataframe loaded with {df.shape[0]} rows and {df.shape[1]} columns.")

# ===============================================================
# Columns to keep
# ===============================================================

columns_to_keep = [
    'Expocode', 'yr', 'mon', 'day', 'hh', 'mm', 'ss',
    'longitude [dec.deg.E]', 'latitude [dec.deg.N]',
    'fCO2rec [uatm]', 'sample_depth [m]', 'sal', 'SST [deg.C]', 'PPPP [hPa]'
]

print("Filtering and renaming columns...")
df = (
    df[columns_to_keep]
    .rename(columns={
        'Expocode': 'expocode',
        'yr': 'year',
        'mon': 'month',
        'day': 'day',
        'hh': 'hour',
        'mm': 'minute',
        'ss': 'second',
        'longitude [dec.deg.E]': 'longitude',
        'latitude [dec.deg.N]': 'latitude',
        'fCO2rec [uatm]': 'fco2_rec',
        'sample_depth [m]': 'sample_depth_m',
        'sal': 'salinity',
        'SST [deg.C]': 'sst_degC',
        'PPPP [hPa]': 'pressure_hpa'
    })
    .dropna(subset=['salinity', 'sst_degC', 'fco2_rec'])
)

print(f"Filtered dataframe has {df.shape[0]} rows after dropping missing values.")

# ===============================================================
# Define independent validation expocodes
# ===============================================================

validation_expocodes = [
    "320620040518",
    "320620040729",
    "069920230226",
    "33RO20080229",
    "49NZ20121128",
    "49KD20181212",
    "09AR19990716",
    "09AR20151028"
]

# ===============================================================
# Load OISO expocodes from Excel file
# ===============================================================
expocode_file = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "_MODEL_SOCAT_OISO/"
    "Expocodes-OISO-dans-SOCAT.xlsx"
)

expocodes_df = pd.read_excel(expocode_file)

oiso_expocodes = (
    expocodes_df["Expocode SOCAT"]
    .dropna()
    .astype(str)
    .str.strip()
    .unique()
    .tolist()
)

print(f"Loaded {len(oiso_expocodes)} OISO expocodes to exclude.")

# ===============================================================
# Combine validation + OISO expocodes
# ===============================================================

all_excluded_expocodes = set(validation_expocodes) | set(oiso_expocodes)
print(f"Total expocodes excluded (validation + OISO): {len(all_excluded_expocodes)}")

# ===============================================================
# Independent validation dataset (ONLY validation cruises)
# ===============================================================

df_validation = df[df["expocode"].isin(validation_expocodes)].copy()
print(f"Independent validation dataset: {df_validation.shape[0]} rows.")

validation_output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "_MODEL_SOCAT_OISO/MODEL_NO_OISO/processing_steps/independent_validation/"
    "pco2_independent_validation_data.csv"
)

df_validation.to_csv(validation_output_path, index=False)
print(f"Independent validation data saved to:\n{validation_output_path}")

# ===============================================================
# Training dataset (SOCAT minus validation + OISO cruises)
# ===============================================================

df_training = df[~df["expocode"].isin(all_excluded_expocodes)].copy()
print(f"Training dataset: {df_training.shape[0]} rows.")

training_output_path = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "_MODEL_SOCAT_OISO/MODEL_NO_OISO/processing_steps/"
    "SOCATv2025_SO_clean_without_ind_or_OISO_expocodes.csv"
)

df_training.to_csv(training_output_path, index=False)
print(f"Clean training data saved to:\n{training_output_path}")

# ===============================================================
# FINAL CHECK: Ensure no validation or OISO expocodes remain
# ===============================================================

remaining = df_training["expocode"].isin(all_excluded_expocodes).any()

if remaining:
    bad_codes = (
        df_training[df_training["expocode"].isin(all_excluded_expocodes)]
        ["expocode"]
        .unique()
    )
    print("\nERROR: Some excluded expocodes remain in the training dataset!")
    print("Expocodes still present:", bad_codes)
else:
    print("\nSUCCESS: No validation or OISO expocodes appear in the training dataset.")
