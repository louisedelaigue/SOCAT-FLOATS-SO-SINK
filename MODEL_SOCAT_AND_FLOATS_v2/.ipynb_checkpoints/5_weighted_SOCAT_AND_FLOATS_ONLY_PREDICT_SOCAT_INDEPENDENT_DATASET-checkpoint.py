# ===============================================================
# Predict pCO2 on Independent Validation Data Using Monte Carlo
# Ensemble Models (100 models = 10 architectures × 100 MC runs)
# Using SOCAT + FLOATS FUSED MC_ensemble Models
# Louise Delaigue – 2025
# ===============================================================

import pandas as pd
import torch
import joblib
import numpy as np
from torch import nn
import os

# ---------------------------------------------------------------
# Paths (UNCHANGED)
# ---------------------------------------------------------------
BASE_PATH = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/independent_validation/"
MC_model_dir = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_AND_FLOATS_v2/models/weighted/MC_ensemble/"

indep_data_path = BASE_PATH + "pco2_independent_validation_data_regridded_SLA_RRS_PAR_WIND_ATM_CO2_feature_engineered_monthly_1deg_pCO2.csv"

# Scaler is shared with training (PATH KEPT AS YOU PROVIDED)
scaler_path = os.path.join(
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_AND_FLOATS_v2/models/weighted/",
    "trained_scaler.joblib"
)

output_csv = f"{MC_model_dir}/SOCAT_independent_predictions/MC_weighted_SOCAT_independent_validation_predictions.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ---------------------------------------------------------------
# Required pipeline input columns
# MATCH SOCAT+FLOATS MC TRAINING INPUTS
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# 1. Load Independent Validation Data
# ---------------------------------------------------------------
print(f"[STEP] Loading independent dataset: {indep_data_path}")
indep_df = pd.read_csv(indep_data_path)
print(f"[INFO] Independent dataset shape: {indep_df.shape}")

# ---------------------------------------------------------------
# Fix column names: remove SOCAT_ prefix for T/S if present
# ---------------------------------------------------------------
rename_map = {}

if "SOCAT_temperature" in indep_df.columns:
    rename_map["SOCAT_temperature"] = "temperature"

if "SOCAT_salinity" in indep_df.columns:
    rename_map["SOCAT_salinity"] = "salinity"

if rename_map:
    print(f"[INFO] Renaming columns: {rename_map}")
    indep_df = indep_df.rename(columns=rename_map)

# Sanity check: ensure all required columns are present
missing_cols = [c for c in input_columns if c not in indep_df.columns]
if missing_cols:
    raise ValueError(f"Missing required input columns in independent dataset: {missing_cols}")

# ---------------------------------------------------------------
# 2. Extract Input Matrix
# ---------------------------------------------------------------
print("[STEP] Extracting feature matrix...")
X_indep = indep_df[input_columns].values
print(f"[INFO] X_indep shape: {X_indep.shape}")

# ---------------------------------------------------------------
# 3. Load Scaler and Scale Data
# ---------------------------------------------------------------
print(f"[STEP] Loading scaler from: {scaler_path}")
scaler = joblib.load(scaler_path)
print("[INFO] Scaler loaded successfully.")

X_indep_scaled = scaler.transform(X_indep)
xindep_tensor = torch.tensor(X_indep_scaled, dtype=torch.float32).to(device)

# ---------------------------------------------------------------
# 4. Load ALL Monte-Carlo Ensemble Models
# ---------------------------------------------------------------
print(f"[STEP] Loading MC ensemble models from: {MC_model_dir}")

# Models trained & saved as:
# model_path = os.path.join(model_output_dir, f"MC_MLP_{l1}_{l2}_{l3}_run{mc}.pth")
model_files = sorted([f for f in os.listdir(MC_model_dir) if f.endswith(".pth")])

print(f"[INFO] Found {len(model_files)} MC models.")
if len(model_files) == 0:
    raise RuntimeError("No MC models found in MC_ensemble directory.")

# ---------------------------------------------------------------
# 5. Define MLP Loader (MATCH TRAINING: GELU, 4 FC LAYERS)
# ---------------------------------------------------------------
def load_mlp(l1, l2, l3, input_dim, model_path):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, l1)
            self.fc2 = nn.Linear(l1, l2)
            self.fc3 = nn.Linear(l2, l3)
            self.fc4 = nn.Linear(l3, 1)
            self.act = nn.GELU()  # MATCH TRAINING

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            return self.fc4(x)

    model = MLP().to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model

# ---------------------------------------------------------------
# 6. Predict Using All MC Models
# ---------------------------------------------------------------
predictions = []
print("[STEP] Running predictions with MC ensemble models...")

for idx, model_file in enumerate(model_files):
    model_path = os.path.join(MC_model_dir, model_file)
    print(f"[MODEL {idx+1}/{len(model_files)}] Loading {model_file}")

    # Parse layer sizes from filename pattern used in training:
    # "MC_MLP_{l1}_{l2}_{l3}_run{mc}.pth"
    name = model_file.replace(".pth", "")
    parts = name.split("_")
    # parts = ["MC", "MLP", l1, l2, l3, "runX"]
    if len(parts) < 6:
        raise ValueError(f"Unexpected model filename format: {model_file}")

    l1 = int(parts[2])
    l2 = int(parts[3])
    l3 = int(parts[4])

    model = load_mlp(l1, l2, l3, X_indep.shape[1], model_path)

    with torch.no_grad():
        pred = model(xindep_tensor).cpu().numpy().flatten()

    predictions.append(pred)

print(f"[INFO] Completed predictions for {len(predictions)} MC models.")

# ---------------------------------------------------------------
# 7. Add Ensemble and Individual Predictions to DataFrame
# ---------------------------------------------------------------
predictions = np.stack(predictions, axis=0)

# Ensemble mean (final prediction)
indep_df["ensemble_pco2_pred"] = predictions.mean(axis=0)

# Individual model predictions (optional)
for i in range(predictions.shape[0]):
    indep_df[f"pco2_pred_MCmodel_{i+1}"] = predictions[i]

# Ensemble spread = uncertainty
indep_df["ensemble_std"] = predictions.std(axis=0)

# ---------------------------------------------------------------
# 8. Save Output
# ---------------------------------------------------------------
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
indep_df.to_csv(output_csv, index=False)

print(f"[DONE] Independent validation MC predictions saved → {output_csv}")
print(indep_df.head())
