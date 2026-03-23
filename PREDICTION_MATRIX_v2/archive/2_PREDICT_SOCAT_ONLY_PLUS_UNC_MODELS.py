# === Predict fCO₂ from Matrix Using Model Files Directly (No Split Scaler) ===

import pandas as pd
import torch
import joblib
import numpy as np
from torch import nn
import os
import re

# --- Paths and Settings ---
model_dir = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/MODEL_SOCAT_ONLY_OBS_PLUS_UNC/models/"
input_csv = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/PREDICTION_MATRIX/processing_steps/ARMOR3D_TS_SouthernOcean_1997_2024_monthly_surface_SLA_RRS_CHL_PAR_WIND_CO2_ATM_MLD_COORDS.csv"
trained_scaler_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/MODEL_SOCAT_ONLY_OBS_PLUS_UNC/processing_steps/trained_scaler.joblib"
output_csv = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/PREDICTION_MATRIX/matrix_predictions/pco2_prediction_matrix_SOCAT_only_models_plus_unc_models.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"[STEP] Device set to: {device}")

# --- Column Renaming and Inputs ---
rename_map = {
    'PAR_mean': 'PAR',
    'mld_da_mean': 'MLD',
    'sst_degC': 'temperature'
}
input_columns = [
    'sla', 'RRS412', 'RRS443', 'RRS490', 'RRS555', 'RRS670',
    'PAR', 'wind_speed', 'atm_co2', 'temperature', 'salinity', 'MLD',
    'doy_sin', 'doy_cos', 'x_cart', 'y_cart', 'z_cart', 'decimal_year', 'bottom_depth_m'
]

# --- 1. Load & Prepare Input Data ---
print(f"[STEP] Loading input matrix from: {input_csv}")
df = pd.read_csv(input_csv)
df.rename(columns=rename_map, inplace=True)

# === Restrict latitude range to match 2019 paper ===
df = df[df['latitude'] <= -35]

df['time'] = pd.to_datetime(df['time'], errors='coerce')
df.dropna(subset=input_columns, inplace=True)
print(f"[INFO] Input shape after dropping NaNs: {df.shape}")

# --- 2. Apply Trained Scaler to Full Input Columns ---
print(f"[STEP] Loading trained scaler from: {trained_scaler_path}")
trained_scaler = joblib.load(trained_scaler_path)

X_final = df[input_columns].values
X_scaled = trained_scaler.transform(X_final)
x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# --- 3. Define MLP Loader (from parsed architecture) ---
def load_mlp(i, j, k, input_dim, path):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, i)
            self.fc2 = nn.Linear(i, j)
            self.fc3 = nn.Linear(j, k)
            self.fc4 = nn.Linear(k, 1)
            self.activation = nn.Tanh()

        def forward(self, x):
            x = self.activation(self.fc1(x))
            x = self.activation(self.fc2(x))
            x = self.activation(self.fc3(x))
            x = self.fc4(x)
            return x

    model = MLP().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model

# --- 4. Loop Over Model Files Matching Pattern ---
model_files = sorted(f for f in os.listdir(model_dir) if f.startswith("model_") and f.endswith(".pth"))
pattern = re.compile(r"model_(\d+)_(\d+)_(\d+)\.pth")

predictions = []
model_ids = []

print(f"[STEP] Found {len(model_files)} model files.")
for fname in model_files:
    match = pattern.match(fname)
    if match:
        i, j, k = map(int, match.groups())
        model_path = os.path.join(model_dir, fname)
        print(f"[MODEL] Using {fname} → architecture: {i}-{j}-{k}")
        model = load_mlp(i, j, k, X_scaled.shape[1], model_path)

        with torch.no_grad():
            pred = model(x_tensor).cpu().numpy().flatten()
            predictions.append(pred)
            model_ids.append(f"{i}_{j}_{k}")
    else:
        print(f"[WARNING] Skipping unmatched file: {fname}")

print(f"[STEP] Predictions complete using {len(predictions)} models.")

# --- 5. Add Predictions to DataFrame ---
pred_array = np.stack(predictions, axis=0)
df['ensemble_pco2_pred'] = pred_array.mean(axis=0)
df['ensemble_pco2_std'] = pred_array.std(axis=0)

for idx, model_id in enumerate(model_ids):
    df[f'pco2_pred_model_{model_id}'] = pred_array[idx]

# --- 6. Save to CSV ---
print(f"[STEP] Saving predictions to: {output_csv}")
df.to_csv(output_csv, index=False)
print("[DONE] Predictions saved.")

# --- 7. Preview Output ---
cols_to_show = ['ensemble_pco2_pred', 'ensemble_pco2_std'] + [col for col in df.columns if col.startswith("pco2_pred_model_")]
print("[PREVIEW] First few rows:")
print(df[cols_to_show].head())
