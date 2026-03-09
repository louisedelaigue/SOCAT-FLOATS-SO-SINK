# === Predict pCO₂ from Matrix Using SOCAT + FLOATS Monte-Carlo Ensemble (SYMMETRIC) ===
# NO source_flag — NO weights — Fully consistent with symmetric training + MC
# Louise Delaigue – 2025

import pandas as pd
import torch
import joblib
import numpy as np
from torch import nn
import os
import re

# ============================================================
# === PATHS & SETTINGS ======================================
# ============================================================

MC_model_dir = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_AND_FLOATS_v2/models/weighted/MC_ensemble/"

input_csv = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/PREDICTION_MATRIX_v2/processing_steps/ARMOR3D_TSMLD_SouthernOcean_2003_2024_monthly_1deg_feature_engineered_monthly_1deg.csv"

trained_scaler_path = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_AND_FLOATS_v2/models/weighted/trained_scaler.joblib"

output_csv = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/PREDICTION_MATRIX_v2/matrix_predictions/pco2_prediction_matrix_SOCAT_AND_FLOATS_weighted_MCensemble.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[STEP] Device set to: {device}")

# ============================================================
# === COLUMN RENAMING =======================================
# ============================================================

rename_map = {
    "lat_center": "latitude",
    "lon_center": "longitude"
}

# MUST MATCH TRAINING (PURE PHYSICAL INPUTS ONLY)
input_columns = [
    "ARMOR3D_temperature",
    "ARMOR3D_salinity",
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

# ============================================================
# === 1. LOAD MATRIX & PREPROCESS ============================
# ============================================================

print(f"[STEP] Loading input matrix from: {input_csv}")
df = pd.read_csv(input_csv)

df.rename(columns=rename_map, inplace=True)

# Southern Ocean only
df = df[df["latitude"] <= -30]

# Drop rows with missing predictors
missing = [c for c in input_columns if c not in df.columns]
if missing:
    raise ValueError(f"Missing required input columns: {missing}")

df.dropna(subset=input_columns, inplace=True)

print(f"[INFO] Input shape after NaN filtering: {df.shape}")

# ============================================================
# === 2. LOAD TRAINED SCALER (PHYSICAL ONLY) =================
# ============================================================

print(f"[STEP] Loading trained scaler from: {trained_scaler_path}")
trained_scaler = joblib.load(trained_scaler_path)

X_final = df[input_columns].values
X_scaled = trained_scaler.transform(X_final)

x_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(device)

# ============================================================
# === 3. FIND ALL MC MODELS =================================
# ============================================================

print(f"[STEP] Searching for MC models in: {MC_model_dir}")

all_files = os.listdir(MC_model_dir)

# Filenames: MC_MLP_88_76_70_run3.pth
pattern = re.compile(r"MLP_(\d+)_(\d+)_(\d+)_MC(\d+)\.pth")

model_files = sorted([f for f in all_files if pattern.match(f)])

print(f"[INFO] Found {len(model_files)} MC ensemble models.")

if len(model_files) == 0:
    raise RuntimeError("No MC models found in MC_ensemble directory!")

# ============================================================
# === 4. DEFINE MLP LOADER (GELU, PHYSICAL ONLY) =============
# ============================================================

def load_mlp(i, j, k, path):

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(len(input_columns), i)
            self.fc2 = nn.Linear(i, j)
            self.fc3 = nn.Linear(j, k)
            self.fc4 = nn.Linear(k, 1)
            self.act = nn.GELU()

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            return self.fc4(x)

    model = MLP().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    return model

# ============================================================
# === 5. LOOP OVER ALL MC MODELS =============================
# ============================================================

predictions = []
model_ids = []

for fname in model_files:

    match = pattern.match(fname)

    if not match:
        print(f"[WARNING] Skipping file: {fname}")
        continue

    i, j, k, run = map(int, match.groups())
    model_path = os.path.join(MC_model_dir, fname)

    print(f"[MODEL] Loading {fname} (layers: {i}-{j}-{k}, run {run})")

    model = load_mlp(i, j, k, model_path)

    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy().flatten()

    predictions.append(pred)
    model_ids.append(f"{i}_{j}_{k}_run{run}")

print(f"[STEP] Predictions complete using {len(predictions)} MC models.")

# ============================================================
# === 6. ADD ENSEMBLE OUTPUTS (DEFRAGMENTED FIX) ============
# ============================================================

pred_array = np.stack(predictions, axis=0)

# Ensemble statistics
df["ensemble_pco2_pred"] = pred_array.mean(axis=0)
df["ensemble_pco2_std"]  = pred_array.std(axis=0)

# Build all MC columns at once (prevents fragmentation)
mc_df = pd.DataFrame(
    pred_array.T,
    columns=[f"pco2_pred_{mid}" for mid in model_ids],
    index=df.index
)

# Single fast concatenation
df = pd.concat([df, mc_df], axis=1)

# Hard defragment before saving large file
df = df.copy()

# ============================================================
# === 7. SAVE OUTPUT (drop per-model cols) ===================
# ============================================================
print(f"[STEP] Saving predictions to: {output_csv}")

# Drop the 1000 per-model prediction columns (keep metadata + ensemble stats)
df_out = df.drop(columns=[c for c in df.columns if c.startswith("pco2_pred_")], errors="ignore")

df_out.to_csv(output_csv, index=False)

print("[DONE] Predictions saved.")

# ============================================================
# === 8. PREVIEW ============================================
# ============================================================

cols_to_show = ["ensemble_pco2_pred", "ensemble_pco2_std"]
print(df[cols_to_show].head())
