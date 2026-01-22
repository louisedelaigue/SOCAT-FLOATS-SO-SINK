# ===============================================================
# Predict pCO2 on Independent Validation Data Using Top 10 Models
# Selected from Random Search (TEST RMSE ranking)
# Louise Delaigue – 2025
# ===============================================================

import pandas as pd
import torch
import joblib
import numpy as np
from torch import nn
import os

# ---------------------------------------------------------------
# Paths
# ---------------------------------------------------------------
BASE_PATH = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
model_output_dir = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/MODEL_SOCAT_ONLY_OBS_v2/models/"

indep_data_path = BASE_PATH + "pco2_independent_validation_data.csv"

# NEW: scaler saved inside models directory
scaler_path = model_output_dir + "trained_scaler.joblib"

# NEW: statistics from random search saved inside models directory
results_path = model_output_dir + "random_search_model_stats.csv"

output_csv = f"{model_output_dir}/independent_predictions/pco2_independent_validation_predictions.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ---------------------------------------------------------------
# Required pipeline input columns
# ---------------------------------------------------------------
input_columns = [
    'SOCAT_temperature',
    'SOCAT_salinity',
    #'ARMOR3D_temperature_clim',
    #'temperature_anom',
    #'ARMOR3D_salinity_clim',
    #'salinity_anom',
    #'MLD_clim',
    #'MLD_anom',
    'MLD',
    'sla_median',
    'rrs412', 'rrs443', 'rrs490', 'rrs555', 'rrs670',
    'PAR_mean',
    #'u10', 'v10',
    'wind_speed',
    'atm_co2',
    #'decimal_year',
    'doy_sin',
    'doy_cos',
    'x_cart', 'y_cart', 'z_cart',
    'bottom_depth_m'
]

# ---------------------------------------------------------------
# 1. Load Independent Validation Data
# ---------------------------------------------------------------
print(f"[STEP] Loading independent dataset: {indep_data_path}")
indep_df = pd.read_csv(indep_data_path)
print(f"[INFO] Independent dataset shape: {indep_df.shape}")

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
# 4. Load Top 10 Models (Ranked by TEST RMSE!)
# ---------------------------------------------------------------
print(f"[STEP] Loading model statistics from {results_path}")
stats_df = pd.read_csv(results_path)

# Sort by TEST RMSE (not validation)
#top_models = stats_df.sort_values("test_rmse").head(10)

top_models = stats_df.sort_values(
    ["test_rmse", "test_r2"],
    ascending=[True, False]
).head(30)

#top_models = stats_df.copy()

print("[INFO] Top 10 architectures (by TEST RMSE):")
print(top_models[["layer1", "layer2", "layer3", "test_rmse", "test_r2"]])

# ---------------------------------------------------------------
# 5. Define Model Loader
# ---------------------------------------------------------------
def load_mlp(i, j, k, input_dim, model_path):
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, i)
            self.fc2 = nn.Linear(i, j)
            self.fc3 = nn.Linear(j, k)
            self.fc4 = nn.Linear(k, 1)
            self.act = nn.Tanh()

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            return self.fc4(x)

    model = MLP().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# ---------------------------------------------------------------
# 6. Predict With All 10 Models
# ---------------------------------------------------------------
predictions = []
print("[STEP] Running predictions with top 10 MLP models...")

for idx, row in top_models.iterrows():
    l1 = int(row["layer1"])
    l2 = int(row["layer2"])
    l3 = int(row["layer3"])

    model_path = f"{model_output_dir}/MLP_{l1}_{l2}_{l3}.pth"
    print(f"[MODEL {idx+1}/10] Loading {model_path}")

    if not os.path.exists(model_path):
        print(f"[WARNING] Missing model file: {model_path} — skipping.")
        continue

    model = load_mlp(l1, l2, l3, X_indep.shape[1], model_path)

    with torch.no_grad():
        pred = model(xindep_tensor).cpu().numpy().flatten()

    predictions.append(pred)
    print(f"[MODEL {idx+1}] Predictions complete. Shape: {pred.shape}")

print(f"[INFO] Completed predictions for {len(predictions)} models.")

# ---------------------------------------------------------------
# 7. Add Ensemble and Individual Predictions to DataFrame
# ---------------------------------------------------------------
predictions = np.stack(predictions, axis=0)
indep_df["ensemble_pco2_pred"] = predictions.mean(axis=0)

for i in range(predictions.shape[0]):
    indep_df[f"pco2_pred_model_{i+1}"] = predictions[i]

# ---------------------------------------------------------------
# 8. Save Output
# ---------------------------------------------------------------
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
indep_df.to_csv(output_csv, index=False)

print(f"[DONE] Independent validation predictions saved → {output_csv}")
print(indep_df.head())
