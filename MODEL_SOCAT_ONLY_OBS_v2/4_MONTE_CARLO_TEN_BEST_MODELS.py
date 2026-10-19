# ===============================================================
# Monte Carlo Ensemble Training of Top 10 Models (VAL RMSE Ranked)
# Using ±5 µatm SOCAT Label Perturbations
# UPDATED: Reuse RANDOM-SEARCH scaler (trained_scaler.joblib)
# Louise Delaigue – 2025
# ===============================================================

import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import joblib
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score

SCRIPT_START_TIME = time.time()

# ===============================================================
# GPU Setup
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ===============================================================
# Paths
# ===============================================================
BASE_PATH = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
MODEL_DIR = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_ONLY_OBS_v2/models/30S/"
STATS_PATH = os.path.join(MODEL_DIR, "random_search_model_stats.csv")

# NEW: reuse the scaler saved during random search
SCALER_PATH = os.path.join(MODEL_DIR, "trained_scaler.joblib")

model_output_dir = os.path.join(MODEL_DIR, "MC_ensemble/")
os.makedirs(model_output_dir, exist_ok=True)

# ===============================================================
# Load Data
# ===============================================================
train_df = pd.read_csv(os.path.join(BASE_PATH, "pco2_train_data.csv"))
val_df   = pd.read_csv(os.path.join(BASE_PATH, "pco2_validation_data.csv"))
test_df  = pd.read_csv(os.path.join(BASE_PATH, "pco2_test_data.csv"))
print(f"[INFO] Shapes — Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

# ===============================================================
# Input Features (must match SOCAT random search training)
# ===============================================================
input_columns = [
    "SOCAT_temperature", "SOCAT_salinity", "MLD", "sla_median",
    "rrs412", "rrs443", "rrs490", "rrs555", "rrs670",
    "PAR_mean", "wind_speed", "atm_co2",
    "doy_sin", "doy_cos",
    "x_cart", "y_cart", "z_cart",
    "bottom_depth_m"
]
output_column = "pco2"

# ===============================================================
# Convert data
# ===============================================================
X_train = train_df[input_columns].values
X_val   = val_df[input_columns].values
X_test  = test_df[input_columns].values

y_train = train_df[output_column].values.reshape(-1, 1)
y_val   = val_df[output_column].values.reshape(-1, 1)
y_test  = test_df[output_column].values.reshape(-1, 1)

# ===============================================================
# Normalize inputs using RANDOM-SEARCH scaler (REUSED)
# ===============================================================
print(f"[STEP] Loading scaler from: {SCALER_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(
        f"trained_scaler.joblib not found at {SCALER_PATH}. "
        "Make sure your random-search script saved it in MODEL_DIR."
    )

scaler = joblib.load(SCALER_PATH)
print("[INFO] Scaler loaded successfully.")

X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

xtrain_tensor = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
xval_tensor   = torch.tensor(X_val_scaled,   dtype=torch.float32, device=device)
xtest_tensor  = torch.tensor(X_test_scaled,  dtype=torch.float32, device=device)

# ===============================================================
# Define the network (SOCAT used TANH in random search)
# ===============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, l1, l2, l3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc4 = nn.Linear(l3, 1)
        self.act = nn.Tanh()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc4(x)

# ===============================================================
# Load Top 10 architectures from stats (BY VAL_RMSE)
# ===============================================================
print("[STEP] Loading model statistics:", STATS_PATH)
stats_df = pd.read_csv(STATS_PATH)

required_stats_cols = {"layer1", "layer2", "layer3", "val_rmse"}
missing_cols = required_stats_cols - set(stats_df.columns)
if missing_cols:
    raise ValueError(f"Stats CSV missing required columns: {sorted(missing_cols)}")

# deterministic tie-break (recommended)
sort_cols = [c for c in ["val_rmse", "test_rmse", "model"] if c in stats_df.columns]
stats_df = stats_df.sort_values(sort_cols, ascending=[True]*len(sort_cols)).reset_index(drop=True)

top_models = stats_df.head(10).copy()

print("[INFO] Top 10 architectures (by VAL RMSE):")
show_cols = [c for c in ["model", "layer1", "layer2", "layer3", "val_rmse", "test_rmse", "test_r2"] if c in top_models.columns]
print(top_models[show_cols])

best_archs = list(zip(top_models.layer1.astype(int), top_models.layer2.astype(int), top_models.layer3.astype(int)))

# ===============================================================
# Monte Carlo parameters
# ===============================================================
sigma_noise = 5.0    # µatm SOCAT uncertainty
n_mc_runs   = 100    # runs per architecture
batch_size  = 256

stats_rows = []
ensemble_id = 0

# ===============================================================
# Main loop — architectures × MC runs
# ===============================================================
for arch_idx, (l1, l2, l3) in enumerate(best_archs, start=1):
    print(f"\n=============================")
    print(f"[ARCH {arch_idx}] {l1}-{l2}-{l3}")
    print(f"=============================")

    for mc in range(1, n_mc_runs + 1):
        print(f"  [MC {mc}/{n_mc_runs}]")
        start = time.time()

        # ---- Sample noise ----
        y_train_noisy = y_train + np.random.normal(0, sigma_noise, size=y_train.shape)
        y_val_noisy   = y_val   + np.random.normal(0, sigma_noise, size=y_val.shape)

        ytrain_tensor = torch.tensor(y_train_noisy, dtype=torch.float32, device=device)

        train_loader = DataLoader(
            TensorDataset(xtrain_tensor, ytrain_tensor),
            batch_size=batch_size,
            shuffle=True
        )

        # ---- Model ----
        model = MLP(len(input_columns), l1, l2, l3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.0002)
        criterion = nn.MSELoss()

        # ---- Early stopping ----
        best_val_rmse = float("inf")
        best_state = None
        patience = 50
        counter = 0
        max_epochs = 2000

        for epoch in range(max_epochs):
            model.train()

            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            # Validation (compare to noisy val, by design)
            model.eval()
            with torch.no_grad():
                pred_val = model(xval_tensor).detach().cpu().numpy().flatten()
            val_rmse = np.sqrt(mean_squared_error(y_val_noisy.flatten(), pred_val))

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        # ---- Restore best state ----
        if best_state is None:
            raise RuntimeError("best_state is None (unexpected). Check training loop.")
        model.load_state_dict(best_state)

        # ---- Final TEST metrics (NO noise on test) ----
        model.eval()
        with torch.no_grad():
            test_pred = model(xtest_tensor).detach().cpu().numpy().flatten()

        test_rmse = np.sqrt(mean_squared_error(y_test.flatten(), test_pred))
        test_r2   = r2_score(y_test.flatten(), test_pred)

        # ---- Save stats ----
        stats_rows.append({
            "ensemble_id": ensemble_id,
            "arch": f"{l1}-{l2}-{l3}",
            "mc_run": mc,
            "val_rmse_noisy": float(best_val_rmse),
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2),
            "time_min": (time.time() - start) / 60
        })

        # ---- Save model ----
        model_path = os.path.join(model_output_dir, f"MLP_{l1}_{l2}_{l3}_MC{mc}.pth")
        torch.save(model.state_dict(), model_path)

        ensemble_id += 1

# ===============================================================
# Save all stats
# ===============================================================
out_stats = os.path.join(model_output_dir, "MC_stats.csv")
pd.DataFrame(stats_rows).to_csv(out_stats, index=False)
print(f"[INFO] Monte Carlo complete. Stats saved → {out_stats}")

total_minutes = (time.time() - SCRIPT_START_TIME) / 60
print(f"[INFO] Total wall-clock time: {total_minutes:.2f} minutes")
