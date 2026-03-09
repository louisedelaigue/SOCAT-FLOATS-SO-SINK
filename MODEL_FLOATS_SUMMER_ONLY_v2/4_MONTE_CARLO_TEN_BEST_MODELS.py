# ===============================================================
# Monte Carlo Ensemble Training of Top 10 Models (VAL RMSE Ranked)
# Using FLOATS Label Perturbations with REAL uncertainties
# MATCHING Random Search Training Parameters
# UPDATED: Reuse RANDOM-SEARCH scaler (trained_scaler.joblib)
# Louise Delaigue – 2025
# ===============================================================

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import time
from copy import deepcopy

SCRIPT_START_TIME = time.time()

# ===============================================================
# GPU Setup
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ===============================================================
# Paths
# ===============================================================
BASE_PATH = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_SUMMER_ONLY_v2/processing_steps/"
MODEL_DIR = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_FLOATS_SUMMER_ONLY_v2/models/30S/"

model_output_dir = os.path.join(MODEL_DIR, "MC_ensemble/")
os.makedirs(model_output_dir, exist_ok=True)

STATS_PATH = os.path.join(MODEL_DIR, "random_search_model_stats.csv")

# NEW: reuse the scaler saved during random search
SCALER_PATH = os.path.join(MODEL_DIR, "trained_scaler.joblib")

# ===============================================================
# Load Data
# ===============================================================
train_df = pd.read_csv(os.path.join(BASE_PATH, "FLOATS_pco2_train_data.csv"))
val_df   = pd.read_csv(os.path.join(BASE_PATH, "FLOATS_pco2_validation_data.csv"))
test_df  = pd.read_csv(os.path.join(BASE_PATH, "FLOATS_pco2_test_data.csv"))

print(f"[INFO] Shapes — Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

# ===============================================================
# Input Features (MATCH RANDOM SEARCH)
# ===============================================================
input_columns = [
    "FLOATS_temperature_median", "FLOATS_salinity_median", "MLD", "sla_median",
    "rrs412", "rrs443", "rrs490", "rrs555", "rrs670",
    "PAR_mean",
    "wind_speed",
    "atm_co2",
    "doy_sin", "doy_cos",
    "x_cart", "y_cart", "z_cart",
    "bottom_depth_m"
]

output_column = "FLOATS_pco2_median"

# ===============================================================
# Prepare Train / Val / Test Data (LINEAR SPACE)
# ===============================================================
X_train = train_df[input_columns].values
X_val   = val_df[input_columns].values
X_test  = test_df[input_columns].values

y_train = train_df[output_column].values.reshape(-1, 1)
y_val   = val_df[output_column].values.reshape(-1, 1)
y_test  = test_df[output_column].values.reshape(-1, 1)

# REAL FLOATS uncertainties
unc_train = train_df["FLOATS_pco2_error_median"].values.reshape(-1, 1)
unc_val   = val_df["FLOATS_pco2_error_median"].values.reshape(-1, 1)

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

batch_size = 256

# ===============================================================
# Define the MLP (MATCH RANDOM SEARCH)
# ===============================================================
class MLP(nn.Module):
    def __init__(self, input_dim, l1, l2, l3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, l1)
        self.fc2 = nn.Linear(l1, l2)
        self.fc3 = nn.Linear(l2, l3)
        self.fc4 = nn.Linear(l3, 1)
        self.act = nn.GELU()

    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return self.fc4(x)

# ===============================================================
# Load TOP 10 architectures — SELECTED BY VAL_RMSE (CORRECT)
# ===============================================================
stats_df = pd.read_csv(STATS_PATH)

top_models = (
    stats_df
    .sort_values("val_rmse", ascending=True)
    .head(10)
)

print("[INFO] Top 10 architectures (ranked by val_rmse):")
show_cols = [c for c in ["layer1", "layer2", "layer3", "val_rmse", "test_rmse", "test_r2"] if c in top_models.columns]
print(top_models[show_cols])

top_archs = list(zip(
    top_models.layer1.astype(int),
    top_models.layer2.astype(int),
    top_models.layer3.astype(int)
))

# ===============================================================
# Monte Carlo Settings
# ===============================================================
n_mc_runs = 100
stats_rows = []
ensemble_id = 0

# ===============================================================
# MAIN LOOP: Architectures × MC Runs
# ===============================================================
for arch_idx, (l1, l2, l3) in enumerate(top_archs, start=1):

    print("\n=============================")
    print(f"[ARCH {arch_idx}] {l1}-{l2}-{l3}")
    print("=============================")

    for mc in range(1, n_mc_runs + 1):

        print(f"  [MC {mc}/{n_mc_runs}]")
        start = time.time()

        # ---- Sample noise using REAL FLOATS uncertainties ----
        # Note: np.random.normal accepts array-like scale; size matches y_*.shape.
        y_train_noisy = y_train + np.random.normal(0, unc_train, size=y_train.shape)
        y_val_noisy   = y_val   + np.random.normal(0, unc_val,   size=y_val.shape)

        ytrain_tensor = torch.tensor(y_train_noisy, dtype=torch.float32, device=device)

        train_loader = DataLoader(
            TensorDataset(xtrain_tensor, ytrain_tensor),
            batch_size=batch_size,
            shuffle=True
        )

        model = MLP(len(input_columns), l1, l2, l3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        criterion = nn.HuberLoss()

        best_val_rmse = float("inf")
        best_state = None
        patience = 100
        counter = 0
        max_epochs = 2000

        # ============================
        # TRAINING LOOP
        # ============================
        for epoch in range(max_epochs):

            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                pred_val = model(xval_tensor).detach().cpu().numpy().flatten()

            # Validation against noisy val targets (by design)
            val_rmse = np.sqrt(mean_squared_error(y_val_noisy.flatten(), pred_val))

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    break

        if best_state is None:
            raise RuntimeError("best_state is None (unexpected). Check training loop.")
        model.load_state_dict(best_state)

        # ---- TEST evaluation (no noise) ----
        model.eval()
        with torch.no_grad():
            test_pred = model(xtest_tensor).detach().cpu().numpy().flatten()

        test_rmse = np.sqrt(mean_squared_error(y_test.flatten(), test_pred))
        test_r2   = r2_score(y_test.flatten(), test_pred)

        stats_rows.append({
            "ensemble_id": ensemble_id,
            "arch": f"{l1}-{l2}-{l3}",
            "mc_run": mc,
            "val_rmse_noisy": float(best_val_rmse),
            "test_rmse": float(test_rmse),
            "test_r2": float(test_r2),
            "time_min": (time.time() - start) / 60
        })

        model_path = os.path.join(model_output_dir, f"MLP_{l1}_{l2}_{l3}_MC{mc}.pth")
        torch.save(model.state_dict(), model_path)

        ensemble_id += 1

# ===============================================================
# Save MC statistics
# ===============================================================
pd.DataFrame(stats_rows).to_csv(
    os.path.join(model_output_dir, "MC_stats.csv"),
    index=False
)

print("[INFO] Monte Carlo Ensemble complete.")

total_minutes = (time.time() - SCRIPT_START_TIME) / 60
print(f"[INFO] Total wall-clock time: {total_minutes:.2f} minutes")
