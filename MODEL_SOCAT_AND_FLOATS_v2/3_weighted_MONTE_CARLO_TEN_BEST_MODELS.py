# ===============================================================
# Monte Carlo Ensemble Training of Top 10 Models (VAL RMSE Ranked) — RESUMABLE
# Same setup as WEIGHTED random search:
# - Train on SOCAT + FLOATS fused
# - SOCAT-only validation & test
# - Weighted Huber loss using sample_weight (from uncertainty_pco2)
# Monte Carlo twist:
# - For each run, perturb train + val targets with N(0, uncertainty_pco2)
# - Evaluate on CLEAN SOCAT test (no noise, no weights in metric)
#
# RESUME FEATURES ADDED:
# - Skips (arch, mc) if model file already exists in MC_OUTPUT_DIR
# - Resumes MC_stats.csv if it exists (continues ensemble_id)
# - Writes MC_stats.csv after EACH completed run (crash-proof)
# ===============================================================

import os
import time
from copy import deepcopy

import joblib
import numpy as np
import pandas as pd
import torch
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
# Paths (weighted random search dir)
# ===============================================================
BASE_PATH       = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_AND_FLOATS_v2/processing_steps/"
SOCAT_ONLY_PATH = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"

RANDOM_SEARCH_DIR = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_AND_FLOATS_v2/models/weighted/"
MC_OUTPUT_DIR     = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_AND_FLOATS_v2/models/weighted/MC_ensemble/"

os.makedirs(MC_OUTPUT_DIR, exist_ok=True)

STATS_PATH  = os.path.join(RANDOM_SEARCH_DIR, "random_search_model_stats.csv")
SCALER_PATH = os.path.join(RANDOM_SEARCH_DIR, "trained_scaler.joblib")  # reuse RS scaler

# ===============================================================
# RESUME MC STATS (if exists)
# ===============================================================
stats_file = os.path.join(MC_OUTPUT_DIR, "MC_stats.csv")
if os.path.exists(stats_file):
    print(f"[INFO] Resuming from existing stats: {stats_file}")
    prev_stats_df = pd.read_csv(stats_file)
    stats_rows = prev_stats_df.to_dict("records")
    ensemble_id = int(prev_stats_df["ensemble_id"].max()) + 1 if len(prev_stats_df) else 0
else:
    stats_rows = []
    ensemble_id = 0

# ===============================================================
# Load Data (MATCHES WEIGHTED RANDOM SEARCH)
# ===============================================================
print("[INFO] Loading datasets...")

train_df = pd.read_csv(os.path.join(BASE_PATH, "pco2_train_data.csv"))  # fused train (has uncertainty + sample_weight)
val_df   = pd.read_csv(os.path.join(SOCAT_ONLY_PATH, "pco2_validation_data.csv"))
test_df  = pd.read_csv(os.path.join(SOCAT_ONLY_PATH, "pco2_test_data.csv"))

print(f"[INFO] Shapes — Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

# Fix SOCAT-only columns to match fused
val_df.rename(columns={"SOCAT_temperature": "temperature", "SOCAT_salinity": "salinity"}, inplace=True)
test_df.rename(columns={"SOCAT_temperature": "temperature", "SOCAT_salinity": "salinity"}, inplace=True)

# SOCAT uncertainty fixed (used for val-noise only; test stays clean)
SIGMA_SOCAT = 5.0
val_df["uncertainty_pco2"]  = SIGMA_SOCAT
test_df["uncertainty_pco2"] = SIGMA_SOCAT

# ===============================================================
# Inputs
# ===============================================================
input_columns = [
    "temperature", "salinity", "MLD", "sla_median",
    "rrs412", "rrs443", "rrs490", "rrs555", "rrs670",
    "PAR_mean", "wind_speed", "atm_co2",
    "doy_sin", "doy_cos",
    "x_cart", "y_cart", "z_cart",
    "bottom_depth_m"
]
output_column = "pco2"

# ===============================================================
# Extract arrays
# ===============================================================
X_train = train_df[input_columns].values
X_val   = val_df[input_columns].values
X_test  = test_df[input_columns].values

y_train = train_df[output_column].values.reshape(-1, 1)
y_val   = val_df[output_column].values.reshape(-1, 1)
y_test  = test_df[output_column].values.reshape(-1, 1)

# Noise sigmas
if "uncertainty_pco2" not in train_df.columns:
    raise ValueError("train_df missing 'uncertainty_pco2' (expected from fusion step).")
unc_train = train_df["uncertainty_pco2"].values.reshape(-1, 1)
unc_val   = val_df["uncertainty_pco2"].values.reshape(-1, 1)

# Weights (ONLY for training loss)
if "sample_weight" not in train_df.columns:
    raise ValueError("train_df missing 'sample_weight' (expected from fusion step).")
w_train = train_df["sample_weight"].values.reshape(-1, 1)

# ===============================================================
# Load RS scaler and transform (KEY CONSISTENCY FIX)
# ===============================================================
print(f"[INFO] Loading scaler from: {SCALER_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"trained_scaler.joblib not found: {SCALER_PATH}")

scaler = joblib.load(SCALER_PATH)

X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

# tensors
xtrain_tensor_base = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
xval_tensor        = torch.tensor(X_val_scaled,   dtype=torch.float32, device=device)
xtest_tensor       = torch.tensor(X_test_scaled,  dtype=torch.float32, device=device)
wtrain_tensor      = torch.tensor(w_train,        dtype=torch.float32, device=device)

# ===============================================================
# Weighted Huber (same as weighted RS)
# ===============================================================
criterion = nn.HuberLoss(delta=5.0, reduction="none")

def weighted_huber(pred, target, weight):
    raw_loss = criterion(pred, target)   # (batch, 1)
    return (raw_loss * weight).mean()

# ===============================================================
# Load Top 10 architectures by VAL_RMSE (SOCAT-only constraint)
# ===============================================================
print(f"[INFO] Loading model statistics: {STATS_PATH}")
if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(f"Stats file not found: {STATS_PATH}")

stats_df = pd.read_csv(STATS_PATH)

required = {"layer1", "layer2", "layer3", "val_rmse"}
missing = required - set(stats_df.columns)
if missing:
    raise ValueError(f"Stats CSV missing required columns: {sorted(missing)}")

sort_cols = [c for c in ["val_rmse", "test_rmse", "model"] if c in stats_df.columns]
stats_df = stats_df.sort_values(sort_cols, ascending=[True] * len(sort_cols)).reset_index(drop=True)

top_models = stats_df.head(10).copy()
print("[INFO] Top 10 architectures (ranked by val_rmse):")
show_cols = [c for c in ["model", "layer1", "layer2", "layer3", "val_rmse", "test_rmse", "test_r2"] if c in top_models.columns]
print(top_models[show_cols])

top_archs = list(zip(
    top_models["layer1"].astype(int),
    top_models["layer2"].astype(int),
    top_models["layer3"].astype(int),
))

# ===============================================================
# Monte Carlo config
# ===============================================================
BATCH_SIZE = 256
N_MC_RUNS  = 100
MAX_EPOCHS = 2000
PATIENCE   = 100
LR         = 2e-4

# ===============================================================
# Model
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
# MAIN LOOP
# ===============================================================
for arch_idx, (l1, l2, l3) in enumerate(top_archs, start=1):
    print("\n===================================")
    print(f"[ARCH {arch_idx}/{len(top_archs)}] {l1}-{l2}-{l3}")
    print("===================================")

    for mc in range(1, N_MC_RUNS + 1):
        # ---- RESUME: skip if model exists ----
        model_path = os.path.join(MC_OUTPUT_DIR, f"MLP_{l1}_{l2}_{l3}_MC{mc}.pth")
        if os.path.exists(model_path):
            print(f"  [MC {mc}/{N_MC_RUNS}] → already done, skipping")
            continue

        print(f"  [MC {mc}/{N_MC_RUNS}] → running")
        start = time.time()

        # Noise on TRAIN + VAL targets
        y_train_noisy = y_train + np.random.normal(0, unc_train, size=y_train.shape)
        y_val_noisy   = y_val   + np.random.normal(0, unc_val,   size=y_val.shape)

        ytrain_tensor = torch.tensor(y_train_noisy, dtype=torch.float32, device=device)
        yval_noisy_np = y_val_noisy.reshape(-1)

        train_loader = DataLoader(
            TensorDataset(xtrain_tensor_base, ytrain_tensor, wtrain_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        model = MLP(len(input_columns), l1, l2, l3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        best_val_rmse = float("inf")
        best_state = None
        counter = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            for xb, yb, wb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = weighted_huber(pred, yb, wb)
                loss.backward()
                optimizer.step()

            # noisy val metric (unweighted RMSE, as in your MC behavior)
            model.eval()
            with torch.no_grad():
                val_pred = model(xval_tensor).detach().cpu().numpy().reshape(-1)

            val_rmse = float(np.sqrt(mean_squared_error(yval_noisy_np, val_pred)))

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= PATIENCE:
                    break

        if best_state is None:
            raise RuntimeError("best_state is None (unexpected).")

        model.load_state_dict(best_state)
        model.eval()

        # Clean SOCAT test eval (no noise)
        with torch.no_grad():
            test_pred = model(xtest_tensor).detach().cpu().numpy().reshape(-1)

        test_rmse = float(np.sqrt(mean_squared_error(y_test.reshape(-1), test_pred)))
        test_r2   = float(r2_score(y_test.reshape(-1), test_pred))

        stats_rows.append({
            "ensemble_id": ensemble_id,
            "arch": f"{l1}-{l2}-{l3}",
            "mc_run": mc,
            "val_rmse_noisy": best_val_rmse,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "time_min": (time.time() - start) / 60.0
        })

        # Save model
        torch.save(model.state_dict(), model_path)

        # Persist stats each run (crash-proof)
        ensemble_id += 1
        pd.DataFrame(stats_rows).to_csv(stats_file, index=False)

# ===============================================================
# Done
# ===============================================================
total_minutes = (time.time() - SCRIPT_START_TIME) / 60.0
print(f"\n[INFO] Weighted MC complete. Stats saved → {stats_file}")
print(f"[INFO] Models saved in → {MC_OUTPUT_DIR}")
print(f"[INFO] Total wall-clock time: {total_minutes:.2f} minutes")
