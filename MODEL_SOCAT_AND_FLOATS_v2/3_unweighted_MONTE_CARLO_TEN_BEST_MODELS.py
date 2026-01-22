# ===============================================================
# Monte Carlo Ensemble Training on SOCAT+FLOATS (FUSED TRAIN) — RESUMABLE
# MATCHES SOCAT-ONLY MC PIPELINE BEHAVIOR:
#   - Noise on TRAIN + VAL labels (Gaussian)
#   - CLEAN TEST evaluation (no noise)
#
# Also MATCHES your SOCAT+FLOATS RANDOM-SEARCH (unweighted) settings:
#   - Uses random-search scaler (trained_scaler.joblib)
#   - Activation: GELU
#   - Loss: Huber(delta=5.0)
#   - Optimizer: Adam(lr=2e-4)
#   - Early stopping: patience=100, max_epochs=2000
#   - Top 10 architectures selected by VAL_RMSE (SOCAT-only validation constraint)
#
# RESUME FEATURES ADDED:
#   - Skips (arch, mc) if model file already exists in MC_MODEL_DIR
#   - Resumes MC_stats.csv if it exists (continues ensemble_id)
#   - Writes MC_stats.csv after EACH completed run (crash-proof)
#
# Louise Delaigue – 2025
# ===============================================================

import os
import time
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import mean_squared_error, r2_score
import joblib

SCRIPT_START_TIME = time.time()

# ===============================================================
# USER CONFIG
# ===============================================================
N_TOP_MODELS = 10
N_MC_RUNS    = 100
BATCH_SIZE   = 256
MAX_EPOCHS   = 2000
PATIENCE     = 100
LR           = 2e-4
HUBER_DELTA  = 5.0

# --- SOCAT-only uncertainty (for SOCAT-only VAL/TEST) ---
SIGMA_SOCAT = 5.0  # µatm

# ===============================================================
# GPU Setup
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ===============================================================
# Paths (ALIGN WITH YOUR SOCAT+FLOATS RANDOM SEARCH 200 SCRIPT)
# ===============================================================
FUSED_BASE_PATH = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "MODEL_SOCAT_AND_FLOATS_v2/processing_steps/"
)

SOCAT_ONLY_PATH = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
)

MODEL_DIR = (
    "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/"
    "MODEL_SOCAT_AND_FLOATS_v2/models/unweighted/"
)

STATS_PATH  = os.path.join(MODEL_DIR, "random_search_model_stats.csv")
SCALER_PATH = os.path.join(MODEL_DIR, "trained_scaler.joblib")  # reuse random-search scaler

MC_MODEL_DIR = os.path.join(MODEL_DIR, "MC_ensemble/")
os.makedirs(MC_MODEL_DIR, exist_ok=True)

# ===============================================================
# RESUME MC STATS (if exists)
# ===============================================================
stats_file = os.path.join(MC_MODEL_DIR, "MC_stats.csv")
if os.path.exists(stats_file):
    print(f"[INFO] Resuming from existing stats: {stats_file}")
    prev_stats_df = pd.read_csv(stats_file)
    stats_rows = prev_stats_df.to_dict("records")
    ensemble_id = int(prev_stats_df["ensemble_id"].max()) + 1 if len(prev_stats_df) else 0
else:
    stats_rows = []
    ensemble_id = 0

# ===============================================================
# Load Data
# ===============================================================
print("[INFO] Loading datasets...")
train_df = pd.read_csv(os.path.join(FUSED_BASE_PATH, "pco2_train_data.csv"))          # SOCAT+FLOATS fused TRAIN
val_df   = pd.read_csv(os.path.join(SOCAT_ONLY_PATH, "pco2_validation_data.csv"))     # SOCAT-only VAL
test_df  = pd.read_csv(os.path.join(SOCAT_ONLY_PATH, "pco2_test_data.csv"))           # SOCAT-only TEST

print(f"[INFO] Shapes — Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

# ===============================================================
# Align SOCAT-only VAL/TEST columns to fused naming
# ===============================================================
val_df  = val_df.rename(columns={"SOCAT_temperature": "temperature", "SOCAT_salinity": "salinity"})
test_df = test_df.rename(columns={"SOCAT_temperature": "temperature", "SOCAT_salinity": "salinity"})

# Add uncertainty column for SOCAT-only VAL/TEST (used for VAL noise, not for TEST eval)
val_df["uncertainty_pco2"]  = SIGMA_SOCAT
test_df["uncertainty_pco2"] = SIGMA_SOCAT

# ===============================================================
# Inputs / Target (MATCH SOCAT+FLOATS RANDOM SEARCH SCRIPT)
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
    "bottom_depth_m",
]
output_column = "pco2"

# ===============================================================
# Extract matrices
# ===============================================================
X_train = train_df[input_columns].values
X_val   = val_df[input_columns].values
X_test  = test_df[input_columns].values

y_train = train_df[output_column].values.reshape(-1, 1)
y_val   = val_df[output_column].values.reshape(-1, 1)
y_test  = test_df[output_column].values.reshape(-1, 1)

# Uncertainties for TRAIN + VAL noise (SOCAT-only MC behavior)
if "uncertainty_pco2" not in train_df.columns:
    raise ValueError("Fused TRAIN dataframe must contain 'uncertainty_pco2' (from fusion step).")

unc_train = train_df["uncertainty_pco2"].values.reshape(-1, 1)
unc_val   = val_df["uncertainty_pco2"].values.reshape(-1, 1)

# ===============================================================
# Load RANDOM-SEARCH SCALER and transform (KEY FOR CONSISTENCY)
# ===============================================================
print(f"[INFO] Loading random-search scaler: {SCALER_PATH}")
if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

scaler = joblib.load(SCALER_PATH)

X_train_scaled = scaler.transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

xtrain_tensor_base = torch.tensor(X_train_scaled, dtype=torch.float32, device=device)
xval_tensor        = torch.tensor(X_val_scaled,   dtype=torch.float32, device=device)
xtest_tensor       = torch.tensor(X_test_scaled,  dtype=torch.float32, device=device)

# ===============================================================
# Model Definition (MATCH SOCAT+FLOATS RANDOM SEARCH: GELU)
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
# Load Top N architectures by VAL_RMSE (SOCAT-only validation constraint)
# ===============================================================
print(f"[INFO] Loading model statistics: {STATS_PATH}")
if not os.path.exists(STATS_PATH):
    raise FileNotFoundError(f"Stats file not found: {STATS_PATH}")

stats_df = pd.read_csv(STATS_PATH)

required_cols = {"layer1", "layer2", "layer3", "val_rmse"}
missing = required_cols - set(stats_df.columns)
if missing:
    raise ValueError(f"Stats CSV missing required columns: {sorted(missing)}")

# deterministic tie-breaks (recommended)
sort_cols = [c for c in ["val_rmse", "test_rmse", "model"] if c in stats_df.columns]
stats_df = stats_df.sort_values(sort_cols, ascending=[True] * len(sort_cols)).reset_index(drop=True)

top_models = stats_df.head(N_TOP_MODELS).copy()

print(f"[INFO] Top {N_TOP_MODELS} architectures (ranked by val_rmse):")
show_cols = [c for c in ["model", "layer1", "layer2", "layer3", "val_rmse", "test_rmse", "test_r2"] if c in top_models.columns]
print(top_models[show_cols])

best_archs = list(zip(
    top_models["layer1"].astype(int),
    top_models["layer2"].astype(int),
    top_models["layer3"].astype(int),
))

# ===============================================================
# Training Setup (MATCH RANDOM SEARCH)
# ===============================================================
criterion = nn.HuberLoss(delta=HUBER_DELTA)

# ===============================================================
# MAIN LOOP — architectures × MC runs
# ===============================================================
for arch_idx, (l1, l2, l3) in enumerate(best_archs, start=1):
    print("\n=============================")
    print(f"[ARCH {arch_idx}/{len(best_archs)}] {l1}-{l2}-{l3}")
    print("=============================")

    for mc in range(1, N_MC_RUNS + 1):
        run_start = time.time()

        # ---- RESUME: skip if model exists ----
        model_path = os.path.join(MC_MODEL_DIR, f"MLP_{l1}_{l2}_{l3}_MC{mc}.pth")
        if os.path.exists(model_path):
            print(f"  [MC {mc}/{N_MC_RUNS}] → already done, skipping")
            continue

        print(f"  [MC {mc}/{N_MC_RUNS}] → running")

        # =======================================================
        # SOCAT-ONLY MC BEHAVIOR: NOISE ON TRAIN + VAL
        # =======================================================
        y_train_noisy = y_train + np.random.normal(0, unc_train, size=y_train.shape)
        y_val_noisy   = y_val   + np.random.normal(0, unc_val,   size=y_val.shape)

        ytrain_tensor = torch.tensor(y_train_noisy, dtype=torch.float32, device=device)
        yval_noisy_np = y_val_noisy.reshape(-1)

        train_loader = DataLoader(
            TensorDataset(xtrain_tensor_base, ytrain_tensor),
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        # ---- Model / optimizer ----
        model = MLP(len(input_columns), l1, l2, l3).to(device)
        optimizer = optim.Adam(model.parameters(), lr=LR)

        # ---- Early stopping ----
        best_val_rmse = float("inf")
        best_state = None
        counter = 0

        for epoch in range(MAX_EPOCHS):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

            # ---- Validation (SOCAT-only, compared to NOISY y_val) ----
            model.eval()
            with torch.no_grad():
                pred_val = model(xval_tensor).detach().cpu().numpy().reshape(-1)

            val_rmse = float(np.sqrt(mean_squared_error(yval_noisy_np, pred_val)))

            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_state = deepcopy(model.state_dict())
                counter = 0
            else:
                counter += 1
                if counter >= PATIENCE:
                    break

        if best_state is None:
            raise RuntimeError("best_state is None. Training did not produce a checkpoint.")

        # ---- Restore best state ----
        model.load_state_dict(best_state)
        model.eval()

        # =======================================================
        # CLEAN TEST (MATCH SOCAT-ONLY MC): NO NOISE ON TEST
        # =======================================================
        with torch.no_grad():
            test_pred = model(xtest_tensor).detach().cpu().numpy().reshape(-1)

        test_rmse = float(np.sqrt(mean_squared_error(y_test.reshape(-1), test_pred)))
        test_r2   = float(r2_score(y_test.reshape(-1), test_pred))

        # ---- Save stats ----
        stats_rows.append({
            "ensemble_id": ensemble_id,
            "arch": f"{l1}-{l2}-{l3}",
            "mc_run": mc,
            "val_rmse_noisy": best_val_rmse,
            "test_rmse": test_rmse,
            "test_r2": test_r2,
            "time_min": (time.time() - run_start) / 60.0
        })

        # ---- Save model ----
        torch.save(model.state_dict(), model_path)

        # ---- Increment ID & persist stats each run (crash-proof) ----
        ensemble_id += 1
        pd.DataFrame(stats_rows).to_csv(stats_file, index=False)

# ===============================================================
# Final message
# ===============================================================
total_minutes = (time.time() - SCRIPT_START_TIME) / 60.0
print(f"\n[INFO] Monte Carlo complete. Stats saved → {stats_file}")
print(f"[INFO] Models saved in → {MC_MODEL_DIR}")
print(f"[INFO] Total wall-clock time: {total_minutes:.2f} minutes")
