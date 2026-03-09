# ===============================================================
# Random Search MLP Training on SOCAT+FLOATS 1×1° Gridded Monthly Data
# Weighted Huber Loss (uncertainty-based), SOCAT Validation Constraint, Early Stopping
# NO LOG TRANSFORM – same setup as unweighted, just with sample weights
# ===============================================================

import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import time
from copy import deepcopy

# ===============================================================
# GPU Setup
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ===============================================================
# Paths
# ===============================================================
BASE_PATH = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_AND_FLOATS_v2/processing_steps/"
SOCAT_ONLY_PATH = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"

model_output_dir = "/remote/unity/bgc-output/DELAIGUE/SOCAT-FLOATS-OSSE/MODEL_SOCAT_AND_FLOATS_v2/models/weighted/"
os.makedirs(model_output_dir, exist_ok=True)

stats_path = f"{model_output_dir}/random_search_model_stats.csv"

if os.path.exists(stats_path):
    os.remove(stats_path)
    print(f"[INFO] Removed old stats file → {stats_path}")

# ===============================================================
# Load Data
# ===============================================================
print("[INFO] Loading datasets...")

train_df = pd.read_csv(BASE_PATH + "pco2_train_data.csv")     # SOCAT + FLOATS mixed (with sample_weight)
val_df   = pd.read_csv(SOCAT_ONLY_PATH + "pco2_validation_data.csv")
test_df  = pd.read_csv(SOCAT_ONLY_PATH + "pco2_test_data.csv")

print(f"[INFO] Shapes — Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}")

# ===============================================================
# Fix SOCAT-only validation & test columns
# ===============================================================
val_df.rename(columns={
    "SOCAT_temperature": "temperature",
    "SOCAT_salinity": "salinity"
}, inplace=True)

test_df.rename(columns={
    "SOCAT_temperature": "temperature",
    "SOCAT_salinity": "salinity"
}, inplace=True)

val_df["uncertainty_pco2"] = 5.0
test_df["uncertainty_pco2"] = 5.0

# ===============================================================
# Input Features (PURE PHYSICAL INPUTS)
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
    "bottom_depth_m"
]

output_column = "pco2"

# ===============================================================
# Extract y and sample weights
# ===============================================================
y_train = train_df[output_column].values.reshape(-1, 1)
y_val   = val_df[output_column].values.reshape(-1, 1)
y_test  = test_df[output_column].values.reshape(-1, 1)

# Use the sample_weight you computed from uncertainty in the fusion step
# sample_weight = 1 / (1 + uncertainty_pco2 / SIGMA0)
w_train = train_df["sample_weight"].values.reshape(-1, 1)

# ===============================================================
# Normalize Inputs (ALL FEATURES SCALED)
# ===============================================================
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(train_df[input_columns])
X_val_scaled   = scaler.transform(val_df[input_columns])
X_test_scaled  = scaler.transform(test_df[input_columns])

scaler_path = os.path.join(model_output_dir, "trained_scaler.joblib")
joblib.dump(scaler, scaler_path)
print(f"[INFO] Scaler saved → {scaler_path}")

# ===============================================================
# Convert to Tensors
# ===============================================================
xtrain_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
ytrain_tensor = torch.tensor(y_train,       dtype=torch.float32).to(device)
wtrain_tensor = torch.tensor(w_train,       dtype=torch.float32).to(device)

xval_tensor   = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
yval_tensor   = torch.tensor(y_val,        dtype=torch.float32).to(device)

xtest_tensor  = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
ytest_tensor  = torch.tensor(y_test,        dtype=torch.float32).to(device)

# ===============================================================
# DataLoader (WITH WEIGHTS)
# ===============================================================
batch_size = 256

train_loader = DataLoader(
    TensorDataset(xtrain_tensor, ytrain_tensor, wtrain_tensor),
    batch_size=batch_size,
    shuffle=True
)

# ===============================================================
# Loss Function – WEIGHTED HUBER
# ===============================================================
# We get per-sample Huber loss (reduction='none'), then apply weights
criterion = nn.HuberLoss(delta=5.0, reduction='none')

def weighted_huber(pred, target, weight):
    # pred, target, weight ~ (batch, 1)
    raw_loss = criterion(pred, target)       # (batch, 1)
    return (raw_loss * weight).mean()

# ===============================================================
# Random Search Space
# ===============================================================
rng = np.random.default_rng(42)
search_space = []

neurons1 = np.arange(90, 121)
neurons2 = np.arange(70, 101)
neurons3 = np.arange(50, 81)

while len(search_space) < 200:
    l1 = int(rng.choice(neurons1))
    l2 = int(rng.choice(neurons2))
    l3 = int(rng.choice(neurons3))
    if l1 >= l2 >= l3:
        search_space.append((l1, l2, l3))

print(f"[INFO] Random search prepared with {len(search_space)} architectures.")

# ===============================================================
# Training Loop
# ===============================================================
for idx, (l1, l2, l3) in enumerate(search_space, start=1):

    print("\n===================================================")
    print(f"[INFO] Training model {idx}/200 | Architecture: {l1}-{l2}-{l3}")
    print("===================================================")

    start_time = time.time()

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(len(input_columns), l1)
            self.fc2 = nn.Linear(l1, l2)
            self.fc3 = nn.Linear(l2, l3)
            self.fc4 = nn.Linear(l3, 1)
            self.act = nn.GELU()

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            return self.fc4(x)

    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=2e-4)

    best_val_rmse = float("inf")
    best_state = None
    patience = 100
    counter = 0
    max_epochs = 2000

    for epoch in range(max_epochs):

        # ---------------- TRAIN ----------------
        model.train()
        epoch_loss = 0.0

        for xb, yb, wb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = weighted_huber(pred, yb, wb)   # WEIGHTED LOSS
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        # ---------------- VALIDATION (SOCAT ONLY, UNWEIGHTED METRIC) ----------------
        model.eval()
        with torch.no_grad():
            val_pred = model(xval_tensor).cpu().numpy().flatten()
            val_rmse = np.sqrt(mean_squared_error(y_val.flatten(), val_pred))

        if epoch % 50 == 0:
            print(f"[Epoch {epoch+1}] Loss={epoch_loss:.4f} | ValRMSE={val_rmse:.4f}")

        # ---------------- EARLY STOPPING ----------------
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_state = deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

    # Restore best checkpoint
    model.load_state_dict(best_state)

    # ---------------- FINAL EVALUATION ----------------
    with torch.no_grad():
        final_pred = model(xval_tensor).cpu().numpy().flatten()

    final_rmse = np.sqrt(mean_squared_error(y_val.flatten(), final_pred))
    final_r2   = r2_score(y_val.flatten(), final_pred)

    with torch.no_grad():
        test_pred = model(xtest_tensor).cpu().numpy().flatten()

    test_rmse = np.sqrt(mean_squared_error(y_test.flatten(), test_pred))
    test_r2   = r2_score(y_test.flatten(), test_pred)

    print(f"[TEST] RMSE={test_rmse:.4f}, R²={test_r2:.4f}")

    # ---------------- SAVE MODEL ----------------
    model_path = f"{model_output_dir}/MLP_{l1}_{l2}_{l3}.pth"
    torch.save(model.state_dict(), model_path)

    # ---------------- SAVE RECORD ----------------
    minutes = (time.time() - start_time) / 60

    row = pd.DataFrame([{
        "model": idx,
        "layer1": l1,
        "layer2": l2,
        "layer3": l3,
        "val_rmse": final_rmse,
        "val_r2": final_r2,
        "test_rmse": test_rmse,
        "test_r2": test_r2,
        "time_min": minutes
    }])

    row.to_csv(stats_path, mode="a", header=not os.path.exists(stats_path), index=False)
    print(f"[INFO] Saved stats for model {idx}")

print("\n[INFO] Random search complete.")
print(f"[INFO] Results saved to → {stats_path}")
