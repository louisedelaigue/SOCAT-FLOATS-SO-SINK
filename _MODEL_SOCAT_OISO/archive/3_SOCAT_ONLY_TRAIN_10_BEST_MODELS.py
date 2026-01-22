# ===============================================================
# Train Top MLP Architectures on SOCAT 1×1° Gridded Monthly Data
# Updated for NEW 2025 Pipeline (MLD, SLA, RRS, PAR, WIND, ATM CO2)
# Louise Delaigue – 2025
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

# ===============================================================
# GPU Setup
# ===============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {device}")

# ===============================================================
# Paths
# ===============================================================
BASE_PATH = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/MODEL_SOCAT_ONLY_OBS_v2/processing_steps/"
results_path = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/archive/MODEL_SOCAT_ONLY_OBS/models/ensemble_results_from_GRID.csv"
model_output_dir = "/home/ldelaigue/Documents/Python/SOCAT-OSSE-SO/MODEL_SOCAT_ONLY_OBS_v2/models/"
os.makedirs(model_output_dir, exist_ok=True)

# ===============================================================
# Load Pre-Split Data
# ===============================================================
print("[INFO] Loading pre-split datasets...")
train_df = pd.read_csv(BASE_PATH + "pco2_train_data.csv")
val_df   = pd.read_csv(BASE_PATH + "pco2_validation_data.csv")
test_df  = pd.read_csv(BASE_PATH + "pco2_test_data.csv")
indep_df = pd.read_csv(BASE_PATH + "pco2_independent_validation_data.csv")

print(f"[INFO] Shapes — Train: {train_df.shape}, Val: {val_df.shape}, Test: {test_df.shape}, Independent: {indep_df.shape}")

# ===============================================================
# Variable Names Already Match Our Pipeline (NO RENAMING NEEDED)
# ===============================================================
# — fco2_rec_median is the new SOCAT fCO₂
# — SOCAT_temperature, SOCAT_salinity
# — MLD already exists
# — sla_mean from SLA step
# — PAR_mean from PAR
# — rrs412 ... rrs670 already lowercase
# — wind_speed, u10, v10 already present

# ===============================================================
# Define ML Input Features
# ===============================================================

input_columns = [
    # ARMOR + SOCAT core ocean state
    'SOCAT_temperature', 'SOCAT_salinity', 'MLD', 'sla_mean',

    # Biology / optics
    'rrs412', 'rrs443', 'rrs490', 'rrs555', 'rrs670',
    'PAR_mean',

    # Physics / forcing
    'u10', 'v10', 'wind_speed',

    # Atmospheric
    'atm_co2',

    # Spatiotemporal features
    'decimal_year', 'doy_sin', 'doy_cos',
    'x_cart', 'y_cart', 'z_cart',
    'bottom_depth_m'
]

output_column = "pco2"

# ===============================================================
# Prepare Data
# ===============================================================
print("[INFO] Preparing input and output arrays...")

X_train = train_df[input_columns].values
y_train = train_df[output_column].values.reshape(-1, 1)

X_val   = val_df[input_columns].values
y_val   = val_df[output_column].values.reshape(-1, 1)

print(f"[INFO] X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"[INFO] X_val:   {X_val.shape}, y_val:   {y_val.shape}")

# ===============================================================
# Normalize Inputs
# ===============================================================
print("[INFO] Normalizing features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# Save scaler
scaler_path = BASE_PATH + "trained_scaler.joblib"
joblib.dump(scaler, scaler_path)
print(f"[INFO] Scaler saved → {scaler_path}")

# ===============================================================
# Convert to Tensors
# ===============================================================
xtrain_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
ytrain_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

xval_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
yval_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# ===============================================================
# DataLoader
# ===============================================================
batch_size = 256
train_dataset = TensorDataset(xtrain_tensor, ytrain_tensor)
train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print(f"[INFO] DataLoader initialized (batch={batch_size})")

# ===============================================================
# Load Top 10 Architectures
# ===============================================================
print("[INFO] Loading top MLP architectures...")
top_models = pd.read_csv(results_path).sort_values("val_rmse").head(10)

# ===============================================================
# Training Loop
# ===============================================================
STATISTICS = []

for idx, row in top_models.iterrows():
    start_time = time.time()
    l1, l2, l3 = int(row['layer1']), int(row['layer2']), int(row['layer3'])

    print(f"\n[INFO] === Training model {idx+1}/10 | Architecture: {l1}-{l2}-{l3} ===")

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(len(input_columns), l1)
            self.fc2 = nn.Linear(l1, l2)
            self.fc3 = nn.Linear(l2, l3)
            self.fc4 = nn.Linear(l3, 1)
            self.act = nn.Tanh()

        def forward(self, x):
            x = self.act(self.fc1(x))
            x = self.act(self.fc2(x))
            x = self.act(self.fc3(x))
            return self.fc4(x)

    model = MLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    criterion = nn.HuberLoss()

    best_loss = float("inf")
    patience = 100
    counter = 0

    # --------------------
    # Training
    # --------------------
    print("[INFO] Training...")
    for epoch in range(1600):
        model.train()
        epoch_loss = 0

        for xb, yb in train_loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss /= len(train_loader)

        if epoch % 25 == 0:
            print(f"[Epoch {epoch+1}] loss = {epoch_loss:.5f}")

        # Early stopping
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"[INFO] Early stopping at epoch {epoch+1}")
                break

    # --------------------
    # Validation
    # --------------------
    model.eval()
    with torch.no_grad():
        preds = model(xval_tensor).cpu().numpy().flatten()
        true  = yval_tensor.cpu().numpy().flatten()

    rmse = mean_squared_error(true, preds, squared=False)
    r2   = r2_score(true, preds)

    print(f"[RESULT] RMSE={rmse:.4f}, R²={r2:.4f}, epochs={epoch+1}")

    # Save model
    model_path = f"{model_output_dir}/MLP_{l1}_{l2}_{l3}.pth"
    torch.save(model.state_dict(), model_path)
    print(f"[INFO] Model saved → {model_path}")
    
    end_time = time.time()
    minutes = (end_time - start_time) / 60
    print(f"[INFO] Training time for model {idx+1}: {minutes:.2f} minutes")

    # Save statistics
    STATISTICS.append({
        'layer1': l1, 'layer2': l2, 'layer3': l3,
        'train_loss': epoch_loss,
        'val_rmse': rmse,
        'val_r2': r2,
        'epochs': epoch+1
    })

# ===============================================================
# Save Summary Statistics
# ===============================================================
results_df = pd.DataFrame(STATISTICS).sort_values("val_rmse")
stats_path = f"{model_output_dir}/retrained_model_stats.csv"
results_df.to_csv(stats_path, index=False)

print("\n[INFO] Training complete.")
print(f"[INFO] Statistics saved → {stats_path}")
print(results_df.head())
