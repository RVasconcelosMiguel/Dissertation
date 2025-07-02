import os
import pickle
from plot_utils import plot_history

# === CONFIGURATION (must match your train.py) ===
model_name = "efficientnetb1"

# === PATHS ===
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}"
history_path = f"models/history_{model_name}.pkl"

# === METRICS TO PLOT ===
metrics = ["accuracy", "loss", "auc", "precision", "recall"]

# === LOAD HISTORY ===
if not os.path.exists(history_path):
    raise FileNotFoundError(f"[ERROR] History file not found at {history_path}")

with open(history_path, "rb") as f:
    history_all = pickle.load(f)

print(f"[INFO] Loaded history from {history_path}")

# === PLOT ===
plot_history(history_all, save_path=output_dir, metrics=metrics)

print(f"[INFO] Finished plotting metrics. Saved to {output_dir}")
