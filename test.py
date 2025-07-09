import os
import pickle
import time
from plot_utils import plot_history_head

# === CONFIGURATION ===
model_name = "efficientnetb4"
MODEL_DIR = "models/head"
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}/head"

history_path = os.path.join(MODEL_DIR, f"history_{model_name}_head.pkl")

# === Load history ===
with open(history_path, "rb") as f:
    history_head = pickle.load(f)
print(f"[INFO] Loaded history from {history_path}")

# === Start timer ===
start_time = time.time()

# === PLOTTING ===
plot_history_head(
    history_head,
    save_path=output_dir,
    metrics=["accuracy", "loss", "auc", "precision", "recall"]
)

# === TRAINING TIME (for this plotting script) ===
elapsed_time = time.time() - start_time
print(f"[INFO] Plotting completed in: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
