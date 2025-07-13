import os
import numpy as np
import tensorflow as tf
from scipy.optimize import minimize

from model import build_model
from data_loader import get_generators

# === CONFIGURATION ===
model_name = "efficientnetb4"
IMG_SIZE = 380
BATCH_SIZE = 16

DROPOUT_H = 0.6
DROPOUT_F = 0.2
L2_REG_H = 1e-3
L2_REG_F = 1e-5

target_recall = 0.75  # Adjust to 0.90 or 0.95 depending on the desired recall

MODEL_DIR = "models/fine"
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}/fine"
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, f"{model_name}_fine_weights")
TEMP_FILE = os.path.join(output_dir, "optimal_temperature.txt")
THRESHOLD_FILE = os.path.join(output_dir, "optimal_threshold_val.txt")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# === TEMPERATURE SCALING UTILS ===
def nll_loss(T, logits, labels):
    scaled_logits = logits / T
    probs = tf.sigmoid(scaled_logits).numpy()
    epsilon = 1e-7
    probs = np.clip(probs, epsilon, 1 - epsilon)
    loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return loss

def optimize_temperature(val_probs, val_labels):
    logits = np.log(val_probs / (1 - val_probs))
    opt_result = minimize(
        nll_loss, x0=[1.0], args=(logits, val_labels),
        bounds=[(0.05, 10)]
    )
    return opt_result.x[0], logits

# === LOAD VALIDATION DATA ===
_, val_df, _, _, val_gen, _ = get_generators(IMG_SIZE, BATCH_SIZE)
val_labels = np.array(val_gen.classes)

# === LOAD MODEL ===
model, base_model = build_model(
    model_name=model_name,
    img_size=IMG_SIZE,
    dropout_head=DROPOUT_H,
    dropout_base=DROPOUT_F,
    l2_lambda_head=L2_REG_H,
    l2_lambda_base=L2_REG_F
)
model.load_weights(MODEL_WEIGHTS_PATH)
print("[INFO] Model weights loaded.")

# === PREDICT PROBABILITIES ON VALIDATION SET ===
val_probs = model.predict(val_gen).squeeze()

# === TEMPERATURE SCALING ===
optimal_T, logits = optimize_temperature(val_probs, val_labels)
with open(TEMP_FILE, "w") as f:
    f.write(f"{optimal_T:.4f}\n")
print(f"[INFO] Temperature scaling applied. T = {optimal_T:.4f}")

scaled_logits = logits / optimal_T
scaled_probs = tf.sigmoid(scaled_logits).numpy()

# === SELECT THRESHOLD BASED ON TARGET RECALL OVER TRUE POSITIVES ONLY ===
positive_indices = np.where(val_labels == 1)[0]
positive_probs = scaled_probs[positive_indices]

# Sort descending to find top k that achieve target recall
sorted_probs = np.sort(positive_probs)[::-1]
num_required = int(np.ceil(target_recall * len(sorted_probs)))
optimal_threshold = sorted_probs[num_required - 1]

print(f"[INFO] Threshold chosen to ensure Recall ≥ {target_recall:.2f} on true positives only.")
print(f"[INFO] Threshold = {optimal_threshold:.4f} yields recall = {target_recall:.2f} by construction.")

# === SAVE THRESHOLD TO FILE ===
with open(THRESHOLD_FILE, "w") as f:
    f.write(f"{optimal_threshold:.4f}\n")
print(f"[INFO] Saved threshold to: {THRESHOLD_FILE}")

# === ANALYSIS: DISPLAY SORTED TRUE POSITIVE PROBABILITIES ===
print("\n[INFO] Predicted probabilities for actual Class 1 samples (after temperature scaling):")
sorted_indices = np.argsort(positive_probs)
for rank in sorted_indices:
    idx = positive_indices[rank]
    print(f"Index: {idx:3d} | Prob: {scaled_probs[idx]:.4f}")
