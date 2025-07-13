import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
from scipy.optimize import minimize
import pickle

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

target_recall = 0.90
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}/fine"
MODEL_WEIGHTS_PATH = f"models/fine/{model_name}_fine_weights"
TEMP_FILE = os.path.join(output_dir, "optimal_temperature.txt")
THRESHOLD_FILE = os.path.join(output_dir, "optimal_threshold_val.txt")

# === ENVIRONMENT ===
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# === TEMPERATURE SCALING ===
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

# === LOAD DATA ===
print("[INFO] Loading validation data...")
_, val_df, _, _, val_gen, _ = get_generators(IMG_SIZE, BATCH_SIZE)
val_labels = np.array(val_gen.classes)

# === LOAD MODEL AND WEIGHTS ===
print("[INFO] Rebuilding model...")
model, base_model = build_model(
    model_name=model_name,
    img_size=IMG_SIZE,
    dropout_head=DROPOUT_H,
    dropout_base=DROPOUT_F,
    l2_lambda_head=L2_REG_H,
    l2_lambda_base=L2_REG_F
)
model.load_weights(MODEL_WEIGHTS_PATH)
print("[INFO] Fine-tuned weights loaded successfully.")

# === PREDICT ===
print("[INFO] Predicting on validation set...")
val_probs = model.predict(val_gen, verbose=1).squeeze()

# === CALIBRATE TEMPERATURE ===
optimal_T, logits = optimize_temperature(val_probs, val_labels)
with open(TEMP_FILE, "w") as f:
    f.write(f"{optimal_T:.4f}\n")
print(f"[INFO] Optimal temperature saved: {optimal_T:.4f}")

scaled_logits = logits / optimal_T
scaled_probs = tf.sigmoid(scaled_logits).numpy()

# === THRESHOLDING BASED ON TARGET RECALL ===
print(f"[INFO] Searching for threshold with recall ≥ {target_recall:.2f}...")
precisions, recalls, thresholds = precision_recall_curve(val_labels, scaled_probs)

recall_condition = recalls >= target_recall
if np.any(recall_condition):
    best_idx = np.argmax(recall_condition)
    optimal_threshold = thresholds[best_idx]
    print(f"[INFO] Selected threshold: {optimal_threshold:.4f}")
    print(f"[INFO] Precision at threshold: {precisions[best_idx]:.4f}")
    print(f"[INFO] Recall at threshold: {recalls[best_idx]:.4f}")
else:
    optimal_threshold = 0.5
    print(f"[WARNING] No threshold found with Recall ≥ {target_recall}. Using default: 0.5")

# === SAVE TO FILE ===
with open(THRESHOLD_FILE, "w") as f:
    f.write(f"{optimal_threshold:.4f}\n")
print(f"[INFO] Threshold saved to {THRESHOLD_FILE}")
