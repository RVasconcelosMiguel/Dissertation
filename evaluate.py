# === evaluate.py ===
import os
import sys
import logging
import warnings
from datetime import datetime

# === Silence TensorFlow logging and warnings ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 3 = errors only
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf_logger = logging.getLogger('tensorflow')
tf_logger.setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# === Set output directory and log file ===
output_dir = "/home/jtstudents/rmiguel/files_to_transfer"
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "evaluate_log.txt")

log_file = open(log_file_path, "w")
sys.stdout = log_file
sys.stderr = log_file

print(f"[INFO] Evaluation started at: {datetime.now().isoformat()}")

# === Imports ===
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve

from model import build_model
from data_loader import get_generators
from plot_utils import save_confusion_matrix
from losses import FocalLoss  # ✅ Needed now that we use compile()

# === CONFIGURATION ===
IMG_SIZE = 224
BATCH_SIZE = 32
WEIGHTS_PATH = "models/efficientnetb1_finetuned_weights"
TRAIN_CSV_NAME = "Augmented_Training_labels.csv"

# === Load Data ===
_, val_gen, test_gen = get_generators(TRAIN_CSV_NAME, IMG_SIZE, BATCH_SIZE)

# === Build & Load Model ===
print("[INFO] Building model architecture...")
model, _ = build_model(img_size=IMG_SIZE)

print(f"[INFO] Loading weights from: {WEIGHTS_PATH}")
if not os.path.isfile(WEIGHTS_PATH + ".index"):
    raise FileNotFoundError("Weights not found. Expected .index and .data* files.")

model.load_weights(WEIGHTS_PATH)

# === Compile the model after loading weights ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss=FocalLoss(gamma=2.0, alpha=0.75),
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

# === Validation Threshold ===
print("[INFO] Predicting validation set...")
y_val_prob = model.predict(val_gen).flatten()
y_val_true = np.array(val_gen.classes)
fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
print(f"[INFO] Optimal threshold (val): {optimal_threshold:.4f}")

# === Test Evaluation ===
print("[INFO] Evaluating on test set...")
results = model.evaluate(test_gen)
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

y_prob = model.predict(test_gen).flatten()
y_pred = (y_prob >= optimal_threshold).astype(int)
y_true = np.array(test_gen.classes)
labels = list(test_gen.class_indices.keys())

print("[INFO] Classification report:")
report = classification_report(y_true, y_pred, target_names=labels, digits=4)
print(report)

# === Save Plots and Metrics ===
save_confusion_matrix(y_true, y_pred, labels, os.path.join(output_dir, "confusion_matrix_tuned.png"))

plt.figure()
plt.plot(fpr, tpr, label=f"Validation AUC = {np.trapz(tpr, fpr):.4f}")
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f"Threshold = {optimal_threshold:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(output_dir, "roc_curve_val_based.png"))
plt.close()

# === Save Report ===
with open(os.path.join(output_dir, "optimal_threshold.txt"), "w") as f:
    f.write(f"Optimal threshold from validation: {optimal_threshold:.4f}\n\n")
    f.write(report)

print(f"[INFO] Evaluation completed at: {datetime.now().isoformat()}")

# === Restore stdout/stderr and close file ===
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()
