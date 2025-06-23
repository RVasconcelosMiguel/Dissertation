# === evaluate.py ===
import os
import sys
from datetime import datetime
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
output_dir = "/home/jtstudents/rmiguel/files_to_transfer"
os.makedirs(output_dir, exist_ok=True)
log_file = open(os.path.join(output_dir, "evaluate_log.txt"), "w")
sys.stdout = log_file
sys.stderr = log_file
print(f"[INFO] Evaluation started at: {datetime.now().isoformat()}")

TRAIN_CSV_NAME = "Augmented_Training_labels.csv"

import tensorflow as tf
from sklearn.metrics import classification_report, roc_curve
from tensorflow.keras.models import load_model
from model import build_model
from data_loader import get_generators
from plot_utils import save_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from train import focal_loss

IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = "models/efficientnetb1_isic16.h5"
if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError("Model not found.")

model = load_model(MODEL_PATH, custom_objects={"focal_loss_fixed": focal_loss()})
_, val_gen, test_gen = get_generators(TRAIN_CSV_NAME, IMG_SIZE, BATCH_SIZE)

print("[INFO] Predicting validation set...")
y_val_prob = model.predict(val_gen).flatten()
y_val_true = np.array(val_gen.classes)
fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]
print(f"[INFO] Optimal threshold (val): {optimal_threshold:.4f}")

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

with open(os.path.join(output_dir, "optimal_threshold.txt"), "w") as f:
    f.write(f"Optimal threshold from validation: {optimal_threshold:.4f}\n\n")
    f.write(report)

print(f"[INFO] Evaluation completed at: {datetime.now().isoformat()}")
log_file.close()