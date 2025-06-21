import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from train import focal_loss
from data_loader import get_generators
from plot_utils import save_confusion_matrix

# --- Config ---
IMG_SIZE = 224
BATCH_SIZE = 32
TRAIN_CSV_NAME = "Augmented_Training_labels.csv"
MODEL_PATH = "models/mobilenetv2_isic16.h5"
OUTPUT_DIR = "/home/jtstudents/rmiguel/files_to_transfer"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load model ---
model = load_model(MODEL_PATH, custom_objects={"focal_loss_fixed": focal_loss(gamma=2.0, alpha=0.75)})

# --- Load test data ---
_, _, test_gen = get_generators(
    train_csv_name=TRAIN_CSV_NAME,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# --- Predict probabilities ---
y_true = test_gen.classes
y_prob = model.predict(test_gen).flatten()

# --- ROC Curve and Threshold Optimization ---
fpr, tpr, thresholds = roc_curve(y_true, y_prob)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]

print(f"[INFO] Optimal threshold found at: {optimal_threshold:.4f}")

# --- Apply new threshold ---
y_pred = (y_prob >= optimal_threshold).astype(int)

# --- Metrics ---
labels = list(test_gen.class_indices.keys())
report = classification_report(y_true, y_pred, target_names=labels, digits=4)
print("\n[INFO] Classification Report with Tuned Threshold:\n")
print(report)

# --- Confusion Matrix ---
conf_matrix = confusion_matrix(y_true, y_pred)
conf_matrix_path = os.path.join(OUTPUT_DIR, "confusion_matrix_tuned.png")
save_confusion_matrix(y_true, y_pred, labels, conf_matrix_path)

# --- Save ROC Curve ---
plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {np.trapz(tpr, fpr):.4f}")
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f"Optimal threshold = {optimal_threshold:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve with Optimal Threshold")
plt.legend()
roc_path = os.path.join(OUTPUT_DIR, "roc_curve_tuned.png")
plt.savefig(roc_path)
plt.close()

# --- Save threshold and report ---
with open(os.path.join(OUTPUT_DIR, "optimal_threshold.txt"), "w") as f:
    f.write(f"Optimal threshold: {optimal_threshold:.4f}\n\n")
    f.write(report)

print("[INFO] Threshold tuning complete. Outputs saved to:", OUTPUT_DIR)
