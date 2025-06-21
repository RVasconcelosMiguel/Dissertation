import os
import sys
from datetime import datetime
from train import focal_loss

# --- Environment config ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Create necessary directories
output_dir = "/home/jtstudents/rmiguel/files_to_transfer"
os.makedirs(output_dir, exist_ok=True)

# --- Logging setup ---
log_filename = "evaluate_log.txt"
log_path = os.path.join(output_dir, log_filename)
log_file = open(log_path, "w")
sys.stdout = log_file
sys.stderr = log_file

print(f"[INFO] Evaluation started at: {datetime.now().isoformat()}")

TRAIN_CSV_NAME = "Augmented_Training_labels.csv"

# --- TensorFlow config ---
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(f"[INFO] Using GPU: {logical_gpus[0].name}")
    except RuntimeError as e:
        print(f"[ERROR] GPU configuration failed: {e}")
else:
    print("[INFO] No GPU found — using CPU.")

# --- Project imports ---
from data_loader import get_generators
from model import build_model
from sklearn.metrics import classification_report, roc_curve, confusion_matrix
from plot_utils import save_confusion_matrix, save_roc_curve
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- Config ---
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = "models/mobilenetv2_isic16.h5"

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"[ERROR] Model not found at {MODEL_PATH}. Please verify training completion.")

# --- Load model ---
model = load_model(MODEL_PATH, custom_objects={"focal_loss_fixed": focal_loss(gamma=2.0, alpha=0.75)})

# --- Load data ---
print("[INFO] Preparing generators...")
_, val_gen, test_gen = get_generators(train_csv_name=TRAIN_CSV_NAME, img_size=IMG_SIZE, batch_size=BATCH_SIZE)

# --- Threshold tuning on validation set ---
y_val_prob = model.predict(val_gen).flatten()
y_val_true = np.array(val_gen.classes)
fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]

print(f"[INFO] Optimal threshold (from validation): {optimal_threshold:.4f}")

# --- Evaluate on test set using tuned threshold ---
print("[INFO] Evaluating model on test set...")
results = model.evaluate(test_gen)
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

print("[INFO] Generating test predictions...")
y_prob = model.predict(test_gen).flatten()
y_pred = (y_prob >= optimal_threshold).astype(int)
y_true = np.array(test_gen.classes)
labels = list(test_gen.class_indices.keys())

print("\n[INFO] Classification Report (test set, tuned threshold):")
report = classification_report(y_true, y_pred, target_names=labels, digits=4)
print(report)

# Save confusion matrix and ROC for test set
save_confusion_matrix(y_true, y_pred, labels, os.path.join(output_dir, "confusion_matrix_tuned.png"))

plt.figure()
plt.plot(fpr, tpr, label=f"Validation AUC = {np.trapz(tpr, fpr):.4f}")
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f"Optimal threshold = {optimal_threshold:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (Validation-based Threshold)")
plt.legend()
plt.savefig(os.path.join(output_dir, "roc_curve_val_based.png"))
plt.close()

with open(os.path.join(output_dir, "optimal_threshold.txt"), "w") as f:
    f.write(f"Optimal threshold from validation: {optimal_threshold:.4f}\n\n")
    f.write(report)

print(f"[INFO] Evaluation completed at: {datetime.now().isoformat()}")
log_file.close()
