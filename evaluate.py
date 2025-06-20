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
from model import build_model  # <--- add this
from sklearn.metrics import classification_report
from plot_utils import save_confusion_matrix, save_roc_curve
import numpy as np
from tensorflow.keras.models import load_model

# --- Config ---
IMG_SIZE = 224
BATCH_SIZE = 32
MODEL_PATH = "models/efficientnetb0_isic16.h5"

if not os.path.isfile(MODEL_PATH):
    raise FileNotFoundError(f"[ERROR] Model not found at {MODEL_PATH}. Please verify training completion.")

# --- Load model ---
model = load_model("models/mobilenetv2_isic16.h5", custom_objects={"focal_loss_fixed": focal_loss(gamma=2.0, alpha=0.25)})

# --- Load data ---
print("[INFO] Preparing test generator...")
_, _, test_gen = get_generators(img_size=IMG_SIZE, batch_size=BATCH_SIZE)

if hasattr(test_gen, 'shuffle') and test_gen.shuffle:
    print("[WARNING] Test generator is shuffled — this may desynchronize labels and predictions.")

# --- Evaluation ---
try:
    print("[INFO] Evaluating model on test set...")
    results = model.evaluate(test_gen)
    for name, val in zip(model.metrics_names, results):
        print(f"{name}: {val:.4f}")
except Exception as e:
    print(f"[ERROR] Evaluation failed: {e}")

# --- Predictions and metrics ---
print("[INFO] Generating predictions...")
y_prob = model.predict(test_gen)
y_pred = np.array((y_prob > 0.5).astype(int).flatten())

y_true = np.array(test_gen.classes)
labels = list(test_gen.class_indices.keys())

y_true = np.array(y_true)
print(f"[INFO] y_true shape: {y_true.shape}, y_pred shape: {y_pred.shape}")

print("\n[INFO] Classification Report:")
print(classification_report(y_true, y_pred, target_names=labels))

# --- Confusion matrix ---
print("[INFO] Saving confusion matrix plot...")
confusion_matrix_filename = "confusion_matrix.png"
confusion_matrix_path = os.path.join(output_dir, confusion_matrix_filename)
save_confusion_matrix(y_true, y_pred, labels, confusion_matrix_path)

# --- ROC curve ---
roc_filename = "roc_curve.png"
roc_path = os.path.join(output_dir, roc_filename)
print("[INFO] Saving ROC curve plot...")
save_roc_curve(y_true, y_prob.flatten(), roc_path)

print(f"[INFO] Evaluation completed at: {datetime.now().isoformat()}")
log_file.close()
