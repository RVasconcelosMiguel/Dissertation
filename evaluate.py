# === evaluate.py ===
import os
import sys
import logging
import warnings
from datetime import datetime

# === Silence TensorFlow logging and warnings ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# === Output Setup ===
output_dir = "/home/jtstudents/rmiguel/files_to_transfer"
os.makedirs(output_dir, exist_ok=True)
log_file_path = os.path.join(output_dir, "evaluate_log.txt")
log_file = open(log_file_path, "w")
sys.stdout = log_file
sys.stderr = log_file

start_time = datetime.now()
print(f"[INFO] Evaluation started at: {start_time.isoformat()}")

# === Imports ===
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_curve, roc_auc_score

from model import build_model
from data_loader import get_generators
from plot_utils import save_confusion_matrix

# === CONFIGURATION ===
IMG_SIZE = 224
BATCH_SIZE = 32
WEIGHTS_PATH = "models/efficientnetb1_finetuned_weights"
TRAIN_CSV_NAME = "Augmented_Training_labels.csv"
CALCULATE_OPTIMAL_THRESHOLD = False  # set True to calculate, False to use fixed
THRESHOLD = 0.5  # fixed threshold if CALCULATE_OPTIMAL_THRESHOLD is False

# === Data Load ===
_, _, _, _, val_gen, test_gen = get_generators(IMG_SIZE, BATCH_SIZE)

# === Build Model ===
print("[INFO] Building model architecture...")
model, _ = build_model(img_size=IMG_SIZE)

# === Load Weights ===
print("[INFO] Loading weights from:", WEIGHTS_PATH)
if not os.path.exists(WEIGHTS_PATH + ".index"):
    raise FileNotFoundError(f"Missing weights: {WEIGHTS_PATH}.index")
model.load_weights(WEIGHTS_PATH)

# === Compile for Evaluation ===
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# === Determine Threshold ===
if CALCULATE_OPTIMAL_THRESHOLD:
    print("[INFO] Calculating optimal threshold from validation set...")
    y_val_prob = model.predict(val_gen).flatten()
    y_val_true = np.array(val_gen.classes)

    fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx]
    roc_auc = roc_auc_score(y_val_true, y_val_prob)

    if not np.isfinite(optimal_threshold):
        print("[WARNING] Invalid threshold detected; defaulting to 0.5")
        optimal_threshold = 0.5

    print(f"[INFO] Optimal threshold (val): {optimal_threshold:.4f}")
    print(f"[INFO] Validation ROC AUC: {roc_auc:.4f}")

else:
    optimal_threshold = THRESHOLD
    print(f"[INFO] Using fixed threshold: {optimal_threshold:.4f}")

# === Compile with Thresholded Metrics ===
thresholded_metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=optimal_threshold),
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision", thresholds=optimal_threshold),
    tf.keras.metrics.Recall(name="recall", thresholds=optimal_threshold),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=3e-5),
    loss="binary_crossentropy",
    metrics=thresholded_metrics
)

# === Evaluate on Test Set ===
print("[INFO] Evaluating on test set...")
results = model.evaluate(test_gen, verbose=1)
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

# === Threshold-based Predictions ===
print("[INFO] Generating test predictions with chosen threshold...")
y_prob = model.predict(test_gen).flatten()
y_pred = (y_prob >= optimal_threshold).astype(int)
y_true = np.array(test_gen.classes)
labels = list(test_gen.class_indices.keys())

print("[INFO] Classification report:")
report = classification_report(y_true, y_pred, target_names=labels, digits=4)
print(report)

# === Save Confusion Matrix ===
save_confusion_matrix(y_true, y_pred, labels, os.path.join(output_dir, "confusion_matrix_tuned.png"))

# === Optionally Save ROC Curve if calculated ===
if CALCULATE_OPTIMAL_THRESHOLD:
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.scatter(fpr[optimal_idx], tpr[optimal_idx], color='red', label=f"Threshold = {optimal_threshold:.4f}")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve (Validation Set)")
    plt.legend()
    plt.savefig(os.path.join(output_dir, "roc_curve_val_based.png"))
    plt.close()

# === Save Report ===
with open(os.path.join(output_dir, "evaluation_report.txt"), "w") as f:
    f.write(f"Threshold used: {optimal_threshold:.4f}\n")
    if CALCULATE_OPTIMAL_THRESHOLD:
        f.write(f"Validation ROC AUC: {roc_auc:.4f}\n")
    f.write("\n" + report)

end_time = datetime.now()
duration = end_time - start_time
print(f"[INFO] Evaluation completed at: {end_time.isoformat()}")
print(f"[INFO] Total evaluation time: {duration}")

sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()
