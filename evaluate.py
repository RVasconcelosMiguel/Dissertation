# === evaluate.py ===
import os
import sys
import logging
import warnings
from datetime import datetime

# === CONFIGURATION ===
model_name = "efficientnetb2"  # options: "efficientnetb0", "efficientnetb1", "efficientnetb2"
IMG_SIZE = 260
BATCH_SIZE = 32

# === Paths ===
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}"
os.makedirs(output_dir, exist_ok=True)

WEIGHTS_PATH = f"models/{model_name}_finetuned_weights"
threshold_path = os.path.join(output_dir, "optimal_threshold_val.txt")
log_file_path = os.path.join(output_dir, "evaluate_log.txt")

# === Silence TensorFlow logging and warnings ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
logging.getLogger('tensorflow').setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# === Redirect stdout and stderr ===
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

# === Load saved optimal threshold from validation ===
with open(threshold_path, "r") as f:
    optimal_threshold = float(f.read().strip())
print(f"[INFO] Loaded optimal threshold from training (validation-based): {optimal_threshold:.4f}")

# === Data Loading ===
_, _, _, _, val_gen, test_gen = get_generators(IMG_SIZE, BATCH_SIZE)

# === Build Model ===
print(f"[INFO] Building model architecture: {model_name}...")
model, _ = build_model(
    model_name=model_name,
    img_size=IMG_SIZE,
    dropout=0.0,
    l2_lambda=1e-4
)

# === Load Trained Weights ===
print(f"[INFO] Loading weights from: {WEIGHTS_PATH}")
if not os.path.exists(WEIGHTS_PATH + ".index"):
    raise FileNotFoundError(f"Missing weights: {WEIGHTS_PATH}.index")
model.load_weights(WEIGHTS_PATH)

# === Compile Model for Evaluation ===
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

# === Evaluate on Test Set (applying threshold from validation) ===
print("[INFO] Evaluating on test set (no further training)...")
results = model.evaluate(test_gen, verbose=1)
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

# === Generate ROC Curve for test set ===
print("[INFO] Generating ROC curve for test set...")
y_prob = model.predict(test_gen).flatten()
y_true = np.array(test_gen.classes)

fpr, tpr, thresholds = roc_curve(y_true, y_prob)
roc_auc = roc_auc_score(y_true, y_prob)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title(f"ROC Curve (Test Set) - {model_name}")
plt.legend()
plt.savefig(os.path.join(output_dir, "roc_curve_test.png"))
plt.close()

# === Threshold-based Predictions and Classification Report ===
print("[INFO] Generating test predictions with loaded threshold (no threshold optimisation on test set)...")
y_pred = (y_prob >= optimal_threshold).astype(int)
labels = list(test_gen.class_indices.keys())

report = classification_report(y_true, y_pred, target_names=labels, digits=4)
print("[INFO] Classification report:")
print(report)

# === Save Confusion Matrix ===
save_confusion_matrix(y_true, y_pred, labels, os.path.join(output_dir, "confusion_matrix_tuned.png"))

# === Save Evaluation Report ===
with open(os.path.join(output_dir, "evaluation_report.txt"), "w") as f:
    f.write(f"Model evaluated: {model_name}\n")
    f.write(f"Threshold used (from validation): {optimal_threshold:.4f}\n")
    f.write(f"Test ROC AUC: {roc_auc:.4f}\n\n")
    f.write(report)

end_time = datetime.now()
duration = end_time - start_time
print(f"[INFO] Evaluation completed at: {end_time.isoformat()}")
print(f"[INFO] Total evaluation time: {duration}")

# === Close Logging ===
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()
