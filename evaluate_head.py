# === evaluate_head.py ===

import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from model import build_model, swish, cbam_block
from data_loader import get_generators
from plot_utils import save_confusion_matrix, save_roc_curve

# === CONFIGURATION ===
model_name = "efficientnetb2"
IMG_SIZE = 260
BATCH_SIZE = 16
THRESHOLD = 0.5

# === Paths ===
output_dir = "outputs/head/results"
MODEL_DIR = "outputs/head/model"
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, f"{model_name}_head_weights")

os.makedirs(output_dir, exist_ok=True)

# === Silence TensorFlow logging ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

start_time = time.time()
print(f"[INFO] Head evaluation started at: {time.ctime(start_time)}")

# === Data Loading ===
_, _, _, _, val_gen, test_gen = get_generators(IMG_SIZE, BATCH_SIZE)

# === Load Model ===
custom_objects = {"swish": swish, "cbam_block": cbam_block}

# === Build architecture and load weights ===
print(f"[INFO] Building model architecture for: {model_name}")
model, _ = build_model(
    model_name=model_name,
    img_size=IMG_SIZE,
    dropout_head=0.6,   # match training
    dropout_base=0.0,
    l2_lambda_head=1e-3,
    l2_lambda_base=0.0
)

# === Load weights ===
print(f"[INFO] Loading head-only weights from: {MODEL_WEIGHTS_PATH}")
model.load_weights(MODEL_WEIGHTS_PATH).expect_partial()

# === Compile Model for Evaluation ===
thresholded_metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
    tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
]

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=thresholded_metrics
)

# === Evaluate on Test Set ===
print("[INFO] Evaluating head model on test set...")
results = model.evaluate(test_gen, verbose=1)
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

# === Generate ROC Curve ===
print("[INFO] Generating ROC curve...")
y_prob = model.predict(test_gen, verbose=1).flatten()
y_true = np.array(test_gen.classes)

roc_curve_path = os.path.join(output_dir, "roc_curve_head_test.png")
save_roc_curve(y_true, y_prob, roc_curve_path)
roc_auc = roc_auc_score(y_true, y_prob)
print(f"[INFO] ROC curve saved to {roc_curve_path}")
print(f"[INFO] Test ROC AUC: {roc_auc:.4f}")

# === Save Prediction Probability Histogram ===
plt.figure(figsize=(8,6))
plt.hist(y_prob, bins=50, color='skyblue', edgecolor='black')
plt.title("Head Test Prediction Probabilities")
plt.xlabel("Predicted probability")
plt.ylabel("Count")
hist_path = os.path.join(output_dir, "test_pred_prob_head_hist.png")
plt.savefig(hist_path)
plt.close()
print(f"[INFO] Histogram saved to {hist_path}")

# === Threshold-based Predictions and Classification Report ===
print("[INFO] Generating classification report...")
y_pred = (y_prob >= THRESHOLD).astype(int)
labels = list(test_gen.class_indices.keys())

report = classification_report(y_true, y_pred, target_names=labels, digits=4)
print("[INFO] Classification report:\n")
print(report)

# === Confusion Matrix and Derived Metrics ===
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"Confusion Matrix: \n{cm}")
print(f"Accuracy: {results[model.metrics_names.index('accuracy')]:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# === Save Confusion Matrix ===
conf_matrix_path = os.path.join(output_dir, "confusion_matrix_head.png")
save_confusion_matrix(y_true, y_pred, labels, conf_matrix_path)
print(f"[INFO] Confusion matrix saved to {conf_matrix_path}")

# === Save Evaluation Report ===
eval_report_path = os.path.join(output_dir, "evaluation_report_head.txt")
with open(eval_report_path, "w") as f:
    f.write(f"Model evaluated: {model_name} (head-only)\n")
    f.write(f"Threshold used: {THRESHOLD:.4f}\n")
    f.write(f"Test ROC AUC: {roc_auc:.4f}\n\n")
    f.write(report)
print(f"[INFO] Evaluation report saved to {eval_report_path}")

# === End Time ===
end_time = time.time()
duration = end_time - start_time
print(f"[INFO] Head evaluation completed at: {time.ctime(end_time)}")
print(f"[INFO] Total evaluation time: {int(duration // 60)}m {int(duration % 60)}s")