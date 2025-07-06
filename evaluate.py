# === evaluate.py ===

import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score
from data_loader import get_generators
from model import build_model
from plot_utils import save_confusion_matrix, save_roc_curve, plot_history
import pickle

# === CONFIGURATION ===
model_name = "efficientnetb3"
IMG_SIZE = 300
BATCH_SIZE = 32

# === Paths ===
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}"
os.makedirs(output_dir, exist_ok=True)

WEIGHTS_PATH = f"models/{model_name}_weights"
history_path = f"models/history_{model_name}.pkl"
threshold_path = os.path.join(output_dir, "optimal_threshold_val.txt")

# === ENVIRONMENT SETUP ===
start_time = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print(f"[INFO] Evaluation started at: {time.ctime(start_time)}")

# === Load optimal threshold ===
if os.path.exists(threshold_path):
    with open(threshold_path, "r") as f:
        optimal_threshold = float(f.read().strip())
else:
    optimal_threshold = 0.5  # fallback
print(f"[INFO] Loaded optimal threshold: {optimal_threshold:.4f}")

# === Load data ===
_, _, _, _, val_gen, test_gen = get_generators(IMG_SIZE, BATCH_SIZE)

# === Build model ===
print("[INFO] Building model...")
model, _ = build_model(model_name=model_name, img_size=IMG_SIZE, dropout=0.5, l2_lambda=5e-4)

# === Load weights ===
print(f"[INFO] Loading weights from {WEIGHTS_PATH}")
model.load_weights(WEIGHTS_PATH).expect_partial()

# === Compile with dummy optimizer for evaluation ===
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=optimal_threshold),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision", thresholds=optimal_threshold),
        tf.keras.metrics.Recall(name="recall", thresholds=optimal_threshold),
    ]
)

# === Evaluate on test set ===
print("[INFO] Evaluating on test set...")
results = model.evaluate(test_gen, verbose=1)
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

# === Predict probabilities ===
print("[INFO] Generating predictions...")
y_prob = model.predict(test_gen).flatten()

# === Extract true labels from test_gen ===
print("[INFO] Extracting true labels...")
if hasattr(test_gen, 'labels'):
    y_true = np.array(test_gen.labels)
else:
    # Recreate test_gen to extract labels if needed
    _, _, _, _, val_gen, test_gen = get_generators(IMG_SIZE, BATCH_SIZE)
    y_true_batches = []
    for batch in test_gen:
        y_true_batches.append(batch[1])
    y_true = np.concatenate(y_true_batches, axis=0)

# === Save ROC curve ===
roc_curve_path = os.path.join(output_dir, "test_roc_curve.png")
save_roc_curve(y_true, y_prob, roc_curve_path)
roc_auc = roc_auc_score(y_true, y_prob)
print(f"[INFO] ROC curve saved to {roc_curve_path}")
print(f"[INFO] Test ROC AUC: {roc_auc:.4f}")

# === Threshold-based predictions ===
y_pred = (y_prob >= optimal_threshold).astype(int)

# === Classification report ===
report = classification_report(y_true, y_pred, target_names=["class_0", "class_1"], digits=4)
print("[INFO] Classification report:\n", report)

# === Confusion matrix ===
conf_matrix_path = os.path.join(output_dir, "test_confusion_matrix.png")
save_confusion_matrix(y_true, y_pred, ["class_0", "class_1"], conf_matrix_path)
print(f"[INFO] Confusion matrix saved to {conf_matrix_path}")

# === Plot training history ===
if os.path.exists(history_path):
    with open(history_path, "rb") as f:
        history_all = pickle.load(f)
    plot_history(history_all, save_path=output_dir, metrics=["accuracy", "loss", "auc", "precision", "recall"])
    print("[INFO] Training history plots saved.")
else:
    print("[WARNING] History file not found. Skipping history plots.")

# === Save evaluation report ===
eval_report_path = os.path.join(output_dir, "test_evaluation_report.txt")
with open(eval_report_path, "w") as f:
    f.write(f"Model evaluated: {model_name}\n")
    f.write(f"Test ROC AUC: {roc_auc:.4f}\n\n")
    f.write(report)
print(f"[INFO] Evaluation report saved to {eval_report_path}")

# === END ===
end_time = time.time()
duration = end_time - start_time
print(f"[INFO] Evaluation completed at: {time.ctime(end_time)}")
print(f"[INFO] Total evaluation time: {int(duration // 60)}m {int(duration % 60)}s")
