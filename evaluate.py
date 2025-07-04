# === evaluate.py ===
import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

from model import build_model
from data_loader import get_generators
from plot_utils import save_confusion_matrix, save_roc_curve

# === CONFIGURATION ===
model_name = "efficientnetb3"
IMG_SIZE = 300
BATCH_SIZE = 32

# === Paths ===
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}"
os.makedirs(output_dir, exist_ok=True)

WEIGHTS_PATH = f"models/{model_name}_weights"
threshold_path = os.path.join(output_dir, "optimal_threshold_val.txt")

# === Silence TensorFlow logging ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

start_time = time.time()
print(f"[INFO] Evaluation started at: {time.ctime(start_time)}")

# === Load saved optimal threshold from validation ===
with open(threshold_path, "r") as f:
    optimal_threshold = float(f.read().strip())
print(f"[INFO] Loaded optimal threshold: {optimal_threshold:.4f}")

# === Data Loading ===
_, _, _, _, val_gen, test_gen = get_generators(IMG_SIZE, BATCH_SIZE)

# === Build Model ===
print(f"[INFO] Building model architecture: {model_name}...")
model, _ = build_model(
    model_name=model_name,
    img_size=IMG_SIZE,
    dropout=0.3,
    l2_lambda=1e-3
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

# === Evaluate on Test Set ===
print("[INFO] Evaluating on test set...")
results = model.evaluate(test_gen, verbose=1)
for name, val in zip(model.metrics_names, results):
    print(f"{name}: {val:.4f}")

# === Generate ROC Curve ===
print("[INFO] Generating ROC curve...")
y_prob = model.predict(test_gen).flatten()
y_true = np.array(test_gen.classes)

roc_curve_path = os.path.join(output_dir, "roc_curve_test.png")
save_roc_curve(y_true, y_prob, roc_curve_path)
roc_auc = roc_auc_score(y_true, y_prob)
print(f"[INFO] ROC curve saved to {roc_curve_path}")
print(f"[INFO] Test ROC AUC: {roc_auc:.4f}")

# === Save prediction probability histogram ===
print("[INFO] Saving prediction probability histogram...")
plt.figure(figsize=(8,6))
plt.hist(y_prob, bins=50, color='skyblue', edgecolor='black')
plt.title("Test Prediction Probabilities")
plt.xlabel("Predicted probability")
plt.ylabel("Count")
hist_path = os.path.join(output_dir, "test_pred_prob_hist.png")
plt.savefig(hist_path)
plt.close()
print(f"[INFO] Histogram saved to {hist_path}")

# === Threshold-based Predictions and Classification Report ===
print("[INFO] Generating classification report...")
y_pred = (y_prob >= optimal_threshold).astype(int)
labels = list(test_gen.class_indices.keys())

report = classification_report(y_true, y_pred, target_names=labels, digits=4)
print("[INFO] Classification report:")
print(report)

# === Save Confusion Matrix ===
conf_matrix_path = os.path.join(output_dir, "confusion_matrix_tuned.png")
save_confusion_matrix(y_true, y_pred, labels, conf_matrix_path)
print(f"[INFO] Confusion matrix saved to {conf_matrix_path}")

# === Save Evaluation Report ===
eval_report_path = os.path.join(output_dir, "evaluation_report.txt")
with open(eval_report_path, "w") as f:
    f.write(f"Model evaluated: {model_name}\n")
    f.write(f"Threshold used: {optimal_threshold:.4f}\n")
    f.write(f"Test ROC AUC: {roc_auc:.4f}\n\n")
    f.write(report)
print(f"[INFO] Evaluation report saved to {eval_report_path}")

end_time = time.time()
duration = end_time - start_time
print(f"[INFO] Evaluation completed at: {time.ctime(end_time)}")
print(f"[INFO] Total evaluation time: {int(duration // 60)}m {int(duration % 60)}s")
