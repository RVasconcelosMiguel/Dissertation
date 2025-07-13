# === evaluate_finetune.py ===

import os
import time
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from model import build_model
from data_loader import get_generators
from plot_utils import save_confusion_matrix, save_roc_curve

# === CONFIGURATION ===
model_name = "efficientnetb4"
IMG_SIZE = 380
BATCH_SIZE = 16

# === Paths ===
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}/fine"
MODEL_DIR = "models/fine"
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, f"{model_name}_fine_weights")
threshold_path = os.path.join(output_dir, "optimal_threshold_val.txt")

# === Ensure output directory exists ===
os.makedirs(output_dir, exist_ok=True)

# === Silence TensorFlow logging ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
tf.get_logger().setLevel('ERROR')

# === Check GPU availability ===
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"[INFO] GPU(s) detected: {[gpu.name for gpu in gpus]}")
else:
    print("[WARNING] No GPU detected. Evaluation will run on CPU.")

# === Start evaluation ===
start_time = time.time()
print(f"[INFO] Fine-tuned model evaluation started at: {time.ctime(start_time)}")

# === Load saved optimal threshold ===
try:
    with open(threshold_path, "r") as f:
        optimal_threshold = float(f.read().strip())
        optimal_threshold = 0.4
    print(f"[INFO] Loaded optimal threshold: {optimal_threshold:.4f}")
except FileNotFoundError:
    raise FileNotFoundError(f"[ERROR] Optimal threshold file not found at {threshold_path}")

# === Data Loading ===
_, _, _, _, val_gen, test_gen = get_generators(IMG_SIZE, BATCH_SIZE)

# === Build Model ===
print(f"[INFO] Building model architecture: {model_name} for evaluation...")
model, _ = build_model(
    model_name=model_name,
    img_size=IMG_SIZE,
    dropout_head=0.0,   # Disable dropout during evaluation
    dropout_base=0.0,
    l2_lambda_head=1e-3,
    l2_lambda_base=1e-3
)

# === Load Fine-tuned Weights ===
print(f"[INFO] Loading fine-tuned weights from: {MODEL_WEIGHTS_PATH}")
if not os.path.exists(MODEL_WEIGHTS_PATH + ".index"):
    raise FileNotFoundError(f"[ERROR] Missing weights: {MODEL_WEIGHTS_PATH}.index")
status = model.load_weights(MODEL_WEIGHTS_PATH)
status.expect_partial()
print("[DEBUG] Fine-tuned weights loaded successfully.")

# === Predict on Test Set ===
print("[INFO] Predicting on test set...")
y_prob = model.predict(test_gen, verbose=1).flatten()
y_true = np.array(test_gen.classes)

# === Generate ROC Curve ===
roc_curve_path = os.path.join(output_dir, "roc_curve_test_finetune.png")
save_roc_curve(y_true, y_prob, roc_curve_path)
roc_auc = roc_auc_score(y_true, y_prob)
print(f"[INFO] ROC curve saved to {roc_curve_path}")
print(f"[INFO] Test ROC AUC: {roc_auc:.4f}")

# === Save prediction probability histogram ===
plt.figure(figsize=(8,6))
plt.hist(y_prob, bins=50, color='skyblue', edgecolor='black')
plt.title("Fine-tuned Test Prediction Probabilities")
plt.xlabel("Predicted probability")
plt.ylabel("Count")
hist_path = os.path.join(output_dir, "test_pred_prob_finetune_hist.png")
plt.savefig(hist_path)
plt.close()
print(f"[INFO] Histogram saved to {hist_path}")

# === Threshold-based Predictions ===
y_pred = (y_prob >= optimal_threshold).astype(int)
labels = list(test_gen.class_indices.keys())

# === Classification Report ===
print("[INFO] Classification report:")
report = classification_report(y_true, y_pred, target_names=labels, digits=4, output_dict=True)

for cls in labels:
    metrics = report[cls]
    print(f"Class '{cls}': Precision={metrics['precision']:.4f}, Recall={metrics['recall']:.4f}, "
          f"F1-score={metrics['f1-score']:.4f}, Support={metrics['support']}")

print(f"Overall Accuracy: {report['accuracy']:.4f}")
print(f"Macro Average: Precision={report['macro avg']['precision']:.4f}, "
      f"Recall={report['macro avg']['recall']:.4f}, "
      f"F1-score={report['macro avg']['f1-score']:.4f}")
print(f"Weighted Average: Precision={report['weighted avg']['precision']:.4f}, "
      f"Recall={report['weighted avg']['recall']:.4f}, "
      f"F1-score={report['weighted avg']['f1-score']:.4f}")

# === Confusion Matrix and Derived Metrics ===
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

print(f"Confusion Matrix: \n{cm}")
print(f"Accuracy: {report['accuracy']:.4f}")
print(f"AUC: {roc_auc:.4f}")
print(f"Sensitivity (Recall): {sensitivity:.4f}")
print(f"Specificity: {specificity:.4f}")

# === Print false negatives with their prediction probabilities ===
false_negatives_idx = np.where((y_true == 1) & (y_pred == 0))[0]

print(f"\n[INFO] Number of False Negatives: {len(false_negatives_idx)}")
print("[INFO] Predicted probabilities for False Negatives (should be class 1):")
for idx in false_negatives_idx:
    print(f"Index: {idx}, Prob: {y_prob[idx]:.4f}")

# === Print false positives with their prediction probabilities ===
false_positives_idx = np.where((y_true == 0) & (y_pred == 1))[0]

print(f"\n[INFO] Number of False Positives: {len(false_positives_idx)}")
print("[INFO] Predicted probabilities for False Positives (should be class 0):")
for idx in false_positives_idx:
    print(f"Index: {idx}, Prob: {y_prob[idx]:.4f}")

# === Save Confusion Matrix ===
conf_matrix_path = os.path.join(output_dir, "confusion_matrix_finetune.png")
save_confusion_matrix(y_true, y_pred, labels, conf_matrix_path)
print(f"[INFO] Confusion matrix saved to {conf_matrix_path}")

# === End evaluation ===
end_time = time.time()
duration = end_time - start_time
print(f"[INFO] Evaluation completed at: {time.ctime(end_time)}")
print(f"[INFO] Total evaluation time: {int(duration // 60)}m {int(duration % 60)}s")
