import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import time
from tqdm import tqdm
from sklearn.metrics import roc_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

from model import build_model
from data_loader import get_generators
from plot_utils import plot_history
from losses import focal_loss

# === CONFIGURATION ===
model_name = "efficientnetb1"
IMG_SIZE = 240
BATCH_SIZE = 64

EPOCHS_HEAD = 15
EPOCHS_FINE = 50

LEARNING_RATE_HEAD = 1e-4
LEARNING_RATE_FINE = 1e-4

DROPOUT = 0.4
L2_REG = 1e-4

CALCULATE_OPTIMAL_THRESHOLD = True
THRESHOLD = 0.5

FINE_TUNE_AT = -100

# === PATHS ===
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}"
os.makedirs(output_dir, exist_ok=True)
MODEL_PATH = f"models/{model_name}_weights"

# === ENVIRONMENT SETUP ===
start_time = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.makedirs("models", exist_ok=True)

log_path = os.path.join(output_dir, "train_log.txt")
log_file = open(log_path, "w")
sys.stdout = log_file
sys.stderr = log_file

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# === HELPER FUNCTIONS ===
def print_distribution(name, df):
    counts = df['label'].astype(int).value_counts().sort_index()
    print(f"[{name}] Class 0: {counts.get(0, 0)} | Class 1: {counts.get(1, 0)}")

def save_history(history, filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(history, f)
        print(f"[DEBUG] History saved to {filename}")
    except Exception as e:
        print(f"[ERROR] Could not save history using pickle: {e}")

class RecallLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        recall = logs.get("val_recall")
        print(f"[Epoch {epoch+1}] val_recall: {recall:.4f}")

def compute_class_weights(df):
    labels = df['label'].astype(int)
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))

# === DATA LOADING ===
train_df, val_df, test_df, train_gen, val_gen, test_gen = get_generators(IMG_SIZE, BATCH_SIZE)
print_distribution("Train", train_df)
print_distribution("Validation", val_df)
print_distribution("Test", test_df)
class_weights = compute_class_weights(train_df)
print(f"Class weights {class_weights}\n")

# === Verify data pipeline sample ===
print("[DEBUG] Displaying a sample training batch labels:")
x_batch, y_batch = next(train_gen)
print("Labels sample:", y_batch[:10])

# === MODEL CONSTRUCTION ===
model, base_model = build_model(model_name, img_size=IMG_SIZE, dropout=DROPOUT, l2_lambda=L2_REG)
model.summary()

# === CALLBACKS ===
callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    RecallLogger()
]

# === HEAD TRAINING ===
if base_model is not None:
    base_model.trainable = False
    print("[INFO] Base model frozen for head training.")

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_HEAD),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
        tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
    ]
)

# === Train head in a single fit call ===
print("[INFO] Starting head training...")
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks,
    class_weight=class_weights,
    verbose=1
)

# === FINE-TUNING ===
if base_model is not None:
    print(f"[INFO] Unfreezing last {abs(FINE_TUNE_AT)} layers for fine-tuning.")
    base_model.trainable = True
    for layer in base_model.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    print("[DEBUG] Layer trainable status after unfreezing:")
    for layer in base_model.layers:
        print(f"{layer.name}: {layer.trainable}")

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FINE),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
            tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
        ]
    )

    # === Train fine-tuning in a single fit call ===
    print("[INFO] Starting fine-tuning...")
    history_fine = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS_FINE,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
else:
    history_fine = None

# === CHECK PREDICTION DISTRIBUTION BEFORE THRESHOLDING ===
print("[DEBUG] Plotting validation prediction probability distribution...")

y_pred_prob = model.predict(val_gen).flatten()

plt.figure(figsize=(8,6))
plt.hist(y_pred_prob, bins=50)
plt.title("Validation Prediction Probabilities")
plt.xlabel("Predicted probability")
plt.ylabel("Count")

hist_path = os.path.join(output_dir, "val_pred_prob_hist.png")
plt.savefig(hist_path)
plt.close()
print(f"[DEBUG] Saved prediction probability histogram to {hist_path}")

# === SAVE HISTORY ===
history_all = {
    'head': history_head.history,
    'fine': history_fine.history if history_fine else {}
}
save_history(history_all, f"models/history_{model_name}.pkl")

# === PLOTTING ===
plot_history(history_all, save_path=output_dir,
             metrics=["accuracy", "loss", "auc", "precision", "recall"])

# === THRESHOLDING ===
print("[INFO] Calculating optimal threshold using Youden's J statistic...")
y_val_prob = model.predict(val_gen).flatten()
y_val_true = np.array(val_gen.classes)
if CALCULATE_OPTIMAL_THRESHOLD:
    fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    optimal_threshold = thresholds[optimal_idx] if np.isfinite(thresholds[optimal_idx]) else 0.5
    print(f"[INFO] Using optimal validation threshold (Youden's J): {optimal_threshold:.4f}")
else:
    optimal_threshold = THRESHOLD
    print(f"[INFO] Using fixed threshold: {optimal_threshold:.4f}")

threshold_path = os.path.join(output_dir, "optimal_threshold_val.txt")
with open(threshold_path, "w") as f:
    f.write(f"{optimal_threshold:.4f}\n")

# === TRAINING TIME ===
elapsed_time = time.time() - start_time
print(f"[INFO] Total training time: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s", file=sys.__stdout__)

print(history_head.history)

# === CLOSE LOG ===
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__
log_file.close()
