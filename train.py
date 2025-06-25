import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from collections import Counter

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

from model import build_model
from data_loader import get_generators, load_dataframes
from plot_utils import plot_history

# === ENVIRONMENT SETUP ===
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
output_dir = "/home/jtstudents/rmiguel/files_to_transfer"
os.makedirs("models", exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

log_path = os.path.join(output_dir, "train_log.txt")
log_file = open(log_path, "w")
sys.stdout = log_file
sys.stderr = log_file

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("Enabled GPU memory growth.")
    except RuntimeError as e:
        print("Error enabling memory growth:", e)
else:
    print("No GPU found — using CPU.")

# === HELPER FUNCTIONS ===
def print_distribution(name, df):
    counts = df['label'].astype(int).value_counts().sort_index()
    print(f"[{name}] Class 0: {counts.get(0, 0)} | Class 1: {counts.get(1, 0)}")

def save_history(history, filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(history.history, f)
        print(f"[DEBUG] History saved to {filename}")
    except Exception as e:
        print(f"[ERROR] Could not save history using pickle: {e}")

class RecallLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        recall = logs.get("val_recall")
        print(f"[Epoch {epoch+1}] val_recall: {recall:.4f}")

def compute_class_weights(df):
    labels = df['label'].astype(int)
    total = len(labels)
    count_0 = (labels == 0).sum()
    count_1 = (labels == 1).sum()
    return {
        0: total / (2.0 * count_0),
        1: total / (2.0 * count_1)
    }

# === CONFIGURATION ===
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS_HEAD = 50
EPOCHS_FINE = 50
LR_HEAD = 1e-3
LR_FINE = 1e-4
UNFREEZE_FROM_LAYER = 100
MODEL_PATH = "models/efficientnetb1_finetuned_weights"
TRAIN_CSV_NAME = "Augmented_Training_labels.csv"
THRESHOLD = 0.52

# === DATA LOADING ===
train_df, val_df, _ = load_dataframes(TRAIN_CSV_NAME)
print_distribution("Train", train_df)
print_distribution("Validation", val_df)
train_gen, val_gen, test_gen = get_generators(TRAIN_CSV_NAME, IMG_SIZE, BATCH_SIZE)
class_weights = compute_class_weights(train_df)

# === MODEL CONSTRUCTION ===
model, base_model = build_model(img_size=IMG_SIZE)
model.summary()

# === METRICS WITH FIXED THRESHOLD ===
metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
    tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
]

# === PHASE 1: HEAD TRAINING ===
model.compile(
    optimizer=Adam(learning_rate=LR_HEAD),
    loss="binary_crossentropy",
    metrics=metrics
)

print("Training classification head...")
history_head = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_HEAD)
model.save_weights("models/efficientnetb1_head_trained_weights")
save_history(history_head, "models/history_efficientnetb1_head.pkl")

# === PHASE 2: FINE-TUNING ===
print("Fine-tuning base model...")
for layer in base_model.layers[:UNFREEZE_FROM_LAYER]:
    layer.trainable = False
for layer in base_model.layers[UNFREEZE_FROM_LAYER:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LR_FINE),
    loss="binary_crossentropy",
    metrics=metrics
)

callbacks_fine = [
    EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=7, min_lr=1e-7, verbose=1),
    RecallLogger()
]

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_fine,
    class_weight=class_weights
)

save_history(history_fine, "models/history_efficientnetb1_fine.pkl")

# === PLOTTING & THRESHOLDING ===
plot_history({"Head": history_head, "Fine": history_fine}, output_dir, ["accuracy", "loss", "auc", "recall"])

y_val_prob = model.predict(val_gen).flatten()
y_val_true = np.array(val_gen.classes)

fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx]

if not np.isfinite(optimal_threshold):
    print("[WARNING] Invalid threshold detected; defaulting to 0.5")
    optimal_threshold = 0.5

with open(os.path.join(output_dir, "optimal_threshold_val.txt"), "w") as f:
    f.write(f"Optimal threshold from validation: {optimal_threshold:.4f}\n")

print(f"[INFO] Saved optimal validation threshold: {optimal_threshold:.4f}")
