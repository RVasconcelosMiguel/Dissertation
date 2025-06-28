import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

from model import build_model
from data_loader import get_generators, load_dataframes
from plot_utils import plot_history

# === PATHS ===
output_dir = "/home/jtstudents/rmiguel/files_to_transfer"
MODEL_PATH = "models/efficientnetb1_finetuned_weights"


# === ENVIRONMENT SETUP ===
start_time = time.time()

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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
BATCH_SIZE = 32
EPOCHS = 100
LR = 1e-4
UNFREEZE_FROM_LAYER = 150
THRESHOLD = 0.45
CALCULATE_OPTIMAL_THRESHOLD = False  # if True, calculate from validation set

# === DATA LOADING ===
train_df, val_df, test_df, train_gen, val_gen, test_gen = get_generators(IMG_SIZE, BATCH_SIZE)

print_distribution("Train", train_df)
print_distribution("Validation", val_df)
print_distribution("Test", test_df)

class_weights = compute_class_weights(train_df)

print(f"Class weights {class_weights}\n")

# === MODEL CONSTRUCTION ===
model, base_model = build_model(img_size=IMG_SIZE)
model.summary()

# === UNFREEZE TOP LAYERS ===
for layer in base_model.layers[:UNFREEZE_FROM_LAYER]:
    layer.trainable = False
for layer in base_model.layers[UNFREEZE_FROM_LAYER:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

# === COMPILE MODEL ===
metrics = [
    tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
    tf.keras.metrics.AUC(name="auc"),
    tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
    tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
]

model.compile(
    optimizer=Adam(learning_rate=LR),
    loss="binary_crossentropy",
    metrics=metrics
)

# === TRAINING ===
print("Training full model with top layers unfrozen...")
callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=20, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=7, min_lr=1e-7, verbose=1),
    RecallLogger()
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

save_history(history, "models/history_efficientnetb1_finetuned.pkl")

# === PLOTTING ===
plot_history({"Finetune": history}, output_dir, ["accuracy", "loss", "auc", "recall"])

# === THRESHOLDING ===
print("[INFO] Calculating or setting threshold...")

y_val_prob = model.predict(val_gen).flatten()
y_val_true = np.array(val_gen.classes)

if CALCULATE_OPTIMAL_THRESHOLD:
    fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    THRESHOLD = thresholds[optimal_idx]
    if not np.isfinite(THRESHOLD):
        print("[WARNING] Invalid threshold detected; defaulting to 0.5")
        THRESHOLD = 0.5
    print(f"[INFO] Using optimal validation threshold: {THRESHOLD:.4f}")
else:
    print(f"[INFO] Using fixed configured threshold: {THRESHOLD:.4f}")

with open(os.path.join(output_dir, "optimal_threshold_val.txt"), "w") as f:
    f.write(f"Threshold used: {THRESHOLD:.4f}\n")

# === TRAINING TIME ===
elapsed_time = time.time() - start_time
minutes = int(elapsed_time // 60)
seconds = int(elapsed_time % 60)
print(f"[INFO] Total training time: {minutes}m {seconds}s")
log_file.close()
