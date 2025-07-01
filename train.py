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

# === CONFIGURATION ===
model_name = "custom_cnn"
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3

DROPOUT = 0.2
L2_REG = 5e-4
CALCULATE_OPTIMAL_THRESHOLD = True
THRESHOLD = 0.5

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

# === MODEL CONSTRUCTION ===
model, base_model = build_model(model_name, img_size=IMG_SIZE, dropout=DROPOUT, l2_lambda=L2_REG)
model.summary()

# === COMPILE MODEL ===
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
        tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
    ]
)

# === CALLBACKS ===
callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    RecallLogger()
]

# === TRAINING WITH TQDM PROGRESS BAR ===
pbar = tqdm(total=EPOCHS, desc="Training", file=sys.__stdout__)
history_all = {'loss': [], 'accuracy': [], 'auc': [], 'precision': [], 'recall': [],
               'val_loss': [], 'val_accuracy': [], 'val_auc': [], 'val_precision': [], 'val_recall': []}

for epoch in range(EPOCHS):
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=1,
        callbacks=[RecallLogger()],
        class_weight=class_weights,
        verbose=0
    )
    for key in history.history:
        if key in history_all:
            history_all[key] += history.history[key]
        else:
            history_all[key] = history.history[key]
    pbar.update(1)

pbar.close()

# === SAVE HISTORY ===
save_history(history, f"models/history_{model_name}.pkl")

# === PLOTTING ===
plot_history(
    {"train": history_all},
    save_path=output_dir,
    metrics=["accuracy", "loss", "auc", "precision", "recall"]
)

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

# === SAVE THRESHOLD ===
threshold_path = os.path.join(output_dir, "optimal_threshold_val.txt")
with open(threshold_path, "w") as f:
    f.write(f"{optimal_threshold:.4f}\n")

# === TRAINING TIME ===
elapsed_time = time.time() - start_time
print(f"[INFO] Total training time: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s", file=sys.__stdout__)

# === CLOSE LOG ===
log_file.close()
