import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import time
from sklearn.metrics import precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

from model import build_model
from data_loader import get_generators
from plot_utils import plot_history

# === CONFIGURATION ===
model_name = "efficientnetb0"
IMG_SIZE = 224
BATCH_SIZE = 32

# Two-stage training hyperparameters
EPOCHS_STAGE1 = 10    # dense head training
EPOCHS_STAGE2 = 100   # fine-tuning entire model
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-5
UNFREEZE_FROM_LAYER = 100

DROPOUT = 0.2
L2_REG = 1e-4
CALCULATE_OPTIMAL_THRESHOLD = True
THRESHOLD = 0.5

# === PATHS ===
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}"
os.makedirs(output_dir, exist_ok=True)
MODEL_PATH = f"models/{model_name}_finetuned_weights"

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

# === STAGE 1: Train head only ===
for layer in base_model.layers:
    layer.trainable = False

print("\n[INFO] Stage 1: Training dense head only...\n")

model.compile(
    optimizer=Adam(learning_rate=LR_STAGE1),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
        tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
    ]
)

history_stage1 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE1,
    callbacks=[RecallLogger()]
)

# === STAGE 2: Fine-tuning ===
for layer in base_model.layers[:UNFREEZE_FROM_LAYER]:
    layer.trainable = False
for layer in base_model.layers[UNFREEZE_FROM_LAYER:]:
    if not isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = True

print("\n[INFO] Stage 2: Fine-tuning model...\n")

model.compile(
    optimizer=Adam(learning_rate=LR_STAGE2),
    loss="binary_crossentropy",
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
        tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
    ]
)

callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=50, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=7, min_lr=1e-7, verbose=1),
    RecallLogger()
]

history_stage2 = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_STAGE2,
    callbacks=callbacks
)

# === SAVE HISTORIES ===
save_history(history_stage1, f"models/history_{model_name}_stage1.pkl")
save_history(history_stage2, f"models/history_{model_name}_stage2.pkl")

# === PLOTTING ===
plot_history({f"{model_name}_stage1": history_stage1, f"{model_name}_stage2": history_stage2}, output_dir, ["accuracy", "loss", "auc", "recall"])

# === THRESHOLDING ===
print("[INFO] Calculating optimal threshold...")

y_val_prob = model.predict(val_gen).flatten()
y_val_true = np.array(val_gen.classes)

if CALCULATE_OPTIMAL_THRESHOLD:
    precision, recall, thresholds = precision_recall_curve(y_val_true, y_val_prob)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    optimal_idx = np.argmax(f1)
    optimal_threshold = thresholds[optimal_idx] if np.isfinite(thresholds[optimal_idx]) else 0.5
    print(f"[INFO] Using optimal validation threshold (F1-maximised): {optimal_threshold:.4f}")
else:
    optimal_threshold = THRESHOLD
    print(f"[INFO] Using fixed threshold: {optimal_threshold:.4f}")

# === SAVE THRESHOLD ===
threshold_path = os.path.join(output_dir, "optimal_threshold_val.txt")
with open(threshold_path, "w") as f:
    f.write(f"{optimal_threshold:.4f}\n")
print(f"[INFO] Threshold saved to: {threshold_path}")

# === TRAINING TIME ===
elapsed_time = time.time() - start_time
print(f"[INFO] Total training time: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
log_file.close()
