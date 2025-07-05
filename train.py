import os
import pickle
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.metrics import roc_curve
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

from model import build_model
from data_loader import get_generators
from plot_utils import plot_history

# === CONFIGURATION ===
model_name = "efficientnetb3"
IMG_SIZE = 300
BATCH_SIZE = 32

EPOCHS_HEAD = 30
EPOCHS_FINE_1 = 40

LEARNING_RATE_HEAD = 3e-3
LEARNING_RATE_FINE = 3e-5

DROPOUT = 0.5  # unified dropout
L2_REG = 5e-4

THRESHOLD = 0.5
LABEL_SMOOTHING_H = 0.05
LABEL_SMOOTHING_F = 0.05

CLASS_WEIGHTS_MULT_HEAD = 1.75
CLASS_WEIGHTS_MULT_FINE = 1.2

FINE_TUNE_STEPS = [0]  # Unfreeze all

# === PATHS ===
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}"
os.makedirs(output_dir, exist_ok=True)
MODEL_PATH = f"models/{model_name}_weights"
os.makedirs("models", exist_ok=True)

# === ENVIRONMENT SETUP ===
start_time = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# === HELPER FUNCTIONS ===
def print_distribution(name, df):
    counts = df['label'].astype(int).value_counts().sort_index()
    print(f"[{name}] Class 0 : {counts.get(0, 0)} | Class 1: {counts.get(1, 0)}")

def save_history(history, filename):
    with open(filename, "wb") as f:
        pickle.dump(history, f)
    print(f"[DEBUG] History saved to {filename}")

class RecallLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        recall = logs.get("val_recall")
        print(f"[Epoch {epoch+1}] val_recall: {recall:.4f}")

def compute_class_weights(df):
    labels = df['label'].astype(int)
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))

# === TEMPERATURE SCALING ===
def nll_loss(T, logits, labels):
    scaled_logits = logits / T
    probs = tf.sigmoid(scaled_logits).numpy()
    epsilon = 1e-7
    probs = np.clip(probs, epsilon, 1 - epsilon)
    loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return loss

def optimize_temperature(val_logits, val_labels):
    opt_result = minimize(
        nll_loss, x0=[1.0], args=(val_logits, val_labels),
        bounds=[(0.05, 10)]
    )
    return opt_result.x[0]

# === DATA LOADING ===
train_df, val_df, test_df, train_gen, val_gen, test_gen = get_generators(IMG_SIZE, BATCH_SIZE)
print_distribution("Train", train_df)
print_distribution("Validation", val_df)
print_distribution("Test", test_df)

# === CLASS WEIGHTS HEAD ===
class_weights_head = compute_class_weights(train_df)
print("Original class weights:", class_weights_head)
class_weights_head[1] *= CLASS_WEIGHTS_MULT_HEAD
print("Adjusted class weights (head):", class_weights_head)

# === MODEL CONSTRUCTION ===
model, base_model = build_model(model_name, img_size=IMG_SIZE, dropout=DROPOUT, l2_lambda=L2_REG)
model.summary()

# === CALLBACKS ===
callbacks_h = [
    EarlyStopping(monitor="val_auc", mode="max", patience=10, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    RecallLogger()
]

callbacks_f = [
    EarlyStopping(monitor="val_auc", mode="max", patience=20, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    RecallLogger()
]

# === HEAD TRAINING ===
base_model.trainable = False
print("[INFO] Base model frozen for head training.")
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_HEAD),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING_H),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
        tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
    ]
)
print("[INFO] Starting head training...")
history_head = model.fit(
    train_gen, validation_data=val_gen, epochs=EPOCHS_HEAD,
    callbacks=callbacks_h, class_weight=class_weights_head, verbose=1
)

# === CLASS WEIGHTS FINE-TUNING ===
class_weights_fine = compute_class_weights(train_df)
class_weights_fine[1] *= CLASS_WEIGHTS_MULT_FINE
print("Adjusted class weights (fine-tuning):", class_weights_fine)

# === GRADUAL FINE-TUNING ===
fine_histories = {}

for idx, fine_tune_at in enumerate(FINE_TUNE_STEPS):
    print(f"[INFO] Unfreezing last {abs(fine_tune_at)} layers for fine-tuning stage {idx+1}.")

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE_FINE),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING_F),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
            tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
        ]
    )

    print(f"[INFO] Starting fine-tuning stage {idx+1}...")
    history_fine = model.fit(
        train_gen, validation_data=val_gen,
        epochs=EPOCHS_FINE_1, callbacks=callbacks_f,
        class_weight=class_weights_fine, verbose=1
    )
    fine_histories[f"fine_{idx+1}"] = history_fine.history

# === SAVE HISTORY ===
history_all = {'head': history_head.history}
history_all.update(fine_histories)
save_history(history_all, f"models/history_{model_name}.pkl")

# === PLOTTING ===
plot_history(history_all, save_path=output_dir, metrics=["accuracy", "loss", "auc", "precision", "recall"])

# === TEMPERATURE SCALING ===
print("[INFO] Starting temperature scaling calibration...")
val_logits = model.predict(val_gen)
val_labels = np.array(val_gen.classes)
optimal_T = optimize_temperature(val_logits, val_labels)
print(f"[INFO] Optimal temperature for calibration: {optimal_T:.4f}")

with open(os.path.join(output_dir, "optimal_temperature.txt"), "w") as f:
    f.write(f"{optimal_T:.4f}\n")

# === THRESHOLDING ===
print("[INFO] Calculating optimal threshold using Youden's J statistic with temperature scaling...")
scaled_logits = val_logits / optimal_T
scaled_probs = tf.sigmoid(scaled_logits).numpy()

fpr, tpr, thresholds = roc_curve(val_labels, scaled_probs)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx] if np.isfinite(thresholds[optimal_idx]) else 0.5
print(f"[INFO] Optimal validation threshold (Youden's J) after temperature scaling: {optimal_threshold:.4f}")

with open(os.path.join(output_dir, "optimal_threshold_val.txt"), "w") as f:
    f.write(f"{optimal_threshold:.4f}\n")

# === TRAINING TIME ===
elapsed_time = time.time() - start_time
print(f"[INFO] Total training time: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
