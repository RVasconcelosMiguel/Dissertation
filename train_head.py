# === train_head.py ===

import os

# === SEED FOR REPRODUCIBILITY ===
from seed_utils import set_global_seed
set_global_seed(42)

import pickle
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from model import build_model
from data_loader import get_generators
from plot_utils import plot_history_head

# === CONFIGURATION ===
model_name = "efficientnetb4"
IMG_SIZE = 380
BATCH_SIZE = 16

# Head training configuration
EPOCHS_HEAD = 25
LEARNING_RATE_HEAD = 5e-5
LABEL_SMOOTHING_H = 0.04

DROPOUT_HEAD = 0.6
DROPOUT_BASE = 0.0  # No dropout on base model during head training

L2_REG_HEAD = 1e-3
L2_REG_BASE = 0.0   # No L2 on base model during head training

THRESHOLD = 0.5
CLASS_WEIGHTS_MULT_HEAD = 1.5

# Learning rate scheduler
lr_schedule = ExponentialDecay(
    initial_learning_rate=LEARNING_RATE_HEAD,
    decay_steps=60,
    decay_rate=0.93,
    staircase=True
)

# === PATHS ===
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}/head"
MODEL_DIR = "models/head"
MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, f"{model_name}_head_weights")
FULL_MODEL_PATH = os.path.join(MODEL_DIR, f"{model_name}_head_model")

# Ensure directories exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# === ENVIRONMENT SETUP ===
start_time = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# === HELPER FUNCTIONS ===
def print_distribution(name, df):
    counts = df['label'].astype(int).value_counts().sort_index()
    print(f"[{name}] Class 0: {counts.get(0, 0)} | Class 1: {counts.get(1, 0)}")

def save_history(history, filename):
    with open(filename, "wb") as f:
        pickle.dump(history, f)
    print(f"[DEBUG] History saved to {filename}")

class RecallLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        recall = logs.get("val_recall")
        lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        print(f"[Epoch {epoch+1}] val_recall: {recall:.4f} - lr: {lr:.8f}")

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

# === CLASS WEIGHTS HEAD ===
class_weights_head = compute_class_weights(train_df)
class_weights_head[1] *= CLASS_WEIGHTS_MULT_HEAD
print("Adjusted class weights (head):", class_weights_head)

# === MODEL CONSTRUCTION ===
model, base_model = build_model(
    model_name=model_name,
    img_size=IMG_SIZE,
    dropout_head=DROPOUT_HEAD,
    dropout_base=DROPOUT_BASE,
    l2_lambda_head=L2_REG_HEAD,
    l2_lambda_base=L2_REG_BASE
)

# === CALLBACKS ===
callbacks_head = [
    EarlyStopping(monitor="val_auc", mode="max", patience=12, restore_best_weights=True),
    ModelCheckpoint(MODEL_WEIGHTS_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True),
    RecallLogger()
]

# === HEAD TRAINING ===
base_model.trainable = False
for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print("[INFO] Base model frozen for head training.")

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=LABEL_SMOOTHING_H),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
        tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
    ]
)

print("[INFO] Starting head training...")
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks_head,
    class_weight=class_weights_head,
    verbose=1
)

# === SAVE HISTORY ===
save_history(history_head.history, os.path.join(MODEL_DIR, f"history_{model_name}_head.pkl"))

# === PLOTTING ===
plot_history_head(
    history_head.history,
    save_path=output_dir,
    metrics=["accuracy", "loss", "auc", "precision", "recall"]
)

# === TRAINING TIME ===
elapsed_time = time.time() - start_time
print(f"[INFO] Total head training time: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
