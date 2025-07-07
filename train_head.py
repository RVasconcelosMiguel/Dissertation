import os
import pickle
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from model import build_model
from data_loader import get_generators
from plot_utils import plot_history_head

# === CONFIGURATION ===
model_name = "efficientnetb4"
IMG_SIZE = 380
BATCH_SIZE = 16

# Head training configuration
EPOCHS_HEAD = 60
LEARNING_RATE_HEAD = 5e-5
LABEL_SMOOTHING_H = 0.04

DROPOUT = 0.6
L2_REG = 1e-3

THRESHOLD = 0.5
CLASS_WEIGHTS_MULT_HEAD = 1.5

DECAY = 1e-6

lr_schedule = ExponentialDecay(
    initial_learning_rate=LEARNING_RATE_HEAD,  # e.g. 5e-5
    decay_steps=1000,                          # adjust as explained
    decay_rate=0.9,                            # adjust as needed
    staircase=True)

# === PATHS ===
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}/head"
os.makedirs(output_dir, exist_ok=True)
MODEL_PATH = f"models/{model_name}_head_weights"
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
    print(f"[{name}] Class 0 : {counts.get(0, 0)} | Class 1 : {counts.get(1, 0)}")

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
model, base_model = build_model(model_name, img_size=IMG_SIZE, dropout=DROPOUT, l2_lambda=L2_REG)

# === CALLBACKS ===
callbacks_head = [
    EarlyStopping(monitor="val_auc", mode="max", patience=12, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True),
    #ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=4, min_lr=1e-7, verbose=1),
    RecallLogger()
]

# === HEAD TRAINING ===
base_model.trainable = False

for layer in base_model.layers:
    if isinstance(layer, tf.keras.layers.BatchNormalization):
        layer.trainable = False

print("[INFO] Base model frozen for head training.")

model.compile(
    optimizer = Adam(learning_rate=lr_schedule),
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
    train_gen, validation_data=val_gen, epochs=EPOCHS_HEAD,
    callbacks=callbacks_head, class_weight=class_weights_head, verbose=1
)

# === SAVE HISTORY ===
save_history(history_head.history, f"models/history_{model_name}_head.pkl")

# === PLOTTING ===
plot_history_head(history_head.history, save_path=output_dir, metrics=["accuracy", "loss", "auc", "precision", "recall"])

# === TRAINING TIME ===
elapsed_time = time.time() - start_time
print(f"[INFO] Total head training time: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
