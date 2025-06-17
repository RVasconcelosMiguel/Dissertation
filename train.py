import os

# Set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Ensure required directories exist
os.makedirs("models", exist_ok=True)
os.makedirs("/home/jtstudents/rmiguel/files_to_transfer", exist_ok=True)

import sys
import json
import numpy as np
import tensorflow as tf

# Redirect stdout and stderr to log file
log_path = "/home/jtstudents/rmiguel/files_to_transfer/train_log.txt"
log_file = open(log_path, "w")
sys.stdout = log_file
sys.stderr = log_file

# --- TensorFlow info ---
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))

# Configure GPU memory growth
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

# --- Imports from project ---
from model import build_model
from data_loader import get_generators
from plot_utils import plot_history
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Training configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 15
EPOCHS_FINE = 10
LR_HEAD = 1e-4
LR_FINE = 1e-5
MODEL_PATH = "models/efficientnetb0_isic16.h5"

# --- Safe history saving utility ---
def make_json_serializable(obj):
    if isinstance(obj, tf.Tensor):
        return obj.numpy().tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.generic,)):
        return obj.item()
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    else:
        return obj

def save_history(history, filename):
    try:
        print("[DEBUG] Saving training history...")
        history_dict = make_json_serializable(history.history)
        with open(filename, "w") as f:
            json.dump(history_dict, f, indent=2)
        print(f"[DEBUG] Training history saved to {filename}.")
    except Exception as e:
        print(f"[ERROR] Could not save history: {e}")

# --- Load data ---
train_gen, val_gen, test_gen = get_generators(img_size=IMG_SIZE, batch_size=BATCH_SIZE)

# --- Build and compile model (head only) ---
model, base_model = build_model(img_size=IMG_SIZE)
model.summary()

model.compile(
    optimizer=Adam(LR_HEAD),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks_head = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint("models/efficientnetb0_head_best.h5", monitor="val_loss", save_best_only=True)
]

# --- Train head ---
print("Training classification head...")
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks_head
)

model.save("models/efficientnetb0_head_trained.h5")
print("Saved model after head training.")

save_history(history_head, "models/history_head.json")

# --- Fine-tune base model ---
print("Fine-tuning base model...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(LR_FINE),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

callbacks_fine = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True)
]

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_fine
)

save_history(history_fine, "models/history_fine.json")

# --- Plot training history ---
plot_history({
    "Head": history_head,
    "Fine": history_fine
})

print("Training complete.")
log_file.close()
