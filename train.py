import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from collections import Counter

# --- Environment Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# --- Directory Setup ---
os.makedirs("models", exist_ok=True)
os.makedirs("/home/jtstudents/rmiguel/files_to_transfer", exist_ok=True)

# --- Logging Setup ---
log_path = "/home/jtstudents/rmiguel/files_to_transfer/train_log.txt"
log_file = open(log_path, "w")
sys.stdout = log_file
sys.stderr = log_file

# --- TensorFlow Info ---
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))

# --- GPU Memory Growth Configuration ---
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

# --- Project Imports ---
from model import build_model
from data_loader import get_generators
from plot_utils import plot_history
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# --- Training Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 30  # doubled from 15
EPOCHS_FINE = 20  # doubled from 10
LR_HEAD = 1e-4
LR_FINE = 1e-5
MODEL_PATH = "models/mobilenetv2_isic16.h5"

# --- Save Training History using Pickle ---
def save_history(history, filename):
    try:
        with open(filename, "wb") as f:
            pickle.dump(history.history, f)
        print(f"[DEBUG] History saved to {filename}")
    except Exception as e:
        print(f"[ERROR] Could not save history using pickle: {e}")

# --- Compute Class Weights ---
def compute_class_weights(generator):
    labels = generator.classes
    counts = Counter(labels)
    total = sum(counts.values())
    class_weights = {
        0: total / (2.0 * counts[0]),
        1: total / (2.0 * counts[1])
    }
    print(f"[INFO] Computed class weights: {class_weights}")
    return class_weights

# --- Load Data ---
train_gen, val_gen, test_gen = get_generators(img_size=IMG_SIZE, batch_size=BATCH_SIZE)
class_weights = compute_class_weights(train_gen)

# --- Build and Compile Classification Head ---
model, base_model = build_model(img_size=IMG_SIZE)
model.summary()

model.compile(
    optimizer=Adam(learning_rate=float(LR_HEAD)),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

callbacks_head = []

print("Training classification head...")
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    callbacks=callbacks_head,
    class_weight=class_weights  # dded
)

model.save("models/mobilenetv2_head_trained.h5")
print("Saved model after head training.")
save_history(history_head, "models/history_mobilenetv2_head.pkl")

# --- Fine-tune Full Model ---
print("Fine-tuning base model...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=float(LR_FINE)),
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
    callbacks=callbacks_fine,
    class_weight=class_weights  #added
)

save_history(history_fine, "models/history_mobilenetv2_fine.pkl")

# --- Plot History ---
plot_history({
    "Head": history_head,
    "Fine": history_fine
})

print("Training complete.")
log_file.close()
