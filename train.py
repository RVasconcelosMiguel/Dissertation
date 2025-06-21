import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from collections import Counter

# --- Environment Setup ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

output_dir = "/home/jtstudents/rmiguel/files_to_transfer"

# --- Directory Setup ---
os.makedirs("models", exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

# --- Logging Setup ---
log_filename = "train_log.txt"
log_path = os.path.join(output_dir, log_filename)
log_file = open(log_path, "w")
sys.stdout = log_file
sys.stderr = log_file

# --- TensorFlow Info ---
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

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
from data_loader import get_generators, load_dataframes
from plot_utils import plot_history
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K

# --- Training Configuration ---
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 50
EPOCHS_FINE = 50
LR_HEAD = 1e-4
LR_FINE = 3e-5
MODEL_PATH = "models/mobilenetv2_isic16.h5"

# --- Dataset Configuration ---
TRAIN_CSV_NAME = "Augmented_Training_labels.csv"  # or "Training_labels.csv"

# --- Class Distribution Output ---
def print_distribution(name, df):
    counts = df['label'].astype(int).value_counts().sort_index()
    print(f"[{name}] Class 0: {counts.get(0, 0)} | Class 1: {counts.get(1, 0)}")

train_df, val_df, _ = load_dataframes(TRAIN_CSV_NAME)
print_distribution("Train", train_df)
print_distribution("Validation", val_df)

# --- Save Training History ---
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

# --- Define Focal Loss ---
def focal_loss(gamma=1.5, alpha=0.5):
    def focal_loss_fixed(y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
        return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
               -K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    return focal_loss_fixed

# --- Load Data ---
train_gen, val_gen, test_gen = get_generators(
    train_csv_name=TRAIN_CSV_NAME,
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)
class_weights = compute_class_weights(train_gen)

# --- Build and Compile Classification Head ---
model, base_model = build_model(img_size=IMG_SIZE)
model.summary()

model.compile(
    optimizer=Adam(learning_rate=LR_HEAD),
    loss=focal_loss(gamma=1.5, alpha=0.5),
    metrics=["accuracy"]
)

print("Training classification head...")
history_head = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_HEAD,
    class_weight=class_weights
)

model.save("models/mobilenetv2_head_trained.h5")
print("Saved model after head training.")
save_history(history_head, "models/history_mobilenetv2_head.pkl")

# --- Fine-tune Deeper Layers of Base Model ---
print("Fine-tuning base model (partial unfreezing)...")

UNFREEZE_FROM_LAYER = 75  # instead of 100
total_layers = len(base_model.layers)
print(f"Total layers in base model: {total_layers}")

for layer in base_model.layers[:UNFREEZE_FROM_LAYER]:
    layer.trainable = False
for layer in base_model.layers[UNFREEZE_FROM_LAYER:]:
    layer.trainable = True

print(f"Unfroze layers from {UNFREEZE_FROM_LAYER} to {total_layers}")

model.compile(
    optimizer=Adam(learning_rate=LR_FINE),
    loss=focal_loss(gamma=1.5, alpha=0.5),
    metrics=[
        "accuracy",
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

callbacks_fine = [
    EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_loss", save_best_only=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=7, min_lr=1e-7, verbose=1)
]

history_fine = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS_FINE,
    callbacks=callbacks_fine,
    class_weight=class_weights
)

save_history(history_fine, "models/history_mobilenetv2_fine.pkl")
