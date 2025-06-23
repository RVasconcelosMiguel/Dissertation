# === train.py ===
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from tensorflow.keras.regularizers import l2

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

# === IMPORTS ===
from data_loader import get_generators, load_dataframes
from plot_utils import plot_history
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K
from model import build_model

# === CUSTOM FOCAL LOSS ===
@tf.keras.utils.register_keras_serializable()
class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=2.0, alpha=0.75, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
        pt_1 = tf.where(K.equal(y_true, 1), y_pred, K.ones_like(y_pred))
        pt_0 = tf.where(K.equal(y_true, 0), y_pred, K.zeros_like(y_pred))
        return -K.mean(self.alpha * K.pow(1. - pt_1, self.gamma) * K.log(pt_1)) \
               -K.mean((1 - self.alpha) * K.pow(pt_0, self.gamma) * K.log(1. - pt_0))

    def get_config(self):
        return {"gamma": self.gamma, "alpha": self.alpha}

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

def compute_class_weights(generator):
    labels = generator.classes
    counts = Counter(labels)
    total = sum(counts.values())
    return {
        0: total / (2.0 * counts[0]),
        1: total / (2.0 * counts[1])
    }

# === CONFIGURATION ===
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 50
EPOCHS_FINE = 50
LR_HEAD = 1e-4
LR_FINE = 3e-5
MODEL_PATH = "models/efficientnetb1_isic16.keras"
TRAIN_CSV_NAME = "Augmented_Training_labels.csv"

# === DATA LOADING ===
train_df, val_df, _ = load_dataframes(TRAIN_CSV_NAME)
print_distribution("Train", train_df)
print_distribution("Validation", val_df)
train_gen, val_gen, test_gen = get_generators(TRAIN_CSV_NAME, IMG_SIZE, BATCH_SIZE)
class_weights = compute_class_weights(train_gen)

# === MODEL CONSTRUCTION ===
model, base_model = build_model(img_size=IMG_SIZE)
model.summary()

# === PHASE 1: HEAD TRAINING ===
loss_fn = FocalLoss(gamma=2.0, alpha=0.75)
model.compile(optimizer=Adam(learning_rate=LR_HEAD), loss=loss_fn, metrics=["accuracy"])
print("Training classification head...")
history_head = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_HEAD)
model.save("models/efficientnetb1_head_trained.keras")
save_history(history_head, "models/history_efficientnetb1_head.pkl")

# === PHASE 2: FINE-TUNING ===
print("Fine-tuning base model...")
UNFREEZE_FROM_LAYER = 300
for layer in base_model.layers[:UNFREEZE_FROM_LAYER]:
    layer.trainable = False
for layer in base_model.layers[UNFREEZE_FROM_LAYER:]:
    layer.trainable = True

model.compile(
    optimizer=Adam(learning_rate=LR_FINE),
    loss=loss_fn,
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc"), tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
)

callbacks_fine = [
    EarlyStopping(monitor="val_recall", mode="max", patience=15, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_recall", mode="max", save_best_only=True),
    ReduceLROnPlateau(monitor="val_recall", mode="max", factor=0.5, patience=7, min_lr=1e-7, verbose=1)
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

with open(os.path.join(output_dir, "optimal_threshold_val.txt"), "w") as f:
    f.write(f"Optimal threshold from validation: {optimal_threshold:.4f}\n")

print(f"[INFO] Saved optimal validation threshold: {optimal_threshold:.4f}")