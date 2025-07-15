# === train_finetune.py ===

import os
import pickle
import numpy as np
import tensorflow as tf
import time
import random
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from model import build_model
from data_loader import get_generators
from plot_utils import plot_history_finetune_stages

# === SEED FOR REPRODUCIBILITY ===
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# === CONFIGURATION ===
model_name = "efficientnetb1"
IMG_SIZE = 240
BATCH_SIZE = 16
FINE_TUNE_EPOCHS = 60
FINE_TUNE_LR = (3e-5)/(0.96**27)
LABEL_SMOOTHING_F = 0

DROPOUT_H = 0.6   # Consistent with head
L2_REG_H = 1e-3
DROPOUT_F = 0.2   # Base dropout during fine-tuning
L2_REG_F = 1e-5

THRESHOLD = 0.5
CLASS_WEIGHTS_MULT_FINE = 4

# === PATHS ===
output_dir = "outputs/fine/results"
MODEL_DIR = "outputs/fine/model"
HEAD_DIR = "outputs/head/model"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_WEIGHTS_PATH = os.path.join(MODEL_DIR, f"{model_name}_fine_weights")
HEAD_WEIGHTS_PATH = os.path.join(HEAD_DIR, f"{model_name}_head_weights")

# === ENVIRONMENT SETUP ===
start_time = time.time()
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
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
        print(f"[Epoch {epoch+1}] val_recall : {recall:.4f} - lr: {lr:.8f}")

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

# === CLASS WEIGHTS ===
class_weights_fine = compute_class_weights(train_df)
class_weights_fine[1] *= CLASS_WEIGHTS_MULT_FINE
print("Adjusted class weights (fine-tuning) :", class_weights_fine)

# === MODEL CONSTRUCTION ===
model, base_model = build_model(
    model_name=model_name,
    img_size=IMG_SIZE,
    dropout_head=DROPOUT_H,
    dropout_base=DROPOUT_F,
    l2_lambda_head=L2_REG_H,
    l2_lambda_base=L2_REG_F
)

# === LOAD HEAD WEIGHTS ===
print(f"[INFO] Loading head-trained weights from: {HEAD_WEIGHTS_PATH}")
model.load_weights(HEAD_WEIGHTS_PATH)
print("[DEBUG] Head weights loaded successfully.")

# === FINE-TUNING SETUP ===
base_model.trainable = True

# === Unfreeze only selected BatchNormalization layers ===
bn_layers = [layer for layer in base_model.layers if isinstance(layer, tf.keras.layers.BatchNormalization)]
num_unfreeze = max(1, int(len(bn_layers) * 0.75))  # Unfreeze last 75% of BN layers

for layer in bn_layers[:-num_unfreeze]:
    layer.trainable = False
for layer in bn_layers[-num_unfreeze:]:
    layer.trainable = True

print(f"[INFO] Unfroze the last {num_unfreeze} BatchNormalization layers for adaptation. ")

# === COMPILE MODEL ===
lr_schedule = ExponentialDecay(
    initial_learning_rate=FINE_TUNE_LR,
    decay_steps=60,
    decay_rate=0.96,
    staircase=True
)

optimizer = Adam(learning_rate=lr_schedule)

model.compile(
    optimizer=optimizer,
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=LABEL_SMOOTHING_F),
    metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
        tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
    ]
)

# === CALLBACKS ===
callbacks = [
    EarlyStopping(monitor="val_auc", mode="max", patience=25, restore_best_weights=True),
    ModelCheckpoint(MODEL_WEIGHTS_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True),
    RecallLogger()
]

# === TRAINING ===
print("[INFO] Starting full fine-tuning...")
history_fine = model.fit(
    train_gen, validation_data=val_gen,
    epochs=FINE_TUNE_EPOCHS, callbacks=callbacks,
    class_weight=class_weights_fine, verbose=1
)

# === SAVE HISTORY ===
save_history(history_fine.history, os.path.join(MODEL_DIR, f"history_{model_name}_fine.pkl"))

# === PLOT METRICS ===
plot_history_finetune_stages(
    {'fine_tune': history_fine.history},
    save_path=output_dir,
    metrics=["accuracy", "loss", "auc", "precision", "recall"]
)

# === TRAINING TIME ===
elapsed_time = time.time() - start_time
print(f"[INFO] Total fine-tuning time: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
