import os
import pickle
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, Callback

from model import build_model
from data_loader import get_generators
from plot_utils import plot_history
from losses import focal_loss

# === CONFIGURATION ===
model_name = "efficientnetb3"
IMG_SIZE = 300
BATCH_SIZE = 16

EPOCHS_HEAD = 15
EPOCHS_FINE_1 = 40

LEARNING_RATE_HEAD = 1e-4

DROPOUT = 0.5
L2_REG = 1e-5

THRESHOLD = 0.5
LABEL_SMOOTHING = 0.1  # Added label smoothing parameter

CLASS_WEIGHTS_MULT = 2

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
    print(f"[{name}] Class 0: {counts.get(0, 0)} | Class 1: {counts.get(1, 0)}")

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

# === LEARNING RATE FINDER CALLBACK ===
class LRFinder(Callback):
    def __init__(self, min_lr=1e-7, max_lr=1e-2, steps=100):
        super().__init__()
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.steps = steps
        self.lr_mult = (max_lr / min_lr) ** (1/steps)
        self.history = {}
        self.best_loss = np.inf

    def on_train_begin(self, logs=None):
        tf.keras.backend.set_value(self.model.optimizer.lr, self.min_lr)
        self.history['lr'] = []
        self.history['loss'] = []

    def on_batch_end(self, batch, logs=None):
        lr = tf.keras.backend.get_value(self.model.optimizer.lr)
        loss = logs.get('loss')

        self.history['lr'].append(lr)
        self.history['loss'].append(loss)

        if loss < self.best_loss:
            self.best_loss = loss

        if loss > 4 * self.best_loss:
            self.model.stop_training = True
            return

        lr *= self.lr_mult
        tf.keras.backend.set_value(self.model.optimizer.lr, lr)

def plot_lr_finder(history):
    plt.figure()
    plt.plot(history['lr'], history['loss'])
    plt.xscale('log')
    plt.xlabel("Learning Rate")
    plt.ylabel("Loss")
    plt.title("Learning Rate Finder")
    plt.savefig(os.path.join(output_dir, "lr_finder.png"))
    plt.close()

# === DATA LOADING ===
train_df, val_df, test_df, train_gen, val_gen, test_gen = get_generators(IMG_SIZE, BATCH_SIZE)
print_distribution("Train", train_df)
print_distribution("Validation", val_df)
print_distribution("Test", test_df)
class_weights = compute_class_weights(train_df)
print("Original class weights:", class_weights)
class_weights[1] *= CLASS_WEIGHTS_MULT
print("Adjusted class weights:", class_weights)

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
    EarlyStopping(monitor="val_auc", mode="max", patience=15, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True),
    ReduceLROnPlateau(monitor="val_auc", mode="max", factor=0.5, patience=5, min_lr=1e-7, verbose=1),
    RecallLogger()
]

# === HEAD TRAINING ===
base_model.trainable = False
print("[INFO] Base model frozen for head training.")
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE_HEAD),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
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
    callbacks=callbacks_h, class_weight=class_weights, verbose=1
)

# === LEARNING RATE FINDER BEFORE FINE-TUNING ===
print("[INFO] Starting Learning Rate Finder...")
lr_finder = LRFinder(min_lr=1e-7, max_lr=1e-3, steps=100)
model.compile(
    optimizer=Adam(),
    loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING)
)
model.fit(train_gen, epochs=1, callbacks=[lr_finder], verbose=1)
plot_lr_finder(lr_finder.history)

# === MANUAL SELECTION OF LR BASED ON LR FINDER ===
# Inspect 'lr_finder.png' and choose LR in optimal stable region
FOUND_FINE_TUNE_LR = 3e-5  # Example, replace with your inspected optimal LR

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
        optimizer=Adam(learning_rate=FOUND_FINE_TUNE_LR),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=LABEL_SMOOTHING),
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
        class_weight=class_weights, verbose=1
    )
    fine_histories[f"fine_{idx+1}"] = history_fine.history

# === SAVE HISTORY ===
history_all = {'head': history_head.history}
history_all.update(fine_histories)
save_history(history_all, f"models/history_{model_name}.pkl")

# === PLOTTING ===
plot_history(history_all, save_path=output_dir, metrics=["accuracy", "loss", "auc", "precision", "recall"])

# === THRESHOLDING ===
print("[INFO] Calculating optimal threshold using Youden's J statistic...")
y_val_prob = model.predict(val_gen).flatten()
y_val_true = np.array(val_gen.classes)
fpr, tpr, thresholds = roc_curve(y_val_true, y_val_prob)
youden_index = tpr - fpr
optimal_idx = np.argmax(youden_index)
optimal_threshold = thresholds[optimal_idx] if np.isfinite(thresholds[optimal_idx]) else 0.5
print(f"[INFO] Optimal validation threshold (Youden's J): {optimal_threshold:.4f}")

with open(os.path.join(output_dir, "optimal_threshold_val.txt"), "w") as f:
    f.write(f"{optimal_threshold:.4f}\n")

# === TRAINING TIME ===
elapsed_time = time.time() - start_time
print(f"[INFO] Total training time: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
