# === train_finetune.py ===

import os
import pickle
import numpy as np
import tensorflow as tf
import time
import random
from scipy.optimize import minimize
from sklearn.metrics import roc_curve
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from model import build_model
from data_loader import get_generators
from plot_utils import plot_history

# === SEED FOR REPRODUCIBILITY ===
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# === CONFIGURATION ===
model_name = "efficientnetb4"
IMG_SIZE = 380
BATCH_SIZE = 16

# Fine-tuning configuration with recommended strategy
FINE_TUNE_UNFREEZE_PERCENTS = [10, 40, 100]
FINE_TUNE_EPOCHS = [15, 20, 30]
FINE_TUNE_LRS = [1e-5, 5e-6, 1e-6]
LABEL_SMOOTHING_F = 0

DROPOUT = 0.6
L2_REG = 1e-3

THRESHOLD = 0.5
CLASS_WEIGHTS_MULT_FINE = 1.5

# === PATHS ===
output_dir = f"/home/jtstudents/rmiguel/files_to_transfer/{model_name}/fine"
os.makedirs(output_dir, exist_ok=True)
MODEL_PATH = f"models/{model_name}_fine_weights"
HEAD_WEIGHTS_PATH = f"models/{model_name}_head_weights"

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
        lr = self.model.optimizer._decayed_lr(tf.float32).numpy()
        print(f"[Epoch {epoch+1}] val_recall: {recall:.4f} - lr: {lr:.8f}")

def compute_class_weights(df):
    labels = df['label'].astype(int)
    classes = np.unique(labels)
    weights = compute_class_weight('balanced', classes=classes, y=labels)
    return dict(zip(classes, weights))

def nll_loss(T, logits, labels):
    scaled_logits = logits / T
    probs = tf.sigmoid(scaled_logits).numpy()
    epsilon = 1e-7
    probs = np.clip(probs, epsilon, 1 - epsilon)
    loss = -np.mean(labels * np.log(probs) + (1 - labels) * np.log(1 - probs))
    return loss

def optimize_temperature(val_probs, val_labels):
    logits = np.log(val_probs / (1 - val_probs))
    opt_result = minimize(
        nll_loss, x0=[1.0], args=(logits, val_labels),
        bounds=[(0.05, 10)]
    )
    return opt_result.x[0], logits

# === DATA LOADING ===
train_df, val_df, test_df, train_gen, val_gen, test_gen = get_generators(IMG_SIZE, BATCH_SIZE)
print_distribution("Train", train_df)
print_distribution("Validation", val_df)
print_distribution("Test", test_df)

# === CLASS WEIGHTS FINE-TUNING ===
class_weights_fine = compute_class_weights(train_df)
class_weights_fine[1] *= CLASS_WEIGHTS_MULT_FINE
print("Adjusted class weights (fine-tuning):", class_weights_fine)

# === MODEL CONSTRUCTION ===
model, base_model = build_model(model_name, img_size=IMG_SIZE, dropout=DROPOUT, l2_lambda=L2_REG)

# === LOAD HEAD WEIGHTS ===
print(f"[INFO] Loading head-trained weights from: {HEAD_WEIGHTS_PATH}")
model.load_weights(HEAD_WEIGHTS_PATH)
print("[DEBUG] Head weights loaded successfully.")

# === CALLBACKS TEMPLATE ===
callbacks_template = lambda: [
    EarlyStopping(monitor="val_auc", mode="max", patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, monitor="val_auc", mode="max", save_best_only=True, save_weights_only=True),
    RecallLogger()
]

# === GRADUAL FINE-TUNING WITH WARMUP LR ===
fine_histories = {}
total_layers = len(base_model.layers)

for idx, (unfreeze_percent, epochs, lr) in enumerate(zip(FINE_TUNE_UNFREEZE_PERCENTS, FINE_TUNE_EPOCHS, FINE_TUNE_LRS)):
    fine_tune_at = int(total_layers * (1 - unfreeze_percent / 100))
    print(f"[INFO] Fine-tuning stage {idx+1}: unfreezing last {unfreeze_percent}% of layers ({total_layers - fine_tune_at}/{total_layers} layers), lr={lr}")

    base_model.trainable = True
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    # Freeze BN layers only in stage 1
    for layer in base_model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False if idx == 0 else True

    # === Warmup LR scheduler ===
    class WarmUpThenDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, base_lr, warmup_steps, decay_steps, decay_rate):
            super().__init__()
            self.base_lr = base_lr
            self.warmup_steps = warmup_steps
            self.decay_steps = decay_steps
            self.decay_rate = decay_rate

        def __call__(self, step):
            warmup_lr = self.base_lr * (tf.cast(step, tf.float32) / tf.cast(self.warmup_steps, tf.float32))
            decayed_lr = self.base_lr * tf.pow(self.decay_rate, (step - self.warmup_steps) / self.decay_steps)
            return tf.cond(step < self.warmup_steps, lambda: warmup_lr, lambda: decayed_lr)

    warmup_steps = 5 * (len(train_gen))  # first 5 epochs warmup
    decay_steps = {0: 100, 1: 200, 2: 300}[idx]
    decay_rate = {0: 0.8, 1: 0.85, 2: 0.9}[idx]

    lr_schedule = WarmUpThenDecay(base_lr=lr, warmup_steps=warmup_steps, decay_steps=decay_steps, decay_rate=decay_rate)

    model.compile(
        optimizer=Adam(learning_rate=lr_schedule),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False, label_smoothing=LABEL_SMOOTHING_F),
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy", threshold=THRESHOLD),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision", thresholds=THRESHOLD),
            tf.keras.metrics.Recall(name="recall", thresholds=THRESHOLD),
        ]
    )

    print(f"[INFO] Starting fine-tuning stage {idx+1} with warmup...")
    history_fine = model.fit(
        train_gen, validation_data=val_gen,
        epochs=epochs, callbacks=callbacks_template(),
        class_weight=class_weights_fine, verbose=1
    )
    fine_histories[f"fine_{idx+1}"] = history_fine.history

# === SAVE HISTORY ===
save_history(fine_histories, f"models/history_{model_name}_fine.pkl")

# === PLOTTING ===
plot_history(fine_histories, save_path=output_dir, metrics=["accuracy", "loss", "auc", "precision", "recall"])

# === TEMPERATURE SCALING ===
print("[INFO] Starting temperature scaling calibration...")
val_probs = model.predict(val_gen)
val_labels = np.array(val_gen.classes)
optimal_T, logits = optimize_temperature(val_probs, val_labels)
print(f"[INFO] Optimal temperature for calibration: {optimal_T:.4f}")

with open(os.path.join(output_dir, "optimal_temperature.txt"), "w") as f:
    f.write(f"{optimal_T:.4f}\n")

# === THRESHOLDING (Youden's J) ===
print("[INFO] Calculating optimal threshold using Youden's J statistic with temperature scaling...")
scaled_logits = logits / optimal_T
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
print(f"[INFO] Total fine-tuning time: {int(elapsed_time // 60)}m {int(elapsed_time % 60)}s")
