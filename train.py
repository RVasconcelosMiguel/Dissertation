import os
# Set environment variables before importing TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

os.makedirs("models", exist_ok=True)
os.makedirs("/home/jtstudents/rmiguel/files_to_transfer", exist_ok=True)

import sys

# Redirect stdout and stderr to log file
log_path = "/home/jtstudents/rmiguel/files_to_transfer/train_log.txt"
log_file = open(log_path, "w")
sys.stdout = log_file
sys.stderr = log_file

import tensorflow as tf
import json  # keep import here as you already have

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
        print(f"GPU memory growth error: {e}")
else:
    print("No GPU found — using CPU.")

from model import build_model
from data_loader import get_generators
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from plot_utils import plot_history

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 15
EPOCHS_FINE = 10
LR_HEAD = 1e-4
LR_FINE = 1e-5
MODEL_PATH = "models/efficientnetb0_isic16.h5"

# --- UPDATED save_history FUNCTION ---
def save_history(history, filename):
    # Convert any EagerTensor to float to avoid JSON serialization error
    history_dict = {}
    for key, values in history.history.items():
        new_values = []
        for v in values:
            if hasattr(v, "numpy"):
                new_values.append(float(v.numpy()))
            else:
                new_values.append(v)
        history_dict[key] = new_values
    with open(filename, "w") as f:
        json.dump(history_dict, f)
# -------------------------------------

train_gen, val_gen, test_gen = get_generators(img_size=IMG_SIZE, batch_size=BATCH_SIZE)

model, base_model = build_model(img_size=IMG_SIZE)
model.summary()

model.compile(optimizer=Adam(LR_HEAD), loss="binary_crossentropy", metrics=["accuracy"])

callbacks_head = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint("models/efficientnetb0_head_best.h5", save_best_only=True, monitor="val_loss", save_weights_only=False)  # explicit save_weights_only=False (optional)
]

print("Training classification head...")
history_head = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_HEAD, callbacks=callbacks_head)
model.save("models/efficientnetb0_head_trained.h5")
print("Saved model after head training.")
save_history(history_head, "models/history_head.json")

print("Fine-tuning base model...")
base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False

model.compile(
    optimizer=Adam(LR_FINE),
    loss="binary_crossentropy",
    metrics=["accuracy", tf.keras.metrics.AUC(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
)

early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss", save_weights_only=False)  # again explicit, optional

history_fine = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FINE,
                         callbacks=[early_stop, checkpoint])

save_history(history_fine, "models/history_fine.json")

plot_history({"Head": history_head, "Fine": history_fine})

print("Training complete.")
log_file.close()
