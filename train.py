import os
os.makedirs("models", exist_ok=True)

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))
print("Num GPUs:", len(tf.config.list_physical_devices('GPU')))

# Your existing GPU memory growth code here...

from model import build_model
from data_loader import get_generators
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from plot_utils import plot_history
import json

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 15
EPOCHS_FINE = 10
LR_HEAD = 1e-4
LR_FINE = 1e-5
MODEL_PATH = "models/efficientnetb0_isic16.h5"

def save_history(history, filename):
    with open(filename, "w") as f:
        json.dump(history.history, f)

train_gen, val_gen, test_gen = get_generators(img_size=IMG_SIZE, batch_size=BATCH_SIZE)

model, base_model = build_model(img_size=IMG_SIZE)
model.summary()

model.compile(optimizer=Adam(LR_HEAD), loss="binary_crossentropy", metrics=["accuracy"])

callbacks_head = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ModelCheckpoint("models/efficientnetb0_head_best.h5", save_best_only=True, monitor="val_loss")
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
checkpoint = ModelCheckpoint(MODEL_PATH, save_best_only=True, monitor="val_loss")

history_fine = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FINE,
                         callbacks=[early_stop, checkpoint])

save_history(history_fine, "models/history_fine.json")

plot_history({"Head": history_head, "Fine": history_fine})
