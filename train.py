from model import build_model
from data_loader import get_generators
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf
from plot_utils import plot_history

# Hyperparams
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 15
EPOCHS_FINE = 10
LR_HEAD = 1e-4
LR_FINE = 1e-5
MODEL_PATH = "models/efficientnetb0_isic16.h5"

# Load data
train_gen, val_gen, test_gen = get_generators(img_size=IMG_SIZE, batch_size=BATCH_SIZE)

# Build and compile model
model, base_model = build_model(img_size=IMG_SIZE)
model.compile(optimizer=Adam(LR_HEAD), loss="binary_crossentropy", metrics=["accuracy"])

# Train classification head
history_head = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_HEAD)

# Fine-tune
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

history_fine = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FINE, callbacks=[early_stop, checkpoint])

# Save performance plots
plot_history({"Head": history_head, "Fine": history_fine})
