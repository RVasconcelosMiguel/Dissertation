from losses import FocalLoss
import tensorflow as tf

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(10,)),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss=FocalLoss())

try:
    model.save("test_model.keras")
    print("✅ Model saved successfully with FocalLoss.")
except Exception as e:
    print("❌ Failed to save model:")
    print(e)
