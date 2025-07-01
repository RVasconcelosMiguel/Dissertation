# === test.py ===

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB1

def test_efficientnetb1_input_output():
    # Initialize model with correct input size
    input_size = 240
    model = EfficientNetB1(include_top=False, weights="imagenet", input_shape=(input_size, input_size, 3))
    
    # Generate dummy input to check compatibility
    dummy_input = np.random.random((1, input_size, input_size, 3)).astype(np.float32)
    output = model(dummy_input)
    
    # Print results clearly for logs
    print(f"[INFO] EfficientNetB1 default input size: {input_size}x{input_size}x3")
    print(f"[INFO] Model output shape: {output.shape}")

if __name__ == "__main__":
    test_efficientnetb1_input_output()
