# === model.py ===
from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

def build_model(img_size=224, dropout=0.5, l2_lambda=1e-4):
    base_model = EfficientNetB1(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout)(x)
    output = Dense(1, activation="sigmoid", kernel_regularizer=l2(l2_lambda))(x)
    return Model(base_model.input, output), base_model
