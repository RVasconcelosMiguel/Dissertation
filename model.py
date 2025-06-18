from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

def build_model(img_size=224, dropout=0.3):
    base_model = MobileNetV2(include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3))
    base_model.trainable = False
    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(dropout)(x)
    output = Dense(1, activation="sigmoid")(x)
    return Model(base_model.input, output), base_model
