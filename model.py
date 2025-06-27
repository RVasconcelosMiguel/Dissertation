from tensorflow.keras.applications import EfficientNetB1
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import swish

def build_model(img_size=224, dropout=0.5, l2_lambda=1e-4):
    input_tensor = Input(shape=(img_size, img_size, 3))
    base_model = EfficientNetB1(include_top=False, weights="imagenet", input_tensor=input_tensor)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation=swish, kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation=swish, kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropout)(x)
    output = Dense(1, activation="sigmoid", kernel_regularizer=l2(l2_lambda))(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model
