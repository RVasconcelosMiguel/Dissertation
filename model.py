from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import swish
import tensorflow as tf

# === EfficientNet builder with separate dropout and L2 for base and head ===
def build_efficientnet_generic(EfficientNetClass, img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base):
    input_tensor = Input(shape=(img_size, img_size, 3))
    base_model = EfficientNetClass(include_top=False, weights="imagenet", input_tensor=input_tensor)

    for layer in base_model.layers:
        if isinstance(layer, Conv2D):
            layer.kernel_regularizer = l2(l2_lambda_base)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)

    if dropout_base > 0:
        x = Dropout(dropout_base)(x)

    x = Dense(256, kernel_regularizer=l2(l2_lambda_head))(x)
    x = BatchNormalization()(x)
    x = swish(x)
    x = Dropout(dropout_head)(x)

    x = Dense(128, kernel_regularizer=l2(l2_lambda_head))(x)
    x = BatchNormalization()(x)
    x = swish(x)
    x = Dropout(dropout_head)(x)

    output = Dense(1, activation="sigmoid", kernel_regularizer=l2(l2_lambda_head))(x)

    model = Model(inputs=base_model.input, outputs=output)
    return model, base_model

# === EfficientNet variants ===
def build_efficientnetb0(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base):
    return build_efficientnet_generic(EfficientNetB0, img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)

def build_efficientnetb1(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base):
    return build_efficientnet_generic(EfficientNetB1, img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)

def build_efficientnetb2(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base):
    return build_efficientnet_generic(EfficientNetB2, img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)

def build_efficientnetb3(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base):
    return build_efficientnet_generic(EfficientNetB3, img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)

def build_efficientnetb4(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base):
    return build_efficientnet_generic(EfficientNetB4, img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)

def build_efficientnetb5(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base):
    return build_efficientnet_generic(EfficientNetB5, img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)

def build_efficientnetb6(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base):
    return build_efficientnet_generic(EfficientNetB6, img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)

def build_efficientnetb7(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base):
    return build_efficientnet_generic(EfficientNetB7, img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)

# === Custom CNN ===
def build_custom_cnn(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base):
    input_tensor = Input(shape=(img_size, img_size, 3))

    x = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(l2_lambda_base))(input_tensor)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(l2_lambda_base))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(l2_lambda_base))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2(l2_lambda_base))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    x = GlobalAveragePooling2D()(x)

    if dropout_base > 0:
        x = Dropout(dropout_base)(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_lambda_head))(x)
    x = Dropout(dropout_head)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_lambda_head))(x)
    x = Dropout(dropout_head)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda_head))(x)
    x = Dropout(dropout_head)(x)

    output = Dense(1, activation="sigmoid", kernel_regularizer=l2(l2_lambda_head))(x)

    model = Model(inputs=input_tensor, outputs=output)
    base_model = None

    return model, base_model

# === build_model dispatcher ===
def build_model(model_name, img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base):
    if model_name == "efficientnetb0":
        return build_efficientnetb0(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)
    elif model_name == "efficientnetb1":
        return build_efficientnetb1(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)
    elif model_name == "efficientnetb2":
        return build_efficientnetb2(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)
    elif model_name == "efficientnetb3":
        return build_efficientnetb3(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)
    elif model_name == "efficientnetb4":
        return build_efficientnetb4(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)
    elif model_name == "efficientnetb5":
        return build_efficientnetb5(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)
    elif model_name == "efficientnetb6":
        return build_efficientnetb6(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)
    elif model_name == "efficientnetb7":
        return build_efficientnetb7(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)
    elif model_name == "custom_cnn":
        return build_custom_cnn(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
