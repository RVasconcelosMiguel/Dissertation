from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import swish

def build_efficientnetb0(img_size, dropout, l2_lambda):
    input_tensor = Input(shape=(img_size, img_size, 3))
    base_model = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=input_tensor)
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

def build_efficientnetb1(img_size, dropout, l2_lambda):
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

def build_efficientnetb2(img_size, dropout, l2_lambda):
    input_tensor = Input(shape=(img_size, img_size, 3))
    base_model = EfficientNetB2(include_top=False, weights="imagenet", input_tensor=input_tensor)
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

def build_efficientnetb3(img_size, dropout, l2_lambda):
    input_tensor = Input(shape=(img_size, img_size, 3))
    base_model = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=input_tensor)
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

def build_efficientnetb4(img_size, dropout, l2_lambda):
    input_tensor = Input(shape=(img_size, img_size, 3))
    base_model = EfficientNetB4(include_top=False, weights="imagenet", input_tensor=input_tensor)
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


def build_custom_cnn(img_size, dropout, l2_lambda):
    input_tensor = Input(shape=(img_size, img_size, 3))

    # === Convolutional Block 1 ===
    x = Conv2D(64, (3,3), activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(input_tensor)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # === Convolutional Block 2 ===
    x = Conv2D(128, (3,3), activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # === Convolutional Block 3 ===
    x = Conv2D(256, (3,3), activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # === Added Convolutional Block 4 ===
    x = Conv2D(512, (3,3), activation='relu', padding='same', kernel_regularizer=l2(l2_lambda))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2,2))(x)

    # === Global Average Pooling ===
    x = GlobalAveragePooling2D()(x)

    # === Fully Connected Layers ===
    x = Dense(256, activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropout)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropout)(x)
    x = Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda))(x)
    x = Dropout(dropout)(x)

    output = Dense(1, activation='sigmoid', kernel_regularizer=l2(l2_lambda))(x)

    model = Model(inputs=input_tensor, outputs=output)
    base_model = None  # No pre-trained base model

    return model, base_model


def build_model(model_name, img_size, dropout, l2_lambda):
    if model_name == "efficientnetb0":
        return build_efficientnetb0(img_size, dropout, l2_lambda)
    elif model_name == "efficientnetb1":
        return build_efficientnetb1(img_size, dropout, l2_lambda)
    elif model_name == "efficientnetb2":
        return build_efficientnetb2(img_size, dropout, l2_lambda)
    elif model_name == "efficientnetb3":
        return build_efficientnetb3(img_size, dropout, l2_lambda)
    elif model_name == "efficientnetb4":
        return build_efficientnetb4(img_size, dropout, l2_lambda)
    elif model_name == "custom_cnn":
        return build_custom_cnn(img_size, dropout, l2_lambda)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
