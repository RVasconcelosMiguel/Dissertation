from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization, GlobalAveragePooling2D, GlobalMaxPooling2D, Multiply, Reshape, Activation, Concatenate, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.activations import swish
import tensorflow as tf

# === CBAM MODULE ===

def cbam_block(input_feature, ratio=8):
    channel = input_feature.shape[-1]

    # Channel attention
    shared_layer_one = Dense(channel // ratio, activation='relu', kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')
    shared_layer_two = Dense(channel, kernel_initializer='he_normal', use_bias=True, bias_initializer='zeros')

    avg_pool = GlobalAveragePooling2D()(input_feature)
    avg_pool = Reshape((1,1,channel))(avg_pool)
    avg_pool = shared_layer_one(avg_pool)
    avg_pool = shared_layer_two(avg_pool)

    max_pool = GlobalMaxPooling2D()(input_feature)
    max_pool = Reshape((1,1,channel))(max_pool)
    max_pool = shared_layer_one(max_pool)
    max_pool = shared_layer_two(max_pool)

    cbam_feature = Add()([avg_pool, max_pool])
    cbam_feature = Activation('sigmoid')(cbam_feature)
    cbam_feature = Multiply()([input_feature, cbam_feature])

    # Spatial attention
    avg_pool = tf.reduce_mean(cbam_feature, axis=3, keepdims=True)
    max_pool = tf.reduce_max(cbam_feature, axis=3, keepdims=True)
    concat = Concatenate(axis=3)([avg_pool, max_pool])
    cbam_feature = Conv2D(filters=1, kernel_size=7, strides=1, padding='same', activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(concat)
    cbam_feature = Multiply()([cbam_feature, input_feature])

    return cbam_feature

# === EfficientNet builders with different dropout and L2 for base vs head ===

def build_efficientnet_generic(EfficientNetClass, img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base):
    input_tensor = Input(shape=(img_size, img_size, 3))
    base_model = EfficientNetClass(include_top=False, weights="imagenet", input_tensor=input_tensor)

    # Apply l2_lambda_base to base model layers if applicable
    for layer in base_model.layers:
        if hasattr(layer, 'kernel_regularizer'):
            layer.kernel_regularizer = l2(l2_lambda_base)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(dropout_base)(x)

    # CBAM integration (only in B3, B4 as in your code)
    if EfficientNetClass in [EfficientNetB3, EfficientNetB4]:
        x = Reshape((1,1,x.shape[-1]))(x)
        x = cbam_block(x)
        x = Reshape((x.shape[-1],))(x)

    x = BatchNormalization()(x)
    x = Dense(256, activation=swish, kernel_regularizer=l2(l2_lambda_head))(x)
    x = Dropout(dropout_head)(x)
    x = Dense(128, activation=swish, kernel_regularizer=l2(l2_lambda_head))(x)
    x = Dropout(dropout_head)(x)

    output = Dense(1, activation="sigmoid", kernel_regularizer=l2(l2_lambda_head))(x)
    model = Model(inputs=base_model.input, outputs=output)

    return model, base_model

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
    elif model_name == "custom_cnn":
        return build_custom_cnn(img_size, dropout_head, dropout_base, l2_lambda_head, l2_lambda_base)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
