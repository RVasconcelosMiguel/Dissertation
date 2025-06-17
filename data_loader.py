import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataframes():
    # Load and format labels
    df_train = pd.read_csv('/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Training_labels.csv',
                           header=None, names=['image', 'label'])
    df_test = pd.read_csv('/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Testing_labels.csv',
                          header=None, names=['image', 'label'])

    df_train['label'] = df_train['label'].map({'benign': '0', 'malignant': '1'})
    df_test['label'] = df_test['label'].astype(str)

    df_train['image'] = df_train['image'].astype(str) + '.jpg'
    df_test['image'] = df_test['image'].astype(str) + '.jpg'

    # Stratified train/val split
    df_train_train, df_train_val = train_test_split(
        df_train,
        stratify=df_train['label'],
        test_size=0.15,
        random_state=42
    )

    return df_train_train, df_train_val, df_test

def get_generators(img_size=224, batch_size=32):
    train_df, val_df, test_df = load_dataframes()

    train_dir = "/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Training_Data"
    test_dir  = "/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Testing_Data"

    # Data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.1,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True
    )
    val_test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_dataframe(
        train_df, train_dir, x_col="image", y_col="label",
        target_size=(img_size, img_size),
        class_mode="binary", batch_size=batch_size, shuffle=True
    )
    val_gen = val_test_datagen.flow_from_dataframe(
        val_df, train_dir, x_col="image", y_col="label",
        target_size=(img_size, img_size),
        class_mode="binary", batch_size=batch_size, shuffle=False
    )
    test_gen = val_test_datagen.flow_from_dataframe(
        test_df, test_dir, x_col="image", y_col="label",
        target_size=(img_size, img_size),
        class_mode="binary", batch_size=batch_size, shuffle=False
    )
    return train_gen, val_gen, test_gen
