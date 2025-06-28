import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

BASE_PATH = "/raid/DATASETS/rmiguel_datasets/ISIC16/Classification/Split"

train_folder = os.path.join(BASE_PATH, "train")
val_folder = os.path.join(BASE_PATH, "val")
test_folder = os.path.join(BASE_PATH, "test")


def load_dataframes(file_path):

    # Load CSV files
    df = pd.read_csv(os.path.join(BASE_PATH, file_path), header=None, names=['image', 'label'])

    # Convert labels to string format '0' and '1'
    df['label'] = df['label'].astype(str)

    return df


def get_generators(img_size, batch_size):

    train_df = load_dataframes(os.path.join(train_folder, "train_labels.csv"))
    val_df = load_dataframes(os.path.join(val_folder, "val_labels.csv"))
    test_df = load_dataframes(os.path.join(test_folder, "test_labels.csv"))

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_folder,
        x_col="image",
        y_col="label",
        target_size=(img_size, img_size),
        class_mode="binary",
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    val_gen = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=val_folder,
        x_col="image",
        y_col="label",
        target_size=(img_size, img_size),
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    test_gen = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=test_folder,
        x_col="image",
        y_col="label",
        target_size=(img_size, img_size),
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    return train_df, val_df, test_df, train_gen, val_gen, test_gen
