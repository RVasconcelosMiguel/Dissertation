import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataframes(train_csv_name):
    """
    Loads and prepares training, validation, and testing DataFrames
    from pre-created split CSVs without leakage.
    """
    base_path = train_csv_name
    test_csv_path = "/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Testing_labels.csv"

    # Load CSV files
    df_train = pd.read_csv(os.path.join(base_path, "train_labels.csv"), header=None, names=['image', 'label'])
    df_val = pd.read_csv(os.path.join(base_path, "val_labels.csv"), header=None, names=['image', 'label'])
    df_test = pd.read_csv(test_csv_path, header=None, names=['image', 'label'])

    # Convert labels to string format '0' and '1'
    df_train['label'] = df_train['label'].astype(str)
    df_val['label'] = df_val['label'].astype(str)
    df_test['label'] = df_test['label'].astype(str)

    return df_train, df_val, df_test


def get_generators(train_csv_name=None, img_size=224, batch_size=64):
    """
    Generates training, validation, and testing generators using ImageDataGenerator.
    Uses fixed split folders to prevent leakage.
    """
    train_df, val_df, test_df = load_dataframes(train_csv_name)

    # Directories
    base_dir = "/raid/DATASETS/rmiguel_datasets/ISIC16/Classification/Split"
    test_dir = "/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Testing_Data"

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=os.path.join(base_dir, "train"),
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
        directory=os.path.join(base_dir, "val"),
        x_col="image",
        y_col="label",
        target_size=(img_size, img_size),
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    test_gen = datagen.flow_from_dataframe(
        dataframe=test_df,
        directory=test_dir,
        x_col="image",
        y_col="label",
        target_size=(img_size, img_size),
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    return train_gen, val_gen, test_gen
