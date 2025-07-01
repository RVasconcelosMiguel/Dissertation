import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# === BASE PATHS ===
BASE_PATH = "/raid/DATASETS/rmiguel_datasets/ISIC16/Classification/Split"

train_folder = os.path.join(BASE_PATH, "train")
val_folder = os.path.join(BASE_PATH, "val")
test_folder = os.path.join(BASE_PATH, "test")

# === LOAD CSV DATAFRAMES ===
def load_dataframes(file_path):
    df = pd.read_csv(os.path.join(BASE_PATH, file_path), header=None, names=['image', 'label'])
    df['label'] = df['label'].astype(str)  # Convert labels to string for flow_from_dataframe compatibility
    return df

# === DATA GENERATORS FUNCTION ===
def get_generators(img_size, batch_size):

    # Load dataframes
    train_df = load_dataframes(os.path.join(train_folder, "train_labels.csv"))
    val_df = load_dataframes(os.path.join(val_folder, "val_labels.csv"))
    test_df = load_dataframes(os.path.join(test_folder, "test_labels.csv"))

    # Define augmentation for training only
    
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=10,             # Added shear augmentation
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,         # Added vertical flip
        fill_mode='nearest'
    )

    # Define rescaling only for validation and test sets
    test_val_datagen = ImageDataGenerator(rescale=1./255)

    # Training generator
    train_gen = train_datagen.flow_from_dataframe(
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

    # Validation generator
    val_gen = test_val_datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=val_folder,
        x_col="image",
        y_col="label",
        target_size=(img_size, img_size),
        class_mode="binary",
        batch_size=batch_size,
        shuffle=False
    )

    # Test generator
    test_gen = test_val_datagen.flow_from_dataframe(
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
