import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input

# === BASE PATHS ===
BASE_PATH = "/raid/DATASETS/rmiguel_datasets/ISIC16/Classification/Split"

train_folder = os.path.join(BASE_PATH, "train")
val_folder = os.path.join(BASE_PATH, "val")
test_folder = os.path.join(BASE_PATH, "test")

# === LOAD CSV DATAFRAMES ===
def load_dataframes(csv_path):
    df = pd.read_csv(csv_path, header=None, names=['image', 'label'])
    df['label'] = df['label'].astype(str)  # Convert labels to string for flow_from_dataframe compatibility
    return df

# === DATA GENERATORS FUNCTION ===
def get_generators(img_size, batch_size):
    # Load dataframes
    train_df = load_dataframes(os.path.join(train_folder, "train_labels.csv"))
    val_df = load_dataframes(os.path.join(val_folder, "val_labels.csv"))
    test_df = load_dataframes(os.path.join(test_folder, "test_labels.csv"))

    # === Verify label distributions ===
    print("Train label distribution:\n", train_df['label'].value_counts())
    print("Val label distribution:\n", val_df['label'].value_counts())
    print("Test label distribution:\n", test_df['label'].value_counts())

    # === Print dataframe samples for path sanity check ===
    print("[DEBUG] Sample train_df:")
    print(train_df.head())

    # === Define augmentation for training with EfficientNet preprocessing ===
    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=90,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=15,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,  # Disable for dermoscopy unless orientation invariant
        brightness_range=[0.8,1.2],
        fill_mode='nearest'
    )

    # === Define preprocessing only for validation and test sets ===
    test_val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    # === Training generator ===
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

    # === Validation generator ===
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

    # === Test generator ===
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

