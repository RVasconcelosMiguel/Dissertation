import os
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataframes(train_csv_name):
    """
    Loads and prepares training and testing DataFrames.
    Splits train/validation based on base images to avoid leakage from augmentations.
    """
    base_path = "/raid/DATASETS/rmiguel_datasets/ISIC16/CSV"
    train_csv_path = os.path.join(base_path, train_csv_name)
    test_csv_path = os.path.join(base_path, "Testing_labels.csv")

    # Load CSV files
    df_train = pd.read_csv(train_csv_path, header=None, names=['image', 'label'])
    df_test = pd.read_csv(test_csv_path, header=None, names=['image', 'label'])

    # Convert labels to string format '0' and '1'
    df_train['label'] = df_train['label'].map({'benign': '0', 'malignant': '1'}).fillna(df_train['label']).astype(str)
    df_test['label'] = df_test['label'].astype(str)

    # Ensure image filenames end with .jpg
    df_train['image'] = df_train['image'].astype(str).apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')
    df_test['image'] = df_test['image'].astype(str).apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

    # Extract base image name (before '_aug_')
    df_train['base_image'] = df_train['image'].apply(lambda x: x.split('_aug_')[0])

    # Drop duplicate base images to get unique originals
    df_unique = df_train.drop_duplicates(subset='base_image')

    # Stratified split based on unique base images
    df_train_base, df_val_base = train_test_split(
        df_unique,
        stratify=df_unique['label'],
        test_size=0.15,
        random_state=42
    )

    # Expand back to include all augmentations of each base image
    df_train_final = df_train[df_train['base_image'].isin(df_train_base['base_image'])].copy()
    df_val_final = df_train[df_train['base_image'].isin(df_val_base['base_image'])].copy()

    # Drop auxiliary column
    df_train_final.drop(columns=['base_image'], inplace=True)
    df_val_final.drop(columns=['base_image'], inplace=True)

    return df_train_final, df_val_final, df_test



def get_generators(train_csv_name, img_size=224, batch_size=64):
    """
    Generates training, validation, and testing generators using ImageDataGenerator.
    Chooses appropriate image directories based on the CSV name (augmented or not).
    """
    train_df, val_df, test_df = load_dataframes(train_csv_name)

    # Explicitly ensure all labels are strings (required for binary classification mode)
    train_df['label'] = train_df['label'].astype(str)
    val_df['label'] = val_df['label'].astype(str)
    test_df['label'] = test_df['label'].astype(str)

    # Choose the appropriate image directory based on augmentation
    train_dir = "/raid/DATASETS/rmiguel_datasets/ISIC16/Augmented_Training_Data" \
        if "Augmented" in train_csv_name else \
        "/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Training_Data"

    test_dir = "/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Testing_Data"

    datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
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
        directory=train_dir,
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
