import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataframes():
    # Load and format labels
    df_train = pd.read_csv('/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Training_labels.csv',
                           header=None, names=['image', 'label'])
    df_test = pd.read_csv('/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Testing_labels.csv',
                          header=None, names=['image', 'label'])

    # Ensure all labels are strings and formatted consistently
    df_train['label'] = df_train['label'].map({'benign': '0', 'malignant': '1'}).astype(str)
    df_test['label'] = df_test['label'].astype(str)

    # Add .jpg to image names if not already present
    df_train['image'] = df_train['image'].astype(str).apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')
    df_test['image'] = df_test['image'].astype(str).apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

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

    # Use augmented images for training and validation
    train_dir = "/raid/DATASETS/rmiguel_datasets/ISIC16/Augmented_Training_Data"
    test_dir  = "/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Testing_Data"

    # Only rescaling — no augmentation (already done offline)
    datagen = ImageDataGenerator(rescale=1./255)

    train_gen = datagen.flow_from_dataframe(
        train_df, train_dir, x_col="image", y_col="label",
        target_size=(img_size, img_size),
        class_mode="binary", batch_size=batch_size, shuffle=True
    )
    val_gen = datagen.flow_from_dataframe(
        val_df, train_dir, x_col="image", y_col="label",
        target_size=(img_size, img_size),
        class_mode="binary", batch_size=batch_size, shuffle=False
    )
    test_gen = datagen.flow_from_dataframe(
        test_df, test_dir, x_col="image", y_col="label",
        target_size=(img_size, img_size),
        class_mode="binary", batch_size=batch_size, shuffle=False
    )

    return train_gen, val_gen, test_gen
