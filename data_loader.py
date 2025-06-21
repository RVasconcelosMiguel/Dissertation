import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_dataframes(train_csv_name):

    # Construct full paths
    base_path = "/raid/DATASETS/rmiguel_datasets/ISIC16/CSV"
    train_csv_path = f"{base_path}/{train_csv_name}"
    test_csv_path  = f"{base_path}/Testing_labels.csv"

    # Load CSVs
    df_train = pd.read_csv(train_csv_path, header=None, names=['image', 'label'])
    df_test  = pd.read_csv(test_csv_path,  header=None, names=['image', 'label'])

    # Format labels as strings: 'benign' -> '0', 'malignant' -> '1'
    df_train['label'] = df_train['label'].map({'benign': '0', 'malignant': '1'}).fillna(df_train['label']).astype(str)
    df_test['label'] = df_test['label'].astype(str)

    # Ensure image names end with .jpg
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


def get_generators(train_csv_name, img_size=224, batch_size=32):

    train_df, val_df, test_df = load_dataframes(train_csv_name)

    train_dir = "/raid/DATASETS/rmiguel_datasets/ISIC16/Augmented_Training_Data" \
        if "Augmented" in train_csv_name else \
        "/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Training_Data"

    test_dir = "/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Testing_Data"

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
