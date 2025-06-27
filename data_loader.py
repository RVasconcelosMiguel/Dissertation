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
