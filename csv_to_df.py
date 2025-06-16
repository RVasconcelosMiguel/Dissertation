# Define paths
labels_path_train = '/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Training_labels.csv'
labels_path_test = '/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Testing_labels.csv'

# === Load CSVs ===
df_train = pd.read_csv(labels_path_train, header=None, names=['image', 'label'])
df_test  = pd.read_csv(labels_path_test,  header=None, names=['image', 'label'])

# === Encode training labels (benign -> 0, malignant -> 1) ===
df_train['label'] = df_train['label'].map({'benign': 0, 'malignant': 1})
df_test['label'] = df_test['label'].astype(int)

# === Add '.jpg' extension to all image names ===
df_train['image'] = df_train['image'].astype(str) + '.jpg'
df_test['image']  = df_test['image'].astype(str)  + '.jpg'