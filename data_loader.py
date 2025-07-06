import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
import tensorflow_probability as tfp

# === BASE PATHS ===
BASE_PATH = "/raid/DATASETS/rmiguel_datasets/ISIC16/Classification/Split"

train_folder = os.path.join(BASE_PATH, "train")
val_folder = os.path.join(BASE_PATH, "val")
test_folder = os.path.join(BASE_PATH, "test")

# === LOAD CSV DATAFRAMES ===
def load_dataframes(csv_path):
    df = pd.read_csv(csv_path, header=None, names=['image', 'label'])
    df['label'] = df['label'].astype(int)
    return df

# === IMAGE PARSE FUNCTION ===
def parse_image(filename, label, folder, img_size):
    image_string = tf.io.read_file(tf.strings.join([folder, '/', filename]))
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, [img_size, img_size])
    image = preprocess_input(image)
    label = tf.cast(label, tf.float32)
    return image, label

# === MIXUP FUNCTION ===
def mixup(ds_one, ds_two, alpha=0.2):
    image1, label1 = ds_one
    image2, label2 = ds_two

    beta = tfp.distributions.Beta(alpha, alpha)
    lambda_val = beta.sample()
    
    mixed_image = lambda_val * image1 + (1 - lambda_val) * image2
    mixed_label = lambda_val * label1 + (1 - lambda_val) * label2
    mixed_label = tf.clip_by_value(mixed_label, 0, 1)  # Ensure labels remain valid

    return mixed_image, mixed_label

# === CUTMIX FUNCTION ===
def cutmix(ds_one, ds_two, img_size, alpha=1.0):
    image1, label1 = ds_one
    image2, label2 = ds_two

    beta = tfp.distributions.Beta(alpha, alpha)
    lambda_val = beta.sample()

    cut_rat = tf.math.sqrt(1. - lambda_val)
    cut_w = tf.cast(img_size * cut_rat, tf.int32)
    cut_h = tf.cast(img_size * cut_rat, tf.int32)

    cx = tf.random.uniform([], 0, img_size, tf.int32)
    cy = tf.random.uniform([], 0, img_size, tf.int32)

    x1 = tf.clip_by_value(cx - cut_w // 2, 0, img_size)
    y1 = tf.clip_by_value(cy - cut_h // 2, 0, img_size)
    x2 = tf.clip_by_value(cx + cut_w // 2, 0, img_size)
    y2 = tf.clip_by_value(cy + cut_h // 2, 0, img_size)

    # Create mask
    bbx1, bby1, bbx2, bby2 = x1, y1, x2, y2

    patch = image2[bby1:bby2, bbx1:bbx2, :]
    pad_top = bby1
    pad_bottom = img_size - bby2
    pad_left = bbx1
    pad_right = img_size - bbx2

    patch_padded = tf.pad(patch, [[pad_top, pad_bottom], [pad_left, pad_right], [0,0]], constant_values=0)
    mask = tf.cast(patch_padded != 0, image1.dtype)

    cutmix_image = image1 * (1 - mask) + patch_padded

    lambda_adjusted = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img_size * img_size))
    mixed_label = lambda_adjusted * label1 + (1 - lambda_adjusted) * label2
    mixed_label = tf.clip_by_value(mixed_label, 0, 1)

    return cutmix_image, mixed_label

# === DATASET BUILDER WITH AUGMENTATION ===
def get_dataset(df, folder, img_size, batch_size, augment_type=None):
    filenames = df['image'].values
    labels = df['label'].values

    ds = tf.data.Dataset.from_tensor_slices((filenames, labels))
    ds = ds.shuffle(len(df))
    ds = ds.map(lambda x, y: parse_image(x, y, folder, img_size), num_parallel_calls=tf.data.AUTOTUNE)

    if augment_type == "mixup":
        ds1 = ds.shuffle(len(df))
        ds2 = ds.shuffle(len(df))
        ds = tf.data.Dataset.zip((ds1, ds2))
        ds = ds.map(mixup, num_parallel_calls=tf.data.AUTOTUNE)

    elif augment_type == "cutmix":
        ds1 = ds.shuffle(len(df))
        ds2 = ds.shuffle(len(df))
        ds = tf.data.Dataset.zip((ds1, ds2))
        ds = ds.map(lambda x, y: cutmix(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

# === MAIN GENERATORS FUNCTION ===
def get_generators(img_size, batch_size):
    train_df = load_dataframes(os.path.join(train_folder, "train_labels.csv"))
    val_df = load_dataframes(os.path.join(val_folder, "val_labels.csv"))
    test_df = load_dataframes(os.path.join(test_folder, "test_labels.csv"))

    print("Train label distribution:\n", train_df['label'].value_counts())
    print("Val label distribution:\n", val_df['label'].value_counts())
    print("Test label distribution:\n", test_df['label'].value_counts())

    train_ds = get_dataset(train_df, train_folder, img_size, batch_size, augment_type="mixup")
    val_ds = get_dataset(val_df, val_folder, img_size, batch_size)
    test_ds = get_dataset(test_df, test_folder, img_size, batch_size)

    return train_df, val_df, test_df, train_ds, val_ds, test_ds
