import os
import random
import shutil
import pandas as pd
from PIL import Image, ImageOps
from tqdm import tqdm

# === Caminhos ===
df_path = "/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Training_labels.csv"
preprocessed_folder = "/raid/DATASETS/rmiguel_datasets/ISIC16/Preprocessed_Training_Data"
augmented_folder = "/raid/DATASETS/rmiguel_datasets/ISIC16/Augmented_Training_Data"

# === Segurança de caminho ===
assert "rmiguel_datasets" in augmented_folder, "Caminho de output inseguro! Abortado."

# === Criar pasta de output ===
if os.path.exists(augmented_folder):
    print(f"⚠️ A pasta {augmented_folder} já existe. Vai ser apagada.")
    shutil.rmtree(augmented_folder)
os.makedirs(augmented_folder, exist_ok=True)

# === Carregar DataFrame ===
df = pd.read_csv(df_path, header=None, names=["image", "label"])

# Garantir que nomes terminam com .jpg
df['image'] = df['image'].astype(str).apply(lambda x: x if x.endswith('.jpg') else x + '.jpg')

# Mapear 'benign' e 'malignant' para 0/1 se necessário
if df['label'].dtype == object:
    df['label'] = df['label'].map({'benign': '0', 'malignant': '1'}).astype(str)

# Copiar imagens originais para pasta de output
for img_file in os.listdir(preprocessed_folder):
    src = os.path.join(preprocessed_folder, img_file)
    dst = os.path.join(augmented_folder, img_file)
    shutil.copy2(src, dst)

# === Contagem por classe ===
label_counts = df['label'].value_counts()
max_count = label_counts.max()

print("Distribuição inicial das classes:")
print(label_counts)

# === Augmentação ===
new_rows = []

for cls, count in label_counts.items():
    if count >= max_count:
        print(f"Classe '{cls}' já está balanceada. Ignorada.")
        continue

    samples = df[df['label'] == cls]
    augment_times = (max_count - count) // count
    print(f"Classe '{cls}': Aumentando {augment_times}x cada amostra.")

    for _, row in tqdm(samples.iterrows(), total=len(samples), desc=f"Augmentando classe {cls}"):
        img_name = row['image']
        img_path = os.path.join(preprocessed_folder, img_name)

        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Erro ao abrir {img_path}: {e}")
            continue

        for i in range(augment_times):
            aug_img = img.copy()

            # Transformações simples
            if random.random() < 0.5:
                aug_img = ImageOps.mirror(aug_img)
            if random.random() < 0.5:
                aug_img = ImageOps.flip(aug_img)

            angle = random.uniform(-30, 30)
            aug_img = aug_img.rotate(angle)

            # Salvar com nome único
            base = os.path.splitext(img_name)[0]
            new_name = f"{base}_aug_{i}.jpg"
            save_path = os.path.join(augmented_folder, new_name)
            aug_img.save(save_path)

            new_rows.append({'image': new_name, 'label': cls})

# === Gerar DataFrame final ===
df_aug = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

print(f"\nTotal de imagens após augmentação: {len(df_aug)}")
print(df_aug['label'].value_counts())

# === (Opcional) Salvar novo CSV ===
out_csv = "/raid/DATASETS/rmiguel_datasets/ISIC16/CSV/Augmented_Training_labels.csv"
df_aug.to_csv(out_csv, index=False, header=False)
print(f"Novo CSV salvo em: {out_csv}")
