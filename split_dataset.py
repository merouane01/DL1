import os
import random
import shutil

# Chemins
source_base = r"C:\Users\PC\final_dataset"
target_base = r"C:\Users\PC\final_dataset_split"

splits = {
    'train': 0.7,
    'val': 0.15,
    'test': 0.15
}

classes = ['real', 'fake']

# Crée l'arborescence cible
for split in splits:
    for cls in classes:
        os.makedirs(os.path.join(target_base, split, cls), exist_ok=True)

def split_class(cls):
    src_folder = os.path.join(source_base, cls)
    files = [f for f in os.listdir(src_folder) if f.lower().endswith('.png')]
    random.shuffle(files)

    n = len(files)
    n_train = int(n * splits['train'])
    n_val = int(n * splits['val'])
    n_test = n - n_train - n_val  # reste

    split_files = {
        'train': files[:n_train],
        'val': files[n_train:n_train + n_val],
        'test': files[n_train + n_val:]
    }

    for split, flist in split_files.items():
        dst_folder = os.path.join(target_base, split, cls)
        for fname in flist:
            src = os.path.join(src_folder, fname)
            dst = os.path.join(dst_folder, fname)
            shutil.copy2(src, dst)

    print(f"{cls}: {n} images → "
          f"{len(split_files['train'])} train, "
          f"{len(split_files['val'])} val, "
          f"{len(split_files['test'])} test")

if __name__ == "__main__":
    random.seed(42)  # reproductible
    for c in classes:
        split_class(c)
    print("✅ Split terminé dans final_dataset_split")
