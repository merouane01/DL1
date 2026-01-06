import os
import cv2
from PIL import Image
import hashlib
import numpy as np

# Chemins
ds1_base_path = r"C:\Users\PC\Dataset"
ds2_base_path = r"C:\Users\PC\images"
final_dataset_path = r"C:\Users\PC\final_dataset"

# SUPPRIME les anciennes images real (on recommence propre)
if os.path.exists(r"C:\Users\PC\final_dataset\real"):
    import shutil
    shutil.rmtree(r"C:\Users\PC\final_dataset\real")
os.makedirs(r"C:\Users\PC\final_dataset\real", exist_ok=True)
os.makedirs(r"C:\Users\PC\final_dataset\fake", exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def compute_hash(img_path):
    with open(img_path, 'rb') as f:
        return hashlib.md5(f.read()).hexdigest()

def detect_and_crop_face(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # ✅ VERSION TOLÉRANTE pour GAN + images modernes
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.3,      # Moins strict (1.1→1.3)
        minNeighbors=3,       # Moins strict (5→3)
        minSize=(30, 30)      # Plus petit (50→30)
    )
    
    if len(faces) > 0:
        # Prend le plus grand visage
        x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
        face_img = img[y:y+h, x:x+w]
    else:
        # ✅ FALLBACK CENTRAL CROP (sauve TOUTES les images)
        h, w = img.shape[:2]
        size = min(h, w) * 3//4
        start_x = (w - size) // 2
        start_y = (h - size) // 2
        face_img = img[start_y:start_y+size, start_x:start_x+size]
    
    # Redimensionne + conversion
    face_img = cv2.resize(face_img, (224, 224))
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(face_img_rgb)

# Fonctions load_images (IDENTIQUES)
def load_images_ds1(base_path):
    categories = ['real', 'fake']
    subsets = ['train', 'test', 'validation']
    images = {'real': [], 'fake': []}
    for subset in subsets:
        for cat in categories:
            folder = os.path.join(base_path, subset, cat)
            if os.path.exists(folder):
                for f in os.listdir(folder):
                    if f.lower().endswith(('jpg','jpeg','png')):  # ✅ lower() = maj/min OK
                        images[cat].append(os.path.join(folder, f))
    return images

def load_images_ds2(base_path):
    images = {'real': [], 'fake': []}
    real_path = os.path.join(base_path, 'real')
    if os.path.exists(real_path):
        for f in os.listdir(real_path):
            if f.lower().endswith(('jpg','jpeg','png')):
                images['real'].append(os.path.join(real_path, f))
    fake_base = os.path.join(base_path, 'fake')
    for sub in ['flux_dev', 'flux_pro', 'sdxl']:
        path = os.path.join(fake_base, sub)
        if os.path.exists(path):
            for f in os.listdir(path):
                if f.lower().endswith(('jpg','jpeg','png')):
                    images['fake'].append(os.path.join(path, f))
    return images

def clean_and_save(images_dict, dest_base):
    hashes = set()
    count = {'real': 0, 'fake': 0}
    
    for label in ['real', 'fake']:
        print(f"\n🔄 Traitement {label.upper()} ({len(images_dict[label])} images)...")
        for i, img_path in enumerate(images_dict[label]):
            if i % 1000 == 0:
                print(f"  {i}/{len(images_dict[label])} ({count[label]} sauvés)")
            
            img_hash = compute_hash(img_path)
            if img_hash in hashes:
                continue
                
            face_img = detect_and_crop_face(img_path)
            if face_img is None:
                continue
                
            hashes.add(img_hash)
            save_path = os.path.join(dest_base, label, f'{label}_{count[label]}.png')
            face_img.save(save_path)
            count[label] += 1
    
    print(f"\n✅ FINAL : {count['real']} REAL, {count['fake']} FAKE sauvés")

# LANCEMENT
print("🚀 NETTOYAGE VERSION 2 (TOLÉRANTE GAN)")
images_ds1 = load_images_ds1(ds1_base_path)
images_ds2 = load_images_ds2(ds2_base_path)
images_merged = {
    'real': images_ds1['real'] + images_ds2['real'],
    'fake': images_ds1['fake'] + images_ds2['fake']
}
print(f"Sources: {len(images_merged['real'])} real, {len(images_merged['fake'])} fake")

clean_and_save(images_merged, final_dataset_path)
print("🎉 TERMINÉ !")
