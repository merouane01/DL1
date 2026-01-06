import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score

# ========= CONFIG =========
DATA_DIR = r"C:\BACKUP"   # <-- adapte ce chemin
BATCH_SIZE = 16
EPOCHS = 10       # commence avec 10, augmente si c'est OK
LR = 1e-3
DEVICE = torch.device("cpu")

# Taille du SOUS-ENSEMBLE (pour aller plus vite sur CPU)
N_TRAIN = 20000    # images max pour train
N_VAL   = 5000      # images max pour val
N_TEST  = 5000      # images max pour test

print("Device :", DEVICE)

# ========= TRANSFORMS =========
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),
])

transform_eval = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5],
                         [0.5, 0.5, 0.5]),
])

# ========= DATASETS COMPLETS =========
train_full = datasets.ImageFolder(os.path.join(DATA_DIR, "train"),
                                  transform=transform_train)
val_full   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"),
                                  transform=transform_eval)
test_full  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"),
                                  transform=transform_eval)

print("Classes :", train_full.classes)
print("Nb images FULL train:", len(train_full),
      "val:", len(val_full),
      "test:", len(test_full))

# ========= SOUS-ENSEMBLES POUR CPU =========
def make_subset(dataset, n_max):
    n = min(n_max, len(dataset))
    indices = np.random.choice(len(dataset), size=n, replace=False)
    return Subset(dataset, indices)

train_dataset = make_subset(train_full, N_TRAIN)
val_dataset   = make_subset(val_full,   N_VAL)
test_dataset  = make_subset(test_full,  N_TEST)

print("Nb images SUBSET train:", len(train_dataset),
      "val:", len(val_dataset),
      "test:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)

# ========= MODELE FiveBlockCNN (Conv2D + MaxPool x5) =========
class FiveBlockCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 256 -> 128
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 128 -> 64
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 64 -> 32
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 32 -> 16
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 16 -> 8
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 8 * 8, 256)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.flatten(x)
        x = self.drop(F.relu(self.fc1(x)))
        x = torch.sigmoid(self.out(x))
        return x

model = FiveBlockCNN().to(DEVICE)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

# ========= ENTRAINEMENT =========
best_val_acc = 0.0

for epoch in range(EPOCHS):
    # ---- TRAIN ----
    model.train()
    running_loss = 0.0

    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.float().unsqueeze(1).to(DEVICE)
                                                       
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_train_loss = running_loss / len(train_loader)

    # ---- VALIDATION ----
    model.eval()
    val_preds, val_trues = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            preds = (outputs.cpu() > 0.5).numpy()
            val_preds.extend(preds)
            val_trues.extend(labels.numpy())

    val_acc = accuracy_score(val_trues, val_preds)
    print(f"Epoch {epoch+1}/{EPOCHS} - "
          f"train_loss={avg_train_loss:.4f} - val_acc={val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "mymodel3.pth")
        print(">> Nouveau meilleur modèle sauvegardé.")

print("Entraînement terminé. Meilleure val_acc:", best_val_acc)

# ========= TEST =========
model.load_state_dict(torch.load("mymodel3.pth",
                                 map_location=DEVICE))
model.eval()
test_preds, test_trues = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(DEVICE)
        outputs = model(imgs)
        preds = (outputs.cpu() > 0.5).numpy()
        test_preds.extend(preds)
        test_trues.extend(labels.numpy())

test_acc = accuracy_score(test_trues, test_preds)
print("Accuracy finale sur le test (subset):", test_acc)




