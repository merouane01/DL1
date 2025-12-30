import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os
import urllib.request

st.set_page_config(page_title="🚀 4x Fake/Real Detector", page_icon="🖼️")

# ---------- Téléchargement automatique des modèles depuis ton HF Space ----------
MODEL_URLS = {
    "densenet_ai_detection.pth": "https://huggingface.co/spaces/merouane02/DL1/resolve/main/densenet_ai_detection.pth",  # ← REMPLACE si nom différent
    "vgg16_binary.pth": "https://huggingface.co/spaces/merouane02/DL1/resolve/main/vgg16_binary.pth",
    "alex_binary.pth": "https://huggingface.co/spaces/merouane02/DL1/resolve/main/alex_binary.pth",
    "mymodel3.pth": "https://huggingface.co/spaces/merouane02/DL1/resolve/main/mymodel3.pth",
}

with st.spinner("Vérification et téléchargement des modèles (premier lancement uniquement)..."):
    for filename, url in MODEL_URLS.items():
        if not os.path.exists(filename):
            st.info(f"Téléchargement de {filename}...")
            urllib.request.urlretrieve(url, filename)
            st.success(f"{filename} téléchargé !")

device = torch.device("cpu")  # Streamlit Cloud & HF Spaces gratuits = CPU only
st.sidebar.write(f"**Device :** {device}")

CLASS_NAMES = ['real', 'fake']

# ========== Transforms ==========
transform_imagenet = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transform_custom = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# ========== Classes des modèles (identiques à ton code original) ==========

class VGG16Custom(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class AlexNetInspo(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True), nn.MaxPool2d(3, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.LazyLinear(4096), nn.ReLU(),
            nn.LazyLinear(4096), nn.ReLU(), nn.LazyLinear(num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class FiveBlockCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.block1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block3 = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block4 = nn.Sequential(nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.block5 = nn.Sequential(nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2))
        self.flatten = nn.Flatten()
        
        # Calcul dynamique de la taille flatten
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            x = self.block1(dummy); x = self.block2(x); x = self.block3(x)
            x = self.block4(x); x = self.block5(x)
            flat_size = x.view(1, -1).size(1)
        
        self.fc1 = nn.Linear(flat_size, 256)
        self.drop = nn.Dropout(0.5)
        self.out = nn.Linear(256, 1)

    def forward(self, x):
        x = self.block1(x); x = self.block2(x); x = self.block3(x)
        x = self.block4(x); x = self.block5(x)
        x = self.flatten(x)
        x = self.drop(torch.relu(self.fc1(x)))
        x = torch.sigmoid(self.out(x))
        return x

# ========== Chargement des modèles ==========

@st.cache_resource
def load_densenet():
    model_path = "densenet_ai_detection.pth"
    model = models.densenet121(weights=None)
    num_ftrs = model.classifier.in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5), nn.Linear(num_ftrs, 512), nn.ReLU(inplace=True),
        nn.Dropout(0.3), nn.Linear(512, 2)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model.to(device).eval()

@st.cache_resource
def load_vgg16_custom():
    model_path = "vgg16_binary.pth"
    if not os.path.exists(model_path):
        return None
    model = VGG16Custom(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        return model.eval()
    except:
        return None

@st.cache_resource
def load_alexnet():
    model_path = "alex_binary.pth"
    if not os.path.exists(model_path):
        return None
    model = AlexNetInspo(num_classes=2).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        return model.eval()
    except:
        return None

@st.cache_resource
def load_fiveblock():
    model_path = "mymodel3.pth"
    model = FiveBlockCNN().to(device)
    if os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        except:
            pass
    return model.eval()

# Chargement effectif
densenet_model = load_densenet()
vgg16_model = load_vgg16_custom()
alexnet_model = load_alexnet()
fiveblock_model = load_fiveblock()

# ========== Interface ==========

st.title("🖼️🚀 4x Fake/Real Image Detector")
st.info("**DenseNet121 + VGG16_Custom + AlexNetInspo + FiveBlockCNN**")

models_list = ["DenseNet121 (ImageNet)"]
if vgg16_model: models_list.append("VGG16_Custom")
if alexnet_model: models_list.append("AlexNetInspo")
models_list.append("FiveBlockCNN (Custom)")

model_choice = st.selectbox("🎯 Choisir le modèle :", models_list)

uploaded_file = st.file_uploader("📁 Charger une image...", type=['png', 'jpg', 'jpeg', 'webp'])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Image chargée", width=400)
    
    if st.button("🔍 CLASSIFIER", type="primary", use_container_width=True):
        with st.spinner(f"Analyse en cours avec {model_choice}..."):
            if "DenseNet" in model_choice:
                transform = transform_imagenet
                model = densenet_model
            elif "VGG16_Custom" in model_choice:
                transform = transform_custom
                model = vgg16_model
            elif "AlexNet" in model_choice:
                transform = transform_custom
                model = alexnet_model
            else:
                transform = transform_custom
                model = fiveblock_model
            
            input_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                if "FiveBlock" in model_choice:
                    prob_fake = model(input_tensor).item()
                    prob_real = 1 - prob_fake
                    pred = 1 if prob_fake > 0.5 else 0
                    confidence = max(prob_real, prob_fake)
                    probs = np.array([prob_real, prob_fake])
                else:
                    outputs = model(input_tensor)
                    probs = torch.softmax(outputs, dim=1)[0].cpu().numpy()
                    confidence = np.max(probs)
                    pred = np.argmax(probs)
            
            col1, col2 = st.columns(2)
            with col1:
                icon = "🚨 FAKE" if pred == 1 else "✅ REAL"
                st.markdown(f"### {icon}")
                st.metric("Confiance", f"{confidence:.1%}")
            with col2:
                st.metric("Real", f"{probs[0]:.1%}")
                st.metric("Fake", f"{probs[1]:.1%}")

# Sidebar : statut des modèles
st.sidebar.markdown("---")
st.sidebar.subheader("Statut des modèles")
for filename in MODEL_URLS.keys():
    if os.path.exists(filename):
        size_mb = os.path.getsize(filename) / (1024 * 1024)
        st.sidebar.success(f"✅ {filename} ({size_mb:.1f} MB)")
    else:
        st.sidebar.error(f"❌ {filename} manquant")
