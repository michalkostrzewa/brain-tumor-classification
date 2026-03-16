import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as F_t
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# 1. Konfiguracja
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Uruchamianie konsylium AI (ResNet18 + VGG16). Proszę czekać...")

# 2. Baza danych (Tylko konwersja na tensor i kolory)
transform_base = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dir = './BrainTumorData/Testing' 
test_dataset = datasets.ImageFolder(test_dir, transform=transform_base)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False)

nazwy_klas = test_dataset.classes
print(f"Rozpoznane klasy: {nazwy_klas}")

# ==========================================
# 3. Wczytywanie Modelu 1: ResNet18 (512x512)
# ==========================================
print("Budzenie doktora ResNet18...")
model_resnet = models.resnet18(weights=None)
num_features_res = model_resnet.fc.in_features
model_resnet.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features_res, len(nazwy_klas))
)
model_resnet.load_state_dict(torch.load('moj_model_resnet18_512.pth', weights_only=True))
model_resnet = model_resnet.to(device)
model_resnet.eval()

# ==========================================
# 4. Wczytywanie Modelu 2: VGG16 (448x448)
# ==========================================
print("Budzenie doktora VGG16...")
model_vgg = models.vgg16(weights=None)
num_features_vgg = model_vgg.classifier[6].in_features
model_vgg.classifier[6] = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features_vgg, len(nazwy_klas))
)
model_vgg.load_state_dict(torch.load('moj_model_vgg16_hd.pth', weights_only=True))
model_vgg = model_vgg.to(device)
model_vgg.eval()

# ==========================================
# 5. Wspólne diagnozowanie i zbieranie statystyk
# ==========================================
wszystkie_prawdziwe = []

# Listy na odpowiedzi każdego modelu
przewidziane_resnet = []
przewidziane_vgg = []
przewidziane_ensemble = []

correct_resnet = 0
correct_vgg = 0
correct_ensemble = 0
total = 0

print("Rozpoczynam diagnozowanie zdjęć...")

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Przygotowanie zdjęć pod konkretne modele w locie
        inputs_resnet = F_t.resize(inputs, [512, 512])
        inputs_vgg = F_t.resize(inputs, [448, 448])
        
        # ---------------- Diagnoza ResNet ----------------
        wyjscie_resnet = model_resnet(inputs_resnet)
        prawdopodobienstwa_resnet = F.softmax(wyjscie_resnet, dim=1)
        _, pred_resnet = torch.max(wyjscie_resnet, 1)
        
        # ---------------- Diagnoza VGG ----------------
        wyjscie_vgg = model_vgg(inputs_vgg)
        prawdopodobienstwa_vgg = F.softmax(wyjscie_vgg, dim=1)
        _, pred_vgg = torch.max(wyjscie_vgg, 1)
        
        # ---------------- Diagnoza ENSEMBLE ----------------
        wynik_zespolowy = (prawdopodobienstwa_resnet *1.1 + prawdopodobienstwa_vgg *0.9) / 2.0
        _, pred_ensemble = torch.max(wynik_zespolowy, 1)
        
        # Zapisywanie statystyk na żywo
        total += labels.size(0)
        correct_resnet += (pred_resnet == labels).sum().item()
        correct_vgg += (pred_vgg == labels).sum().item()
        correct_ensemble += (pred_ensemble == labels).sum().item()
        
        wszystkie_prawdziwe.extend(labels.cpu().numpy())
        przewidziane_resnet.extend(pred_resnet.cpu().numpy())
        przewidziane_vgg.extend(pred_vgg.cpu().numpy())
        przewidziane_ensemble.extend(pred_ensemble.cpu().numpy())

# Obliczanie skuteczności procentowej
acc_resnet = 100 * correct_resnet / total
acc_vgg = 100 * correct_vgg / total
acc_ensemble = 100 * correct_ensemble / total

print(f"\n================ PODSUMOWANIE ================")
print(f"Skuteczność ResNet18: {acc_resnet:.2f}%")
print(f"Skuteczność VGG16:    {acc_vgg:.2f}%")
print(f"Skuteczność ENSEMBLE: {acc_ensemble:.2f}%")
print("==============================================")

# ==========================================
# 6. Rysowanie i zapisywanie 3 Macierzy Pomyłek
# ==========================================

def narysuj_macierz(prawdziwe, przewidziane, nazwa_modelu, kolor, plik, skutecznosc):
    cm = confusion_matrix(prawdziwe, przewidziane)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap=kolor, 
                xticklabels=nazwy_klas, yticklabels=nazwy_klas, annot_kws={"size": 14})
    plt.title(f'Macierz Pomyłek - {nazwa_modelu} | Skuteczność: {skutecznosc:.2f}%', fontsize=15, pad=20)
    plt.ylabel('Prawdziwa klasa (True)', fontsize=14)
    plt.xlabel('Przewidziana klasa przez AI (Predicted)', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    plt.savefig(plik, dpi=300)
    plt.close() # Zamyka wykres, żeby kolejny narysował się na czysto

print("Generowanie plików graficznych...")

# Rysujemy 3 macierze w różnych kolorach dla łatwego odróżnienia
narysuj_macierz(wszystkie_prawdziwe, przewidziane_resnet, "ResNet18", "Blues", "macierz_1_resnet.png", acc_resnet)
narysuj_macierz(wszystkie_prawdziwe, przewidziane_vgg, "VGG16", "Greens", "macierz_2_vgg.png", acc_vgg)
narysuj_macierz(wszystkie_prawdziwe, przewidziane_ensemble, "ENSEMBLE", "Purples", "macierz_3_ensemble.png", acc_ensemble)

print("Sukces! Zapisano 3 pliki: macierz_1_resnet.png, macierz_2_vgg.png oraz macierz_3_ensemble.png")