import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# 1. Konfiguracja sprzętu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Generowanie macierzy pomyłek dla ulepszonego ResNet18. Proszę czekać...")

# 2. Przygotowanie danych testowych
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_dir = './BrainTumorData/Testing'
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Załadowanie Twojego ulepszonego modelu ResNet18
model = models.resnet18(weights=None)
num_features = model.fc.in_features

# POPRAWKA: Struktura musi idealnie pasować do tej z pliku treningowego!
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, len(test_dataset.classes))
)

# Wczytujemy "mózg" zapisany z ostatniego, najlepszego treningu
model.load_state_dict(torch.load('moj_model_resnet18.pth', weights_only=True))
model = model.to(device)
model.eval()

# 4. Zbieranie prawdziwych i przewidzianych etykiet
wszystkie_prawdziwe = []
wszystkie_przewidziane = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        
        wszystkie_prawdziwe.extend(labels.cpu().numpy())
        wszystkie_przewidziane.extend(predicted.cpu().numpy())

# 5. Rysowanie Macierzy Pomyłek
cm = confusion_matrix(wszystkie_prawdziwe, wszystkie_przewidziane)
nazwy_klas = test_dataset.classes

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=nazwy_klas, yticklabels=nazwy_klas,
            annot_kws={"size": 14})

plt.title('Macierz Pomyłek - Ulepszony ResNet18', fontsize=16, pad=20)
plt.ylabel('Prawdziwa klasa (True)', fontsize=14)
plt.xlabel('Przewidziana klasa przez AI (Predicted)', fontsize=14)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(rotation=0, fontsize=12)
plt.tight_layout()

# Zapis do pliku
nazwa_pliku = 'macierz_pomylek_resnet.png'
plt.savefig(nazwa_pliku, dpi=300)
print(f"Sukces! Macierz została wygenerowana i zapisana jako '{nazwa_pliku}'")