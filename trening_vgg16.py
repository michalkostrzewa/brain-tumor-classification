import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights
import time

# 1. Konfiguracja sprzętu
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Rozpoczynam pracę na WYSOKIEJ ROZDZIELCZOŚCI (512x512). Używane urządzenie: {device}")

# 2. Przygotowanie danych (Wysoka Rozdzielczość 448x448)
train_transform = transforms.Compose([
    transforms.Resize((480, 480)),         # Powiększamy z zapasem
    transforms.RandomCrop(448),            # Wycinamy docelowe 448x448 (podzielne przez 7 w VGG16!)
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = './BrainTumorData/Training' 

try:
    dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    # UWAGA: VGG16 zjada dużo pamięci, zwłaszcza przy 512x512. Batch_size = 16 to konieczność.
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"Znaleziono {len(dataset)} obrazów treningowych w {len(dataset.classes)} klasach: {dataset.classes}")
except Exception as e:
    print(f"Błąd ładowania danych: {e}")
    exit()

# 3. Pobranie i konfigurowanie modelu VGG16 (Transfer Learning + Fine-Tuning)
print("Pobieranie modelu VGG16...")
model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# Zamrażamy wszystkie wagi początkowe (ekstrakcja cech)
for param in model.parameters():
    param.requires_grad = False

# GŁĘBOKI FINE-TUNING: Odmrażamy "oczy" VGG16 (od warstwy 24 w górę) - TEGO BRAKOWAŁO!
for param in model.features[24:].parameters():
    param.requires_grad = True

# Odmrażamy cały blok klasyfikatora
for param in model.classifier.parameters():
    param.requires_grad = True

# Przebudowa ostatniej warstwy (dodanie Dropout zapobiegającego przeuczeniu)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, len(dataset.classes))
)

# Przenosimy model na procesor M4 Pro (MPS)
model = model.to(device)

# 4. Konfiguracja zaawansowanego uczenia

# SYSTEM KAR (Class Weights) - Waga 3.0 dla 'glioma', 1.0 dla reszty
# Uwaga: Musisz się upewnić, że 'glioma' to rzeczywiście klasa o indeksie 0 w dataset.classes
# W standardowym datasecie Kaggle: 0-glioma, 1-meningioma, 2-notumor, 3-pituitary
wagi_klas = torch.tensor([3.0, 1.0, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=wagi_klas)

# ZMIANA W OPTYMALIZATORZE: Uczymy "oczy" bardzo powoli (1e-5), a klasyfikator normalnie (1e-4)
optimizer = optim.Adam([
    {'params': model.features[24:].parameters(), 'lr': 1e-5},
    {'params': model.classifier.parameters(), 'lr': 1e-4}
], weight_decay=1e-4)
# 5. Pętla ucząca
epochs = 20 # Zwiększono do 20, podobnie jak w ResNet18
print(f"Rozpoczynamy trening na {epochs} epok!")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_time = time.time() - start_time
    accuracy = 100 * correct / total
    print(f"Epoka {epoch+1}/{epochs} | Strata: {running_loss/len(dataloader):.4f} | Dokładność treningowa: {accuracy:.2f}% | Czas: {epoch_time:.1f}s")

# 6. Zapisanie modelu
torch.save(model.state_dict(), 'moj_model_vgg16_hd.pth')
print("Trening zakończony! Model HD (512x512) został zapisany do pliku moj_model_vgg16_hd.pth")