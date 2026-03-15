import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Rozpoczynam Zaawansowany Trening ResNet. Używane urządzenie: {device}")

# 1. ZAAWANSOWANE Przygotowanie danych (Data Augmentation)
train_transform = transforms.Compose([
    transforms.Resize((230, 230)),         # Lekko powiększamy obraz...
    transforms.RandomCrop(224),            # ...aby losowo wyciąć fragment 224x224
    transforms.RandomHorizontalFlip(),     # Losowe odbicie lustrzane
    transforms.RandomRotation(15),         # Losowy obrót o max 15 stopni
    transforms.ColorJitter(brightness=0.1, contrast=0.1), # Delikatna zmiana jasności/kontrastu
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = './BrainTumorData/Training' 
dataset = datasets.ImageFolder(data_dir, transform=train_transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

# 2. Pobranie i modyfikacja modelu ResNet18 (Fine-Tuning)
print("Pobieranie i konfigurowanie modelu ResNet18...")
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Najpierw "zamrażamy" całą sieć
for param in model.parameters():
    param.requires_grad = False

# FINE-TUNING: Odmrażamy ostatni duży blok konwolucyjny (layer4) 
# Pozwoli to modelowi nauczyć się specyficznych, medycznych tekstur guzów
for param in model.layer4.parameters():
    param.requires_grad = True

# Podmieniamy i automatycznie "odmrażamy" ostatnią warstwę klasyfikatora
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5), # Losowo wyłącza 50% połączeń podczas treningu
    nn.Linear(num_features, len(dataset.classes))
)
model = model.to(device)

# 3. Konfiguracja zaawansowanego uczenia
# TWORZENIE SYSTEMU KAR (Class Weights)
# Kolejność alfabetyczna: [glioma, meningioma, notumor, pituitary]
# Dajemy wagę 3.0 dla 'glioma', a 1.0 dla pozostałych klas.
wagi_klas = torch.tensor([3.0, 1.0, 1.0, 1.0]).to(device)

# Podpinamy system kar do funkcji obliczającej błąd modelu
criterion = nn.CrossEntropyLoss(weight=wagi_klas)

# Definiujemy różne prędkości uczenia dla różnych części modelu:
# - layer4 uczy się BARDZO powoli (żeby nie zniszczyć pre-trenowanej wiedzy)
# - warstwa fc (nasza nowa) uczy się normalnym tempem
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], weight_decay=1e-4) # <- Dodana kara za przeuczenie

# 4. Pętla ucząca
epochs = 20 # Zostawiamy 20 epok, by model miał czas przyswoić augmentację
print("Rozpoczynamy trening z Fine-Tuningiem!")

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
    print(f"Epoka {epoch+1}/{epochs} | Strata (Loss): {running_loss/len(dataloader):.4f} | Dokładność: {accuracy:.2f}% | Czas: {epoch_time:.1f}s")

# 5. Zapisanie modelu nadpisze stary, słabszy model
torch.save(model.state_dict(), 'moj_model_resnet18.pth')
print("Trening zakończony! Ulepszony model ResNet został zapisany do pliku moj_model_resnet18.pth")