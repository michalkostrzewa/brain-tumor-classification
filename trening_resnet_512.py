import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
import time

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Rozpoczynam pracę na WYSOKIEJ ROZDZIELCZOŚCI (512x512). Używane urządzenie: {device}")

# 1. Przygotowanie danych (Wysoka Rozdzielczość 512x512)
train_transform = transforms.Compose([
    transforms.Resize((550, 550)),         # Powiększamy z zapasem
    transforms.RandomCrop(512),            # Wycinamy docelowe 512x512
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = './BrainTumorData/Training' 
dataset = datasets.ImageFolder(data_dir, transform=train_transform)

# ZMNIEJSZONY BATCH SIZE DO 16 (aby nie zapchać pamięci RAM przy dużych zdjęciach)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

# 2. Pobranie i konfigurowanie modelu ResNet18
print("Pobieranie modelu ResNet18...")
model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

# Zamrażamy wagi
for param in model.parameters():
    param.requires_grad = False

# Fine-Tuning: Odmrażamy ostatni blok (layer4)
for param in model.layer4.parameters():
    param.requires_grad = True

# Podmieniamy ostatnią warstwę dodając mechanizm DROPOUT (zapobiega przeuczeniu)
num_features = model.fc.in_features
model.fc = nn.Sequential(
    nn.Dropout(p=0.5),
    nn.Linear(num_features, len(dataset.classes))
)

model = model.to(device)

# 3. Konfiguracja zaawansowanego uczenia

# SYSTEM KAR (Class Weights) - Waga 3.0 dla 'glioma', 1.0 dla reszty
wagi_klas = torch.tensor([3.0, 1.0, 1.0, 1.0]).to(device)
criterion = nn.CrossEntropyLoss(weight=wagi_klas)

# Optymalizator z systemem weight_decay (kara za zbyt dużą pewność siebie)
optimizer = optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-4},
    {'params': model.fc.parameters(), 'lr': 1e-3}
], weight_decay=1e-4)

# 4. Pętla ucząca (ZMNIEJSZONA DO 10 EPOK)
epochs = 20
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

# 5. Zapisanie modelu
torch.save(model.state_dict(), 'moj_model_resnet18.pth')
print("Trening zakończony! Model HD (512x512) został zapisany do pliku moj_model_resnet18.pth")