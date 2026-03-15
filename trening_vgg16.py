import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights
import time

# 1. Konfiguracja sprzętu - wykorzystujemy moc Twojego M4 Pro!
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Rozpoczynam pracę. Używane urządzenie: {device}")

# 2. Przygotowanie danych (Transformacje)
# VGG16 wymaga obrazów 224x224 i konkretnej normalizacji kolorów
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Wskazujemy folder z danymi treningowymi (Zmień 'ścieżka_do_danych' na swoją)
# Zakładam, że w folderze 'Training' są podfoldery z klasami guzów
data_dir = './BrainTumorData/Training' 

try:
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    print(f"Znaleziono {len(dataset)} obrazów treningowych w {len(dataset.classes)} klasach: {dataset.classes}")
except Exception as e:
    print(f"Błąd ładowania danych. Upewnij się, że folder istnieje i ma odpowiednią strukturę. Szczegóły: {e}")
    exit()

# 3. Pobranie modelu VGG16 (Transfer Learning)
print("Pobieranie modelu VGG16...")
model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)

# "Zamrażamy" początkowe warstwy, żeby ich nie trenować od nowa (oszczędza to mnóstwo czasu)
for param in model.parameters():
    param.requires_grad = False

# Podmieniamy ostatnią warstwę klasyfikatora na naszą (4 klasy guzów)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, len(dataset.classes))

# Przenosimy model na procesor M4 Pro (MPS)
model = model.to(device)

# 4. Konfiguracja uczenia
criterion = nn.CrossEntropyLoss()
# Trenujemy tylko nowo dodaną warstwę
optimizer = optim.Adam(model.classifier[6].parameters(), lr=0.001)

# 5. Pętla ucząca (Trening)
epochs = 3 # Zaczynamy od 3 epok dla testu
print("Rozpoczynamy trening!")

for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for inputs, labels in dataloader:
        # Przenosimy dane na kartę graficzną Apple
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zerujemy gradienty
        optimizer.zero_grad()
        
        # Propagacja w przód (Forward)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Propagacja wstecz i aktualizacja wag (Backward & Optimize)
        loss.backward()
        optimizer.step()
        
        # Obliczanie statystyk
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    epoch_time = time.time() - start_time
    accuracy = 100 * correct / total
    print(f"Epoka {epoch+1}/{epochs} | Strata (Loss): {running_loss/len(dataloader):.4f} | Dokładność: {accuracy:.2f}% | Czas: {epoch_time:.1f}s")

print("Trening zakończony!")

torch.save(model.state_dict(), 'moj_model_vgg16.pth')
print("Model został zapisany do pliku moj_model_vgg16.pth!")