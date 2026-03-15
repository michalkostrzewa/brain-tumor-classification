import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torchvision.models import VGG16_Weights

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Rozpoczynam testowanie. Używane urządzenie: {device}")

# 1. Transformacje (takie same jak przy treningu, to kluczowe!)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 2. Wczytanie danych TESTOWYCH
test_dir = './BrainTumorData/Testing'
test_dataset = datasets.ImageFolder(test_dir, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

# 3. Odtworzenie struktury modelu VGG16
model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
num_features = model.classifier[6].in_features
model.classifier[6] = nn.Linear(num_features, len(test_dataset.classes))

# 4. Wczytanie zapisanego "mózgu" z pliku .pth
model.load_state_dict(torch.load('moj_model_vgg16.pth', weights_only=True))
model = model.to(device)

# Ważne: Przełączamy model w tryb ewaluacji (wyłącza pewne mechanizmy używane tylko w treningu)
model.eval()

# 5. Egzamin!
correct = 0
total = 0

print("Sprawdzam zdjęcia testowe. Proszę czekać...")
# Wyłączamy obliczanie gradientów - przyspiesza to testowanie i oszczędza pamięć
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f"Konfiguracja testowa zakończona!")
print(f"Skuteczność modelu na {total} nowych, niewidzianych wcześniej zdjęciach wynosi: {accuracy:.2f}%")