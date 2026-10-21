"""Arquitectura compartida por inspección, entrenamiento y predicción."""
from pathlib import Path

import torch
from torch import nn

# Modelo CNN simple para clasificación binaria de gatos y perros
class CatDogCNN(nn.Module):
    def __init__(self):
        super().__init__()

        # Cuatro bloques convolucionales con normalización y reducción espacial.
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )

        # Capa totalmente conectada para clasificación binaria
        self.classifier = nn.Sequential(
            # El pooling global deja una característica por cada uno de los 256 canales.
            nn.Flatten(), nn.Linear(256, 128), nn.ReLU(),
            # Dropout con probabilidad de 0.5 para evitar sobreajuste
            nn.Dropout(0.5), nn.Linear(128, 2),
        )

    # Método forward que define cómo los datos pasan a través del modelo
    def forward(self, images):
        return self.classifier(self.features(images)) # Pasa las imágenes a través de las capas convolucionales y luego a través del clasificador


# Para seleccionar el dispositivo (CPU o GPU) para la inferencia
def select_device(name="auto"):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA no está disponible; usa --device cpu.")
    if name == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    return torch.device(name)

# Cargar un checkpoint y devolver el modelo y el checkpoint (es decir los pesos del modelo y la información de las clases)
def load_checkpoint(path: Path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["class_to_idx"] != {"cat": 0, "dog": 1}:
        raise ValueError("El checkpoint no tiene las clases esperadas.")
    model = CatDogCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device).eval(), checkpoint
