"""Muestra la arquitectura y verifica sus dimensiones sin entrenar."""
import torch
from model import CatDogCNN, select_device


def main():
    select_device("cpu")
    model = CatDogCNN().eval()
    print(model)
    x = torch.zeros(1, 3, 224, 224)
    print("Entrada:", tuple(x.shape))
    with torch.inference_mode():
        for layer in [*model.features, *model.classifier]:
            x = layer(x)
            print(f"{type(layer).__name__:12s} → {tuple(x.shape)}")
    assert x.shape == (1, 2)
    print(f"Parámetros entrenables: {sum(p.numel() for p in model.parameters()):,}")


if __name__ == "__main__":
    main()
