"""Clasifica una imagen nueva con el modelo entrenado."""
import argparse
from pathlib import Path

from PIL import Image
import torch

from data import DEFAULT_CHECKPOINT, image_tensor
from model import load_checkpoint, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    with Image.open(args.image) as image:
        tensor = image_tensor(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(tensor).softmax(dim=1)[0].cpu()
    classes = sorted(checkpoint["class_to_idx"], key=checkpoint["class_to_idx"].get)
    if checkpoint["config"].get("smoke_test", False):
        print("PRUEBA TÉCNICA: modelo sin entrenamiento completo.")
    print(f"Predicción: {classes[probabilities.argmax().item()]}")
    for label, probability in zip(classes, probabilities.tolist()):
        print(f"{label}: {probability:.2%}")


if __name__ == "__main__":
    main()
