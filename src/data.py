"""Carga consistente de imágenes y utilidades compartidas del experimento."""
import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "dataset/split"
DEFAULT_OUTPUT = ROOT / "artifacts/default"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT / "best_model.pt"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def image_tensor(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.size != (224, 224):
        image = image.resize((224, 224), Image.Resampling.LANCZOS)
    return ToTensor()(image)


def load_dataset(root, split):
    dataset = ImageFolder(Path(root) / split, transform=image_tensor)
    if dataset.class_to_idx != {"cat": 0, "dog": 1}:
        raise ValueError(f"Clases inesperadas en {split}: {dataset.class_to_idx}")
    return dataset


def make_loader(dataset, batch_size, *, shuffle=False, seed=42):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(seed), num_workers=0)


def positive_int(value):
    import argparse
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Debe ser un entero mayor que cero.")
    return number


def new_output(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise ValueError(f"La salida contiene resultados; elige otra con --output: {path}")
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
