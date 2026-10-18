"""Verifica conteos y tamaños; guarda una cuadrícula de ejemplos de train."""
import argparse
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from data import DEFAULT_DATA, ROOT, load_dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "artifacts/inspection/examples.png")
    args = parser.parse_args()
    expected = {"train": 17500, "val": 3750, "test": 3750}
    names = set()
    datasets = {}
    for split, count in expected.items():
        dataset = datasets[split] = load_dataset(args.data, split)
        counts = Counter(dataset.targets)
        print(f"{split}: {len(dataset)} imágenes; "
              f"{dict((label, counts[idx]) for label, idx in dataset.class_to_idx.items())}", flush=True)
        if len(dataset) != count or set(counts.values()) != {count // 2}:
            raise ValueError(f"Conteos inesperados en {split}.")
        for path, _ in dataset.samples:
            name = Path(path).name
            if name in names:
                raise ValueError(f"Archivo repetido entre conjuntos: {name}")
            names.add(name)
            with Image.open(path) as image:
                if image.size != (224, 224) or image.mode != "RGB":
                    raise ValueError(f"Se esperaba RGB 224×224: {path}")
    if args.output.exists():
        raise ValueError(f"Ya existe {args.output}; elige otra ruta con --output.")
    fig, axes = plt.subplots(2, 4, figsize=(10, 5))
    dataset = datasets["train"]
    for label in range(2):
        examples = [p for p, target in dataset.samples if target == label][:4]
        for ax, path in zip(axes[label], examples):
            with Image.open(path) as image:
                ax.imshow(image)
            ax.set_title(dataset.classes[label])
            ax.axis("off")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    plt.close(fig)
    print(f"Verificación correcta. Ejemplos: {args.output}")


if __name__ == "__main__":
    main()
