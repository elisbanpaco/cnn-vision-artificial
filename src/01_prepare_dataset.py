"""Divide las imágenes por clase (70/15/15) y las guarda en RGB a 224×224.

Uso desde la raíz: python src/01_prepare_dataset.py
Los originales se conservan en dataset/train; la salida es dataset/split.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parents[1]


def prepare_dataset(source: Path, output: Path, seed: int = 42) -> None:
    source, output = source.resolve(), output.resolve()
    if not source.is_dir():
        raise ValueError(f"No existe el directorio de entrada: {source}")
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Entrada y salida deben estar en directorios separados.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"La salida debe estar vacía para evitar mezclar divisiones: {output}")

    classes = defaultdict(list)
    for path in sorted(source.iterdir()):
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            classes[path.name.split(".")[0]].append(path)
    if not classes:
        raise ValueError(f"No se encontraron imágenes en {source}")

    rng = random.Random(seed)
    assignments = []
    counts = {}
    for label, paths in sorted(classes.items()):
        rng.shuffle(paths)
        train_end = int(len(paths) * 0.70)
        val_end = train_end + int(len(paths) * 0.15)
        partitions = {"train": paths[:train_end], "val": paths[train_end:val_end], "test": paths[val_end:]}
        counts[label] = {split: len(items) for split, items in partitions.items()}
        for split, items in partitions.items():
            for path in items:
                assignments.append((path, split, label))

    original_sizes = Counter()
    records = []
    for index, (path, split, label) in enumerate(assignments, 1):
        target = output / split / label / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            original_sizes[f"{image.width}x{image.height}"] += 1
            image = ImageOps.exif_transpose(image).convert("RGB")
            if image.size != (224, 224):
                image = image.resize((224, 224), Image.Resampling.LANCZOS)
            image.save(target)
        records.append({"source": path.name, "split": split, "class": label,
                        "output": target.relative_to(output).as_posix()})
        if index % 1000 == 0:
            print(f"Procesadas {index}/{len(assignments)} imágenes", flush=True)

    manifest = {"seed": seed, "ratios": {"train": 0.70, "val": 0.15, "test": 0.15},
                "size": [224, 224], "counts": counts,
                "original_sizes": dict(original_sizes), "images": records}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(counts, indent=2))
    print(f"Dataset guardado en {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "dataset/train")
    parser.add_argument("--output", type=Path, default=ROOT / "dataset/split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    prepare_dataset(args.source, args.output, args.seed)
