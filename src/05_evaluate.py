"""Evalúa un checkpoint con test y guarda métricas y matriz de confusión."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay

from data import (DEFAULT_CHECKPOINT, DEFAULT_DATA, load_dataset, make_loader,
                  new_output, positive_int, write_json)
from model import load_checkpoint, select_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, help="Por defecto: carpeta evaluation junto al checkpoint.")
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-batches", type=positive_int, help="Evaluación parcial para prueba técnica.")
    args = parser.parse_args()
    device = select_device(args.device)
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    test = load_dataset(args.data, "test")
    output = new_output(args.output or args.checkpoint.parent / "evaluation")
    loader = make_loader(test, args.batch_size)
    truth, predictions = [], []
    with torch.inference_mode():
        for index, (images, labels) in enumerate(loader):
            if args.max_batches is not None and index >= args.max_batches:
                break
            predictions.extend(model(images.to(device)).argmax(1).cpu().tolist())
            truth.extend(labels.tolist())
    smoke = checkpoint["config"].get("smoke_test", False) or args.max_batches is not None
    report = classification_report(truth, predictions, labels=[0, 1],
                                   target_names=test.classes, output_dict=True, zero_division=0)
    matrix = confusion_matrix(truth, predictions, labels=[0, 1])
    results = {"smoke_test": smoke, "checkpoint": str(args.checkpoint.resolve()),
               "checkpoint_epoch": checkpoint["epoch"], "samples_evaluated": len(truth),
               "total_test_samples": len(test), "accuracy": accuracy_score(truth, predictions),
               "classification_report": report, "confusion_matrix": matrix.tolist()}
    write_json(output / "metrics.json", results)
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay(matrix, display_labels=test.classes).plot(ax=ax, colorbar=False)
    ax.set_title("PRUEBA TÉCNICA" if smoke else "Evaluación en test")
    fig.tight_layout()
    fig.savefig(output / "confusion_matrix.png")
    plt.close(fig)
    if smoke:
        print("PRUEBA TÉCNICA: estas métricas no representan rendimiento final.")
    print(f"Accuracy: {results['accuracy']:.4f}; imágenes evaluadas: {len(truth)}/{len(test)}")
    print(f"Métricas por clase (precision, recall, F1): {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
