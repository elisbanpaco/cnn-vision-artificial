"""Entrena con train y selecciona el mejor modelo usando exclusivamente val."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse
import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn

from data import (DEFAULT_DATA, DEFAULT_OUTPUT, load_dataset, make_loader,
                  new_output, positive_int, seed_everything, write_json)
from model import CatDogCNN, select_device


def run_epoch(model, loader, device, optimizer=None, max_batches=None):
    training = optimizer is not None
    model.train(training)
    loss_sum = correct = total = 0
    criterion = nn.CrossEntropyLoss()
    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise ValueError("La pérdida no es finita.")
            if training:
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * labels.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-batches", type=positive_int,
                        help="Limita lotes de train y val: solo para prueba técnica.")
    args = parser.parse_args()
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr debe ser positivo y finito")
    device = select_device(args.device)
    seed_everything(args.seed)
    train = load_dataset(args.data, "train")
    val = load_dataset(args.data, "val")
    train_loader = make_loader(train, args.batch_size, shuffle=True, seed=args.seed)
    val_loader = make_loader(val, args.batch_size)
    output = new_output(args.output)
    config = {key: str(value) if isinstance(value, Path) else value
              for key, value in vars(args).items()}
    config.update(device_used=str(device), smoke_test=args.max_batches is not None,
                  class_to_idx=train.class_to_idx,
                  architecture="conv32-conv64-fc128-dropout0.5-fc2", input_size=[3, 224, 224])
    write_json(output / "config.json", config)
    print(f"Dispositivo: {device}; train={len(train)}, val={len(val)}", flush=True)
    if config["smoke_test"]:
        print("PRUEBA TÉCNICA: las métricas no representan rendimiento final.", flush=True)
    model = CatDogCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    history, best_loss = [], float("inf")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"])
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = run_epoch(model, train_loader, device, optimizer, args.max_batches)
            val_loss, val_acc = run_epoch(model, val_loader, device, max_batches=args.max_batches)
            row = dict(epoch=epoch, train_loss=train_loss, train_accuracy=train_acc,
                       val_loss=val_loss, val_accuracy=val_acc)
            history.append(row)
            writer.writerow(row)
            handle.flush()
            print(f"Época {epoch}/{args.epochs}: train loss={train_loss:.4f} acc={train_acc:.4f}; "
                  f"val loss={val_loss:.4f} acc={val_acc:.4f}", flush=True)
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save({"model_state_dict": model.state_dict(), "epoch": epoch,
                            "val_loss": val_loss, "class_to_idx": train.class_to_idx,
                            "config": config}, output / "best_model.pt")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, metric in zip(axes, ("loss", "accuracy")):
        for split in ("train", "val"):
            ax.plot([r["epoch"] for r in history], [r[f"{split}_{metric}"] for r in history], label=split)
        ax.set(xlabel="Época", ylabel=metric)
        ax.legend()
    fig.suptitle("PRUEBA TÉCNICA" if config["smoke_test"] else "Entrenamiento CNN")
    fig.tight_layout()
    fig.savefig(output / "curves.png")
    plt.close(fig)
    print(f"Mejor modelo: {output / 'best_model.pt'}")


if __name__ == "__main__":
    main()
