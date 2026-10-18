"""Orquesta preparación e inspección; --smoke prueba el flujo y --train entrena."""
import argparse
import math
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_step(command, cwd=ROOT):
    print(f"\n[ORQUESTADOR] Ejecutando: {shlex.join(str(arg) for arg in command)}", flush=True)
    result = subprocess.run([str(arg) for arg in command], cwd=cwd)
    if result.returncode != 0:
        print(f"[ERROR] El paso falló con código {result.returncode}.", flush=True)
        raise SystemExit(result.returncode)


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Debe ser mayor que cero.")
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--smoke", action="store_true", help="Una época con dos lotes de dos imágenes.")
    mode.add_argument("--train", action="store_true", help="Autoriza entrenamiento completo y evaluación en test.")
    parser.add_argument("--source", type=Path, default=ROOT / "dataset/train")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset/split")
    parser.add_argument("--output", type=Path, help="Directorio nuevo de resultados.")
    parser.add_argument("--epochs", type=positive_int, default=20)
    parser.add_argument("--batch-size", type=positive_int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--image", type=Path, help="Predicción opcional al terminar --train o --smoke.")
    args = parser.parse_args()
    if args.image and not (args.train or args.smoke):
        parser.error("--image requiere --train o --smoke; para un modelo existente usa src/06_predict.py")
    if args.lr <= 0 or not math.isfinite(args.lr):
        parser.error("--lr debe ser positivo y finito")
    if args.image and not args.image.is_file():
        parser.error(f"No existe la imagen: {args.image}")
    output = (args.output or ROOT / "artifacts" / datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")).resolve()
    data = args.data.resolve()
    source = args.source.resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error(f"La salida debe estar vacía: {output}")
    for protected in (data, source):
        if output == protected or protected in output.parents or output in protected.parents:
            parser.error("La salida de resultados debe estar separada del dataset.")
    if data.exists() and not data.is_dir():
        parser.error(f"La ruta de datos no es un directorio: {data}")
    if data.exists() and any(data.iterdir()) and not (data / "manifest.json").is_file():
        parser.error(f"Dataset incompleto: {data}. Revisa la preparación antes de continuar.")

    def script(name, *arguments):
        run_step([sys.executable, ROOT / "src" / name, *arguments])

    print("=== PIPELINE CNN: GATOS Y PERROS ===", flush=True)
    print(f"Resultados: {output}", flush=True)
    print("\n[1] Preparación del dataset", flush=True)
    if (data / "manifest.json").is_file():
        print("Dataset preparado: se reutiliza y se verificará en el paso 2.", flush=True)
    else:
        script("01_prepare_dataset.py", "--source", source, "--output", data, "--seed", args.seed)
    print("\n[2] Inspección del dataset", flush=True)
    script("02_inspect_dataset.py", "--data", data, "--output", output / "inspection/examples.png")
    print("\n[3] Inspección del modelo", flush=True)
    script("03_inspect_model.py")
    if not (args.train or args.smoke):
        print("\n=== INSPECCIONES COMPLETADAS; ENTRENAMIENTO NO INICIADO ===", flush=True)
        return
    training_output = output / "training"
    checkpoint = training_output / "best_model.pt"
    print("\n[4] Prueba técnica limitada" if args.smoke else "\n[4] Entrenamiento completo", flush=True)
    train_args = ["--data", data, "--output", training_output,
                  "--device", args.device, "--seed", args.seed, "--lr", args.lr,
                  "--epochs", 1 if args.smoke else args.epochs,
                  "--batch-size", 2 if args.smoke else args.batch_size]
    if args.smoke:
        train_args += ["--max-batches", 2]
    script("04_train.py", *train_args)
    print("\n[5] Evaluación", flush=True)
    eval_args = ["--data", data, "--checkpoint", checkpoint, "--output", output / "evaluation",
                 "--device", args.device, "--batch-size", 2 if args.smoke else args.batch_size]
    if args.smoke:
        eval_args += ["--max-batches", 2]
    script("05_evaluate.py", *eval_args)
    if args.image:
        print("\n[6] Predicción de una imagen", flush=True)
        script("06_predict.py", "--checkpoint", checkpoint, "--image", args.image.resolve(), "--device", args.device)
    print("\n=== PIPELINE COMPLETADO ===", flush=True)
    if args.smoke:
        print("Prueba técnica: las métricas no representan rendimiento final.", flush=True)
    print(f"Resultados: {output}", flush=True)


if __name__ == "__main__":
    main()
