# CNN: clasificación de gatos y perros

CNN entrenada desde cero con PyTorch. Todos los archivos Python están directamente
en `src/`.

## Instalación

```bash
uv sync
```

Los ejemplos usan `uv run python` para utilizar las dependencias del proyecto.
También puedes activar el entorno con `source .venv/bin/activate` y ejecutar `python`.

## Ejecutar mediante el orquestador

`pipeline.py` ejecuta los scripts en orden, muestra cada comando y se detiene si
un paso falla. Reutiliza el dataset cuando existe `manifest.json`; si no está
preparado, ejecuta el paso 01. No borra directorios ni resultados anteriores.

```bash
# Preparar si hace falta e inspeccionar: NO entrena
uv run python pipeline.py

# Prueba técnica: 1 época, solo 2 lotes de 2 imágenes en train, val y test
uv run python pipeline.py --smoke

# Para cuando decidas iniciar el entrenamiento completo
uv run python pipeline.py --train --epochs 20 --batch-size 32
```

Cada ejecución guarda resultados en `artifacts/run_FECHA_HORA/`: `inspection/`,
`training/` y `evaluation/` según el modo. Puedes elegir una carpeta vacía con
`--output artifacts/mi_experimento`. No se eliminan resultados para repetir una
corrida: elige otra carpeta. La opción `--seed` se usa para preparar un dataset
nuevo y entrenar; reutilizar un dataset existente conserva su división original.

Añade `--image /ruta/foto.jpg` a `--smoke` o `--train` para ejecutar también la
predicción al terminar. Para predecir después con un checkpoint existente:

```bash
uv run python src/06_predict.py --checkpoint artifacts/mi_experimento/training/best_model.pt --image /ruta/foto.jpg
```

## Orden de uso

Ejecuta los comandos desde la raíz del proyecto. Los pasos 01–03 se realizan para
preparar y revisar el experimento. Después de entrenar, puedes evaluar o predecir
sin volver a preparar el dataset ni entrenar otra vez.

### 01. Preparar imágenes (ya realizado)

```bash
uv run python src/01_prepare_dataset.py
```

Conserva los originales en `dataset/train/` y genera:

```text
dataset/split/
├── train/  # 17 500 imágenes
│   ├── cat/
│   └── dog/
├── val/    # 3 750 imágenes; cat/ y dog/
├── test/   # 3 750 imágenes; cat/ y dog/
└── manifest.json
```

La división 70/15/15 se realiza por clase con semilla 42. Cada conjunto contiene
la misma cantidad de gatos y perros. La salida debe estar vacía: no ejecutes de
nuevo este paso sobre el dataset ya generado. Puedes usar `--output otra/ruta`.
Todas las copias son RGB de 224 × 224. El redimensionado directo puede cambiar
la relación de aspecto. El manifiesto registra asignaciones y tamaños originales.

### 02. Inspeccionar el dataset

```bash
uv run python src/02_inspect_dataset.py
```

Comprueba conteos, clases, tamaños RGB 224 × 224 y ausencia de nombres repetidos
entre conjuntos. Guarda ejemplos de train en `artifacts/inspection/examples.png`.
Para repetir sin sobrescribir, usa `--output artifacts/inspection/examples_2.png`.

### 03. Inspeccionar el modelo

```bash
uv run python src/03_inspect_model.py
```

Muestra las capas, dimensiones y parámetros. No entrena ni guarda pesos.

| Operación | Salida por imagen (canales × alto × ancho) |
|---|---|
| Entrada RGB | 3 × 224 × 224 |
| Conv 5 × 5, 32 filtros, sin padding + ReLU | 32 × 220 × 220 |
| MaxPooling 2 × 2 | 32 × 110 × 110 |
| Conv 5 × 5, 64 filtros, sin padding + ReLU | 64 × 106 × 106 |
| MaxPooling 2 × 2 | 64 × 53 × 53 |
| Flatten | 179 776 |
| Densa + ReLU | 128 |
| Dropout 0.5 | 128 |
| Salida | 2 logits |

Se conserva la arquitectura de referencia adaptada a dos clases. La primera capa
densa tiene aproximadamente 23 millones de pesos debido a la entrada de 224 × 224.

### 04. Entrenar

```bash
uv run python src/04_train.py
```

Usa train para ajustar pesos y val para elegir el checkpoint con menor pérdida.
No carga test. Valores predeterminados: Adam, learning rate 0.001, batch 32,
20 épocas y semilla 42. Las imágenes se convierten a tensores en [0, 1], sin
normalización adicional ni aumentación. Solo train se mezcla.
Se usa `CrossEntropyLoss` sobre logits; no se aplica softmax durante entrenamiento.

Detecta CUDA automáticamente y usa CPU cuando no está disponible. Puedes elegir
`--device cpu` o `--device cuda`. La semilla y los algoritmos deterministas fijan
la aleatoriedad dentro del mismo entorno; no garantizan igualdad entre hardware
o versiones diferentes.

```bash
uv run python src/04_train.py --epochs 20 --batch-size 16 --lr 0.001 --seed 42 --output artifacts/experimento_2
```

La salida predeterminada es `artifacts/default/` y debe estar vacía. Contiene:

- `best_model.pt`: pesos, clases, configuración y época del mejor modelo.
- `config.json`: parámetros del experimento y dispositivo utilizado.
- `history.csv`: pérdida y accuracy de train y val por época.
- `curves.png`: curvas de entrenamiento y validación.

### 05. Evaluar con test

```bash
uv run python src/05_evaluate.py
```

Carga `artifacts/default/best_model.pt` y evalúa las 3 750 imágenes de test.
Guarda `metrics.json` (accuracy, precision, recall, F1 por clase y promedios) y
`confusion_matrix.png` en `artifacts/default/evaluation/`.
Las filas de la matriz son clases reales y las columnas predicciones: cat, dog.
Usa test para la evaluación final, después de seleccionar el modelo con val.

Para otro experimento o repetir la evaluación:

```bash
uv run python src/05_evaluate.py --checkpoint artifacts/experimento_2/best_model.pt --output artifacts/experimento_2/evaluation_2
```

### 06. Predecir una imagen nueva

```bash
uv run python src/06_predict.py --image /ruta/a/foto.jpg
```

Carga el checkpoint, convierte la imagen a RGB 224 × 224 y muestra la clase
predicha junto con las probabilidades softmax de `cat` y `dog`.
Acepta `--checkpoint artifacts/experimento_2/best_model.pt` y `--device cpu`.
Las probabilidades no son una medida calibrada de certeza.

## Prueba corta del flujo

Comprueba que los scripts funcionan sin realizar el entrenamiento completo:

```bash
uv run python src/04_train.py --epochs 1 --batch-size 2 --max-batches 2 --output artifacts/smoke
uv run python src/05_evaluate.py --checkpoint artifacts/smoke/best_model.pt --batch-size 2 --max-batches 2
uv run python src/06_predict.py --checkpoint artifacts/smoke/best_model.pt --image dataset/split/test/cat/cat.9831.jpg
```

Para el último comando puedes sustituir la ruta por cualquier imagen existente. `--max-batches` limita los lotes de train y
val (o test en evaluación). Los checkpoints, curvas y métricas quedan marcados
como **prueba técnica**; sus resultados no representan rendimiento final.
Usa una salida diferente si repites una prueba.

## Código compartido

- `src/model.py` define `CatDogCNN`, selección de dispositivo y carga del checkpoint.
- `src/data.py` comparte rutas, carga `ImageFolder`, transformaciones y semillas.

Los scripts numerados importan estos módulos para usar la misma arquitectura y
preprocesamiento. `cat=0`, `dog=1`. Todos los scripts de datos aceptan `--data`
para otra división preparada; `--help` muestra los argumentos de cada comando.

Referencias: [ImageFolder](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)
y [guardado de modelos de PyTorch](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html).

Dataset original: Se uso el dataset de [Kaggle Dogs vs. Cats](https://www.kaggle.com/competitions/dogs-vs-cats/data), solo de descargo la parte train/ la cual fue dividida en train/ val/ test/ (70/15/15) con una semilla 42.
