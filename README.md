# 🌿 Detección de Enfermedades en Hojas con YOLO11

Sistema de visión por computadora para detectar y clasificar **6 enfermedades en hojas de cultivo**, 
basado en YOLOv8/YOLO11 (Ultralytics).

---

## 📋 Clases Detectadas

| ID | Nombre técnico | Nombre en español | Severidad |
|:--:|---|---|---|
| 0 | `bercak_daun` | Mancha Foliar | ⚠️ Moderado |
| 1 | `defisiensi_kalsium` | Deficiencia de Calcio | ⚠️ Moderado |
| 2 | `hangus_daun` | Quemadura de Hoja | 🔴 Grave |
| 3 | `hawar_daun` | Tizón Foliar | 🔴 Grave |
| 4 | `mosaik_vena_kuning` | Mosaico Vena Amarilla | ⚠️ Moderado |
| 5 | `virus_kuning_keriting` | Virus Rizado Amarillo | 🔴 Grave |

---

## 🗂️ Estructura del Proyecto

```
YOLO/
├── dataset/
│   └── data.yaml              ← Configuración del dataset
├── images/
│   ├── train/
│   │   ├── images/            ← Imágenes de entrenamiento
│   │   └── labels/            ← Etiquetas de entrenamiento
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── models/
│   └── best.pt                ← Mejor modelo entrenado
├── src/
│   ├── train.py               ← Script de entrenamiento
│   ├── predict.py             ← Inferencia imagen/video/webcam
│   ├── evaluate.py            ← Evaluación y métricas
│   └── utils/
│       ├── __init__.py        ← Constantes compartidas
│       └── visualizer.py      ← Visualización de resultados
├── app/
│   └── app.py                 ← Aplicación web (Gradio)
├── runs/                      ← Resultados de entrenamiento/evaluación
├── requirements.txt
└── README.md
```

---

## ⚡ Instalación

```bash
# 1. Clonar / entrar al proyecto
cd YOLO

# 2. Crear entorno virtual (recomendado)
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # Linux/Mac

# 3. Instalar dependencias
pip install -r requirements.txt
```

---

## 🏋️ Entrenamiento

```bash
# Con GPU (recomendado)
python src/train.py --model yolo11n.pt --epochs 50 --device 0

# Sin GPU (más lento)
python src/train.py --model yolo11n.pt --epochs 50 --device cpu

# Modelo más grande (más preciso)
python src/train.py --model yolo11s.pt --epochs 100 --batch 8

# Ver todas las opciones
python src/train.py --help
```

El mejor modelo se guarda en `models/best.pt` automáticamente.

---

## 🔍 Inferencia

```bash
# Imagen individual
python src/predict.py --source mi_hoja.jpg

# Carpeta de imágenes
python src/predict.py --source images/test/images/

# Webcam en tiempo real
python src/predict.py --source 0

# Video (guardar resultado)
python src/predict.py --source video.mp4 --save-video

# Con confianza personalizada
python src/predict.py --source imagen.jpg --conf 0.5
```

---

## 📊 Evaluación

```bash
python src/evaluate.py --split test
```

Genera:
- mAP50 y mAP50-95
- Precision / Recall por clase
- Matriz de confusión
- Curvas PR y F1
- Archivo `metrics.json`

---

## 🌐 Aplicación Web

```bash
python app/app.py

# Con enlace público (Gradio Share)
python app/app.py --share
```

Abre tu navegador en: **http://127.0.0.1:7860**

La aplicación incluye:
- 🔍 Pestaña de **Detección** con imagen y bounding boxes
- 📜 **Historial** de detecciones con exportación CSV
- 📚 **Catálogo** de enfermedades y recomendaciones
- ❓ **Ayuda** y documentación de uso

---

## 📁 Dataset

- **Fuente:** [Roboflow - Plant Disease TMYQ8](https://universe.roboflow.com/learning-eri4b/plant-disease-tmyq8/dataset/4)
- **Licencia:** Public Domain
- **Versión:** 4

---

## 🛠️ Tecnologías

| Componente | Tecnología |
|---|---|
| Detección | YOLO11 / YOLOv8 (Ultralytics) |
| Visión | OpenCV |
| Deep Learning | PyTorch |
| Interfaz | Gradio |
| Lenguaje | Python 3.10+ |
