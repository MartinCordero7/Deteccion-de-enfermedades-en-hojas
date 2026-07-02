"""
Script de entrenamiento YOLOv8/YOLO11 para detección de enfermedades en hojas.

Uso:
    python src/train.py
    python src/train.py --model yolo11n.pt --epochs 100 --batch 16
"""
import argparse
import sys
from pathlib import Path

# Asegurar que el directorio raíz del proyecto esté en sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO


# ─────────────────────────────────────────────
# Configuración por defecto
# ─────────────────────────────────────────────
DATA_YAML    = ROOT / "dataset" / "data.yaml"
MODELS_DIR   = ROOT / "models"
RUNS_DIR     = ROOT / "runs"

DEFAULT_MODEL  = "yolo11n.pt"   # nano → rápido; cambiar a yolo11s/m para más precisión
DEFAULT_EPOCHS = 50
DEFAULT_BATCH  = 16
DEFAULT_IMGSZ  = 640
DEFAULT_DEVICE = "0"            # "0" = GPU cuda:0 | "cpu" = sin GPU


def parse_args():
    parser = argparse.ArgumentParser(
        description="Entrenar YOLOv8/YOLO11 para detección de enfermedades en hojas"
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"Modelo base (default: {DEFAULT_MODEL}). "
             "Opciones: yolo11n.pt, yolo11s.pt, yolo11m.pt, yolov8n.pt, yolov8s.pt"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch",  type=int, default=DEFAULT_BATCH)
    parser.add_argument("--imgsz",  type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--device", default=DEFAULT_DEVICE,
                        help="'0' para GPU, 'cpu' para CPU")
    parser.add_argument("--name",   default="leaf_disease",
                        help="Nombre del experimento (subcarpeta en runs/)")
    parser.add_argument("--resume", action="store_true",
                        help="Reanudar entrenamiento desde el último checkpoint")
    return parser.parse_args()


def train(args):
    print("=" * 60)
    print("  🌿 Detección de Enfermedades en Hojas — Entrenamiento")
    print("=" * 60)
    print(f"  Modelo    : {args.model}")
    print(f"  Épocas    : {args.epochs}")
    print(f"  Batch     : {args.batch}")
    print(f"  Img size  : {args.imgsz}")
    print(f"  Dispositivo: {args.device}")
    print(f"  Dataset   : {DATA_YAML}")
    print("=" * 60)

    # Verificar que data.yaml existe
    if not DATA_YAML.exists():
        print(f"\n❌ ERROR: No se encontró {DATA_YAML}")
        sys.exit(1)

    # Cargar modelo (descarga automática si no existe)
    model = YOLO(args.model)

    # Entrenar
    results = model.train(
        data    = str(DATA_YAML),
        epochs  = args.epochs,
        batch   = args.batch,
        imgsz   = args.imgsz,
        device  = args.device,
        project = str(RUNS_DIR / "train"),
        name    = args.name,
        resume  = args.resume,

        # Hiperparámetros optimizados para enfermedades en hojas
        lr0        = 0.01,
        lrf        = 0.01,
        momentum   = 0.937,
        weight_decay = 0.0005,
        warmup_epochs = 3.0,
        patience   = 20,           # Early stopping

        # Augmentación de datos
        hsv_h      = 0.015,
        hsv_s      = 0.7,
        hsv_v      = 0.4,
        flipud     = 0.2,
        fliplr     = 0.5,
        mosaic     = 1.0,
        mixup      = 0.15,

        # Guardado
        save       = True,
        save_period = 10,          # Guardar checkpoint cada N épocas
        plots      = True,
        verbose    = True,
    )

    # Copiar el mejor modelo a models/
    best_pt = Path(results.save_dir) / "weights" / "best.pt"
    if best_pt.exists():
        MODELS_DIR.mkdir(exist_ok=True)
        dest = MODELS_DIR / "best.pt"
        import shutil
        shutil.copy(best_pt, dest)
        print(f"\n✅ Mejor modelo guardado en: {dest}")
    else:
        print(f"\n⚠️  No se encontró best.pt en {results.save_dir}")

    print(f"\n📊 Resultados completos en: {results.save_dir}")
    return results


if __name__ == "__main__":
    args = parse_args()
    train(args)
