"""
Evaluación del modelo YOLO entrenado.
Genera métricas, matriz de confusión y curvas de rendimiento.

Uso:
    python src/evaluate.py
    python src/evaluate.py --model models/best.pt --split test
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

DEFAULT_MODEL = ROOT / "models" / "best.pt"
DATA_YAML     = ROOT / "dataset" / "data.yaml"
EVAL_DIR      = ROOT / "runs" / "evaluate"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluar el modelo YOLO en el conjunto de prueba"
    )
    parser.add_argument("--model",  default=str(DEFAULT_MODEL))
    parser.add_argument("--data",   default=str(DATA_YAML))
    parser.add_argument("--split",  default="test",
                        choices=["train", "val", "test"],
                        help="Conjunto a evaluar")
    parser.add_argument("--conf",   type=float, default=0.001,
                        help="Umbral de confianza para evaluación (bajo para mAP)")
    parser.add_argument("--iou",    type=float, default=0.6)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def evaluate(args):
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        sys.exit(1)

    print("=" * 60)
    print("  📊 Evaluación del Modelo — Enfermedades en Hojas")
    print("=" * 60)
    print(f"  Modelo  : {model_path}")
    print(f"  Dataset : {args.data}")
    print(f"  Split   : {args.split}")
    print("=" * 60)

    model = YOLO(str(model_path))

    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    metrics = model.val(
        data    = args.data,
        split   = args.split,
        conf    = args.conf,
        iou     = args.iou,
        device  = args.device,
        project = str(EVAL_DIR),
        name    = f"eval_{args.split}",
        plots   = True,
        verbose = True,
    )

    # ─── Mostrar resumen de métricas ────────────────────────────────
    print("\n" + "=" * 60)
    print("  📈 RESULTADOS")
    print("=" * 60)
    print(f"  mAP50       : {metrics.box.map50:.4f}")
    print(f"  mAP50-95    : {metrics.box.map:.4f}")
    print(f"  Precision   : {metrics.box.mp:.4f}")
    print(f"  Recall      : {metrics.box.mr:.4f}")
    print("=" * 60)

    # Métricas por clase
    print("\n  Métricas por clase:")
    print(f"  {'Clase':<30} {'AP50':>8} {'Precision':>10} {'Recall':>8}")
    print("  " + "-" * 60)
    names = model.names
    for i, (ap50, p, r) in enumerate(zip(
        metrics.box.ap50,
        metrics.box.p,
        metrics.box.r,
    )):
        cls_name = names.get(i, f"class_{i}")
        print(f"  {cls_name:<30} {ap50:>8.4f} {p:>10.4f} {r:>8.4f}")

    # Guardar métricas en JSON
    metrics_dict = {
        "mAP50":     float(metrics.box.map50),
        "mAP50_95":  float(metrics.box.map),
        "precision": float(metrics.box.mp),
        "recall":    float(metrics.box.mr),
        "per_class": {
            names.get(i, f"class_{i}"): {
                "ap50":      float(ap50),
                "precision": float(p),
                "recall":    float(r),
            }
            for i, (ap50, p, r) in enumerate(zip(
                metrics.box.ap50,
                metrics.box.p,
                metrics.box.r,
            ))
        },
    }

    json_out = EVAL_DIR / f"eval_{args.split}" / "metrics.json"
    json_out.parent.mkdir(parents=True, exist_ok=True)
    json_out.write_text(json.dumps(metrics_dict, indent=2, ensure_ascii=False))
    print(f"\n✅ Métricas guardadas en: {json_out}")
    print(f"📊 Gráficas guardadas en: {EVAL_DIR / f'eval_{args.split}'}")

    return metrics_dict


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
