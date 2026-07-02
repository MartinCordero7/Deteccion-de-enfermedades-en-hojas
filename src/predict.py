"""
Script de inferencia/predicción con el modelo YOLO entrenado.

Uso:
    # Imagen individual
    python src/predict.py --source mi_hoja.jpg

    # Carpeta de imágenes
    python src/predict.py --source images/test/images/

    # Webcam en tiempo real
    python src/predict.py --source 0

    # Archivo de video
    python src/predict.py --source video.mp4 --save-video

    # Con modelo personalizado
    python src/predict.py --source imagen.jpg --model models/best.pt
"""
import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from src.utils import CLASS_NAMES_ES, RECOMMENDATIONS
from src.utils.visualizer import draw_detections, build_summary_text, save_annotated_image

# Modelo por defecto
DEFAULT_MODEL  = ROOT / "models" / "best.pt"
DEFAULT_CONF   = 0.40
DEFAULT_IOU    = 0.45
RESULTS_DIR    = ROOT / "runs" / "predict"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inferencia de detección de enfermedades en hojas"
    )
    parser.add_argument("--source", required=True,
                        help="Imagen, carpeta, video o índice de cámara (0)")
    parser.add_argument("--model",  default=str(DEFAULT_MODEL),
                        help=f"Ruta al modelo .pt (default: {DEFAULT_MODEL})")
    parser.add_argument("--conf",   type=float, default=DEFAULT_CONF,
                        help=f"Umbral de confianza (default: {DEFAULT_CONF})")
    parser.add_argument("--iou",    type=float, default=DEFAULT_IOU,
                        help=f"Umbral IoU para NMS (default: {DEFAULT_IOU})")
    parser.add_argument("--device", default="cpu",
                        help="Dispositivo: '0' (GPU) o 'cpu'")
    parser.add_argument("--save-video", action="store_true",
                        help="Guardar el video anotado")
    parser.add_argument("--no-show", action="store_true",
                        help="No mostrar ventana de previsualización")
    return parser.parse_args()


def predict_image(model: YOLO, source: str, conf: float, iou: float,
                  save: bool = True, show: bool = True) -> list[dict]:
    """
    Ejecuta inferencia en una imagen o carpeta de imágenes.
    Retorna lista de dicts con resultados por imagen.
    """
    results_data = []

    results = model.predict(
        source=source,
        conf=conf,
        iou=iou,
        verbose=False,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for result in results:
        img_bgr = result.orig_img.copy()
        annotated = draw_detections(img_bgr, [result], conf_threshold=conf)
        summary   = build_summary_text([result], conf_threshold=conf)

        # Nombre del archivo de salida
        src_path  = Path(result.path) if result.path else Path("frame")
        out_path  = RESULTS_DIR / f"pred_{src_path.name}"
        if save:
            save_annotated_image(annotated, out_path)

        # Mostrar ventana
        if show:
            cv2.imshow("Detección de Enfermedades en Hojas", annotated)
            cv2.waitKey(0)

        # Construir datos de resultados
        detections = []
        if result.boxes is not None:
            for box in result.boxes:
                if float(box.conf[0]) >= conf:
                    cls_name = result.names[int(box.cls[0])]
                    detections.append({
                        "clase":       cls_name,
                        "nombre_es":   CLASS_NAMES_ES.get(cls_name, cls_name),
                        "confianza":   float(box.conf[0]),
                        "bbox":        box.xyxy[0].tolist(),
                        "recomendacion": RECOMMENDATIONS.get(cls_name, ""),
                    })

        results_data.append({
            "imagen":      str(src_path.name),
            "detecciones": detections,
            "resumen":     summary,
            "guardado_en": str(out_path) if save else None,
        })

        print(f"\n{'='*55}")
        print(f"📸 Imagen: {src_path.name}")
        print(summary)
        for det in detections:
            print(f"   → {det['nombre_es']} ({det['confianza']:.1%})")
            print(f"     💡 {det['recomendacion']}")

    cv2.destroyAllWindows()
    return results_data


def predict_video(model: YOLO, source: str, conf: float, iou: float,
                  save_video: bool = False, show: bool = True):
    """
    Ejecuta inferencia en video o cámara en tiempo real.
    """
    source_val = int(source) if source.isdigit() else source
    cap = cv2.VideoCapture(source_val)

    if not cap.isOpened():
        print(f"❌ No se pudo abrir la fuente: {source}")
        sys.exit(1)

    # Configuración del escritor de video
    writer = None
    if save_video:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_name   = f"pred_video_{int(time.time())}.mp4"
        out_video  = str(RESULTS_DIR / out_name)
        fourcc     = cv2.VideoWriter_fourcc(*"mp4v")
        fps        = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer     = cv2.VideoWriter(out_video, fourcc, fps, (w, h))
        print(f"📹 Guardando video en: {out_video}")

    print("\n🎬 Iniciando detección en video. Presiona 'q' para salir.\n")
    frame_count = 0
    fps_timer   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=conf, iou=iou, verbose=False)
        annotated = draw_detections(frame, results, conf_threshold=conf)

        # FPS overlay
        frame_count += 1
        elapsed = time.time() - fps_timer
        if elapsed > 0:
            fps_val = frame_count / elapsed
            cv2.putText(
                annotated, f"FPS: {fps_val:.1f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2
            )

        if writer:
            writer.write(annotated)

        if show:
            cv2.imshow("Detección de Enfermedades — Video", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("\n✅ Video finalizado.")


def main():
    args = parse_args()

    # Verificar modelo
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"❌ Modelo no encontrado: {model_path}")
        print("   Entrena primero con: python src/train.py")
        sys.exit(1)

    print(f"\n🌿 Cargando modelo: {model_path}")
    model = YOLO(str(model_path))

    source = args.source

    # Detectar si la fuente es video/cámara
    is_video = (
        source.isdigit()
        or source.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".webm"))
    )

    if is_video:
        predict_video(
            model, source,
            conf=args.conf, iou=args.iou,
            save_video=args.save_video,
            show=not args.no_show,
        )
    else:
        predict_image(
            model, source,
            conf=args.conf, iou=args.iou,
            save=True,
            show=not args.no_show,
        )


if __name__ == "__main__":
    main()
