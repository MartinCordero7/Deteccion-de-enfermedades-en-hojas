"""
Módulo de visualización de resultados YOLO para detección de enfermedades en hojas.
"""
import cv2
import numpy as np
from pathlib import Path
from src.utils import CLASS_NAMES_ES, CLASS_COLORS_BGR, SEVERITY_LEVEL


def draw_detections(image: np.ndarray, results, conf_threshold: float = 0.25) -> np.ndarray:
    """
    Dibuja los bounding boxes y etiquetas sobre la imagen.

    Args:
        image: Imagen BGR de OpenCV.
        results: Resultados de YOLO (ultralytics).
        conf_threshold: Umbral mínimo de confianza para mostrar detecciones.

    Returns:
        Imagen con anotaciones dibujadas.
    """
    annotated = image.copy()

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue

        for box in boxes:
            conf = float(box.conf[0])
            if conf < conf_threshold:
                continue

            cls_id  = int(box.cls[0])
            cls_name = result.names[cls_id]
            color   = CLASS_COLORS_BGR.get(cls_name, (0, 255, 0))
            label_es = CLASS_NAMES_ES.get(cls_name, cls_name)

            # Coordenadas del bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Dibujar rectángulo con borde grueso
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)

            # Preparar texto
            text = f"{label_es}  {conf:.0%}"
            font       = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 0.6
            thickness  = 1
            (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)

            # Fondo sólido para el texto
            pad = 6
            cv2.rectangle(
                annotated,
                (x1, y1 - th - 2 * pad),
                (x1 + tw + 2 * pad, y1),
                color,
                cv2.FILLED,
            )

            # Texto en blanco sobre el fondo de color
            cv2.putText(
                annotated,
                text,
                (x1 + pad, y1 - pad),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

    return annotated


def build_summary_text(results, conf_threshold: float = 0.25) -> str:
    """
    Construye un resumen de texto de las detecciones.

    Returns:
        Cadena con el conteo y severidad de cada clase detectada.
    """
    counts: dict[str, int] = {}

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            if float(box.conf[0]) < conf_threshold:
                continue
            cls_name = result.names[int(box.cls[0])]
            counts[cls_name] = counts.get(cls_name, 0) + 1

    if not counts:
        return "✅ No se detectaron enfermedades con la confianza establecida."

    lines = ["### 🔍 Enfermedades Detectadas\n"]
    for cls_name, count in counts.items():
        label = CLASS_NAMES_ES.get(cls_name, cls_name)
        severity = SEVERITY_LEVEL.get(cls_name, "❓ Desconocido")
        lines.append(f"- **{label}**: {count} detección(es)  |  {severity}")

    return "\n".join(lines)


def save_annotated_image(image_bgr: np.ndarray, output_path: str | Path) -> Path:
    """Guarda la imagen anotada en disco."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image_bgr)
    return output_path
