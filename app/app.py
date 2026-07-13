"""
Aplicación Web para Detección de Enfermedades en Hojas
Interfaz Gradio moderna con tema oscuro y panel de resultados completo.

Uso:
    python app/app.py
    python app/app.py --model models/best.pt --port 7860
"""
import argparse
import sys
import json
import csv
import io
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import gradio as gr
from ultralytics import YOLO
from src.utils import CLASS_NAMES_ES, SEVERITY_LEVEL, RECOMMENDATIONS
from src.utils.visualizer import draw_detections, build_summary_text
import src.db.mongo as mongo_db
import src.ai_analyst as ai_analyst

# ─────────────────────────────────────────────────────────────
# Configuración
# ─────────────────────────────────────────────────────────────
DEFAULT_MODEL = ROOT / "models" / "best.pt"
_model: YOLO | None = None        # Modelo cargado en memoria


def load_model(model_path: str) -> YOLO:
    global _model
    _model = YOLO(model_path)
    return _model


def get_model() -> YOLO:
    if _model is None:
        raise RuntimeError("Modelo no cargado.")
    return _model


# ─────────────────────────────────────────────────────────────
# Funciones de inferencia
# ─────────────────────────────────────────────────────────────
def predict_from_pil(
    pil_image: Image.Image,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[Image.Image, str, str, str]:
    """
    Realiza la inferencia sobre una imagen PIL.

    Returns:
        (imagen_anotada, markdown_detecciones, markdown_recomendaciones, json_raw)
    """
    if pil_image is None:
        return None, "⬆️ Sube una imagen para analizar.", "", "{}"

    model = get_model()

    # PIL → BGR (OpenCV)
    img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Inferencia
    results = model.predict(
        img_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )

    # Imagen anotada
    annotated_bgr = draw_detections(img_bgr, results, conf_threshold)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    # Resumen de detecciones
    summary_md = build_summary_text(results, conf_threshold)

    # Recomendaciones y detalle
    detections_raw = []
    recom_lines    = ["### 💡 Recomendaciones de Acción\n"]
    seen_classes   = set()

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            if float(box.conf[0]) < conf_threshold:
                continue
            cls_name = result.names[int(box.cls[0])]
            conf_val = float(box.conf[0])
            label_es = CLASS_NAMES_ES.get(cls_name, cls_name)
            severity = SEVERITY_LEVEL.get(cls_name, "")
            recom    = RECOMMENDATIONS.get(cls_name, "")
            bbox     = box.xyxy[0].tolist()

            detections_raw.append({
                "clase":       cls_name,
                "nombre_es":   label_es,
                "confianza":   round(conf_val, 4),
                "severidad":   severity,
                "bbox":        [round(v, 1) for v in bbox],
            })

            if cls_name not in seen_classes:
                seen_classes.add(cls_name)
                recom_lines.append(
                    f"**{label_es}** ({severity})\n> {recom}\n"
                )

    if not seen_classes:
        recom_lines = ["### ✅ Sin Enfermedades Detectadas\n",
                       "La planta parece estar sana con el umbral de confianza establecido."]

    recom_md = "\n".join(recom_lines)

    # Guardar en historial
    mongo_db.save_detection(
        img_original_rgb=np.array(pil_image),
        img_annotated_rgb=annotated_rgb,
        detections_data=detections_raw
    )

    return annotated_pil, summary_md, recom_md, json.dumps(detections_raw, indent=2, ensure_ascii=False)


def predict_multi_images(
    files,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[list[Image.Image], str, str, str]:
    """
    Realiza la inferencia sobre múltiples imágenes (hasta 4).
    """
    if not files:
        return [], "⬆️ Sube al menos una imagen para analizar.", "", "{}"

    model = get_model()
    
    # Procesar máximo 4 archivos
    files = files[:4]
    
    all_annotated_pil = []
    combined_summary = []
    combined_recom = []
    all_detections_raw = []
    
    for i, file_obj in enumerate(files):
        img_path = file_obj.name if hasattr(file_obj, 'name') else str(file_obj)
        
        try:
            pil_image = Image.open(img_path).convert("RGB")
        except Exception as e:
            combined_summary.append(f"### Imagen {i+1}\nError al cargar la imagen.")
            continue
            
        img_bgr = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        
        results = model.predict(
            img_bgr,
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False,
        )
        
        annotated_bgr = draw_detections(img_bgr, results, conf_threshold)
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        all_annotated_pil.append(Image.fromarray(annotated_rgb))
        
        # Resumen
        summary_md = build_summary_text(results, conf_threshold)
        combined_summary.append(f"### Imagen {i+1}\n" + summary_md)
        
        # Recomendaciones
        detections_raw = []
        recom_lines = [f"### 💡 Recomendaciones - Imagen {i+1}\n"]
        seen_classes = set()
        
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                if float(box.conf[0]) < conf_threshold:
                    continue
                cls_name = result.names[int(box.cls[0])]
                conf_val = float(box.conf[0])
                label_es = CLASS_NAMES_ES.get(cls_name, cls_name)
                severity = SEVERITY_LEVEL.get(cls_name, "")
                recom    = RECOMMENDATIONS.get(cls_name, "")
                bbox     = box.xyxy[0].tolist()

                detections_raw.append({
                    "imagen":      i+1,
                    "clase":       cls_name,
                    "nombre_es":   label_es,
                    "confianza":   round(conf_val, 4),
                    "severidad":   severity,
                    "bbox":        [round(v, 1) for v in bbox],
                })

                if cls_name not in seen_classes:
                    seen_classes.add(cls_name)
                    recom_lines.append(
                        f"**{label_es}** ({severity})\n> {recom}\n"
                    )
                    
        if not seen_classes:
            recom_lines.append("✅ **Sin Enfermedades Detectadas**\nLa planta parece estar sana.\n")
            
        combined_recom.append("\n".join(recom_lines))
        all_detections_raw.extend(detections_raw)
        
        # Guardar en historial
        mongo_db.save_detection(
            img_original_rgb=np.array(pil_image),
            img_annotated_rgb=annotated_rgb,
            detections_data=detections_raw
        )
        
    return all_annotated_pil, "\n\n---\n\n".join(combined_summary), "\n\n---\n\n".join(combined_recom), json.dumps(all_detections_raw, indent=2, ensure_ascii=False)


def predict_live(
    frame: np.ndarray,
    conf_threshold: float,
    iou_threshold: float,
) -> np.ndarray:
    """Inferencia rápida para tiempo real (webcam)."""
    if frame is None:
        return None
    model = get_model()
    # Gradio proporciona frame en RGB (numpy array) cuando streaming=True
    img_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    results = model.predict(
        img_bgr,
        conf=conf_threshold,
        iou=iou_threshold,
        verbose=False,
    )
    annotated_bgr = draw_detections(img_bgr, results, conf_threshold)
    return cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)


def export_history_csv() -> str:
    """Exporta el historial de detecciones como CSV."""
    import io
    import csv
    history_data = mongo_db.get_recent_history(limit=1000)
    if not history_data:
        return "No hay historial para exportar."

    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["Timestamp", "Num Detecciones", "Clases", "Nombre ES", "Confianza", "Severidad"])

    for entry in history_data:
        if entry.get("resultados"):
            for det in entry["resultados"]:
                writer.writerow([
                    entry["timestamp"],
                    entry["detecciones"],
                    det["clase"],
                    det["nombre_es"],
                    det["confianza"],
                    det["severidad"],
                ])
        else:
            writer.writerow([entry["timestamp"], 0, "—", "Sin enfermedades", "", ""])

    return output.getvalue()


def get_history_table(start_date=None, end_date=None) -> list[list]:
    """Construye la tabla del historial para Gradio Dataframe."""
    rows = []
    history_data = mongo_db.get_recent_history(limit=50, start_date=start_date, end_date=end_date)
    for entry in history_data:
        clases_str = ", ".join(
            CLASS_NAMES_ES.get(c, c) for c in entry.get("clases", [])
        ) or "✅ Sana"
        rows.append([
            entry["timestamp"],
            entry["detecciones"],
            clases_str,
        ])
    return rows


def restore_session(evt: gr.SelectData, history_data_df):
    """
    Restaura una sesión haciendo clic en una fila de la tabla de historial.
    Retorna los componentes actualizados de la pestaña Detección.
    """
    if evt.index[0] >= len(history_data_df):
        return [gr.update()]*12

    # history_data_df is a pandas DataFrame that Gradio sends automatically
    timestamp = history_data_df.iloc[evt.index[0], 0]
    doc = mongo_db.get_detection_by_timestamp(timestamp)
    if not doc:
        return [gr.update()]*12
    
    img_orig = mongo_db.decode_image_from_base64(doc.get("img_original", ""))
    img_annotated = mongo_db.decode_image_from_base64(doc.get("img_annotated", ""))
    
    import json
    
    results_raw = doc.get("resultados", [])
    
    if img_annotated is None:
        img_annotated = img_orig

    # Reconstruir summary y recomendaciones
    recom_lines = ["### 💡 Recomendaciones de Acción\n"]
    seen_classes = set()
    for det in results_raw:
        seen_classes.add(det["clase"])
        recom_lines.append(f"**{det['nombre_es']}** ({det['severidad']})\n> {RECOMMENDATIONS.get(det['clase'], '')}\n")
    
    if not seen_classes:
        recom_lines = ["### ✅ Sin Enfermedades Detectadas\nLa planta parece estar sana."]
    
    recom_md = "\n".join(recom_lines)
    
    num_det = len(results_raw)
    summary_md = f"### 📊 Resumen de Detección\n* **Objetos detectados:** {num_det}\n"
    if num_det == 0:
         summary_md += "* **Estado:** ✅ Planta sana\n"
    
    json_out = json.dumps(results_raw, indent=2, ensure_ascii=False)
    
    return [
        img_orig,                 # input_image
        img_annotated,            # output_image
        summary_md,               # summary_md
        recom_md,                 # recom_md
        json_out,                 # json_out
        gr.update(visible=False), # group_vivo
        gr.update(visible=True),  # group_foto
        gr.update(visible=False), # group_multi
        gr.update(visible=False), # group_history
        gr.update(visible=False), # group_classes
        gr.update(visible=False), # group_help
        gr.update(value="Detección por Foto"), # mode_selector
    ]


# ─────────────────────────────────────────────────────────────
# CSS personalizado
# ─────────────────────────────────────────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&display=swap');

/* ── Variables de color Premium ─────────────────────── */
:root {
    --green-primary: #10b981;
    --green-dark: #059669;
    --green-glow: rgba(16, 185, 129, 0.4);
    --bg-dark: #09090b;
    --bg-card: rgba(255, 255, 255, 0.03);
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
    --border: rgba(255, 255, 255, 0.08);
    --radius: 16px;
}

/* ── Layout base ────────────────────────────────────── */
body, .gradio-container {
    background: radial-gradient(circle at 50% -20%, #0d281a, var(--bg-dark) 60%) !important;
    background-attachment: fixed !important;
    font-family: 'Outfit', sans-serif !important;
    color: var(--text-primary) !important;
}

/* ── Header Premium ─────────────────────────────────── */
#header-html {
    background: rgba(10, 15, 20, 0.6);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(16, 185, 129, 0.2);
    border-top: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 20px;
    padding: 36px 40px;
    margin-bottom: 30px;
    box-shadow: 0 15px 35px rgba(0,0,0,0.4), inset 0 0 40px rgba(16, 185, 129, 0.05);
    position: relative;
    overflow: hidden;
}
#header-html::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
    transform: skewX(-20deg);
    animation: shine 6s infinite;
}
@keyframes shine {
    0% { left: -100%; }
    20% { left: 200%; }
    100% { left: 200%; }
}
#header-html h1 {
    font-family: 'Outfit', sans-serif;
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(135deg, #34d399, #10b981, #059669);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 10px 0;
    letter-spacing: -0.5px;
}
#header-html p {
    color: var(--text-muted);
    font-size: 1.05rem;
    font-weight: 300;
    margin: 0;
}

/* ── Tarjetas Glassmorphism ─────────────────────────── */
.panel-card {
    background: var(--bg-card) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 24px !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
}
.panel-card:hover {
    transform: translateY(-3px);
    border-color: rgba(16, 185, 129, 0.3) !important;
    box-shadow: 0 8px 30px rgba(16, 185, 129, 0.15) !important;
}

/* ── Botón principal ─────────────────────────────────── */
#btn-detect {
    background: linear-gradient(135deg, #10b981, #059669) !important;
    color: #fff !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    padding: 14px 32px !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 8px 25px var(--green-glow), inset 0 2px 4px rgba(255,255,255,0.2) !important;
    letter-spacing: 0.5px;
}
#btn-detect:hover { 
    transform: translateY(-2px) scale(1.02);
    box-shadow: 0 12px 35px rgba(16,185,129,0.5), inset 0 2px 4px rgba(255,255,255,0.3) !important;
}
#btn-detect:active {
    transform: translateY(1px);
}

/* ── Sliders Premium ─────────────────────────────────── */
input[type=range]::-webkit-slider-thumb {
    background: #fff !important;
    border: 2px solid var(--green-primary) !important;
    box-shadow: 0 0 10px var(--green-glow) !important;
}
input[type=range]::-webkit-slider-runnable-track {
    background: rgba(255,255,255,0.1) !important;
}

/* ── Tabs Modernos ───────────────────────────────────── */
.tab-nav {
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    margin-bottom: 20px !important;
}
.tab-nav button {
    color: var(--text-muted) !important;
    border: none !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    padding: 12px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s ease !important;
}
.tab-nav button:hover {
    color: #fff !important;
    background: rgba(255,255,255,0.03) !important;
}
.tab-nav button.selected {
    color: var(--green-primary) !important;
    background: rgba(16,185,129,0.05) !important;
    border-bottom: 3px solid var(--green-primary) !important;
}

/* ── Badges de clase ─────────────────────────────────── */
.badge {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 99px;
    font-size: 0.8rem;
    font-weight: 600;
    margin: 2px;
    backdrop-filter: blur(4px);
}
.badge-grave   { background: rgba(239,68,68,0.15); color: #fca5a5; border: 1px solid rgba(248,113,113,0.3); box-shadow: 0 0 10px rgba(239,68,68,0.2); }
.badge-moderad { background: rgba(234,179,8,0.15);  color: #fde047; border: 1px solid rgba(251,191,36,0.3); box-shadow: 0 0 10px rgba(234,179,8,0.2); }
.badge-sano    { background: rgba(34,197,94,0.15);  color: #86efac; border: 1px solid rgba(74,222,128,0.3); box-shadow: 0 0 10px rgba(34,197,94,0.2); }

/* ── Ajustes de Gradio Internos ──────────────────────── */
.gradio-container .prose h3, .gradio-container .prose h4 {
    color: #fff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
}

/* ── Sobrescribir fondos por defecto de Gradio ───────── */
:root, .dark, body.dark, .gradio-container {
    --background-fill-secondary: #1f222b !important;
    --block-background-fill: #1f222b !important;
    --panel-background-fill: #1f222b !important;
    --input-background-fill: #1f222b !important;
}

/* ── Custom Nav ──────────────────────────────────────── */
#custom-nav {
    margin-bottom: 20px !important;
    align-items: center;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
    padding-bottom: 5px !important;
}
#nav-links {
    justify-content: flex-end;
    align-items: center;
    gap: 5px;
}
.nav-btn {
    color: var(--text-muted) !important;
    border: none !important;
    font-family: 'Outfit', sans-serif !important;
    font-size: 1.05rem !important;
    font-weight: 500 !important;
    padding: 12px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    background: transparent !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: none !important;
}
.nav-btn:hover {
    color: #fff !important;
    background: rgba(255,255,255,0.03) !important;
}
.dropdown-elegant {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    box-shadow: 0 4px 20px rgba(0,0,0,0.2) !important;
}
.dropdown-elegant > div, .dropdown-elegant .wrap, .dropdown-elegant .container {
    background: transparent !important;
    border: none !important;
}
"""

# ─────────────────────────────────────────────────────────────
# Interfaz Gradio
# ─────────────────────────────────────────────────────────────
def build_ui(model_path: str) -> gr.Blocks:
    load_model(model_path)
    print(f"✅ Modelo cargado: {model_path}")

    with gr.Blocks(
        css=CUSTOM_CSS,
        title="🌿 Detección de Enfermedades en Hojas",
        theme=gr.themes.Base(
            primary_hue="green",
            neutral_hue="slate",
        ),
    ) as demo:

        # ── Header ──────────────────────────────────────────────
        gr.HTML(
            """
            <div id="header-html">
              <h1>🌿 Detección de Enfermedades en Hojas</h1>
              <p>Sistema de inteligencia artificial para monitoreo fitosanitario
                 · Powered by YOLO11 + OpenCV · Agricultura de Precisión</p>
            </div>
            """,
            elem_id="header-section",
        )

        # ── Contenedor Principal ────────────────────────────────
        with gr.Column():

            # ════════════════════════════════════════════════════
            # NAVEGACIÓN SUPERIOR
            # ════════════════════════════════════════════════════
            with gr.Row(elem_id="custom-nav"):
                with gr.Column(scale=3, min_width=250):
                    mode_selector = gr.Dropdown(
                        choices=["Detección por Foto", "Detección en Vivo", "Detección Multi-Imagen"],
                        value="Detección por Foto",
                        show_label=False,
                        interactive=True,
                        filterable=False,
                        elem_classes=["dropdown-elegant"],
                    )
                with gr.Column(scale=7):
                    with gr.Row(elem_id="nav-links"):
                        btn_historial = gr.Button("📜 Historial", elem_classes=["nav-btn"])
                        btn_clases = gr.Button("📚 Clases del Modelo", elem_classes=["nav-btn"])
                        btn_ayuda = gr.Button("❓ Ayuda", elem_classes=["nav-btn"])
            
            # ════════════════════════════════════════════════════
            # PANELES DE CONTENIDO
            # ════════════════════════════════════════════════════
            with gr.Column():
                
                # ── Grupo 1: En Vivo ──
                with gr.Group(visible=False) as group_vivo:
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5):
                            gr.Markdown("#### 📷 Cámara")
                            live_image = gr.Image(
                                type="numpy",
                                label="Transmisión",
                                sources=["webcam"],
                                streaming=True,
                                elem_classes=["panel-card"],
                            )
                            with gr.Group(elem_classes=["panel-card"]):
                                with gr.Row():
                                    live_conf = gr.Slider(minimum=0.1, maximum=0.95, value=0.20, step=0.05, label="🎯 Confianza mínima")
                                    live_iou = gr.Slider(minimum=0.1, maximum=0.95, value=0.45, step=0.05, label="📐 Umbral IoU")
                        
                        with gr.Column(scale=5):
                            gr.Markdown("#### 🖼️ Detección en Tiempo Real")
                            live_output = gr.Image(
                                type="numpy",
                                label="Salida",
                                interactive=False,
                                elem_classes=["panel-card"],
                            )
                    
                    # Evento de streaming
                    live_image.stream(
                        fn=predict_live,
                        inputs=[live_image, live_conf, live_iou],
                        outputs=[live_output],
                    )

                # ── Grupo 2: Por Foto ──
                with gr.Group(visible=True) as group_foto:
                    # ── Fila 1: Imágenes (Entrada y Salida) ─────────
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5):
                            gr.Markdown("#### 📷 Imagen de la hoja")
                            input_image = gr.Image(
                                type="pil",
                                label="Cargar imagen",
                                sources=["upload", "clipboard", "webcam"],
                                height=360,
                                elem_classes=["panel-card"],
                            )
                        with gr.Column(scale=5):
                            gr.Markdown("#### 🖼️ Resultado de Detección")
                            output_image = gr.Image(
                                type="pil",
                                label="Imagen anotada",
                                height=360,
                                interactive=False,
                                elem_classes=["panel-card"],
                            )

                    # ── Fila 2: Controles ────────────────────────────
                    with gr.Row():
                        with gr.Column(scale=10):
                            with gr.Group(elem_classes=["panel-card"]):
                                with gr.Row():
                                    conf_slider = gr.Slider(
                                        minimum=0.10, maximum=0.95, value=0.40, step=0.05,
                                        label="🎯 Confianza mínima",
                                        info="Mayor valor → menos detecciones pero más precisas",
                                    )
                                    iou_slider = gr.Slider(
                                        minimum=0.10, maximum=0.95, value=0.45, step=0.05,
                                        label="📐 Umbral IoU (NMS)",
                                        info="Controla superposición de bounding boxes",
                                    )
                                btn_detect = gr.Button(
                                    "🔬 Analizar Hoja",
                                    variant="primary",
                                    elem_id="btn-detect",
                                )
                                gr.Examples(
                                    examples=[],
                                    inputs=input_image,
                                    label="Ejemplos rápidos",
                                )

                    # ── Fila 3: Resultados (Resumen + Recomendaciones) ────────
                    with gr.Row():
                        with gr.Column(scale=5):
                            summary_md = gr.Markdown(
                                "**Resultado:** Esperando imagen...",
                                label="Resumen",
                                elem_classes=["panel-card"],
                            )
                        with gr.Column(scale=5):
                            recom_md = gr.Markdown(
                                "**Recomendaciones:** —",
                                label="Acciones recomendadas",
                                elem_classes=["panel-card"],
                            )

                    # ── JSON raw ──────────────────────────────────────
                    with gr.Accordion("📋 Datos raw (JSON)", open=False):
                        json_out = gr.Code(language="json", label="Detecciones JSON")

                    # ── Evento principal ──────────────────────────────
                    btn_detect.click(
                        fn=predict_from_pil,
                        inputs=[input_image, conf_slider, iou_slider],
                        outputs=[output_image, summary_md, recom_md, json_out],
                        show_progress="full",
                    )

                # ── Grupo 3: Multi-Imagen ──
                with gr.Group(visible=False) as group_multi:
                    with gr.Row(equal_height=False):
                        with gr.Column(scale=5):
                            gr.Markdown("#### 📷 Subir Múltiples Imágenes (Máximo 4)")
                            input_files = gr.File(
                                file_count="multiple",
                                file_types=["image"],
                                label="Seleccionar imágenes",
                                elem_classes=["panel-card"],
                            )
                        with gr.Column(scale=5):
                            gr.Markdown("#### 🖼️ Resultados en Galería")
                            output_gallery = gr.Gallery(
                                label="Imágenes anotadas",
                                columns=2,
                                rows=2,
                                object_fit="contain",
                                interactive=False,
                                elem_classes=["panel-card"],
                            )

                    with gr.Row():
                        with gr.Column(scale=10):
                            with gr.Group(elem_classes=["panel-card"]):
                                with gr.Row():
                                    multi_conf_slider = gr.Slider(
                                        minimum=0.10, maximum=0.95, value=0.40, step=0.05,
                                        label="🎯 Confianza mínima",
                                        info="Mayor valor → menos detecciones pero más precisas",
                                    )
                                    multi_iou_slider = gr.Slider(
                                        minimum=0.10, maximum=0.95, value=0.45, step=0.05,
                                        label="📐 Umbral IoU (NMS)",
                                        info="Controla superposición de bounding boxes",
                                    )
                                btn_multi_detect = gr.Button(
                                    "🔬 Analizar Imágenes",
                                    variant="primary",
                                    elem_id="btn-detect",
                                )

                    with gr.Row():
                        with gr.Column(scale=5):
                            multi_summary_md = gr.Markdown(
                                "**Resultado:** Esperando imágenes...",
                                label="Resumen",
                                elem_classes=["panel-card"],
                            )
                        with gr.Column(scale=5):
                            multi_recom_md = gr.Markdown(
                                "**Recomendaciones:** —",
                                label="Acciones recomendadas",
                                elem_classes=["panel-card"],
                            )

                    with gr.Accordion("📋 Datos raw (JSON)", open=False):
                        multi_json_out = gr.Code(language="json", label="Detecciones JSON")

                    def check_file_count(files):
                        if files and len(files) >= 4:
                            try:
                                if len(files) > 4:
                                    gr.Warning("Límite excedido. Se procesarán solo las primeras 4 imágenes.")
                            except AttributeError:
                                pass
                            return gr.update(interactive=False)
                        return gr.update(interactive=True)

                    input_files.change(
                        fn=check_file_count,
                        inputs=[input_files],
                        outputs=[input_files]
                    )

                    btn_multi_detect.click(
                        fn=predict_multi_images,
                        inputs=[input_files, multi_conf_slider, multi_iou_slider],
                        outputs=[output_gallery, multi_summary_md, multi_recom_md, multi_json_out],
                        show_progress="full",
                    )
                    
                pass # Lógica de navegación al final

            # ════════════════════════════════════════════════════
            # TAB 2: HISTORIAL
            # ════════════════════════════════════════════════════
            with gr.Group(visible=False) as group_history:
                gr.Markdown("### 📊 Historial de detecciones en esta sesión")

                with gr.Row():
                    filter_start_date = gr.DateTime(label="Fecha inicio", include_time=False, type="string")
                    filter_end_date = gr.DateTime(label="Fecha fin", include_time=False, type="string")

                history_table = gr.Dataframe(
                    headers=["Timestamp", "Nº Detecciones", "Enfermedades detectadas"],
                    datatype=["str", "number", "str"],
                    value=[],
                    interactive=False,
                    label="Historial",
                )
                
                with gr.Row():
                    btn_export  = gr.Button("💾 Exportar CSV", variant="secondary")
                    btn_ai = gr.Button("🤖 Analizar tendencias con IA", variant="primary")
                
                ai_output = gr.Markdown("Presiona el botón para generar un análisis agronómico basado en los filtros seleccionados.")
                
                csv_output = gr.Textbox(
                    label="CSV generado (copiar o descargar)",
                    lines=10,
                    visible=False,
                )

                filter_start_date.change(
                    fn=get_history_table,
                    inputs=[filter_start_date, filter_end_date],
                    outputs=[history_table],
                )
                filter_end_date.change(
                    fn=get_history_table,
                    inputs=[filter_start_date, filter_end_date],
                    outputs=[history_table],
                )

                btn_export.click(
                    fn=lambda: (export_history_csv(), gr.update(visible=True)),
                    inputs=[],
                    outputs=[csv_output, csv_output],
                )

                def run_ai_analysis(start_d, end_d):
                    history_data = mongo_db.get_recent_history(limit=500, start_date=start_d, end_date=end_d)
                    return ai_analyst.analyze_history(history_data, start_d, end_d)

                btn_ai.click(
                    fn=run_ai_analysis,
                    inputs=[filter_start_date, filter_end_date],
                    outputs=[ai_output],
                )

            # ════════════════════════════════════════════════════
            # TAB 3: CLASES
            # ════════════════════════════════════════════════════
            with gr.Group(visible=False) as group_classes:
                gr.Markdown("### 🌱 Enfermedades que puede detectar el modelo")

                classes_info = ""
                for cls_id, (cls_key, cls_es) in enumerate(CLASS_NAMES_ES.items()):
                    sev   = SEVERITY_LEVEL.get(cls_key, "")
                    recom = RECOMMENDATIONS.get(cls_key, "")
                    classes_info += f"""
**{cls_id}. {cls_es}** (`{cls_key}`)
- Severidad: {sev}
- Recomendación: {recom}

---
"""
                gr.Markdown(classes_info)

            # ════════════════════════════════════════════════════
            # TAB 4: AYUDA
            # ════════════════════════════════════════════════════
            with gr.Group(visible=False) as group_help:
                gr.Markdown("""
### 🚀 Guía de Uso

#### 1️⃣ Detección en imagen
1. Ve a la pestaña **Detección**
2. Sube una foto de hoja (JPG, PNG, WEBP)
3. Ajusta los umbrales si lo necesitas
4. Haz clic en **Analizar Hoja**
5. Revisa la imagen anotada y las recomendaciones

#### 2️⃣ Parámetros
| Parámetro | Descripción | Valor recomendado |
|---|---|---|
| Confianza | Umbral mínimo de certeza | 0.40 – 0.60 |
| IoU | Supresión de cajas duplicadas | 0.45 |

#### 3️⃣ Entrenamiento propio
```bash
python src/train.py --model yolo11n.pt --epochs 50
```

#### 4️⃣ Inferencia por línea de comandos
```bash
# Imagen
python src/predict.py --source mi_hoja.jpg

# Webcam en tiempo real
python src/predict.py --source 0

# Video
python src/predict.py --source video.mp4 --save-video
```

#### 5️⃣ Evaluación del modelo
```bash
python src/evaluate.py --split test
```

---
> **Nota:** El modelo fue entrenado con el dataset
> [Plant Disease TMYQ8](https://universe.roboflow.com/learning-eri4b/plant-disease-tmyq8)
> (Roboflow, Public Domain).
""")

        # ── Footer ───────────────────────────────────────────────
        gr.HTML("""
        <div style="text-align:center; color:#475569; font-size:0.8rem; margin-top:24px; padding:16px;
                    border-top: 1px solid rgba(255,255,255,0.06);">
          🌿 Plant Disease Detection · YOLO11 · OpenCV · Gradio
          &nbsp;|&nbsp; Proyecto académico — SEXTO SEMESTRE
        </div>
        """)

        # ── Lógica de Navegación Custom ──
        all_groups = [group_vivo, group_foto, group_multi, group_history, group_classes, group_help]
        
        def show_mode(choice):
            if not choice:
                return [gr.update() for _ in range(6)]
            return (
                gr.update(visible=(choice == "Detección en Vivo")),
                gr.update(visible=(choice == "Detección por Foto")),
                gr.update(visible=(choice == "Detección Multi-Imagen")),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
            )
            
        def show_nav(nav_id):
            return (
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=(nav_id == "historial")),
                gr.update(visible=(nav_id == "clases")),
                gr.update(visible=(nav_id == "ayuda")),
            )
            
        mode_selector.change(fn=show_mode, inputs=[mode_selector], outputs=all_groups)
        def nav_to_history(start_d, end_d):
            groups = show_nav("historial")
            table_data = get_history_table(start_d, end_d)
            return (*groups, table_data)

        btn_historial.click(
            fn=nav_to_history,
            inputs=[filter_start_date, filter_end_date],
            outputs=all_groups + [history_table]
        )
        btn_clases.click(fn=lambda: show_nav("clases"), inputs=[], outputs=all_groups)
        btn_ayuda.click(fn=lambda: show_nav("ayuda"), inputs=[], outputs=all_groups)

        # Restaurar sesión desde el historial
        history_table.select(
            fn=restore_session,
            inputs=[history_table],
            outputs=[
                input_image, 
                output_image, 
                summary_md, 
                recom_md, 
                json_out, 
            ] + all_groups + [mode_selector]
        )

    return demo


# ─────────────────────────────────────────────────────────────
# Punto de entrada
# ─────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(description="Lanzar aplicación web de detección")
    parser.add_argument("--model",  default=str(DEFAULT_MODEL),
                        help="Ruta al modelo .pt entrenado")
    parser.add_argument("--port",   type=int, default=7860)
    parser.add_argument("--share",  action="store_true",
                        help="Crear enlace público temporal (Gradio Share)")
    parser.add_argument("--host",   default="127.0.0.1")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n⚠️  Modelo no encontrado: {model_path}")
        print("   Entrena primero con:  python src/train.py")
        print("   O especifica la ruta: python app/app.py --model ruta/al/modelo.pt")
        sys.exit(1)

    print(f"\n{'='*60}")
    print("  🌿 Iniciando aplicación de detección de enfermedades")
    print(f"  Modelo : {model_path}")
    print(f"  URL    : http://{args.host}:{args.port}")
    print(f"{'='*60}\n")

    demo = build_ui(str(model_path))
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        show_error=True,
    )
