import os
from google import genai
from typing import List, Dict, Any
from dotenv import load_dotenv

from src.utils import CLASS_NAMES_ES

load_dotenv()

# Configurar API Key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    # No es necesario configurar de forma global con el nuevo SDK.
    # El cliente se instanciará en la función.
    pass
else:
    print("AVISO: GEMINI_API_KEY no configurada. El análisis con IA no funcionará.")

def analyze_history(history_data: List[Dict[str, Any]], start_date: str = None, end_date: str = None) -> str:
    """
    Toma los datos del historial (y opcionalmente las fechas) para generar un 
    análisis agronómico utilizando Google Gemini.
    """
    if not GEMINI_API_KEY:
        return "⚠️ **Error:** No se ha configurado la clave de API de Gemini (`GEMINI_API_KEY`). Por favor, añádela a tu archivo `.env` en la raíz del proyecto."
    
    if not history_data:
        return "ℹ️ No hay datos suficientes en el rango de fechas seleccionado para realizar un análisis."

    # Resumir los datos para el prompt
    total_detections = len(history_data)
    disease_counts = {}
    
    for entry in history_data:
        # history_data es lo que devuelve mongo_db.get_recent_history
        # que tiene 'detecciones' y 'clases' (lista de keys)
        for cls in entry.get("clases", []):
            name_es = CLASS_NAMES_ES.get(cls, cls)
            disease_counts[name_es] = disease_counts.get(name_es, 0) + 1

    summary_text = f"- Total de sesiones de detección: {total_detections}\n"
    summary_text += "- Frecuencia de enfermedades encontradas:\n"
    for cls, count in disease_counts.items():
        summary_text += f"  * {cls}: {count} veces\n"
        
    if not disease_counts:
        summary_text += "  * Ninguna (Plantas sanas)\n"

    # Preparar el prompt
    date_context = "el periodo seleccionado"
    if start_date and end_date:
        date_context = f"el periodo del {str(start_date)[:10]} al {str(end_date)[:10]}"
    elif start_date:
        date_context = f"el periodo desde {str(start_date)[:10]}"
    elif end_date:
        date_context = f"el periodo hasta {str(end_date)[:10]}"

    prompt = f"""
Eres un experto agrónomo virtual. Analiza el siguiente resumen de detecciones de enfermedades en hojas de plantas para {date_context}.

**Resumen de los datos:**
{summary_text}

**Instrucciones:**
1. Proporciona un breve análisis de las tendencias observadas.
2. Basado en las enfermedades más frecuentes, sugiere posibles causas ambientales o de cuidado que podrían estar provocándolas.
3. Brinda 3 recomendaciones prácticas generales para el agricultor o cuidador de las plantas para mitigar estos problemas o mantener las plantas sanas.

Usa formato Markdown, sé profesional pero amigable, y conciso.
"""

    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model='gemini-3.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"❌ **Error al generar el análisis:** {str(e)}"
