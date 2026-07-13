import os
import time
from groq import Groq
from typing import List, Dict, Any
from dotenv import load_dotenv

from src.utils import CLASS_NAMES_ES

load_dotenv()

# Configurar API Key
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("AVISO: GROQ_API_KEY no configurada. El análisis con IA no funcionará.")

def analyze_history(history_data: List[Dict[str, Any]], start_date: str = None, end_date: str = None) -> str:
    """
    Toma los datos del historial (y opcionalmente las fechas) para generar un 
    análisis agronómico utilizando Google Gemini.
    """
    if not GROQ_API_KEY:
        return "⚠️ **Error:** No se ha configurado la clave de API de Groq (`GROQ_API_KEY`). Por favor, añádela a tu archivo `.env` en la raíz del proyecto."
    
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

    max_retries = 3
    fallback_models = ['llama-3.3-70b-versatile', 'llama3-70b-8192', 'llama3-8b-8192', 'mixtral-8x7b-32768']
    
    last_error = ""
    for attempt in range(max_retries):
        for model_name in fallback_models:
            try:
                client = Groq(api_key=GROQ_API_KEY)
                chat_completion = client.chat.completions.create(
                    messages=[
                        {
                            "role": "user",
                            "content": prompt,
                        }
                    ],
                    model=model_name,
                )
                return chat_completion.choices[0].message.content
            except Exception as e:
                err_str = str(e)
                last_error = err_str
                # Si es error 503 (Unavailable), 429 (Rate limit) pasamos al siguiente modelo
                if "503" in err_str or "429" in err_str or "not found" in err_str.lower():
                    continue
                # Otro tipo de error (ej. clave inválida), retornamos de inmediato
                return f"❌ **Error al generar el análisis:** {err_str}"
                
        # Si fallaron todos los modelos, aplicamos backoff antes del siguiente intento general
        if attempt < max_retries - 1:
            time.sleep(2 ** attempt)
            
    return f"❌ **Error al generar el análisis (Todos los modelos fallaron):** {last_error}"
