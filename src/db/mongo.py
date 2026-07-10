import base64
from datetime import datetime
import cv2
import numpy as np
from pymongo import MongoClient

# Use the connection string provided by the user
MONGO_URI = "mongodb+srv://macordero:espe123@cluster0.fwu2dep.mongodb.net/"
DB_NAME = "YOLO"
COLLECTION_NAME = "Historial"

try:
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    history_collection = db[COLLECTION_NAME]
    # Ping the server to verify connection
    client.admin.command('ping')
    print("✅ Conectado a MongoDB exitosamente")
except Exception as e:
    print(f"❌ Error conectando a MongoDB: {e}")
    history_collection = None

def encode_image_to_base64(image_rgb: np.ndarray) -> str:
    """Convierte una imagen de OpenCV (RGB) a string base64 para MongoDB."""
    if image_rgb is None:
        return ""
    # Convert RGB to BGR for OpenCV encoding
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
    return base64.b64encode(buffer).decode('utf-8')

def decode_image_from_base64(base64_str: str) -> np.ndarray:
    """Convierte un string base64 de MongoDB a imagen OpenCV (RGB)."""
    if not base64_str:
        return None
    image_bytes = base64.b64decode(base64_str)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image_bgr is not None:
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return None

def save_detection(img_original_rgb, img_annotated_rgb, detections_data):
    """
    Guarda la detección en MongoDB.
    
    Args:
        img_original_rgb (numpy array): Imagen original en RGB.
        img_annotated_rgb (numpy array): Imagen con las detecciones dibujadas en RGB.
        detections_data (list): Lista de diccionarios con la info de cada detección.
    """
    if history_collection is None:
        print("⚠️ No hay conexión a MongoDB. No se guardará el historial.")
        return

    # Extraer clases detectadas para resumen
    clases_detectadas = [det["clase"] for det in detections_data]
    
    document = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "detecciones": len(detections_data),
        "clases": clases_detectadas,
        "resultados": detections_data,
        "img_original": encode_image_to_base64(img_original_rgb),
        "img_annotated": encode_image_to_base64(img_annotated_rgb),
    }

    try:
        history_collection.insert_one(document)
        print("✅ Detección guardada en MongoDB.")
    except Exception as e:
        print(f"❌ Error al guardar en MongoDB: {e}")

def get_recent_history(limit=20):
    """
    Recupera los últimos registros del historial, sin cargar las imágenes para 
    no saturar la memoria (usado para la tabla).
    """
    if history_collection is None:
        return []
    
    try:
        # Exclude images to make the query fast
        cursor = history_collection.find({}, {"img_original": 0, "img_annotated": 0}).sort("timestamp", -1).limit(limit)
        return list(cursor)
    except Exception as e:
        print(f"❌ Error consultando MongoDB: {e}")
        return []

def get_detection_by_timestamp(timestamp: str):
    """
    Recupera una detección completa por su timestamp, incluyendo imágenes.
    """
    if history_collection is None:
        return None
    
    try:
        return history_collection.find_one({"timestamp": timestamp})
    except Exception as e:
        print(f"❌ Error recuperando sesión de MongoDB: {e}")
        return None
