"""
Utilidades compartidas para el proyecto de detección de enfermedades en hojas.
"""

# Mapeo de nombres técnicos (indonesio) → nombres en español para la UI
CLASS_NAMES_ES = {
    "bercak_daun":          "Mancha Foliar",
    "defisiensi_kalsium":   "Deficiencia de Calcio",
    "hangus_daun":          "Quemadura de Hoja",
    "hawar_daun":           "Tizón Foliar",
    "mosaik_vena_kuning":   "Mosaico Vena Amarilla",
    "virus_kuning_keriting": "Virus Rizado Amarillo",
}

# Colores BGR para cada clase (OpenCV)
CLASS_COLORS_BGR = {
    "bercak_daun":           (0,   165, 255),   # Naranja
    "defisiensi_kalsium":    (0,   255, 255),   # Amarillo
    "hangus_daun":           (0,   0,   200),   # Rojo oscuro
    "hawar_daun":            (128, 0,   128),   # Púrpura
    "mosaik_vena_kuning":    (0,   200, 100),   # Verde-amarillo
    "virus_kuning_keriting": (255, 0,   0),     # Azul
}

# Nivel de severidad de cada enfermedad
SEVERITY_LEVEL = {
    "bercak_daun":           "⚠️ Moderado",
    "defisiensi_kalsium":    "⚠️ Moderado",
    "hangus_daun":           "🔴 Grave",
    "hawar_daun":            "🔴 Grave",
    "mosaik_vena_kuning":    "⚠️ Moderado",
    "virus_kuning_keriting": "🔴 Grave",
}

# Recomendaciones de acción
RECOMMENDATIONS = {
    "bercak_daun": (
        "Aplicar fungicida de contacto. "
        "Retirar hojas afectadas. "
        "Evitar riego por aspersión."
    ),
    "defisiensi_kalsium": (
        "Aplicar fertilizante con calcio (CaCl₂). "
        "Ajustar pH del suelo a 6.0-6.5. "
        "Mejorar el drenaje."
    ),
    "hangus_daun": (
        "Aplicar bactericida cúprico. "
        "Eliminar hojas quemadas. "
        "Evitar el exceso de fertilizante nitrogenado."
    ),
    "hawar_daun": (
        "Aplicar fungicida sistémico. "
        "Realizar rotación de cultivos. "
        "Destruir restos de cosecha infectados."
    ),
    "mosaik_vena_kuning": (
        "Controlar insectos vectores (mosca blanca). "
        "Eliminar plantas infectadas. "
        "Usar variedades resistentes."
    ),
    "virus_kuning_keriting": (
        "Eliminar plantas infectadas inmediatamente. "
        "Controlar la mosca blanca con insecticidas. "
        "Usar malla antiáfidos."
    ),
}
