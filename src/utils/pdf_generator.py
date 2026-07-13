import os
from datetime import datetime
from PIL import Image
import tempfile

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

def generate_report(image_pil: Image.Image, summary_md: str, recom_md: str) -> str:
    """
    Genera un archivo PDF temporal con los resultados del análisis y retorna su ruta.
    """
    temp_fd, temp_path = tempfile.mkstemp(suffix=".pdf", prefix="reporte_agronomico_")
    os.close(temp_fd)
    
    doc = SimpleDocTemplate(temp_path, pagesize=letter,
                            rightMargin=50, leftMargin=50,
                            topMargin=50, bottomMargin=50)
    Story = []
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor("#059669"),
        spaceAfter=20,
        alignment=1 # Center
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#10b981"),
        spaceBefore=15,
        spaceAfter=10
    )
    body_style = ParagraphStyle(
        'CustomBody',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=8,
        leading=15
    )

    Story.append(Paragraph("Reporte de Análisis Agronómico", title_style))
    Story.append(Paragraph(f"Fecha de Análisis: {datetime.now().strftime('%d/%m/%Y %H:%M')}", body_style))
    Story.append(Spacer(1, 0.2*inch))
    
    img_path = None
    if image_pil:
        img_fd, img_path = tempfile.mkstemp(suffix=".jpg")
        os.close(img_fd)
        # Fix transparent images if needed
        if image_pil.mode in ('RGBA', 'LA') or (image_pil.mode == 'P' and 'transparency' in image_pil.info):
            bg = Image.new('RGB', image_pil.size, (255, 255, 255))
            bg.paste(image_pil, mask=image_pil.split()[3])
            bg.save(img_path, format="JPEG")
        else:
            image_pil.convert('RGB').save(img_path, format="JPEG")
        
        img_width, img_height = image_pil.size
        aspect = img_height / float(img_width)
        target_width = 5.0 * inch
        target_height = target_width * aspect
        
        rl_img = RLImage(img_path, width=target_width, height=target_height)
        Story.append(rl_img)
        Story.append(Spacer(1, 0.2*inch))
        
    def parse_and_add_markdown(text):
        if not text: return
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            if line.startswith('### '):
                Story.append(Paragraph(line[4:].replace('*', ''), heading_style))
            elif line.startswith('**') and ':**' in line:
                Story.append(Paragraph("<b>" + line.replace('**', '').replace(':', ':</b>'), body_style))
            elif line.startswith('* **'):
                Story.append(Paragraph("• " + line[2:].replace('**', '<b>', 1).replace('**', '</b>', 1), body_style))
            elif line.startswith('> '):
                Story.append(Paragraph("<i>" + line[2:] + "</i>", body_style))
            elif line.startswith('**'):
                Story.append(Paragraph("<b>" + line.replace('**', '') + "</b>", body_style))
            else:
                Story.append(Paragraph(line.replace('*', ''), body_style))
                
    Story.append(Paragraph("Resumen de Detección", heading_style))
    parse_and_add_markdown(summary_md)
    Story.append(Spacer(1, 0.1*inch))
    
    Story.append(Paragraph("Recomendaciones y Acciones", heading_style))
    parse_and_add_markdown(recom_md)
    
    doc.build(Story)
    
    if img_path:
        try:
            os.remove(img_path)
        except:
            pass
            
    return temp_path
