import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from src.utils import CLASS_NAMES_ES
import src.db.mongo as mongo_db

import matplotlib
matplotlib.use('Agg')

def get_stats_plots():
    """
    Fetch history from MongoDB and generate two Matplotlib figures.
    Returns: (fig_pie, fig_line)
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)
    
    history = mongo_db.get_recent_history(limit=5000, start_date=start_date, end_date=end_date)
    
    if not history:
        fig_pie, ax_pie = plt.subplots(figsize=(6, 4), facecolor='#1f222b')
        ax_pie.text(0.5, 0.5, "Sin datos suficientes", color='white', ha='center')
        ax_pie.set_axis_off()
        
        fig_line, ax_line = plt.subplots(figsize=(8, 4), facecolor='#1f222b')
        ax_line.text(0.5, 0.5, "Sin datos suficientes", color='white', ha='center')
        ax_line.set_axis_off()
        return fig_pie, fig_line

    df = pd.DataFrame(history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    all_classes = []
    for cls_list in df['clases']:
        if not cls_list:
             all_classes.append("Sana")
        for c in cls_list:
            all_classes.append(CLASS_NAMES_ES.get(c, c))
            
    counts = pd.Series(all_classes).value_counts()
    
    fig_pie, ax_pie = plt.subplots(figsize=(6, 5), facecolor='#1f222b')
    ax_pie.set_facecolor('#1f222b')
    
    colors = ['#10b981', '#ef4444', '#eab308', '#3b82f6', '#8b5cf6', '#f97316', '#14b8a6', '#6366f1']
    wedges, texts, autotexts = ax_pie.pie(
        counts, 
        labels=counts.index, 
        autopct='%1.1f%%', 
        startangle=90,
        colors=colors[:len(counts)],
        textprops={'color': 'white'}
    )
    plt.setp(autotexts, size=9, weight="bold", color="white")
    ax_pie.set_title("Distribución de Enfermedades (Últimos 30 días)", color='white', pad=20)
    
    df['date'] = df['timestamp'].dt.date
    daily_counts = df.groupby('date').size()
    
    idx = pd.date_range(start_date.date(), end_date.date())
    daily_counts.index = pd.DatetimeIndex(daily_counts.index)
    daily_counts = daily_counts.reindex(idx, fill_value=0)
    
    fig_line, ax_line = plt.subplots(figsize=(8, 5), facecolor='#1f222b')
    ax_line.set_facecolor('#1f222b')
    
    ax_line.plot(daily_counts.index, daily_counts.values, marker='o', color='#10b981', linewidth=2, markersize=6)
    ax_line.fill_between(daily_counts.index, daily_counts.values, alpha=0.2, color='#10b981')
    
    ax_line.set_title("Evolución de Análisis Diarios", color='white', pad=20)
    ax_line.set_ylabel("Análisis Realizados", color='white')
    ax_line.tick_params(colors='white')
    ax_line.spines['bottom'].set_color('white')
    ax_line.spines['left'].set_color('white') 
    ax_line.spines['top'].set_visible(False)
    ax_line.spines['right'].set_visible(False)
    ax_line.grid(color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
    
    fig_line.autofmt_xdate()
    
    fig_pie.tight_layout()
    fig_line.tight_layout()
    
    return fig_pie, fig_line
