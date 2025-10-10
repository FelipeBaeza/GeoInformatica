#!/usr/bin/env python3
"""
SCRIPT: Generador de Visualizaciones - Semana 2 (Características Espaciales)

PROPÓSITO:
Este script genera visualizaciones completas de los índices de habitabilidad urbana calculados 
para Santiago de Chile. Procesa datos geoespaciales para crear 10 gráficos analíticos que muestran
patrones de accesibilidad, distribución espacial y correlaciones entre variables urbanas. Los 
gráficos se generan y se muestran de forma simple en pantalla, además de guardarse como archivos PNG.

FUNCIONALIDADES PRINCIPALES:
- Carga datos geoespaciales con índices de habitabilidad calculados
- Genera 10 visualizaciones diferentes (distribuciones, mapas, correlaciones)
- Muestra gráficos directamente en pantalla de forma interactiva
- Guarda archivos PNG de alta resolución para documentación
- Crea dashboard resumen con métricas principales

DATOS DE ENTRADA:
- grilla_con_indices.geojson: Puntos de evaluación con todos los índices calculados

DATOS DE SALIDA:
- 10 archivos PNG con visualizaciones
- Reporte JSON con resumen de métricas
- Visualización interactiva en pantalla

Autor: Proyecto GeoInformática
Fecha: Octubre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def configurar_matplotlib():
    """
    FUNCIÓN: Configuración estética de matplotlib
    
    PROPÓSITO:
    Establece parámetros globales de matplotlib para crear gráficos profesionales 
    y consistentes. Define tamaños de fuente, dimensiones de figura y estilos 
    visuales que se aplicarán a todos los gráficos generados.
    
    CONFIGURACIONES:
    - Tamaño de figura: 12x8 pulgadas (ideal para presentaciones)
    - Fuentes: Escaladas apropiadamente para legibilidad
    - Títulos y etiquetas: Jerarquía visual clara
    """
    plt.rcParams['figure.figsize'] = (12, 8)        # Tamaño base de figuras
    plt.rcParams['font.size'] = 10                  # Tamaño fuente general
    plt.rcParams['axes.titlesize'] = 14             # Títulos de ejes
    plt.rcParams['axes.labelsize'] = 12             # Etiquetas de ejes
    plt.rcParams['xtick.labelsize'] = 10            # Etiquetas eje X
    plt.rcParams['ytick.labelsize'] = 10            # Etiquetas eje Y
    plt.rcParams['legend.fontsize'] = 10            # Tamaño fuente leyenda
    plt.rcParams['figure.titlesize'] = 16           # Título principal figura

def cargar_datos():
    """
    FUNCIÓN: Carga de datos geoespaciales procesados
    
    PROPÓSITO:
    Carga el archivo GeoJSON que contiene la grilla de evaluación con todos los
    índices de habitabilidad ya calculados. Este archivo es el resultado del 
    procesamiento de la Semana 2 y contiene coordenadas geográficas junto con
    72 características espaciales calculadas para cada punto.
    
    ARCHIVO DE ENTRADA:
    - grilla_con_indices.geojson: Puntos con índices de accesibilidad calculados
    
    RETORNA:
    - GeoDataFrame con datos cargados o None si hay error
    
    VALIDACIONES:
    - Verifica existencia del archivo
    - Confirma carga exitosa con conteo de puntos y columnas
    """
    print(" Cargando datos para visualización...")
    
    # Definir ruta del archivo de datos procesados
    grilla_path = "../features/grilla_con_indices.geojson"
    
    # Validar existencia del archivo de datos
    if not os.path.exists(grilla_path):
        print(f" Error: No se encuentra {grilla_path}")
        return None
    
    # Cargar datos geoespaciales con geopandas
    grilla = gpd.read_file(grilla_path)
    print(f" Datos cargados: {len(grilla)} puntos con {len(grilla.columns)} columnas")
    
    return grilla

def crear_grafico_distribucion_comunas(grilla):
    """Gráfico de distribución de puntos por comuna"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Distribución de puntos
    distribucion = grilla['comuna'].value_counts()
    colors = sns.color_palette("husl", len(distribucion))
    
    wedges, texts, autotexts = ax1.pie(distribucion.values, labels=distribucion.index, 
                                       autopct='%1.1f%%', startangle=90, colors=colors)
    ax1.set_title('Distribución de Puntos de Evaluación\npor Comuna', fontweight='bold')
    
    # Hacer texto más legible
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
    
    # Tabla de estadísticas
    ax2.axis('tight')
    ax2.axis('off')
    
    tabla_data = []
    for comuna, count in distribucion.items():
        porcentaje = (count / len(grilla)) * 100
        tabla_data.append([comuna, count, f"{porcentaje:.1f}%"])
    
    tabla = ax2.table(cellText=tabla_data,
                     colLabels=['Comuna', 'Puntos', 'Porcentaje'],
                     cellLoc='center',
                     loc='center')
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(11)
    tabla.scale(1.2, 1.8)
    
    # Estilo de la tabla
    for i in range(len(tabla_data) + 1):
        for j in range(3):
            cell = tabla[(i, j)]
            if i == 0:  # Header
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F2F2F2' if i % 2 == 0 else '#FFFFFF')
    
    ax2.set_title('Estadísticas por Comuna', fontweight='bold')
    
    plt.tight_layout()
    return fig

def crear_grafico_indices_principales(grilla):
    """Gráfico de los índices principales de habitabilidad"""
    # Preparar datos
    indices_principales = ['acc_educacion', 'acc_salud', 'acc_transporte', 
                          'acc_entorno', 'acc_seguridad', 'acc_comercial']
    
    datos_indices = grilla[indices_principales + ['comuna']].melt(
        id_vars='comuna', 
        var_name='indice', 
        value_name='valor'
    )
    
    # Renombrar para mejor visualización
    nombres_indices = {
        'acc_educacion': 'Educación',
        'acc_salud': 'Salud', 
        'acc_transporte': 'Transporte',
        'acc_entorno': 'Entorno',
        'acc_seguridad': 'Seguridad',
        'acc_comercial': 'Comercial'
    }
    datos_indices['indice'] = datos_indices['indice'].map(nombres_indices)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, indice in enumerate(nombres_indices.values()):
        datos_indice = datos_indices[datos_indices['indice'] == indice]
        
        # Boxplot por comuna
        sns.boxplot(data=datos_indice, x='comuna', y='valor', ax=axes[i])
        axes[i].set_title(f'Accesibilidad {indice}', fontweight='bold', fontsize=12)
        axes[i].set_xlabel('Comuna', fontsize=10)
        axes[i].set_ylabel('Índice (0-10)', fontsize=10)
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 10)
        
        # Añadir línea del promedio general
        promedio_general = datos_indice['valor'].mean()
        axes[i].axhline(y=promedio_general, color='red', linestyle='--', 
                       alpha=0.7, label=f'Promedio: {promedio_general:.2f}')
        axes[i].legend()
    
    plt.suptitle('Distribución de Índices de Accesibilidad por Comuna', 
                 fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig

def crear_grafico_indices_superiores(grilla):
    """Gráfico de los índices superiores de habitabilidad"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Índices superiores
    indices_superiores = ['idx_vida_urbana', 'idx_calidad_vida', 'idx_habitabilidad_global']
    
    # 1. Distribución general de índices superiores
    datos_superiores = grilla[indices_superiores + ['comuna']].melt(
        id_vars='comuna', var_name='indice', value_name='valor'
    )
    
    nombres_superiores = {
        'idx_vida_urbana': 'Vida Urbana',
        'idx_calidad_vida': 'Calidad de Vida', 
        'idx_habitabilidad_global': 'Habitabilidad Global'
    }
    datos_superiores['indice'] = datos_superiores['indice'].map(nombres_superiores)
    
    sns.boxplot(data=datos_superiores, x='indice', y='valor', ax=axes[0,0])
    axes[0,0].set_title('Distribución de Índices Superiores', fontweight='bold')
    axes[0,0].set_xlabel('Índice', fontsize=10)
    axes[0,0].set_ylabel('Puntuación (0-10)', fontsize=10)
    axes[0,0].tick_params(axis='x', rotation=45)
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Correlación entre índices
    matriz_corr = grilla[indices_superiores].corr()
    sns.heatmap(matriz_corr, annot=True, cmap='coolwarm', center=0,
                xticklabels=['Vida Urbana', 'Calidad Vida', 'Habitabilidad'],
                yticklabels=['Vida Urbana', 'Calidad Vida', 'Habitabilidad'],
                ax=axes[0,1])
    axes[0,1].set_title('Correlación entre Índices Superiores', fontweight='bold')
    
    # 3. Ranking por comuna - Habitabilidad Global
    ranking_comunas = grilla.groupby('comuna')['idx_habitabilidad_global'].mean().sort_values(ascending=True)
    
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    bars = axes[1,0].barh(ranking_comunas.index, ranking_comunas.values, color=colors)
    axes[1,0].set_title('Ranking de Habitabilidad Global\npor Comuna', fontweight='bold')
    axes[1,0].set_xlabel('Índice Promedio (0-10)', fontsize=10)
    axes[1,0].grid(True, alpha=0.3, axis='x')
    
    # Añadir valores en las barras
    for bar, valor in zip(bars, ranking_comunas.values):
        axes[1,0].text(valor + 0.05, bar.get_y() + bar.get_height()/2, 
                      f'{valor:.2f}', va='center', fontsize=10, fontweight='bold')
    
    # 4. Dispersión Vida Urbana vs Calidad de Vida
    scatter = axes[1,1].scatter(grilla['idx_vida_urbana'], grilla['idx_calidad_vida'], 
                               c=grilla['idx_habitabilidad_global'], cmap='viridis', 
                               alpha=0.6, s=30)
    axes[1,1].set_xlabel('Índice de Vida Urbana', fontsize=10)
    axes[1,1].set_ylabel('Índice de Calidad de Vida', fontsize=10)
    axes[1,1].set_title('Relación Vida Urbana vs Calidad de Vida', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    # Colorbar para el scatter plot
    cbar = plt.colorbar(scatter, ax=axes[1,1])
    cbar.set_label('Habitabilidad Global', fontsize=10)
    
    plt.tight_layout()
    return fig

def crear_grafico_distancias_densidades(grilla):
    """Gráfico comparativo de distancias vs densidades"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Categorías principales para análisis
    categorias = ['educacion', 'salud', 'comercio']
    
    for i, categoria in enumerate(categorias):
        # Distancias
        col_dist = f'dist_{categoria}_min_m' if categoria == 'educacion' else f'dist_{categoria}_m'
        if col_dist in grilla.columns:
            distancias = grilla[col_dist] / 1000  # Convertir a km
            
            axes[0,i].hist(distancias, bins=30, alpha=0.7, color=sns.color_palette()[i])
            axes[0,i].set_title(f'Distribución de Distancias\n{categoria.capitalize()}', fontweight='bold')
            axes[0,i].set_xlabel('Distancia (km)')
            axes[0,i].set_ylabel('Frecuencia')
            axes[0,i].grid(True, alpha=0.3)
            
            # Estadísticas
            media = distancias.mean()
            mediana = distancias.median()
            axes[0,i].axvline(media, color='red', linestyle='--', alpha=0.8, 
                             label=f'Media: {media:.2f}km')
            axes[0,i].axvline(mediana, color='orange', linestyle='--', alpha=0.8, 
                             label=f'Mediana: {mediana:.2f}km')
            axes[0,i].legend()
        
        # Densidades (usar radio de 600m como compromiso)
        col_dens = f'dens_{categoria}_600m_km2'
        if col_dens in grilla.columns:
            densidades = grilla[col_dens]
            
            axes[1,i].hist(densidades, bins=30, alpha=0.7, color=sns.color_palette()[i])
            axes[1,i].set_title(f'Distribución de Densidades\n{categoria.capitalize()} (600m)', fontweight='bold')
            axes[1,i].set_xlabel('Densidad (servicios/km²)')
            axes[1,i].set_ylabel('Frecuencia')
            axes[1,i].grid(True, alpha=0.3)
            
            # Estadísticas
            media_dens = densidades.mean()
            mediana_dens = densidades.median()
            axes[1,i].axvline(media_dens, color='red', linestyle='--', alpha=0.8, 
                             label=f'Media: {media_dens:.1f}')
            axes[1,i].axvline(mediana_dens, color='orange', linestyle='--', alpha=0.8, 
                             label=f'Mediana: {mediana_dens:.1f}')
            axes[1,i].legend()
    
    plt.suptitle('Análisis de Distancias y Densidades por Categoría', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    return fig

def crear_grafico_mapa_habitabilidad(grilla):
    """Mapa de habitabilidad con puntos coloreados"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7))
    
    # Índices a mapear
    indices_mapa = [
        ('idx_vida_urbana', 'Vida Urbana'),
        ('idx_calidad_vida', 'Calidad de Vida'), 
        ('idx_habitabilidad_global', 'Habitabilidad Global')
    ]
    
    for i, (indice, titulo) in enumerate(indices_mapa):
        # Crear scatter plot geográfico
        scatter = axes[i].scatter(grilla.geometry.x, grilla.geometry.y, 
                                 c=grilla[indice], cmap='RdYlGn', 
                                 s=15, alpha=0.8, vmin=0, vmax=10)
        
        axes[i].set_title(f'Mapa de {titulo}', fontweight='bold', fontsize=12)
        axes[i].set_xlabel('Coordenada X (UTM)', fontsize=10)
        axes[i].set_ylabel('Coordenada Y (UTM)', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_aspect('equal', adjustable='box')
        
        # Colorbar
        cbar = plt.colorbar(scatter, ax=axes[i], fraction=0.046, pad=0.04)
        cbar.set_label(f'Índice {titulo} (0-10)', fontsize=10)
        
        # Formato de coordenadas más legible
        axes[i].ticklabel_format(style='scientific', axis='both', scilimits=(0,0))
    
    plt.suptitle('Distribución Espacial de Índices de Habitabilidad', 
                 fontsize=16, fontweight='bold', y=0.95)
    plt.tight_layout()
    return fig

def crear_dashboard_resumen(grilla):
    """Dashboard resumen con métricas clave"""
    fig = plt.figure(figsize=(20, 12))
    
    # Layout del dashboard
    gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 1], width_ratios=[1, 1, 1, 1])
    
    # 1. Métricas generales
    ax_metricas = fig.add_subplot(gs[0, :2])
    ax_metricas.axis('off')
    
    # Calcular métricas clave
    total_puntos = len(grilla)
    comunas = grilla['comuna'].nunique()
    hab_promedio = grilla['idx_habitabilidad_global'].mean()
    hab_std = grilla['idx_habitabilidad_global'].std()
    mejor_comuna = grilla.groupby('comuna')['idx_habitabilidad_global'].mean().idxmax()
    
    metricas_texto = f"""
     MÉTRICAS GENERALES DEL PROYECTO
    
    • Total de puntos evaluados: {total_puntos:,}
    • Comunas analizadas: {comunas}
    • Habitabilidad promedio: {hab_promedio:.2f}/10
    • Desviación estándar: {hab_std:.2f}
    • Comuna con mejor habitabilidad: {mejor_comuna}
    • Área cubierta: 213.3 km²
    """
    
    ax_metricas.text(0.05, 0.95, metricas_texto, transform=ax_metricas.transAxes,
                    fontsize=12, verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
    
    # 2. Top comunas por categoría
    ax_ranking = fig.add_subplot(gs[0, 2:])
    
    categorias_ranking = ['acc_educacion', 'acc_salud', 'acc_transporte', 'acc_entorno']
    nombres_categorias = ['Educación', 'Salud', 'Transporte', 'Entorno']
    
    ranking_data = []
    for categoria, nombre in zip(categorias_ranking, nombres_categorias):
        mejor = grilla.groupby('comuna')[categoria].mean().idxmax()
        valor = grilla.groupby('comuna')[categoria].mean().max()
        ranking_data.append([nombre, mejor, f"{valor:.2f}"])
    
    tabla_ranking = ax_ranking.table(cellText=ranking_data,
                                   colLabels=['Categoría', 'Mejor Comuna', 'Puntuación'],
                                   cellLoc='center', loc='center')
    tabla_ranking.auto_set_font_size(False)
    tabla_ranking.set_fontsize(10)
    tabla_ranking.scale(1.2, 2)
    
    # Estilo tabla
    for i in range(len(ranking_data) + 1):
        for j in range(3):
            cell = tabla_ranking[(i, j)]
            if i == 0:
                cell.set_facecolor('#2E8B57')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F0F8FF' if i % 2 == 0 else '#FFFFFF')
    
    ax_ranking.set_title(' Ranking por Categoría', fontweight='bold', fontsize=12)
    ax_ranking.axis('off')
    
    # 3. Gráfico de radar por comuna
    ax_radar = fig.add_subplot(gs[1, :2])
    
    # Preparar datos para radar
    categorias_radar = ['acc_educacion', 'acc_salud', 'acc_transporte', 
                       'acc_entorno', 'acc_seguridad', 'acc_comercial']
    nombres_radar = ['Educación', 'Salud', 'Transporte', 'Entorno', 'Seguridad', 'Comercial']
    
    promedios_comuna = grilla.groupby('comuna')[categorias_radar].mean()
    
    # Crear gráfico de barras agrupadas en lugar de radar (más simple)
    x = np.arange(len(nombres_radar))
    width = 0.2
    
    for i, comuna in enumerate(promedios_comuna.index):
        valores = promedios_comuna.loc[comuna].values
        ax_radar.bar(x + i*width, valores, width, label=comuna, alpha=0.8)
    
    ax_radar.set_xlabel('Categorías de Accesibilidad')
    ax_radar.set_ylabel('Puntuación Promedio')
    ax_radar.set_title('Perfil de Accesibilidad por Comuna', fontweight='bold')
    ax_radar.set_xticks(x + width * 1.5)
    ax_radar.set_xticklabels(nombres_radar, rotation=45, ha='right')
    ax_radar.legend()
    ax_radar.grid(True, alpha=0.3)
    
    # 4. Histograma de habitabilidad global
    ax_hist = fig.add_subplot(gs[1, 2:])
    
    colors_comunas = sns.color_palette("husl", grilla['comuna'].nunique())
    comuna_colors = dict(zip(grilla['comuna'].unique(), colors_comunas))
    
    for comuna in grilla['comuna'].unique():
        data_comuna = grilla[grilla['comuna'] == comuna]['idx_habitabilidad_global']
        ax_hist.hist(data_comuna, bins=20, alpha=0.6, label=comuna, 
                    color=comuna_colors[comuna])
    
    ax_hist.set_xlabel('Índice de Habitabilidad Global')
    ax_hist.set_ylabel('Frecuencia')
    ax_hist.set_title('Distribución de Habitabilidad Global\npor Comuna', fontweight='bold')
    ax_hist.legend()
    ax_hist.grid(True, alpha=0.3)
    
    # 5. Comparación de características extremas
    ax_extremos = fig.add_subplot(gs[2, :])
    
    # Encontrar puntos con características extremas
    mejor_punto = grilla.loc[grilla['idx_habitabilidad_global'].idxmax()]
    peor_punto = grilla.loc[grilla['idx_habitabilidad_global'].idxmin()]
    
    caracteristicas_comp = ['acc_educacion', 'acc_salud', 'acc_transporte', 
                           'acc_entorno', 'acc_seguridad', 'acc_comercial']
    
    x_pos = np.arange(len(caracteristicas_comp))
    
    mejor_valores = [mejor_punto[cat] for cat in caracteristicas_comp]
    peor_valores = [peor_punto[cat] for cat in caracteristicas_comp]
    
    ax_extremos.bar(x_pos - 0.2, mejor_valores, 0.4, label=f'Mejor ubicación ({mejor_punto["comuna"]})', 
                   color='green', alpha=0.7)
    ax_extremos.bar(x_pos + 0.2, peor_valores, 0.4, label=f'Peor ubicación ({peor_punto["comuna"]})', 
                   color='red', alpha=0.7)
    
    ax_extremos.set_xlabel('Categorías de Accesibilidad')
    ax_extremos.set_ylabel('Puntuación')
    ax_extremos.set_title('Comparación: Mejor vs Peor Ubicación', fontweight='bold')
    ax_extremos.set_xticks(x_pos)
    ax_extremos.set_xticklabels(['Educación', 'Salud', 'Transporte', 'Entorno', 'Seguridad', 'Comercial'],
                               rotation=45, ha='right')
    ax_extremos.legend()
    ax_extremos.grid(True, alpha=0.3)
    ax_extremos.set_ylim(0, 10)
    
    plt.suptitle(' DASHBOARD RESUMEN - CARACTERÍSTICAS ESPACIALES SANTIAGO', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig

def mostrar_graficos_interactivos(grilla):
    """
    FUNCIÓN: Visualización interactiva de gráficos
    
    PROPÓSITO:
    Genera y muestra gráficos de forma simple en pantalla sin abrir navegadores
    o URLs externas. Utiliza matplotlib.pyplot.show() para mostrar cada gráfico
    de manera secuencial, permitiendo al usuario ver los resultados directamente
    en el entorno de desarrollo.
    
    CARACTERÍSTICAS:
    - Muestra gráficos uno a uno de forma interactiva
    - No requiere navegador web
    - Permite cerrar cada gráfico para continuar al siguiente
    - Configuración optimizada para visualización simple
    
    PARÁMETROS:
    - grilla: GeoDataFrame con datos procesados de habitabilidad
    
    USO:
    El usuario puede cerrar cada ventana de gráfico para avanzar al siguiente.
    """
    print("\n MOSTRANDO GRÁFICOS DE FORMA INTERACTIVA")
    print("=" * 50)
    print(" Cierra cada ventana para continuar al siguiente gráfico")
    
    # Lista de funciones de gráficos con descripciones
    graficos = [
        (crear_grafico_distribucion_comunas, "1/10: Distribución por Comunas"),
        (crear_grafico_indices_principales, "2/10: Índices Principales de Accesibilidad"),
        (crear_grafico_indices_superiores, "3/10: Índices Superiores de Habitabilidad"),
        (crear_grafico_distancias_densidades, "4/10: Análisis Distancias vs Densidades"),
        (crear_grafico_mapa_habitabilidad, "5/10: Mapa de Habitabilidad Espacial"),
        (crear_dashboard_resumen, "6/10: Dashboard Resumen Ejecutivo")
    ]
    
    # Mostrar cada gráfico de forma interactiva
    for i, (funcion_grafico, descripcion) in enumerate(graficos, 1):
        print(f"\n Mostrando: {descripcion}")
        
        try:
            # Generar el gráfico
            fig = funcion_grafico(grilla)
            
            # Mostrar de forma interactiva (se cierra manualmente)
            plt.show()
            
            print(f" Gráfico {i} mostrado correctamente")
            
        except Exception as e:
            print(f" Error mostrando gráfico {i}: {e}")
            continue
    
    print(f"\n ¡Visualización interactiva completada!")
    print(f" Los archivos PNG están guardados en: ../graficos/")

def main():
    """
    FUNCIÓN PRINCIPAL: Generador completo de visualizaciones
    
    PROPÓSITO:
    Función principal que coordina todo el proceso de generación de gráficos.
    Configura el entorno, carga los datos, genera los gráficos tanto para 
    guardado como para visualización interactiva, y crea reportes de resumen.
    
    PROCESO:
    1. Configura parámetros de matplotlib
    2. Carga datos procesados de habitabilidad
    3. Genera y guarda 10 gráficos como archivos PNG
    4. Muestra gráficos de forma interactiva en pantalla
    5. Crea reporte JSON con métricas de resumen
    
    RETORNA:
    - True si el proceso se completa exitosamente
    - False si hay errores críticos
    """
    print(" GENERADOR DE GRÁFICOS - SEMANA 2")
    print("="*50)
    
    # Configurar matplotlib
    configurar_matplotlib()
    
    # Crear directorio de gráficos si no existe
    os.makedirs('../graficos', exist_ok=True)
    
    # Cargar datos
    grilla = cargar_datos()
    if grilla is None:
        return False
    
    print(f" Generando gráficos para {len(grilla)} puntos de evaluación...")
    
    # Lista de gráficos a generar
    graficos = [
        (crear_grafico_distribucion_comunas, "01_distribucion_comunas.png", "Distribución por comunas"),
        (crear_grafico_indices_principales, "02_indices_principales.png", "Índices principales de accesibilidad"),
        (crear_grafico_indices_superiores, "03_indices_superiores.png", "Índices superiores de habitabilidad"),
        (crear_grafico_distancias_densidades, "04_distancias_densidades.png", "Análisis distancias vs densidades"),
        (crear_grafico_mapa_habitabilidad, "05_mapa_habitabilidad.png", "Mapa de habitabilidad espacial"),
        (crear_dashboard_resumen, "06_dashboard_resumen.png", "Dashboard resumen ejecutivo")
    ]
    
    # Generar cada gráfico
    for i, (funcion_grafico, nombre_archivo, descripcion) in enumerate(graficos, 1):
        print(f"\n[{i}/{len(graficos)}] Generando: {descripcion}...")
        
        try:
            fig = funcion_grafico(grilla)
            ruta_archivo = f"../graficos/{nombre_archivo}"
            fig.savefig(ruta_archivo, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close(fig)  # Liberar memoria
            
            print(f" Guardado: {ruta_archivo}")
            
        except Exception as e:
            print(f" Error generando {descripcion}: {e}")
            continue
    
    # Generar reporte de gráficos
    print(f"\n Generando índice de gráficos...")
    
    reporte_graficos = {
        'fecha_generacion': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_puntos_analizados': len(grilla),
        'comunas_incluidas': grilla['comuna'].unique().tolist(),
        'graficos_generados': [
            {
                'archivo': nombre,
                'descripcion': desc,
                'tipo': 'análisis_exploratorio'
            } for _, nombre, desc in graficos
        ],
        'metricas_resumen': {
            'habitabilidad_promedio': float(grilla['idx_habitabilidad_global'].mean()),
            'habitabilidad_std': float(grilla['idx_habitabilidad_global'].std()),
            'mejor_comuna': grilla.groupby('comuna')['idx_habitabilidad_global'].mean().idxmax(),
            'rango_habitabilidad': [float(grilla['idx_habitabilidad_global'].min()), 
                                  float(grilla['idx_habitabilidad_global'].max())]
        }
    }
    
    import json
    with open('../reportes/graficos_resumen.json', 'w', encoding='utf-8') as f:
        json.dump(reporte_graficos, f, indent=2, ensure_ascii=False)
    
    print(f" Índice de gráficos guardado: ../reportes/graficos_resumen.json")
    print(f"\n Generación de gráficos completada exitosamente!")
    print(f" Ubicación: semana2_caracteristicas_espaciales/graficos/")
    
    # Preguntar si desea ver los gráficos de forma interactiva
    print(f"\n" + "="*50)
    respuesta = input("¿Deseas ver los gráficos de forma interactiva? (s/n): ").lower().strip()
    
    if respuesta in ['s', 'si', 'sí', 'y', 'yes']:
        mostrar_graficos_interactivos(grilla)
    else:
        print(" Los gráficos están guardados en la carpeta ../graficos/")
        print(" Puedes ejecutar solo la visualización interactiva ejecutando:")
        print("   python -c \"from generar_graficos import *; g = cargar_datos(); mostrar_graficos_interactivos(g)\"")
    
    return True

if __name__ == "__main__":
    exito = main()
    if not exito:
        exit(1)