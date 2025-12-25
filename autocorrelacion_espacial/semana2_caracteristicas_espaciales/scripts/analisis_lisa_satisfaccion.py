#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Análisis LISA (Local Indicators of Spatial Association) - Factores de Satisfacción
==================================================================================

Este script realiza un análisis de autocorrelación espacial para determinar si los 
factores que influyen en la satisfacción residencial están correlacionados con 
zonas geográficas específicas (comunas).

Pregunta de investigación:
¿Los factores que harían que una propiedad sea más satisfactoria están ligados 
a una zona/comuna en particular?

Metodología:
1. Moran's I Global - Determina si existe autocorrelación espacial general
2. LISA (Local Moran's I) - Identifica clústeres locales significativos
3. Análisis por comuna - Caracterización de cada zona

Clústeres LISA:
- High-High (HH): Zonas donde propiedades con alta satisfacción están rodeadas de otras similares
- Low-Low (LL): Zonas donde propiedades con baja satisfacción están agrupadas
- High-Low (HL): Propiedades satisfactorias en zonas de baja satisfacción (outliers)
- Low-High (LH): Propiedades poco satisfactorias en zonas premium (outliers)

Autores: Equipo Geoinformática
Fecha: Diciembre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar libspatialindex para análisis espacial
try:
    from libpysal.weights import KNN, Queen, DistanceBand
    from esda.moran import Moran, Moran_Local
    from esda.getisord import G_Local
    ESDA_DISPONIBLE = True
except ImportError:
    print("Instalando libpysal y esda...")
    os.system('pip install libpysal esda --quiet')
    from libpysal.weights import KNN, Queen, DistanceBand
    from esda.moran import Moran, Moran_Local
    ESDA_DISPONIBLE = True

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SEMANA3_DIR = os.path.join(os.path.dirname(BASE_DIR), 'semana3_modelo_satisfaccion')
DATA_DIR = os.path.join(SEMANA3_DIR, 'data')
DATOS_NORMALIZADOS = os.path.join(os.path.dirname(BASE_DIR), 'semana1_preparacion_datos', 
                                   'datos_normalizados', 'datos_normalizados')
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados_analisis')
GRAFICOS_DIR = os.path.join(BASE_DIR, 'graficos')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# Colores para clústeres LISA
COLORES_LISA = {
    'High-High': '#d7191c',      # Rojo - Hot spots
    'Low-Low': '#2c7bb6',        # Azul - Cold spots
    'High-Low': '#fdae61',       # Naranja - Outliers positivos
    'Low-High': '#abd9e9',       # Celeste - Outliers negativos
    'No Significativo': '#ffffbf' # Amarillo claro - No significativo
}

# Colores para comunas
COLORES_COMUNAS = {
    'Santiago': '#3498db',
    'Ñuñoa': '#2ecc71', 
    'La Reina': '#9b59b6',
    'Estación Central': '#e74c3c'
}

print("=" * 80)
print("   ANÁLISIS LISA - FACTORES DE SATISFACCIÓN RESIDENCIAL")
print("   Autocorrelación Espacial por Comunas")
print("=" * 80)


def cargar_datos():
    """Carga los datos de propiedades con factores espaciales."""
    print("\nCARGANDO DATOS...")
    
    # Intentar cargar el GeoJSON con factores espaciales
    geojson_path = os.path.join(DATA_DIR, 'propiedades_con_factores_espaciales.geojson')
    csv_path = os.path.join(DATA_DIR, 'propiedades_con_factores_espaciales.csv')
    
    if os.path.exists(geojson_path):
        gdf = gpd.read_file(geojson_path)
        print(f"Cargado GeoJSON: {len(gdf)} propiedades")
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        # Crear geometría desde coordenadas
        from shapely.geometry import Point
        geometry = [Point(xy) for xy in zip(df['x_utm'], df['y_utm'])]
        gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:32719")
        print(f"Cargado CSV y convertido a GeoDataFrame: {len(gdf)} propiedades")
    else:
        raise FileNotFoundError(f"No se encontraron datos en {DATA_DIR}")
    
    # Cargar comunas
    comunas_path = os.path.join(DATOS_NORMALIZADOS, 'comunas_buffer.geojson')
    comunas = gpd.read_file(comunas_path)
    print(f"Comunas cargadas: {len(comunas)}")
    
    return gdf, comunas


def calcular_indice_satisfaccion(gdf):
    """
    Calcula un índice de satisfacción compuesto basado en los factores espaciales.
    
    Factores considerados (ponderados):
    1. Accesibilidad a servicios (distancias mínimas) - peso negativo
    2. Densidad de servicios (1000m) - peso positivo
    3. Diversidad de servicios - peso positivo
    """
    print("\nCALCULANDO ÍNDICE DE SATISFACCIÓN COMPUESTO...")
    
    # Columnas de distancias (menor = mejor)
    dist_cols = [
        'dist_educacion_min_m', 'dist_salud_min_m', 'dist_seguridad_min_m',
        'dist_transporte_min_m', 'dist_areas_verdes_m', 'dist_comercio_m'
    ]
    
    # Columnas de densidades normalizadas (mayor = mejor)
    dens_cols = [
        'dens_norm_educacion_1000m_km2', 'dens_norm_salud_1000m_km2',
        'dens_norm_comercio_1000m_km2', 'dens_norm_seguridad_1000m_km2',
        'dens_norm_transporte_1000m_km2', 'dens_norm_recreacion_1000m_km2'
    ]
    
    # Columnas de diversidad (mayor = mejor)
    div_cols = ['div_norm_servicios_300m', 'div_norm_servicios_600m', 'div_norm_servicios_1000m']
    
    # Filtrar columnas existentes
    dist_cols = [c for c in dist_cols if c in gdf.columns]
    dens_cols = [c for c in dens_cols if c in gdf.columns]
    div_cols = [c for c in div_cols if c in gdf.columns]
    
    # Normalizar distancias (invertir: menor distancia = mayor valor)
    for col in dist_cols:
        if gdf[col].max() > 0:
            gdf[f'{col}_norm'] = 1 - (gdf[col] / gdf[col].max())
        else:
            gdf[f'{col}_norm'] = 0
    
    dist_norm_cols = [f'{c}_norm' for c in dist_cols]
    
    # Calcular sub-índices
    gdf['idx_accesibilidad'] = gdf[dist_norm_cols].mean(axis=1) if dist_norm_cols else 0
    gdf['idx_densidad'] = gdf[dens_cols].mean(axis=1) if dens_cols else 0
    gdf['idx_diversidad'] = gdf[div_cols].mean(axis=1) if div_cols else 0
    
    # Índice compuesto de satisfacción (0-1)
    # Ponderación: 40% accesibilidad, 35% densidad, 25% diversidad
    gdf['indice_satisfaccion'] = (
        0.40 * gdf['idx_accesibilidad'] +
        0.35 * gdf['idx_densidad'] +
        0.25 * gdf['idx_diversidad']
    )
    
    # Normalizar a rango 0-1
    gdf['indice_satisfaccion'] = (gdf['indice_satisfaccion'] - gdf['indice_satisfaccion'].min()) / \
                                  (gdf['indice_satisfaccion'].max() - gdf['indice_satisfaccion'].min())
    
    print(f"Índice de satisfacción calculado")
    print(f"     - Rango: {gdf['indice_satisfaccion'].min():.4f} - {gdf['indice_satisfaccion'].max():.4f}")
    print(f"     - Media: {gdf['indice_satisfaccion'].mean():.4f}")
    print(f"     - Desv. Std: {gdf['indice_satisfaccion'].std():.4f}")
    
    return gdf


def crear_matriz_pesos(gdf, k=8):
    """Crea matriz de pesos espaciales usando K vecinos más cercanos."""
    print(f"\n CREANDO MATRIZ DE PESOS ESPACIALES (K={k})...")
    
    # Asegurar que no hay geometrías nulas
    gdf_valid = gdf[~gdf.geometry.isna()].copy()
    
    # Crear matriz de pesos K-NN
    w = KNN.from_dataframe(gdf_valid, k=k)
    w.transform = 'r'  # Normalización por fila
    
    print(f"Matriz creada: {w.n} observaciones")
    print(f"Vecinos promedio: {w.mean_neighbors:.2f}")
    
    return w, gdf_valid


def analisis_moran_global(gdf, w, variable='indice_satisfaccion'):
    """Calcula el estadístico Moran's I global."""
    print(f"\n MORAN'S I GLOBAL - Variable: {variable}")
    print("-" * 60)
    
    # Calcular Moran's I
    y = gdf[variable].values
    moran = Moran(y, w, permutations=999)
    
    resultado = {
        'I': round(float(moran.I), 4),
        'valor_esperado': round(float(moran.EI), 4),
        'varianza': round(float(moran.VI_norm), 6),
        'z_score': round(float(moran.z_norm), 4),
        'p_value': round(float(moran.p_norm), 4),
        'significativo': bool(moran.p_norm < 0.05)
    }
    
    # Interpretar resultado
    if resultado['significativo']:
        if resultado['I'] > 0:
            resultado['tipo_autocorrelacion'] = 'positiva'
            interpretacion = "✓ AUTOCORRELACIÓN POSITIVA SIGNIFICATIVA"
            descripcion = "Las propiedades con características similares tienden a agruparse espacialmente"
        else:
            resultado['tipo_autocorrelacion'] = 'negativa'
            interpretacion = "✓ AUTOCORRELACIÓN NEGATIVA SIGNIFICATIVA"
            descripcion = "Las propiedades con características diferentes tienden a estar cerca"
    else:
        resultado['tipo_autocorrelacion'] = 'ninguna'
        interpretacion = "✗ NO HAY AUTOCORRELACIÓN SIGNIFICATIVA"
        descripcion = "La distribución espacial es aleatoria"
    
    print(f"\n   Moran's I: {resultado['I']}")
    print(f"   Valor esperado: {resultado['valor_esperado']}")
    print(f"   Z-score: {resultado['z_score']}")
    print(f"   P-value: {resultado['p_value']}")
    print(f"\n   {interpretacion}")
    print(f"   → {descripcion}")
    
    return resultado, moran


def analisis_lisa_local(gdf, w, variable='indice_satisfaccion', alpha=0.05):
    """Calcula LISA (Local Moran's I) para identificar clústeres locales."""
    print(f"\n ANÁLISIS LISA LOCAL - Variable: {variable}")
    print("-" * 60)
    
    y = gdf[variable].values
    lisa = Moran_Local(y, w, permutations=999)
    
    # Asignar tipo de clúster
    sig = lisa.p_sim < alpha
    hotspot = (lisa.q == 1) & sig  # High-High
    coldspot = (lisa.q == 3) & sig  # Low-Low
    doughnut = (lisa.q == 2) & sig  # Low-High
    diamond = (lisa.q == 4) & sig  # High-Low
    
    # Crear columna de clústeres
    gdf['lisa_cluster'] = 'No Significativo'
    gdf.loc[hotspot, 'lisa_cluster'] = 'High-High'
    gdf.loc[coldspot, 'lisa_cluster'] = 'Low-Low'
    gdf.loc[doughnut, 'lisa_cluster'] = 'Low-High'
    gdf.loc[diamond, 'lisa_cluster'] = 'High-Low'
    
    # Agregar valores LISA
    gdf['lisa_I'] = lisa.Is
    gdf['lisa_p_value'] = lisa.p_sim
    gdf['lisa_q'] = lisa.q
    
    # Estadísticas de clústeres
    cluster_counts = gdf['lisa_cluster'].value_counts()
    
    print("\n   DISTRIBUCIÓN DE CLÚSTERES:")
    for cluster, count in cluster_counts.items():
        pct = count / len(gdf) * 100
        print(f"      {cluster}: {count} propiedades ({pct:.1f}%)")
    
    return gdf, lisa


def analisis_por_comuna(gdf, comunas):
    """Analiza la distribución de clústeres LISA por comuna."""
    print("\n ANÁLISIS DE CLÚSTERES POR COMUNA")
    print("-" * 60)
    
    # Asegurar que tienen el mismo CRS
    if gdf.crs != comunas.crs:
        gdf = gdf.to_crs(comunas.crs)
    
    # Spatial join para asignar comuna
    gdf_con_comuna = gpd.sjoin(gdf, comunas[['comuna', 'geometry']], 
                                how='left', predicate='within')
    
    # Crear resumen por comuna
    resumen_comunas = {}
    
    for comuna in comunas['comuna'].unique():
        props_comuna = gdf_con_comuna[gdf_con_comuna['comuna'] == comuna]
        
        if len(props_comuna) == 0:
            continue
            
        cluster_dist = props_comuna['lisa_cluster'].value_counts().to_dict()
        
        resumen_comunas[comuna] = {
            'total_propiedades': len(props_comuna),
            'distribucion_clusters': cluster_dist,
            'indice_satisfaccion_promedio': round(props_comuna['indice_satisfaccion'].mean(), 4),
            'indice_satisfaccion_std': round(props_comuna['indice_satisfaccion'].std(), 4),
            'pct_high_high': round(cluster_dist.get('High-High', 0) / len(props_comuna) * 100, 2),
            'pct_low_low': round(cluster_dist.get('Low-Low', 0) / len(props_comuna) * 100, 2),
            'idx_accesibilidad_prom': round(props_comuna['idx_accesibilidad'].mean(), 4),
            'idx_densidad_prom': round(props_comuna['idx_densidad'].mean(), 4),
            'idx_diversidad_prom': round(props_comuna['idx_diversidad'].mean(), 4)
        }
        
        print(f"\n    {comuna}:")
        print(f"      • Total propiedades: {resumen_comunas[comuna]['total_propiedades']}")
        print(f"      • Satisfacción promedio: {resumen_comunas[comuna]['indice_satisfaccion_promedio']:.4f}")
        print(f"      • % Hot Spots (High-High): {resumen_comunas[comuna]['pct_high_high']:.1f}%")
        print(f"      • % Cold Spots (Low-Low): {resumen_comunas[comuna]['pct_low_low']:.1f}%")
    
    return gdf_con_comuna, resumen_comunas


def generar_mapa_lisa(gdf, comunas, variable='indice_satisfaccion'):
    """Genera mapa de clústeres LISA."""
    print("\n GENERANDO MAPAS DE CLÚSTERES LISA...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # --- Mapa 1: Índice de Satisfacción ---
    ax1 = axes[0]
    
    # Plotear comunas base
    comunas.plot(ax=ax1, color='lightgray', edgecolor='black', linewidth=0.5)
    
    # Plotear propiedades coloreadas por satisfacción
    scatter1 = gdf.plot(column=variable, ax=ax1, cmap='RdYlGn', 
                        markersize=8, alpha=0.7, legend=True,
                        legend_kwds={'label': 'Índice de Satisfacción', 
                                    'orientation': 'horizontal',
                                    'shrink': 0.6, 'pad': 0.05})
    
    # Añadir nombres de comunas
    for idx, row in comunas.iterrows():
        centroid = row['geometry'].centroid
        ax1.annotate(row['comuna'], xy=(centroid.x, centroid.y),
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.8, edgecolor='gray'))
    
    ax1.set_title('Distribución del Índice de Satisfacción\npor Propiedad', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Coordenada Este (m)')
    ax1.set_ylabel('Coordenada Norte (m)')
    ax1.set_aspect('equal')
    
    # --- Mapa 2: Clústeres LISA ---
    ax2 = axes[1]
    
    # Plotear comunas base
    comunas.plot(ax=ax2, color='lightgray', edgecolor='black', linewidth=0.5)
    
    # Ordenar para plotear en orden correcto
    cluster_order = ['High-High', 'Low-Low', 'High-Low', 'Low-High', 'No Significativo']
    
    for cluster in cluster_order:
        subset = gdf[gdf['lisa_cluster'] == cluster]
        if len(subset) > 0:
            color = COLORES_LISA[cluster]
            subset.plot(ax=ax2, color=color, markersize=10, alpha=0.7, label=cluster)
    
    # Añadir nombres de comunas
    for idx, row in comunas.iterrows():
        centroid = row['geometry'].centroid
        ax2.annotate(row['comuna'], xy=(centroid.x, centroid.y),
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.8, edgecolor='gray'))
    
    ax2.set_title('Clústeres LISA de Satisfacción Residencial\n(Autocorrelación Espacial Local)', 
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Coordenada Este (m)')
    ax2.set_ylabel('Coordenada Norte (m)')
    ax2.legend(loc='lower left', fontsize=9, framealpha=0.9)
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    # Guardar
    output_path = os.path.join(GRAFICOS_DIR, 'mapa_lisa_satisfaccion.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Mapa guardado: {output_path}")
    
    plt.close()
    
    return output_path


def generar_mapa_por_comuna(gdf, comunas, resumen_comunas):
    """Genera mapa resumen por comuna con estadísticas."""
    print("\n GENERANDO MAPA RESUMEN POR COMUNAS...")
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    
    # --- Mapa 1: Satisfacción promedio por comuna ---
    ax1 = axes[0]
    
    # Agregar satisfacción promedio a comunas
    satisfaccion_por_comuna = {}
    for comuna, data in resumen_comunas.items():
        satisfaccion_por_comuna[comuna] = data['indice_satisfaccion_promedio']
    
    comunas['satisfaccion_promedio'] = comunas['comuna'].map(satisfaccion_por_comuna)
    
    comunas.plot(column='satisfaccion_promedio', ax=ax1, cmap='RdYlGn',
                 edgecolor='black', linewidth=1, legend=True,
                 legend_kwds={'label': 'Satisfacción Promedio', 
                             'orientation': 'horizontal',
                             'shrink': 0.6, 'pad': 0.05})
    
    # Añadir nombres y valores
    for idx, row in comunas.iterrows():
        centroid = row['geometry'].centroid
        sat = row.get('satisfaccion_promedio', 0)
        ax1.annotate(f"{row['comuna']}\n{sat:.3f}", xy=(centroid.x, centroid.y),
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.9, edgecolor='gray'))
    
    ax1.set_title('Índice de Satisfacción Promedio\npor Comuna', 
                  fontsize=13, fontweight='bold')
    ax1.set_xlabel('Coordenada Este (m)')
    ax1.set_ylabel('Coordenada Norte (m)')
    ax1.set_aspect('equal')
    
    # --- Mapa 2: Porcentaje de Hot Spots por comuna ---
    ax2 = axes[1]
    
    pct_hotspots = {}
    for comuna, data in resumen_comunas.items():
        pct_hotspots[comuna] = data['pct_high_high']
    
    comunas['pct_hotspots'] = comunas['comuna'].map(pct_hotspots)
    
    comunas.plot(column='pct_hotspots', ax=ax2, cmap='Reds',
                 edgecolor='black', linewidth=1, legend=True,
                 legend_kwds={'label': '% Hot Spots (High-High)', 
                             'orientation': 'horizontal',
                             'shrink': 0.6, 'pad': 0.05})
    
    # Añadir nombres y valores
    for idx, row in comunas.iterrows():
        centroid = row['geometry'].centroid
        pct = row.get('pct_hotspots', 0)
        ax2.annotate(f"{row['comuna']}\n{pct:.1f}%", xy=(centroid.x, centroid.y),
                    ha='center', va='center', fontsize=10, fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                            alpha=0.9, edgecolor='gray'))
    
    ax2.set_title('Concentración de Hot Spots (High-High)\npor Comuna', 
                  fontsize=13, fontweight='bold')
    ax2.set_xlabel('Coordenada Este (m)')
    ax2.set_ylabel('Coordenada Norte (m)')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    
    # Guardar
    output_path = os.path.join(GRAFICOS_DIR, 'mapa_lisa_por_comunas.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Mapa guardado: {output_path}")
    
    plt.close()
    
    return output_path


def generar_graficos_estadisticos(gdf, resumen_comunas):
    """Genera gráficos estadísticos del análisis LISA."""
    print("\nGENERANDO GRÁFICOS ESTADÍSTICOS...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # --- Gráfico 1: Distribución global de clústeres ---
    ax1 = axes[0, 0]
    cluster_counts = gdf['lisa_cluster'].value_counts()
    colors = [COLORES_LISA[c] for c in cluster_counts.index]
    bars1 = ax1.bar(cluster_counts.index, cluster_counts.values, color=colors, edgecolor='black')
    ax1.set_title('Distribución Global de Clústeres LISA', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Tipo de Clúster')
    ax1.set_ylabel('Número de Propiedades')
    ax1.tick_params(axis='x', rotation=45)
    
    # Añadir valores
    for bar, val in zip(bars1, cluster_counts.values):
        ax1.annotate(f'{val}\n({val/len(gdf)*100:.1f}%)', 
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9)
    
    # --- Gráfico 2: Satisfacción promedio por comuna ---
    ax2 = axes[0, 1]
    comunas_list = list(resumen_comunas.keys())
    satisfaccion = [resumen_comunas[c]['indice_satisfaccion_promedio'] for c in comunas_list]
    colores_comunas = [COLORES_COMUNAS.get(c, 'gray') for c in comunas_list]
    
    bars2 = ax2.bar(comunas_list, satisfaccion, color=colores_comunas, edgecolor='black')
    ax2.axhline(y=np.mean(satisfaccion), color='red', linestyle='--', 
                label=f'Promedio: {np.mean(satisfaccion):.3f}')
    ax2.set_title('Índice de Satisfacción Promedio por Comuna', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Comuna')
    ax2.set_ylabel('Índice de Satisfacción')
    ax2.legend()
    ax2.tick_params(axis='x', rotation=45)
    
    for bar, val in zip(bars2, satisfaccion):
        ax2.annotate(f'{val:.3f}', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # --- Gráfico 3: Hot Spots y Cold Spots por comuna ---
    ax3 = axes[1, 0]
    pct_hh = [resumen_comunas[c]['pct_high_high'] for c in comunas_list]
    pct_ll = [resumen_comunas[c]['pct_low_low'] for c in comunas_list]
    
    x = np.arange(len(comunas_list))
    width = 0.35
    
    bars_hh = ax3.bar(x - width/2, pct_hh, width, label='Hot Spots (High-High)', 
                      color=COLORES_LISA['High-High'], edgecolor='black')
    bars_ll = ax3.bar(x + width/2, pct_ll, width, label='Cold Spots (Low-Low)', 
                      color=COLORES_LISA['Low-Low'], edgecolor='black')
    
    ax3.set_title('Concentración de Clústeres Significativos por Comuna', 
                  fontsize=12, fontweight='bold')
    ax3.set_xlabel('Comuna')
    ax3.set_ylabel('Porcentaje de Propiedades (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(comunas_list, rotation=45)
    ax3.legend()
    
    # --- Gráfico 4: Componentes del índice por comuna ---
    ax4 = axes[1, 1]
    
    idx_acc = [resumen_comunas[c]['idx_accesibilidad_prom'] for c in comunas_list]
    idx_dens = [resumen_comunas[c]['idx_densidad_prom'] for c in comunas_list]
    idx_div = [resumen_comunas[c]['idx_diversidad_prom'] for c in comunas_list]
    
    x = np.arange(len(comunas_list))
    width = 0.25
    
    ax4.bar(x - width, idx_acc, width, label='Accesibilidad', color='#2ecc71', edgecolor='black')
    ax4.bar(x, idx_dens, width, label='Densidad Servicios', color='#3498db', edgecolor='black')
    ax4.bar(x + width, idx_div, width, label='Diversidad', color='#9b59b6', edgecolor='black')
    
    ax4.set_title('Componentes del Índice de Satisfacción por Comuna', 
                  fontsize=12, fontweight='bold')
    ax4.set_xlabel('Comuna')
    ax4.set_ylabel('Valor del Subíndice (0-1)')
    ax4.set_xticks(x)
    ax4.set_xticklabels(comunas_list, rotation=45)
    ax4.legend()
    
    plt.tight_layout()
    
    # Guardar
    output_path = os.path.join(GRAFICOS_DIR, 'estadisticas_lisa_satisfaccion.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path.replace('.png', '.pdf'), bbox_inches='tight')
    print(f"Gráficos guardados: {output_path}")
    
    plt.close()
    
    return output_path


def generar_reporte(gdf, moran_global, resumen_comunas):
    """Genera reporte JSON y Markdown del análisis."""
    print("\n GENERANDO REPORTES...")
    
    # Distribución de clústeres
    cluster_dist = gdf['lisa_cluster'].value_counts().to_dict()
    cluster_dist_pct = {k: round(v/len(gdf)*100, 2) for k, v in cluster_dist.items()}
    
    reporte = {
        'fecha_analisis': datetime.now().isoformat(),
        'pregunta_investigacion': '¿Los factores de satisfacción están correlacionados espacialmente con zonas específicas (comunas)?',
        'metodologia': {
            'nombre': 'LISA (Local Indicators of Spatial Association)',
            'estadistico': "Local Moran's I",
            'matriz_pesos': 'K-Nearest Neighbors (k=8)',
            'permutaciones': 999,
            'alpha': 0.05,
            'indice_satisfaccion': {
                'descripcion': 'Índice compuesto de satisfacción residencial',
                'componentes': {
                    'accesibilidad': '40% - Distancias mínimas a servicios (invertido)',
                    'densidad': '35% - Densidad de servicios en 1000m',
                    'diversidad': '25% - Diversidad de servicios disponibles'
                }
            }
        },
        'moran_global': moran_global,
        'total_propiedades': len(gdf),
        'distribucion_clusters': {
            'conteos': cluster_dist,
            'porcentajes': cluster_dist_pct
        },
        'analisis_por_comuna': resumen_comunas,
        'conclusiones': {
            'autocorrelacion_detectada': moran_global['significativo'],
            'tipo': moran_global['tipo_autocorrelacion'],
            'interpretacion': []
        }
    }
    
    # Generar conclusiones
    if moran_global['significativo'] and moran_global['I'] > 0:
        reporte['conclusiones']['interpretacion'] = [
            "✓ EXISTE AUTOCORRELACIÓN ESPACIAL POSITIVA SIGNIFICATIVA",
            "→ Los factores de satisfacción SÍ están ligados a zonas específicas",
            "→ Las propiedades con alta satisfacción tienden a agruparse (Hot Spots)",
            "→ Las propiedades con baja satisfacción también se agrupan (Cold Spots)",
            "→ La ubicación geográfica es un determinante clave de la satisfacción"
        ]
        
        # Identificar comuna con más hot spots
        max_hh = max(resumen_comunas.items(), key=lambda x: x[1]['pct_high_high'])
        max_ll = max(resumen_comunas.items(), key=lambda x: x[1]['pct_low_low'])
        
        reporte['conclusiones']['zonas_premium'] = {
            'comuna': max_hh[0],
            'pct_hotspots': max_hh[1]['pct_high_high'],
            'descripcion': f"{max_hh[0]} tiene la mayor concentración de Hot Spots ({max_hh[1]['pct_high_high']:.1f}%)"
        }
        
        reporte['conclusiones']['zonas_desarrollo'] = {
            'comuna': max_ll[0],
            'pct_coldspots': max_ll[1]['pct_low_low'],
            'descripcion': f"{max_ll[0]} tiene mayor potencial de desarrollo ({max_ll[1]['pct_low_low']:.1f}% Cold Spots)"
        }
    
    # Guardar JSON
    json_path = os.path.join(OUTPUT_DIR, 'analisis_lisa_satisfaccion.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    print(f"Reporte JSON: {json_path}")
    
    # Generar Markdown
    md_content = f"""# Análisis LISA - Factores de Satisfacción Residencial

## Pregunta de Investigación

**¿Los factores que harían que una propiedad sea más satisfactoria están ligados a una zona/comuna en particular?**

---

## Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Moran's I Global | {moran_global['I']:.4f} |
| Z-score | {moran_global['z_score']:.4f} |
| P-value | {moran_global['p_value']:.4f} |
| Autocorrelación | {'**SIGNIFICATIVA**' if moran_global['significativo'] else 'No significativa'} |
| Tipo | {moran_global['tipo_autocorrelacion'].capitalize()} |

---

## Respuesta a la Pregunta de Investigación

"""
    
    if moran_global['significativo'] and moran_global['I'] > 0:
        md_content += """### SÍ, existe una correlación espacial significativa

Los factores de satisfacción **están fuertemente ligados a zonas geográficas específicas**. Esto significa que:

1. **Las propiedades con alta satisfacción se agrupan** en ciertas zonas (Hot Spots)
2. **Las propiedades con baja satisfacción también se agrupan** en otras zonas (Cold Spots)
3. **La ubicación es un determinante clave** de la satisfacción potencial

"""
    
    md_content += """---

## Distribución de Clústeres LISA

| Clúster | Propiedades | Porcentaje | Interpretación |
|---------|-------------|------------|----------------|
"""
    
    for cluster, count in cluster_dist.items():
        pct = cluster_dist_pct[cluster]
        interp = {
            'High-High': 'Hot Spots - Zonas premium',
            'Low-Low': 'Cold Spots - Zonas con potencial',
            'High-Low': 'Outliers positivos',
            'Low-High': 'Outliers negativos',
            'No Significativo': 'Sin patrón definido'
        }.get(cluster, '')
        md_content += f"| {cluster} | {count} | {pct}% | {interp} |\n"
    
    md_content += """
---

## Análisis por Comuna

"""
    
    for comuna, data in sorted(resumen_comunas.items(), 
                                key=lambda x: x[1]['indice_satisfaccion_promedio'], 
                                reverse=True):
        md_content += f"""### {comuna}

| Métrica | Valor |
|---------|-------|
| Total propiedades | {data['total_propiedades']} |
| Satisfacción promedio | {data['indice_satisfaccion_promedio']:.4f} |
| % Hot Spots (High-High) | {data['pct_high_high']:.1f}% |
| % Cold Spots (Low-Low) | {data['pct_low_low']:.1f}% |
| Índice Accesibilidad | {data['idx_accesibilidad_prom']:.4f} |
| Índice Densidad | {data['idx_densidad_prom']:.4f} |
| Índice Diversidad | {data['idx_diversidad_prom']:.4f} |

"""
    
    md_content += """---

## Implicaciones para la Recomendación de Propiedades

1. **Usuarios que priorizan servicios cercanos**: Recomendar propiedades en zonas High-High (Hot Spots)
2. **Usuarios con presupuesto limitado**: Considerar zonas Low-Low con potencial de desarrollo
3. **La comuna es un proxy de satisfacción**: Santiago y Ñuñoa tienden a tener mejores indicadores

---

*Análisis generado automáticamente - Proyecto GeoInformática*
*Fecha: """ + datetime.now().strftime('%Y-%m-%d %H:%M') + "*"
    
    md_path = os.path.join(OUTPUT_DIR, 'ANALISIS_LISA_SATISFACCION.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    print(f"Reporte Markdown: {md_path}")
    
    return reporte


def guardar_geodataframe(gdf):
    """Guarda el GeoDataFrame con los resultados LISA."""
    output_path = os.path.join(OUTPUT_DIR, 'propiedades_con_lisa_satisfaccion.geojson')
    gdf.to_file(output_path, driver='GeoJSON')
    print(f"GeoDataFrame guardado: {output_path}")
    return output_path


def main():
    """Ejecuta el análisis LISA completo."""
    try:
        # 1. Cargar datos
        gdf, comunas = cargar_datos()
        
        # 2. Calcular índice de satisfacción
        gdf = calcular_indice_satisfaccion(gdf)
        
        # 3. Crear matriz de pesos espaciales
        w, gdf = crear_matriz_pesos(gdf, k=8)
        
        # 4. Análisis Moran Global
        moran_resultado, moran_obj = analisis_moran_global(gdf, w, 'indice_satisfaccion')
        
        # 5. Análisis LISA Local
        gdf, lisa = analisis_lisa_local(gdf, w, 'indice_satisfaccion')
        
        # 6. Análisis por comuna
        gdf, resumen_comunas = analisis_por_comuna(gdf, comunas)
        
        # 7. Generar visualizaciones
        generar_mapa_lisa(gdf, comunas)
        generar_mapa_por_comuna(gdf, comunas, resumen_comunas)
        generar_graficos_estadisticos(gdf, resumen_comunas)
        
        # 8. Generar reportes
        reporte = generar_reporte(gdf, moran_resultado, resumen_comunas)
        
        # 9. Guardar resultados
        guardar_geodataframe(gdf)
        
        # Resumen final
        print("\n" + "=" * 80)
        print("ANÁLISIS LISA COMPLETADO")
        print("=" * 80)
        
        print("\nRESPUESTA A LA PREGUNTA DE INVESTIGACIÓN:")
        print("-" * 60)
        
        if moran_resultado['significativo'] and moran_resultado['I'] > 0:
            print("\nSÍ, LOS FACTORES DE SATISFACCIÓN ESTÁN LIGADOS A ZONAS ESPECÍFICAS")
            print(f"\n   Moran's I = {moran_resultado['I']:.4f} (p-value = {moran_resultado['p_value']:.4f})")
            print("\n   Esto significa que:")
            print("   → Las propiedades satisfactorias tienden a agruparse geográficamente")
            print("   → Elegir una comuna específica influye en la satisfacción esperada")
            print("   → Las zonas Hot Spots concentran alta satisfacción")
            
            # Mostrar ranking de comunas
            print("\nRANKING DE COMUNAS POR SATISFACCIÓN:")
            ranking = sorted(resumen_comunas.items(), 
                           key=lambda x: x[1]['indice_satisfaccion_promedio'], 
                           reverse=True)
            for i, (comuna, data) in enumerate(ranking, 1):
                print(f"      {i}. {comuna}: {data['indice_satisfaccion_promedio']:.4f} "
                      f"(Hot Spots: {data['pct_high_high']:.1f}%)")
        
        print("\nArchivos generados:")
        print(f"   • {GRAFICOS_DIR}/mapa_lisa_satisfaccion.png")
        print(f"   • {GRAFICOS_DIR}/mapa_lisa_por_comunas.png")
        print(f"   • {GRAFICOS_DIR}/estadisticas_lisa_satisfaccion.png")
        print(f"   • {OUTPUT_DIR}/analisis_lisa_satisfaccion.json")
        print(f"   • {OUTPUT_DIR}/ANALISIS_LISA_SATISFACCION.md")
        print(f"   • {OUTPUT_DIR}/propiedades_con_lisa_satisfaccion.geojson")
        
        return gdf, reporte
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


if __name__ == "__main__":
    gdf, reporte = main()
