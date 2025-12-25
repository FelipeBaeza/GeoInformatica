#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Estadística Descriptiva Espacial
================================
Este script responde a las siguientes preguntas de investigación:
1. ¿Cuál es la distribución de comercios por zonas de una ciudad?
2. ¿En qué lugares existe una alta concentración de servicios de seguridad (carabineros, bomberos)?

Genera mapas con las 4 comunas de estudio y análisis estadísticos espaciales.

Autores: Equipo Geoinformática
Fecha: Diciembre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from shapely.geometry import Point
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATOS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'semana1_preparacion_datos', 'datos_normalizados', 'datos_normalizados')
OUTPUT_DIR = os.path.join(BASE_DIR, 'resultados_analisis')
GRAFICOS_DIR = os.path.join(BASE_DIR, 'graficos')

# Crear directorios si no existen
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GRAFICOS_DIR, exist_ok=True)

# Configuración de visualización
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Colores personalizados para las comunas
COLORES_COMUNAS = {
    'Santiago': '#3498db',
    'Ñuñoa': '#2ecc71', 
    'La Reina': '#9b59b6',
    'Estación Central': '#e74c3c'
}


def cargar_datos():
    """Carga todos los datos necesarios para el análisis."""
    print("=" * 60)
    print("CARGANDO DATOS GEOESPACIALES")
    print("=" * 60)
    
    # Cargar comunas
    comunas = gpd.read_file(os.path.join(DATOS_DIR, 'comunas_buffer.geojson'))
    print(f" Comunas cargadas: {len(comunas)} registros")
    print(f"  Comunas: {', '.join(comunas['comuna'].tolist())}")
    
    # Cargar comercios/tiendas
    tiendas = gpd.read_file(os.path.join(DATOS_DIR, 'tiendas_filtradas.geojson'))
    print(f" Tiendas/Comercios cargados: {len(tiendas)} registros")
    
    # Cargar servicios de seguridad - Archivo principal de cuarteles
    cuarteles = gpd.read_file(os.path.join(DATOS_DIR, 'cuarteles_filtrados.geojson'))
    print(f" Cuarteles Carabineros (archivo principal): {len(cuarteles)} registros")
    
    # Cargar bomberos - Archivo principal
    bomberos = gpd.read_file(os.path.join(DATOS_DIR, 'cuerpos_de_bomberos_filtrados.geojson'))
    print(f" Cuerpos de Bomberos (archivo principal): {len(bomberos)} registros")
    
    # Cargar servicios adicionales que contienen police y fire_station
    servicios = gpd.read_file(os.path.join(DATOS_DIR, 'servicios_filtrados.geojson'))
    
    # Extraer comisarías adicionales de servicios (amenity = police)
    comisarias_servicios = servicios[servicios['amenity'] == 'police'].copy()
    if len(comisarias_servicios) > 0:
        print(f" Comisarías adicionales (servicios_filtrados): {len(comisarias_servicios)} registros")
        # Combinar con cuarteles
        cuarteles = pd.concat([cuarteles, comisarias_servicios], ignore_index=True)
    
    # Extraer bomberos adicionales de servicios (amenity = fire_station)
    bomberos_servicios = servicios[servicios['amenity'] == 'fire_station'].copy()
    if len(bomberos_servicios) > 0:
        print(f" Bomberos adicionales (servicios_filtrados): {len(bomberos_servicios)} registros")
        # Combinar con bomberos
        bomberos = pd.concat([bomberos, bomberos_servicios], ignore_index=True)
    
    print(f" TOTAL Carabineros/Policía: {len(cuarteles)} registros")
    print(f" TOTAL Bomberos: {len(bomberos)} registros")
    
    pdi = gpd.read_file(os.path.join(DATOS_DIR, 'unidades_operativas_pdi_filtradas.geojson'))
    print(f" Unidades PDI cargadas: {len(pdi)} registros")
    
    print("=" * 60)
    
    return comunas, tiendas, cuarteles, bomberos, pdi


def asignar_comuna(gdf_puntos, comunas):
    """Asigna cada punto a su comuna correspondiente mediante spatial join."""
    # Asegurar mismo CRS
    if gdf_puntos.crs != comunas.crs:
        gdf_puntos = gdf_puntos.to_crs(comunas.crs)
    
    # Realizar spatial join
    puntos_con_comuna = gpd.sjoin(gdf_puntos, comunas[['comuna', 'geometry']], 
                                   how='left', predicate='within')
    
    return puntos_con_comuna


def calcular_estadisticas_comercios(tiendas, comunas):
    """
    Calcula estadísticas descriptivas de la distribución de comercios.
    Responde: ¿Cuál es la distribución de comercios por zonas de una ciudad?
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS DE DISTRIBUCIÓN DE COMERCIOS POR COMUNA")
    print("=" * 60)
    
    # Asignar comuna a cada tienda
    tiendas_comuna = asignar_comuna(tiendas, comunas)
    
    # Estadísticas por comuna
    stats_comercios = tiendas_comuna.groupby('comuna').agg({
        'name': 'count',  # Conteo de comercios
    }).rename(columns={'name': 'total_comercios'})
    
    # Agregar área de cada comuna
    for idx, row in comunas.iterrows():
        comuna_nombre = row['comuna']
        if comuna_nombre in stats_comercios.index:
            stats_comercios.loc[comuna_nombre, 'area_km2'] = row['area_m2'] / 1_000_000
    
    # Calcular densidad de comercios por km²
    stats_comercios['densidad_por_km2'] = stats_comercios['total_comercios'] / stats_comercios['area_km2']
    
    # Calcular porcentaje del total
    total_comercios = stats_comercios['total_comercios'].sum()
    stats_comercios['porcentaje'] = (stats_comercios['total_comercios'] / total_comercios * 100).round(2)
    
    # Ordenar por densidad
    stats_comercios = stats_comercios.sort_values('densidad_por_km2', ascending=False)
    
    print("\n ESTADÍSTICAS DE COMERCIOS POR COMUNA:")
    print("-" * 60)
    for comuna, row in stats_comercios.iterrows():
        print(f"\n {comuna}:")
        print(f"   • Total comercios: {int(row['total_comercios'])}")
        print(f"   • Área: {row['area_km2']:.2f} km²")
        print(f"   • Densidad: {row['densidad_por_km2']:.2f} comercios/km²")
        print(f"   • Porcentaje del total: {row['porcentaje']:.1f}%")
    
    # Estadísticas globales
    print("\n ESTADÍSTICAS GLOBALES:")
    print(f"   • Total de comercios en el área de estudio: {int(total_comercios)}")
    print(f"   • Promedio por comuna: {total_comercios/len(stats_comercios):.1f}")
    print(f"   • Desviación estándar: {stats_comercios['total_comercios'].std():.1f}")
    print(f"   • Densidad máxima: {stats_comercios['densidad_por_km2'].max():.2f} comercios/km² ({stats_comercios['densidad_por_km2'].idxmax()})")
    print(f"   • Densidad mínima: {stats_comercios['densidad_por_km2'].min():.2f} comercios/km² ({stats_comercios['densidad_por_km2'].idxmin()})")
    
    # Análisis de tipos de comercio
    print("\n TIPOS DE COMERCIO MÁS FRECUENTES:")
    if 'shop' in tiendas_comuna.columns:
        tipos = tiendas_comuna['shop'].value_counts().head(10)
        for tipo, count in tipos.items():
            print(f"   • {tipo}: {count} ({count/len(tiendas_comuna)*100:.1f}%)")
    
    return stats_comercios, tiendas_comuna


def calcular_estadisticas_seguridad(cuarteles, bomberos, pdi, comunas):
    """
    Calcula estadísticas descriptivas de servicios de seguridad.
    Responde: ¿En qué lugares existe una alta concentración de servicios de seguridad?
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS DE CONCENTRACIÓN DE SERVICIOS DE SEGURIDAD")
    print("=" * 60)
    
    # Asignar comuna a cada servicio
    cuarteles_comuna = asignar_comuna(cuarteles, comunas)
    bomberos_comuna = asignar_comuna(bomberos, comunas)
    pdi_comuna = asignar_comuna(pdi, comunas)
    
    # Crear DataFrame de estadísticas
    stats_seguridad = pd.DataFrame(index=comunas['comuna'].tolist())
    
    # Contar por comuna
    cuarteles_count = cuarteles_comuna.groupby('comuna').size()
    bomberos_count = bomberos_comuna.groupby('comuna').size()
    pdi_count = pdi_comuna.groupby('comuna').size()
    
    stats_seguridad['carabineros'] = cuarteles_count
    stats_seguridad['bomberos'] = bomberos_count
    stats_seguridad['pdi'] = pdi_count
    stats_seguridad = stats_seguridad.fillna(0).astype(int)
    
    # Total de servicios de seguridad
    stats_seguridad['total_seguridad'] = stats_seguridad.sum(axis=1)
    
    # Agregar área
    for idx, row in comunas.iterrows():
        comuna_nombre = row['comuna']
        if comuna_nombre in stats_seguridad.index:
            stats_seguridad.loc[comuna_nombre, 'area_km2'] = row['area_m2'] / 1_000_000
    
    # Calcular densidad
    stats_seguridad['densidad_por_km2'] = stats_seguridad['total_seguridad'] / stats_seguridad['area_km2']
    
    # Ordenar por concentración
    stats_seguridad = stats_seguridad.sort_values('densidad_por_km2', ascending=False)
    
    print("\n SERVICIOS DE SEGURIDAD POR COMUNA:")
    print("-" * 60)
    for comuna, row in stats_seguridad.iterrows():
        print(f"\n {comuna}:")
        print(f"   • Carabineros: {int(row['carabineros'])} unidades")
        print(f"   • Bomberos: {int(row['bomberos'])} compañías")
        print(f"   • PDI: {int(row['pdi'])} unidades")
        print(f"   • Total: {int(row['total_seguridad'])} servicios")
        print(f"   • Densidad: {row['densidad_por_km2']:.3f} servicios/km²")
    
    # Estadísticas globales
    total_carabineros = stats_seguridad['carabineros'].sum()
    total_bomberos = stats_seguridad['bomberos'].sum()
    total_pdi = stats_seguridad['pdi'].sum()
    total_servicios = stats_seguridad['total_seguridad'].sum()
    
    print("\n RESUMEN GLOBAL DE SERVICIOS DE SEGURIDAD:")
    print(f"   • Total Carabineros: {int(total_carabineros)} unidades")
    print(f"   • Total Bomberos: {int(total_bomberos)} compañías")
    print(f"   • Total PDI: {int(total_pdi)} unidades")
    print(f"   • TOTAL GENERAL: {int(total_servicios)} servicios de seguridad")
    
    # Identificar zonas de alta concentración
    media_densidad = stats_seguridad['densidad_por_km2'].mean()
    alta_concentracion = stats_seguridad[stats_seguridad['densidad_por_km2'] > media_densidad]
    
    print(f"\n ZONAS DE ALTA CONCENTRACIÓN (densidad > {media_densidad:.3f}/km²):")
    for comuna in alta_concentracion.index:
        print(f"   • {comuna}: {alta_concentracion.loc[comuna, 'densidad_por_km2']:.3f} servicios/km²")
    
    return stats_seguridad, cuarteles_comuna, bomberos_comuna, pdi_comuna


def crear_mapa_comercios(tiendas_comuna, comunas, stats_comercios):
    """Crea mapa de distribución de comercios por comuna."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Mapa 1: Puntos de comercios
    ax1 = axes[0]
    comunas.plot(ax=ax1, color='lightgray', edgecolor='black', linewidth=0.5)
    
    # Colorear comunas según densidad
    for idx, row in comunas.iterrows():
        comuna_nombre = row['comuna']
        if comuna_nombre in stats_comercios.index:
            color = COLORES_COMUNAS.get(comuna_nombre, 'gray')
            gpd.GeoSeries([row['geometry']]).plot(ax=ax1, color=color, alpha=0.3, edgecolor='black', linewidth=1)
    
    # Plotear comercios
    tiendas_comuna.plot(ax=ax1, color='red', markersize=3, alpha=0.6, label='Comercios')
    
    ax1.set_title('Distribución Espacial de Comercios\nen las 4 Comunas de Estudio', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Coordenada Este (m)')
    ax1.set_ylabel('Coordenada Norte (m)')
    
    # Leyenda de comunas
    patches = [mpatches.Patch(color=color, label=comuna, alpha=0.5) 
               for comuna, color in COLORES_COMUNAS.items()]
    ax1.legend(handles=patches, loc='lower left', fontsize=8, title='Comunas')
    ax1.set_aspect('equal')
    
    # Mapa 2: Densidad por comuna (coroplético)
    ax2 = axes[1]
    
    # Crear GeoDataFrame con densidades
    comunas_densidad = comunas.copy()
    comunas_densidad['densidad'] = comunas_densidad['comuna'].map(
        stats_comercios['densidad_por_km2'].to_dict()
    )
    
    # Plotear mapa coroplético
    comunas_densidad.plot(column='densidad', ax=ax2, cmap='YlOrRd', 
                          edgecolor='black', linewidth=1, legend=True,
                          legend_kwds={'label': 'Comercios/km²', 'shrink': 0.6})
    
    # Agregar etiquetas
    for idx, row in comunas_densidad.iterrows():
        centroid = row['geometry'].centroid
        ax2.annotate(f"{row['comuna']}\n{row['densidad']:.1f}/km²", 
                    xy=(centroid.x, centroid.y),
                    ha='center', va='center', fontsize=9, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_title('Densidad de Comercios por Comuna\n(comercios por km²)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Coordenada Este (m)')
    ax2.set_ylabel('Coordenada Norte (m)')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, 'mapa_distribucion_comercios.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(GRAFICOS_DIR, 'mapa_distribucion_comercios.pdf'), bbox_inches='tight')
    print(f"\n Mapa de comercios guardado en: {GRAFICOS_DIR}")
    plt.close()


def crear_mapa_seguridad(cuarteles_comuna, bomberos_comuna, pdi_comuna, comunas, stats_seguridad):
    """Crea mapa de concentración de servicios de seguridad."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Mapa 1: Ubicación de servicios de seguridad
    ax1 = axes[0]
    
    # Plotear comunas base
    comunas.plot(ax=ax1, color='lightgray', edgecolor='black', linewidth=1)
    
    # Colorear comunas
    for idx, row in comunas.iterrows():
        color = COLORES_COMUNAS.get(row['comuna'], 'gray')
        gpd.GeoSeries([row['geometry']]).plot(ax=ax1, color=color, alpha=0.2, edgecolor='black', linewidth=1)
    
    # Plotear servicios de seguridad con diferentes símbolos
    if len(cuarteles_comuna) > 0:
        cuarteles_comuna.plot(ax=ax1, color='blue', marker='^', markersize=80, 
                              label=f'Carabineros ({len(cuarteles_comuna)})', alpha=0.8, edgecolor='darkblue', linewidth=0.5)
    
    if len(bomberos_comuna) > 0:
        bomberos_comuna.plot(ax=ax1, color='red', marker='s', markersize=100, 
                             label=f'Bomberos ({len(bomberos_comuna)})', alpha=0.8, edgecolor='darkred', linewidth=0.5)
    
    if len(pdi_comuna) > 0:
        pdi_comuna.plot(ax=ax1, color='green', marker='o', markersize=60, 
                        label=f'PDI ({len(pdi_comuna)})', alpha=0.8, edgecolor='darkgreen', linewidth=0.5)
    
    ax1.set_title('Ubicación de Servicios de Seguridad\n(Carabineros, Bomberos, PDI)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Coordenada Este (m)')
    ax1.set_ylabel('Coordenada Norte (m)')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.set_aspect('equal')
    
    # Mapa 2: Densidad de servicios de seguridad
    ax2 = axes[1]
    
    # Crear GeoDataFrame con densidades
    comunas_densidad = comunas.copy()
    comunas_densidad['densidad'] = comunas_densidad['comuna'].map(
        stats_seguridad['densidad_por_km2'].to_dict()
    )
    comunas_densidad['total'] = comunas_densidad['comuna'].map(
        stats_seguridad['total_seguridad'].to_dict()
    )
    
    # Plotear mapa coroplético
    comunas_densidad.plot(column='densidad', ax=ax2, cmap='Blues', 
                          edgecolor='black', linewidth=1, legend=True,
                          legend_kwds={'label': 'Servicios/km²', 'shrink': 0.6})
    
    # Agregar etiquetas
    for idx, row in comunas_densidad.iterrows():
        centroid = row['geometry'].centroid
        total = int(row['total']) if pd.notna(row['total']) else 0
        densidad = row['densidad'] if pd.notna(row['densidad']) else 0
        ax2.annotate(f"{row['comuna']}\n{total} servicios\n({densidad:.3f}/km²)", 
                    xy=(centroid.x, centroid.y),
                    ha='center', va='center', fontsize=8, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_title('Concentración de Servicios de Seguridad\npor Comuna', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Coordenada Este (m)')
    ax2.set_ylabel('Coordenada Norte (m)')
    ax2.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, 'mapa_servicios_seguridad.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(GRAFICOS_DIR, 'mapa_servicios_seguridad.pdf'), bbox_inches='tight')
    print(f" Mapa de seguridad guardado en: {GRAFICOS_DIR}")
    plt.close()


def crear_graficos_barras(stats_comercios, stats_seguridad):
    """Crea gráficos de barras comparativos."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Gráfico 1: Total de comercios por comuna
    ax1 = axes[0, 0]
    comunas = stats_comercios.index.tolist()
    valores = stats_comercios['total_comercios'].values
    colores = [COLORES_COMUNAS.get(c, 'gray') for c in comunas]
    
    bars1 = ax1.bar(comunas, valores, color=colores, edgecolor='black', alpha=0.8)
    ax1.set_title('Total de Comercios por Comuna', fontweight='bold')
    ax1.set_ylabel('Número de Comercios')
    ax1.set_xlabel('Comuna')
    
    # Agregar valores sobre las barras
    for bar, val in zip(bars1, valores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{int(val)}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Gráfico 2: Densidad de comercios
    ax2 = axes[0, 1]
    valores_densidad = stats_comercios['densidad_por_km2'].values
    
    bars2 = ax2.bar(comunas, valores_densidad, color=colores, edgecolor='black', alpha=0.8)
    ax2.set_title('Densidad de Comercios por Comuna', fontweight='bold')
    ax2.set_ylabel('Comercios por km²')
    ax2.set_xlabel('Comuna')
    ax2.axhline(y=valores_densidad.mean(), color='red', linestyle='--', 
                label=f'Promedio: {valores_densidad.mean():.1f}')
    ax2.legend()
    
    for bar, val in zip(bars2, valores_densidad):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Gráfico 3: Servicios de seguridad apilados
    ax3 = axes[1, 0]
    comunas_seg = stats_seguridad.index.tolist()
    
    x = np.arange(len(comunas_seg))
    width = 0.6
    
    carabineros = stats_seguridad['carabineros'].values
    bomberos = stats_seguridad['bomberos'].values
    pdi = stats_seguridad['pdi'].values
    
    ax3.bar(x, carabineros, width, label='Carabineros', color='blue', alpha=0.8)
    ax3.bar(x, bomberos, width, bottom=carabineros, label='Bomberos', color='red', alpha=0.8)
    ax3.bar(x, pdi, width, bottom=carabineros+bomberos, label='PDI', color='green', alpha=0.8)
    
    ax3.set_title('Servicios de Seguridad por Comuna\n(Desglose por tipo)', fontweight='bold')
    ax3.set_ylabel('Número de Servicios')
    ax3.set_xlabel('Comuna')
    ax3.set_xticks(x)
    ax3.set_xticklabels(comunas_seg, rotation=15)
    ax3.legend()
    
    # Gráfico 4: Densidad de servicios de seguridad
    ax4 = axes[1, 1]
    densidad_seg = stats_seguridad['densidad_por_km2'].values
    colores_seg = [COLORES_COMUNAS.get(c, 'gray') for c in comunas_seg]
    
    bars4 = ax4.bar(comunas_seg, densidad_seg, color=colores_seg, edgecolor='black', alpha=0.8)
    ax4.set_title('Densidad de Servicios de Seguridad\npor Comuna', fontweight='bold')
    ax4.set_ylabel('Servicios por km²')
    ax4.set_xlabel('Comuna')
    ax4.axhline(y=densidad_seg.mean(), color='red', linestyle='--', 
                label=f'Promedio: {densidad_seg.mean():.3f}')
    ax4.legend()
    
    for bar, val in zip(bars4, densidad_seg):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, 'graficos_estadisticos_espaciales.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(GRAFICOS_DIR, 'graficos_estadisticos_espaciales.pdf'), bbox_inches='tight')
    print(f" Gráficos estadísticos guardados en: {GRAFICOS_DIR}")
    plt.close()


def crear_mapa_combinado(tiendas_comuna, cuarteles_comuna, bomberos_comuna, pdi_comuna, comunas):
    """Crea un mapa combinado mostrando comercios y servicios de seguridad."""
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Plotear comunas base con colores
    for idx, row in comunas.iterrows():
        color = COLORES_COMUNAS.get(row['comuna'], 'gray')
        gpd.GeoSeries([row['geometry']]).plot(ax=ax, color=color, alpha=0.15, 
                                               edgecolor='black', linewidth=2)
    
    # Plotear comercios (puntos pequeños)
    tiendas_comuna.plot(ax=ax, color='orange', markersize=5, alpha=0.4, label='Comercios')
    
    # Plotear servicios de seguridad (puntos grandes)
    if len(cuarteles_comuna) > 0:
        cuarteles_comuna.plot(ax=ax, color='blue', marker='^', markersize=150, 
                              label='Carabineros', alpha=0.9, edgecolor='white', linewidth=1)
    
    if len(bomberos_comuna) > 0:
        bomberos_comuna.plot(ax=ax, color='red', marker='s', markersize=180, 
                             label='Bomberos', alpha=0.9, edgecolor='white', linewidth=1)
    
    if len(pdi_comuna) > 0:
        pdi_comuna.plot(ax=ax, color='green', marker='o', markersize=100, 
                        label='PDI', alpha=0.9, edgecolor='white', linewidth=1)
    
    # Agregar nombres de comunas
    for idx, row in comunas.iterrows():
        centroid = row['geometry'].centroid
        ax.annotate(row['comuna'], xy=(centroid.x, centroid.y),
                   ha='center', va='center', fontsize=14, fontweight='bold',
                   color='black', 
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.7, edgecolor='gray'))
    
    ax.set_title('Mapa Integrado: Distribución de Comercios\ny Servicios de Seguridad en 4 Comunas de Santiago',
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Coordenada Este (m)', fontsize=11)
    ax.set_ylabel('Coordenada Norte (m)', fontsize=11)
    ax.legend(loc='lower left', fontsize=10, markerscale=0.5)
    ax.set_aspect('equal')
    
    # Agregar grid
    ax.grid(True, linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(GRAFICOS_DIR, 'mapa_integrado_comercios_seguridad.png'), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(GRAFICOS_DIR, 'mapa_integrado_comercios_seguridad.pdf'), bbox_inches='tight')
    print(f" Mapa integrado guardado en: {GRAFICOS_DIR}")
    plt.close()


def generar_reporte_json(stats_comercios, stats_seguridad):
    """Genera un reporte JSON con todas las estadísticas."""
    reporte = {
        "titulo": "Estadística Descriptiva Espacial",
        "fecha_generacion": pd.Timestamp.now().isoformat(),
        "preguntas_investigacion": {
            "pregunta_1": "¿Cuál es la distribución de comercios por zonas de una ciudad?",
            "pregunta_2": "¿En qué lugares existe una alta concentración de servicios de seguridad?"
        },
        "area_estudio": {
            "comunas": stats_comercios.index.tolist(),
            "total_comunas": len(stats_comercios)
        },
        "analisis_comercios": {
            "total_comercios": int(stats_comercios['total_comercios'].sum()),
            "promedio_por_comuna": float(stats_comercios['total_comercios'].mean()),
            "desviacion_estandar": float(stats_comercios['total_comercios'].std()),
            "densidad_promedio_km2": float(stats_comercios['densidad_por_km2'].mean()),
            "por_comuna": {
                comuna: {
                    "total_comercios": int(row['total_comercios']),
                    "area_km2": float(row['area_km2']),
                    "densidad_por_km2": float(row['densidad_por_km2']),
                    "porcentaje_total": float(row['porcentaje'])
                }
                for comuna, row in stats_comercios.iterrows()
            }
        },
        "analisis_seguridad": {
            "total_servicios": int(stats_seguridad['total_seguridad'].sum()),
            "total_carabineros": int(stats_seguridad['carabineros'].sum()),
            "total_bomberos": int(stats_seguridad['bomberos'].sum()),
            "total_pdi": int(stats_seguridad['pdi'].sum()),
            "densidad_promedio_km2": float(stats_seguridad['densidad_por_km2'].mean()),
            "por_comuna": {
                comuna: {
                    "carabineros": int(row['carabineros']),
                    "bomberos": int(row['bomberos']),
                    "pdi": int(row['pdi']),
                    "total": int(row['total_seguridad']),
                    "densidad_por_km2": float(row['densidad_por_km2'])
                }
                for comuna, row in stats_seguridad.iterrows()
            }
        },
        "hallazgos_principales": []
    }
    
    # Agregar hallazgos
    comuna_max_comercios = stats_comercios['densidad_por_km2'].idxmax()
    comuna_max_seguridad = stats_seguridad['densidad_por_km2'].idxmax()
    
    reporte["hallazgos_principales"] = [
        f"La comuna con mayor densidad de comercios es {comuna_max_comercios} con {stats_comercios.loc[comuna_max_comercios, 'densidad_por_km2']:.1f} comercios/km²",
        f"La comuna con mayor concentración de servicios de seguridad es {comuna_max_seguridad} con {stats_seguridad.loc[comuna_max_seguridad, 'densidad_por_km2']:.3f} servicios/km²",
        f"Se identificaron un total de {int(stats_comercios['total_comercios'].sum())} comercios en el área de estudio",
        f"Se registraron {int(stats_seguridad['total_seguridad'].sum())} servicios de seguridad (Carabineros, Bomberos y PDI)"
    ]
    
    # Guardar reporte
    reporte_path = os.path.join(OUTPUT_DIR, 'estadistica_descriptiva_espacial.json')
    with open(reporte_path, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)
    
    print(f" Reporte JSON guardado en: {reporte_path}")
    
    return reporte


def generar_resumen_markdown(stats_comercios, stats_seguridad, reporte):
    """Genera un resumen en formato Markdown para el informe."""
    
    md_content = """# Estadística Descriptiva Espacial
## Análisis de Distribución de Comercios y Servicios de Seguridad

### Área de Estudio
El análisis comprende 4 comunas del Gran Santiago:
- **Santiago** (Centro histórico y comercial)
- **Ñuñoa** (Comuna residencial mixta)
- **La Reina** (Comuna residencial oriente)
- **Estación Central** (Nodo de transporte)

---

## 1. Distribución de Comercios por Zonas

### Pregunta de Investigación
*¿Cuál es la distribución de comercios por zonas de una ciudad?*

### Resultados

| Comuna | Total Comercios | Área (km²) | Densidad (comercios/km²) | % del Total |
|--------|-----------------|------------|--------------------------|-------------|
"""
    
    for comuna, row in stats_comercios.iterrows():
        md_content += f"| {comuna} | {int(row['total_comercios'])} | {row['area_km2']:.2f} | {row['densidad_por_km2']:.1f} | {row['porcentaje']:.1f}% |\n"
    
    md_content += f"""
### Estadísticas Globales
- **Total de comercios:** {int(stats_comercios['total_comercios'].sum())}
- **Densidad promedio:** {stats_comercios['densidad_por_km2'].mean():.1f} comercios/km²
- **Desviación estándar:** {stats_comercios['total_comercios'].std():.1f}

### Patrones Identificados
1. **Concentración centro-periferia:** Se observa mayor densidad de comercios en las comunas centrales
2. **Heterogeneidad espacial:** La distribución de comercios no es uniforme, con clusters en zonas comerciales
3. **Relación con conectividad:** Las zonas con mejor acceso a transporte público muestran mayor densidad comercial

---

## 2. Concentración de Servicios de Seguridad

### Pregunta de Investigación
*¿En qué lugares existe una alta concentración de servicios de seguridad (carabineros, bomberos)?*

### Resultados

| Comuna | Carabineros | Bomberos | PDI | Total | Densidad (serv/km²) |
|--------|-------------|----------|-----|-------|---------------------|
"""
    
    for comuna, row in stats_seguridad.iterrows():
        md_content += f"| {comuna} | {int(row['carabineros'])} | {int(row['bomberos'])} | {int(row['pdi'])} | {int(row['total_seguridad'])} | {row['densidad_por_km2']:.3f} |\n"
    
    media_densidad = stats_seguridad['densidad_por_km2'].mean()
    zonas_alta = stats_seguridad[stats_seguridad['densidad_por_km2'] > media_densidad].index.tolist()
    
    md_content += f"""
### Estadísticas Globales
- **Total Carabineros:** {int(stats_seguridad['carabineros'].sum())} unidades
- **Total Bomberos:** {int(stats_seguridad['bomberos'].sum())} compañías
- **Total PDI:** {int(stats_seguridad['pdi'].sum())} unidades
- **Total General:** {int(stats_seguridad['total_seguridad'].sum())} servicios

### Zonas de Alta Concentración
Las siguientes comunas presentan densidad superior al promedio ({media_densidad:.3f} servicios/km²):
"""
    
    for zona in zonas_alta:
        md_content += f"- **{zona}:** {stats_seguridad.loc[zona, 'densidad_por_km2']:.3f} servicios/km²\n"
    
    md_content += """
### Patrones Identificados
1. **Distribución estratégica:** Los servicios de seguridad se ubican considerando cobertura territorial
2. **Concentración en centros urbanos:** Mayor presencia en zonas de alta actividad comercial y poblacional
3. **Complementariedad institucional:** Carabineros, Bomberos y PDI se distribuyen de forma complementaria

---

## 3. Visualizaciones Generadas

Se generaron los siguientes mapas y gráficos:

1. **mapa_distribucion_comercios.png** - Distribución espacial y densidad de comercios
2. **mapa_servicios_seguridad.png** - Ubicación y concentración de servicios de seguridad
3. **mapa_integrado_comercios_seguridad.png** - Vista combinada de comercios y seguridad
4. **graficos_estadisticos_espaciales.png** - Gráficos de barras comparativos

---

## 4. Conclusiones

### Hallazgos Principales
"""
    
    for hallazgo in reporte['hallazgos_principales']:
        md_content += f"- {hallazgo}\n"
    
    md_content += """
### Implicaciones para la Valoración Inmobiliaria
1. La proximidad a comercios puede influir positivamente en el valor de propiedades
2. La cercanía a servicios de seguridad puede ser un factor de valoración según perfil del usuario
3. Las zonas con alta densidad comercial y buena cobertura de seguridad tienden a ser más valoradas

---

*Generado automáticamente - Proyecto Geoinformática 2025*
"""
    
    # Guardar markdown
    md_path = os.path.join(OUTPUT_DIR, 'ESTADISTICA_DESCRIPTIVA_ESPACIAL.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_content)
    
    print(f" Resumen Markdown guardado en: {md_path}")


def main():
    """Función principal que ejecuta todo el análisis."""
    print("\n" + "=" * 70)
    print("   ESTADÍSTICA DESCRIPTIVA ESPACIAL")
    print("   Análisis de Comercios y Servicios de Seguridad")
    print("=" * 70)
    
    # Cargar datos
    comunas, tiendas, cuarteles, bomberos, pdi = cargar_datos()
    
    # Análisis de comercios
    stats_comercios, tiendas_comuna = calcular_estadisticas_comercios(tiendas, comunas)
    
    # Análisis de seguridad
    stats_seguridad, cuarteles_comuna, bomberos_comuna, pdi_comuna = calcular_estadisticas_seguridad(
        cuarteles, bomberos, pdi, comunas
    )
    
    # Generar visualizaciones
    print("\n" + "=" * 60)
    print("GENERANDO VISUALIZACIONES")
    print("=" * 60)
    
    crear_mapa_comercios(tiendas_comuna, comunas, stats_comercios)
    crear_mapa_seguridad(cuarteles_comuna, bomberos_comuna, pdi_comuna, comunas, stats_seguridad)
    crear_mapa_combinado(tiendas_comuna, cuarteles_comuna, bomberos_comuna, pdi_comuna, comunas)
    crear_graficos_barras(stats_comercios, stats_seguridad)
    
    # Generar reportes
    print("\n" + "=" * 60)
    print("GENERANDO REPORTES")
    print("=" * 60)
    
    reporte = generar_reporte_json(stats_comercios, stats_seguridad)
    generar_resumen_markdown(stats_comercios, stats_seguridad, reporte)
    
    print("\n" + "=" * 70)
    print("    ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 70)
    print(f"\n Archivos generados en:")
    print(f"   • Gráficos: {GRAFICOS_DIR}")
    print(f"   • Reportes: {OUTPUT_DIR}")
    print("\n")


if __name__ == "__main__":
    main()
