#!/usr/bin/env python3
"""
Script para generar grilla de evaluación espacial para Santiago Metropolitano.
Crea una grilla regular de puntos donde se calcularán características espaciales.

Autor: Proyecto GeoInformática
Fecha: Octubre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point, box
import os
import sys

def cargar_limites_santiago():
    """Carga los límites de las comunas para definir área de estudio"""
    try:
        # Cargar comunas normalizadas (límites administrativos)
        comunas_path = "datos_normalizados/comunas_buffer.geojson"
        comunas = gpd.read_file(comunas_path)
        
        # Crear bounding box general
        total_bounds = comunas.total_bounds
        
        return comunas, total_bounds
        
    except Exception as e:
        print(f"Error cargando límites de Santiago: {e}")
        return None, None

def crear_grilla_regular(bounds, espaciado=200):
    """
    Crea grilla regular de puntos con espaciado definido
    
    Args:
        bounds: [minx, miny, maxx, maxy] en metros (UTM 19S)
        espaciado: distancia entre puntos en metros
    
    Returns:
        GeoDataFrame con puntos de la grilla
    """
    minx, miny, maxx, maxy = bounds
    
    # Expandir bounds ligeramente para cubrir completamente el área
    buffer = espaciado * 2
    minx -= buffer
    miny -= buffer
    maxx += buffer
    maxy += buffer
    
    # Generar coordenadas X e Y
    x_coords = np.arange(minx, maxx + espaciado, espaciado)
    y_coords = np.arange(miny, maxy + espaciado, espaciado)
    
    # Crear mesh grid
    xx, yy = np.meshgrid(x_coords, y_coords)
    
    # Crear puntos
    puntos = []
    ids = []
    coordenadas_x = []
    coordenadas_y = []
    
    for i, (x_row, y_row) in enumerate(zip(xx, yy)):
        for j, (x, y) in enumerate(zip(x_row, y_row)):
            puntos.append(Point(x, y))
            ids.append(f"GRID_{i:04d}_{j:04d}")
            coordenadas_x.append(x)
            coordenadas_y.append(y)
    
    # Crear GeoDataFrame
    grilla_df = gpd.GeoDataFrame({
        'grid_id': ids,
        'x_utm': coordenadas_x,
        'y_utm': coordenadas_y,
        'geometry': puntos
    }, crs='EPSG:32719')
    
    print(f"Grilla inicial creada: {len(grilla_df)} puntos")
    return grilla_df

def filtrar_grilla_por_comunas(grilla, comunas):
    """
    Filtra grilla para mantener solo puntos dentro de las comunas
    """
    # Hacer intersección espacial
    puntos_dentro = gpd.sjoin(grilla, comunas, how='inner', predicate='within')
    
    # Limpiar columnas duplicadas del join
    columnas_a_mantener = ['grid_id', 'x_utm', 'y_utm', 'geometry', 'Comuna', 'Provincia']
    
    # Verificar qué columnas existen
    columnas_disponibles = [col for col in columnas_a_mantener if col in puntos_dentro.columns]
    if 'Comuna' not in columnas_disponibles and 'COMUNA' in puntos_dentro.columns:
        puntos_dentro['Comuna'] = puntos_dentro['COMUNA']
        columnas_disponibles.append('Comuna')
    
    # Usar columnas básicas si las específicas no existen
    columnas_finales = ['grid_id', 'x_utm', 'y_utm', 'geometry']
    if 'comuna' in puntos_dentro.columns:
        columnas_finales.append('comuna')
    
    puntos_filtrados = puntos_dentro[columnas_finales].copy()
    
    # Eliminar duplicados (un punto podría estar en borde de múltiples comunas)
    puntos_filtrados = puntos_filtrados.drop_duplicates(subset=['grid_id'])
    
    print(f"Puntos filtrados dentro de comunas: {len(puntos_filtrados)}")
    return puntos_filtrados

def agregar_metadatos_grilla(grilla):
    """Agrega metadatos útiles a la grilla"""
    # Calcular zona UTM (para referencia)
    grilla['zona_utm'] = '19S'
    grilla['crs_epsg'] = 32719
    
    # Agregar timestamp
    from datetime import datetime
    grilla['fecha_creacion'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Inicializar columnas para características futuras
    grilla['procesado'] = False
    
    return grilla

def validar_grilla(grilla):
    """Validaciones básicas de la grilla generada"""
    print("\n=== VALIDACIÓN DE GRILLA ===")
    
    # Verificar CRS
    print(f"CRS: {grilla.crs}")
    assert grilla.crs.to_epsg() == 32719, "CRS debe ser EPSG:32719"
    
    # Verificar bounds
    bounds = grilla.total_bounds
    print(f"Bounds: {bounds}")
    
    # Verificar que esté en rango de Santiago
    assert 280000 < bounds[0] < 420000, f"MinX fuera de rango: {bounds[0]}"
    assert 6240000 < bounds[1] < 6360000, f"MinY fuera de rango: {bounds[1]}"
    
    # Verificar que todos los puntos tengan geometría válida
    geometrias_validas = grilla.geometry.is_valid.all()
    print(f"Todas las geometrías válidas: {geometrias_validas}")
    
    # Estadísticas de distribución
    print(f"Total de puntos: {len(grilla)}")
    print(f"Área aproximada cubierta: {(bounds[2]-bounds[0])*(bounds[3]-bounds[1])/1e6:.1f} km²")
    
    # Verificar distribución por comuna
    if 'comuna' in grilla.columns:
        distribucion = grilla['comuna'].value_counts()
        print(f"Distribución por comuna:")
        print(distribucion)
    
    print(" Validación completada exitosamente")

def main():
    """Función principal"""
    print("=== GENERACIÓN DE GRILLA DE EVALUACIÓN ESPACIAL ===\n")
    
    # Verificar entorno virtual
    if 'venv_semana1' not in sys.executable:
        print("  Recomendado: Activar entorno virtual")
        print("   source ../venv_semana1/bin/activate")
    
    # Crear directorio de salida si no existe
    os.makedirs("features", exist_ok=True)
    
    # 1. Cargar límites de Santiago
    print("1. Cargando límites de comunas...")
    comunas, bounds = cargar_limites_santiago()
    
    if comunas is None:
        print(" Error: No se pudieron cargar los límites de Santiago")
        return False
    
    print(f"    Cargadas {len(comunas)} comunas")
    print(f"    Bounds: {bounds}")
    
    # 2. Crear grilla regular
    print("\n2. Creando grilla regular (espaciado: 200m)...")
    grilla = crear_grilla_regular(bounds, espaciado=200)
    
    # 3. Filtrar por límites comunales
    print("\n3. Filtrando puntos dentro de comunas...")
    grilla_filtrada = filtrar_grilla_por_comunas(grilla, comunas)
    
    # 4. Agregar metadatos
    print("\n4. Agregando metadatos...")
    grilla_final = agregar_metadatos_grilla(grilla_filtrada)
    
    # 5. Validar resultado
    print("\n5. Validando grilla generada...")
    validar_grilla(grilla_final)
    
    # 6. Guardar resultado
    print("\n6. Guardando grilla...")
    output_path = "features/grilla_evaluacion_santiago.geojson"
    grilla_final.to_file(output_path, driver='GeoJSON')
    
    print(f"    Grilla guardada en: {output_path}")
    print(f"    Total de puntos: {len(grilla_final)}")
    
    # 7. Generar reporte resumido
    print("\n7. Generando reporte...")
    reporte = {
        'total_puntos': len(grilla_final),
        'espaciado_metros': 200,
        'area_km2': round((bounds[2]-bounds[0])*(bounds[3]-bounds[1])/1e6, 2),
        'densidad_puntos_km2': round(len(grilla_final) / ((bounds[2]-bounds[0])*(bounds[3]-bounds[1])/1e6), 1),
        'crs': 'EPSG:32719',
        'fecha_creacion': grilla_final['fecha_creacion'].iloc[0]
    }
    
    import json
    reporte_path = "reportes/grilla_evaluacion_reporte.json"
    with open(reporte_path, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    print(f"    Reporte guardado en: {reporte_path}")
    print("\n Grilla de evaluación creada exitosamente!")
    
    return True

if __name__ == "__main__":
    exito = main()
    if not exito:
        sys.exit(1)