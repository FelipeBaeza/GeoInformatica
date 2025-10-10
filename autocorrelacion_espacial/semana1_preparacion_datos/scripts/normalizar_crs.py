#!/usr/bin/env python3
"""
Normalización de CRS a EPSG:32719 para todos los GeoJSON
Corrige geometrías inválidas y estandariza estructura
"""

import os
import json
import geopandas as gpd
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def detectar_crs_probable(gdf, archivo_nombre):
    """Detecta CRS más probable basado en bounds y contexto"""
    bounds = gdf.total_bounds
    
    # Si bounds están en rango de coordenadas geográficas de Chile
    if (-75 <= bounds[0] <= -66) and (-56 <= bounds[1] <= -17):
        return 'EPSG:4326'  # WGS84
    
    # Si bounds están en rango de Web Mercator de Chile
    elif (-8400000 <= bounds[0] <= -7300000) and (-6200000 <= bounds[1] <= -1900000):
        return 'EPSG:3857'  # Web Mercator
    
    # Si bounds están en rango UTM 19S de Chile  
    elif (200000 <= bounds[0] <= 800000) and (6000000 <= bounds[1] <= 8000000):
        return 'EPSG:32719'  # UTM 19S
    
    # Casos especiales por nombre de archivo
    if 'vertedores' in archivo_nombre.lower():
        # Vertederos tiene x,y en UTM pero geometría en WGS84
        return 'EPSG:4326'
    
    # Default: asumir WGS84
    print(f"     No se pudo detectar CRS para {archivo_nombre}, asumiendo WGS84")
    return 'EPSG:4326'

def normalizar_archivo(archivo_origen, carpeta_destino):
    """Normaliza un archivo GeoJSON individual"""
    try:
        print(f"    Procesando: {archivo_origen.name}")
        
        # Leer archivo
        gdf = gpd.read_file(archivo_origen)
        
        if gdf.empty:
            print(f"        Archivo vacío, copiando sin cambios")
            gdf.to_file(carpeta_destino / archivo_origen.name, driver='GeoJSON')
            return {'archivo': archivo_origen.name, 'estado': 'vacio', 'filas_procesadas': 0}
        
        # Detectar/asignar CRS si falta
        if not gdf.crs:
            crs_detectado = detectar_crs_probable(gdf, archivo_origen.name)
            gdf = gdf.set_crs(crs_detectado)
            print(f"       CRS asignado: {crs_detectado}")
        
        crs_original = str(gdf.crs)
        
        # Reproyectar a UTM 19S si no está ya
        if gdf.crs.to_epsg() != 32719:
            print(f"        Reproyectando de {gdf.crs.to_epsg()} a 32719")
            gdf = gdf.to_crs('EPSG:32719')
        else:
            print(f"       Ya en UTM 19S")
        
        # Validar y reparar geometrías
        geometrias_invalidas = (~gdf.geometry.is_valid).sum()
        if geometrias_invalidas > 0:
            print(f"       Reparando {geometrias_invalidas} geometrías inválidas")
            gdf['geometry'] = gdf.geometry.buffer(0)  # Truco para reparar
        
        # Eliminar geometrías nulas
        gdf = gdf[~gdf.geometry.isna()]
        
        # Limpiar columnas problemáticas
        if 'Shape__Area' in gdf.columns:
            gdf = gdf.drop('Shape__Area', axis=1)  # Área en grados, será recalculada
        if 'Shape__Length' in gdf.columns:
            gdf = gdf.drop('Shape__Length', axis=1)  # Perímetro en grados
        
        # Recalcular área y perímetro en metros para polígonos
        if gdf.geom_type.isin(['Polygon', 'MultiPolygon']).any():
            poligonos_mask = gdf.geom_type.isin(['Polygon', 'MultiPolygon'])
            gdf.loc[poligonos_mask, 'area_m2'] = gdf.loc[poligonos_mask, 'geometry'].area
            gdf.loc[poligonos_mask, 'perimeter_m'] = gdf.loc[poligonos_mask, 'geometry'].length
        
        # Agregar coordenadas centroides para análisis
        gdf['centroide_x'] = gdf.geometry.centroid.x
        gdf['centroide_y'] = gdf.geometry.centroid.y
        
        # Agregar metadatos de procesamiento
        gdf['crs_original'] = crs_original
        gdf['fecha_normalizacion'] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Guardar archivo normalizado
        archivo_salida = carpeta_destino / archivo_origen.name
        gdf.to_file(archivo_salida, driver='GeoJSON')
        
        return {
            'archivo': archivo_origen.name,
            'estado': 'normalizado', 
            'filas_procesadas': len(gdf),
            'crs_original': crs_original,
            'crs_final': 'EPSG:32719',
            'geometrias_reparadas': geometrias_invalidas,
            'area_total_m2': gdf.get('area_m2', pd.Series([])).sum() if 'area_m2' in gdf.columns else None
        }
        
    except Exception as e:
        print(f"       Error procesando {archivo_origen.name}: {str(e)}")
        return {
            'archivo': archivo_origen.name,
            'estado': 'error',
            'error': str(e)
        }

def normalizar_todos_los_archivos(carpeta_origen, carpeta_destino):
    """Normaliza todos los archivos GeoJSON de una carpeta"""
    
    carpeta_origen = Path(carpeta_origen)
    carpeta_destino = Path(carpeta_destino)
    
    # Crear carpeta destino
    carpeta_destino.mkdir(exist_ok=True)
    
    # Obtener todos los archivos GeoJSON
    archivos_geojson = list(carpeta_origen.glob('*.geojson'))
    
    print(f" Iniciando normalización de {len(archivos_geojson)} archivos")
    print(f" Origen: {carpeta_origen}")
    print(f" Destino: {carpeta_destino}")
    print("="*60)
    
    resultados = []
    
    for archivo in sorted(archivos_geojson):
        resultado = normalizar_archivo(archivo, carpeta_destino)
        resultados.append(resultado)
    
    return resultados

def generar_reporte_normalizacion(resultados, carpeta_reportes):
    """Genera reporte de la normalización"""
    
    carpeta_reportes.mkdir(exist_ok=True)
    
    # Estadísticas
    total_archivos = len(resultados)
    exitosos = len([r for r in resultados if r['estado'] == 'normalizado'])
    errores = len([r for r in resultados if r['estado'] == 'error'])
    vacios = len([r for r in resultados if r['estado'] == 'vacio'])
    
    total_filas = sum(r.get('filas_procesadas', 0) for r in resultados)
    total_reparaciones = sum(r.get('geometrias_reparadas', 0) for r in resultados)
    
    # Reporte resumen
    resumen = {
        'fecha_procesamiento': pd.Timestamp.now().isoformat(),
        'estadisticas': {
            'total_archivos': total_archivos,
            'exitosos': exitosos,
            'errores': errores,
            'vacios': vacios,
            'total_geometrias': total_filas,
            'total_reparaciones': total_reparaciones
        },
        'archivos_procesados': resultados
    }
    
    # Guardar reporte JSON
    with open(carpeta_reportes / 'normalizacion_crs.json', 'w', encoding='utf-8') as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2, default=str)
    
    # Guardar CSV resumen
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(carpeta_reportes / 'normalizacion_resumen.csv', index=False, encoding='utf-8')
    
    # Imprimir resumen
    print("\n" + "="*60)
    print(" RESUMEN DE NORMALIZACIÓN")
    print("="*60)
    print(f" Archivos procesados exitosamente: {exitosos}/{total_archivos}")
    print(f" Archivos con errores: {errores}")
    print(f" Archivos vacíos: {vacios}")
    print(f" Total geometrías procesadas: {total_filas:,}")
    print(f" Geometrías reparadas: {total_reparaciones}")
    print(f"\n Reportes guardados en: {carpeta_reportes}")
    
    # Mostrar errores si los hay
    if errores > 0:
        print(f"\n ARCHIVOS CON ERRORES:")
        for r in resultados:
            if r['estado'] == 'error':
                print(f"   - {r['archivo']}: {r.get('error', 'Error desconocido')}")

if __name__ == "__main__":
    # Configuración de rutas
    base_path = Path(__file__).parent.parent
    carpeta_origen = base_path / "datos_filtrados"
    carpeta_destino = base_path / "datos_normalizados"
    carpeta_reportes = base_path / "reportes"
    
    # Ejecutar normalización
    resultados = normalizar_todos_los_archivos(carpeta_origen, carpeta_destino)
    
    # Generar reporte
    generar_reporte_normalizacion(resultados, carpeta_reportes)
    
    print(f"\n Normalización completada!")
    print(f" Archivos normalizados en: {carpeta_destino}")