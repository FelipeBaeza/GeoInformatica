#!/usr/bin/env python3
"""
Análisis detallado de CRS y geometrías de todos los GeoJSON en datos_filtrados
Genera reporte completo para identificar problemas antes de normalización
"""

import os
import json
import geopandas as gpd
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

def analizar_archivo_geojson(archivo_path):
    """Analiza un archivo GeoJSON y retorna diagnóstico completo"""
    try:
        # Leer con geopandas
        gdf = gpd.read_file(archivo_path)
        
        # Información básica
        info = {
            'archivo': archivo_path.name,
            'filas': len(gdf),
            'columnas': len(gdf.columns),
            'crs_original': str(gdf.crs) if gdf.crs else 'Sin CRS',
            'crs_epsg': gdf.crs.to_epsg() if gdf.crs else None,
            'bounds': gdf.total_bounds.tolist() if not gdf.empty else None,
            'tipos_geometria': gdf.geom_type.value_counts().to_dict(),
            'geometrias_validas': gdf.geometry.is_valid.sum(),
            'geometrias_invalidas': (~gdf.geometry.is_valid).sum(),
            'geometrias_nulas': gdf.geometry.isna().sum(),
            'area_total_original': None,
            'necesita_reproyeccion': False,
            'problema_identificado': None
        }
        
        # Calcular área total si hay polígonos (solo para diagnóstico)
        poligonos = gdf[gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]
        if not poligonos.empty and gdf.crs:
            try:
                area_total = poligonos.geometry.area.sum()
                info['area_total_original'] = area_total
            except:
                info['area_total_original'] = 'Error al calcular'
        
        # Identificar problemas de CRS
        if not gdf.crs:
            info['problema_identificado'] = 'Sin CRS definido'
            info['necesita_reproyeccion'] = True
        elif gdf.crs.to_epsg() == 4326:
            info['problema_identificado'] = 'En WGS84 (grados) - necesita UTM para métricas'
            info['necesita_reproyeccion'] = True
        elif gdf.crs.to_epsg() == 3857:
            info['problema_identificado'] = 'En Web Mercator - necesita UTM'
            info['necesita_reproyeccion'] = True
        elif gdf.crs.to_epsg() == 32719:
            info['problema_identificado'] = 'Ya en UTM 19S - OK'
            info['necesita_reproyeccion'] = False
        else:
            info['problema_identificado'] = f'CRS desconocido: {gdf.crs.to_epsg()}'
            info['necesita_reproyeccion'] = True
            
        # Análisis de columnas importantes
        columnas_interes = ['objectid', 'osm_id', 'osm_id2', 'name', 'nombre', 'amenity', 'shop', 'healthcare', 'superficie', 'precio']
        columnas_encontradas = [col for col in columnas_interes if col.lower() in [c.lower() for c in gdf.columns]]
        info['columnas_clave'] = columnas_encontradas
        
        # Muestra de datos
        if not gdf.empty:
            muestra = gdf.drop('geometry', axis=1).head(2).to_dict('records')
            info['muestra_datos'] = muestra
        
        return info
        
    except Exception as e:
        return {
            'archivo': archivo_path.name,
            'error': str(e),
            'problema_identificado': f'Error de lectura: {str(e)}'
        }

def generar_reporte_completo(carpeta_datos):
    """Genera reporte completo de todos los archivos"""
    carpeta = Path(carpeta_datos)
    archivos_geojson = list(carpeta.glob('*.geojson'))
    
    print(f"🔍 Analizando {len(archivos_geojson)} archivos GeoJSON...")
    
    resultados = []
    problemas_por_tipo = defaultdict(list)
    
    for archivo in sorted(archivos_geojson):
        print(f"   Procesando: {archivo.name}")
        info = analizar_archivo_geojson(archivo)
        resultados.append(info)
        
        # Categorizar problemas
        if 'error' in info:
            problemas_por_tipo['errores_lectura'].append(archivo.name)
        elif info.get('necesita_reproyeccion'):
            problemas_por_tipo['necesita_reproyeccion'].append(archivo.name)
        if info.get('geometrias_invalidas', 0) > 0:
            problemas_por_tipo['geometrias_invalidas'].append(archivo.name)
        if info.get('geometrias_nulas', 0) > 0:
            problemas_por_tipo['geometrias_nulas'].append(archivo.name)
    
    return resultados, problemas_por_tipo

def imprimir_resumen(resultados, problemas_por_tipo):
    """Imprime resumen ejecutivo del análisis"""
    print("\n" + "="*80)
    print("📊 RESUMEN EJECUTIVO - ANÁLISIS CRS Y GEOMETRÍAS")
    print("="*80)
    
    # Estadísticas generales
    total_archivos = len(resultados)
    total_filas = sum(r.get('filas', 0) for r in resultados)
    archivos_sin_crs = len([r for r in resultados if not r.get('crs_epsg')])
    archivos_utm = len([r for r in resultados if r.get('crs_epsg') == 32719])
    archivos_wgs84 = len([r for r in resultados if r.get('crs_epsg') == 4326])
    archivos_mercator = len([r for r in resultados if r.get('crs_epsg') == 3857])
    
    print(f"\n📈 ESTADÍSTICAS GENERALES:")
    print(f"   Total archivos: {total_archivos}")
    print(f"   Total registros: {total_filas:,}")
    print(f"   Sin CRS: {archivos_sin_crs} archivos")
    print(f"   En UTM 19S (32719): {archivos_utm} archivos ✅")
    print(f"   En WGS84 (4326): {archivos_wgs84} archivos ⚠️")  
    print(f"   En Web Mercator (3857): {archivos_mercator} archivos ⚠️")
    
    # Problemas identificados
    print(f"\n🚨 PROBLEMAS IDENTIFICADOS:")
    for tipo_problema, archivos in problemas_por_tipo.items():
        print(f"   {tipo_problema}: {len(archivos)} archivos")
        for archivo in archivos[:5]:  # Mostrar hasta 5
            print(f"      - {archivo}")
        if len(archivos) > 5:
            print(f"      ... y {len(archivos)-5} más")
    
    # Archivos por tipo de geometría
    print(f"\n📍 TIPOS DE GEOMETRÍA:")
    tipos_geom = defaultdict(int)
    for r in resultados:
        for tipo, cantidad in r.get('tipos_geometria', {}).items():
            tipos_geom[tipo] += cantidad
    
    for tipo, cantidad in sorted(tipos_geom.items(), key=lambda x: x[1], reverse=True):
        print(f"   {tipo}: {cantidad:,} geometrías")

def exportar_reporte_detallado(resultados, ruta_salida):
    """Exporta reporte detallado a JSON y CSV"""
    
    # JSON completo
    with open(ruta_salida / 'analisis_crs_detallado.json', 'w', encoding='utf-8') as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2, default=str)
    
    # CSV resumen
    resumen_df = []
    for r in resultados:
        fila = {
            'archivo': r.get('archivo', ''),
            'filas': r.get('filas', 0),
            'crs_original': r.get('crs_original', ''),
            'crs_epsg': r.get('crs_epsg', ''),
            'necesita_reproyeccion': r.get('necesita_reproyeccion', False),
            'geometrias_invalidas': r.get('geometrias_invalidas', 0),
            'geometrias_nulas': r.get('geometrias_nulas', 0),
            'problema_identificado': r.get('problema_identificado', ''),
            'tipos_geometria': str(r.get('tipos_geometria', {}))
        }
        resumen_df.append(fila)
    
    df_resumen = pd.DataFrame(resumen_df)
    df_resumen.to_csv(ruta_salida / 'resumen_crs_geometrias.csv', index=False, encoding='utf-8')
    
    print(f"\n💾 Reportes exportados:")
    print(f"   📄 Detallado: {ruta_salida / 'analisis_crs_detallado.json'}")
    print(f"   📊 Resumen: {ruta_salida / 'resumen_crs_geometrias.csv'}")

if __name__ == "__main__":
    # Configuración
    carpeta_datos = Path(__file__).parent.parent / "datos_filtrados"
    carpeta_reportes = Path(__file__).parent.parent / "reportes"
    
    # Crear carpeta reportes si no existe
    carpeta_reportes.mkdir(exist_ok=True)
    
    # Ejecutar análisis
    resultados, problemas = generar_reporte_completo(carpeta_datos)
    
    # Mostrar resumen en consola
    imprimir_resumen(resultados, problemas)
    
    # Exportar reportes
    exportar_reporte_detallado(resultados, carpeta_reportes)
    
    print(f"\n✅ Análisis completado. Revisar reportes en: {carpeta_reportes}")