#!/usr/bin/env python3
"""
Validación de calidad de datos geoespaciales normalizados
Detecta duplicados, outliers espaciales, y problemas de calidad
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

def validar_calidad_archivo(archivo_path):
    """Valida calidad de un archivo GeoJSON normalizado"""
    
    try:
        gdf = gpd.read_file(archivo_path)
        
        if gdf.empty:
            return {'archivo': archivo_path.name, 'estado': 'vacio'}
        
        validacion = {
            'archivo': archivo_path.name,
            'filas_total': len(gdf),
            'crs': str(gdf.crs),
            'tipos_geometria': gdf.geom_type.value_counts().to_dict(),
            'bounds': gdf.total_bounds.tolist(),
            'area_total_m2': None,
            'validaciones': {}
        }
        
        # 1. Validación de CRS
        if gdf.crs and gdf.crs.to_epsg() == 32719:
            validacion['validaciones']['crs_correcto'] = True
        else:
            validacion['validaciones']['crs_correcto'] = False
            validacion['validaciones']['crs_problema'] = f"CRS incorrecto: {gdf.crs}"
        
        # 2. Geometrías válidas
        geometrias_validas = gdf.geometry.is_valid.sum()
        geometrias_invalidas = len(gdf) - geometrias_validas
        validacion['validaciones']['geometrias_validas'] = geometrias_validas
        validacion['validaciones']['geometrias_invalidas'] = geometrias_invalidas
        
        # 3. Geometrías nulas
        geometrias_nulas = gdf.geometry.isna().sum()
        validacion['validaciones']['geometrias_nulas'] = geometrias_nulas
        
        # 4. Duplicados por geometría
        if len(gdf) > 1:
            duplicados_geom = gdf.duplicated('geometry').sum()
            validacion['validaciones']['duplicados_geometria'] = duplicados_geom
        
        # 5. Duplicados por ID (si existe)
        campos_id = ['objectid', 'osm_id', 'osm_id2', 'id', 'ID']
        for campo in campos_id:
            if campo in gdf.columns:
                duplicados_id = gdf.duplicated(campo).sum()
                validacion['validaciones'][f'duplicados_{campo}'] = duplicados_id
                break
        
        # 6. Completitud de campos importantes
        campos_importantes = ['name', 'nombre', 'amenity', 'shop', 'healthcare', 'tipo']
        for campo in campos_importantes:
            if campo in gdf.columns:
                nulos = gdf[campo].isna().sum()
                completitud = ((len(gdf) - nulos) / len(gdf)) * 100
                validacion['validaciones'][f'completitud_{campo}'] = completitud
        
        # 7. Outliers espaciales (geometrías muy alejadas del centro)
        if len(gdf) > 5:  # Solo si hay suficientes datos
            centroides = gdf.geometry.centroid
            centro_x, centro_y = centroides.x.median(), centroides.y.median()
            
            # Distancia al centro mediano
            distancias = np.sqrt((centroides.x - centro_x)**2 + (centroides.y - centro_y)**2)
            q75 = distancias.quantile(0.75)
            q25 = distancias.quantile(0.25)
            iqr = q75 - q25
            limite_superior = q75 + 1.5 * iqr
            
            outliers_espaciales = (distancias > limite_superior).sum()
            validacion['validaciones']['outliers_espaciales'] = outliers_espaciales
            validacion['validaciones']['distancia_maxima_centro'] = distancias.max()
        
        # 8. Área total para polígonos
        poligonos = gdf[gdf.geom_type.isin(['Polygon', 'MultiPolygon'])]
        if not poligonos.empty:
            area_total = poligonos.geometry.area.sum()
            validacion['area_total_m2'] = area_total
        
        # 9. Densidad de puntos por km²
        if gdf.geom_type.eq('Point').all():
            bounds = gdf.total_bounds
            area_bbox_km2 = ((bounds[2] - bounds[0]) * (bounds[3] - bounds[1])) / 1e6
            densidad = len(gdf) / area_bbox_km2 if area_bbox_km2 > 0 else 0
            validacion['validaciones']['densidad_puntos_km2'] = densidad
        
        return validacion
        
    except Exception as e:
        return {
            'archivo': archivo_path.name,
            'estado': 'error',
            'error': str(e)
        }

def validar_consistencia_entre_archivos(resultados_validacion):
    """Valida consistencia entre archivos"""
    
    consistencia = {
        'archivos_analizados': len(resultados_validacion),
        'crs_consistente': True,
        'bounds_coherentes': True,
        'problemas_detectados': []
    }
    
    # Verificar CRS consistente
    crs_diferentes = set()
    bounds_todos = []
    
    for resultado in resultados_validacion:
        if 'error' in resultado:
            continue
            
        # CRS
        crs = resultado.get('crs')
        if crs:
            crs_diferentes.add(crs)
        
        # Bounds
        bounds = resultado.get('bounds')
        if bounds and len(bounds) == 4:
            bounds_todos.append(bounds)
    
    # Verificar CRS
    if len(crs_diferentes) > 1:
        consistencia['crs_consistente'] = False
        consistencia['problemas_detectados'].append(f"CRS inconsistentes: {list(crs_diferentes)}")
    
    # Verificar bounds coherentes (todos en Chile)
    if bounds_todos:
        all_bounds = np.array(bounds_todos)
        
        # Bounds esperados para Chile en UTM 19S
        chile_bounds_utm = [200000, 6000000, 800000, 8000000]  # Aproximado
        
        for i, bounds in enumerate(bounds_todos):
            if not (chile_bounds_utm[0] <= bounds[0] <= chile_bounds_utm[2] and 
                   chile_bounds_utm[1] <= bounds[1] <= chile_bounds_utm[3]):
                consistencia['bounds_coherentes'] = False
                archivo = resultados_validacion[i]['archivo']
                consistencia['problemas_detectados'].append(
                    f"Bounds fuera de Chile: {archivo} - {bounds}"
                )
    
    return consistencia

def generar_reporte_calidad(validaciones, consistencia, carpeta_reportes):
    """Genera reporte completo de calidad"""
    
    # Estadísticas generales
    total_archivos = len(validaciones)
    archivos_error = len([v for v in validaciones if 'error' in v])
    archivos_vacios = len([v for v in validaciones if v.get('estado') == 'vacio'])
    archivos_ok = total_archivos - archivos_error - archivos_vacios
    
    total_geometrias = sum(v.get('filas_total', 0) for v in validaciones if 'filas_total' in v)
    
    # Problemas críticos
    problemas_criticos = []
    problemas_menores = []
    
    for v in validaciones:
        if 'error' in v:
            problemas_criticos.append(f"{v['archivo']}: Error de lectura - {v.get('error', '')}")
            continue
            
        validaciones_archivo = v.get('validaciones', {})
        
        # Problemas críticos
        if not validaciones_archivo.get('crs_correcto', True):
            problemas_criticos.append(f"{v['archivo']}: CRS incorrecto")
        
        if validaciones_archivo.get('geometrias_invalidas', 0) > 0:
            problemas_criticos.append(f"{v['archivo']}: {validaciones_archivo['geometrias_invalidas']} geometrías inválidas")
        
        if validaciones_archivo.get('geometrias_nulas', 0) > 0:
            problemas_criticos.append(f"{v['archivo']}: {validaciones_archivo['geometrias_nulas']} geometrías nulas")
        
        # Problemas menores
        if validaciones_archivo.get('duplicados_geometria', 0) > 0:
            problemas_menores.append(f"{v['archivo']}: {validaciones_archivo['duplicados_geometria']} duplicados por geometría")
        
        if validaciones_archivo.get('outliers_espaciales', 0) > 0:
            problemas_menores.append(f"{v['archivo']}: {validaciones_archivo['outliers_espaciales']} outliers espaciales")
    
    # Compilar reporte
    reporte = {
        'fecha_validacion': pd.Timestamp.now().isoformat(),
        'resumen': {
            'total_archivos': total_archivos,
            'archivos_ok': archivos_ok,
            'archivos_error': archivos_error,
            'archivos_vacios': archivos_vacios,
            'total_geometrias': total_geometrias,
            'problemas_criticos': len(problemas_criticos),
            'problemas_menores': len(problemas_menores)
        },
        'consistencia_entre_archivos': consistencia,
        'problemas_detectados': {
            'criticos': problemas_criticos,
            'menores': problemas_menores
        },
        'validaciones_detalladas': validaciones
    }
    
    # Guardar reporte completo
    with open(carpeta_reportes / 'validacion_calidad.json', 'w', encoding='utf-8') as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2, default=str)
    
    # Guardar resumen CSV
    df_resumen = []
    for v in validaciones:
        if 'error' in v or 'estado' in v:
            continue
        
        fila = {
            'archivo': v['archivo'],
            'filas': v.get('filas_total', 0),
            'crs_correcto': v['validaciones'].get('crs_correcto', False),
            'geometrias_validas': v['validaciones'].get('geometrias_validas', 0),
            'geometrias_invalidas': v['validaciones'].get('geometrias_invalidas', 0),
            'duplicados_geometria': v['validaciones'].get('duplicados_geometria', 0),
            'outliers_espaciales': v['validaciones'].get('outliers_espaciales', 0)
        }
        df_resumen.append(fila)
    
    if df_resumen:
        pd.DataFrame(df_resumen).to_csv(
            carpeta_reportes / 'validacion_resumen.csv', 
            index=False, encoding='utf-8'
        )
    
    return reporte

def imprimir_resumen_validacion(reporte):
    """Imprime resumen de validación en consola"""
    
    print("\n" + "="*80)
    print(" REPORTE DE VALIDACIÓN DE CALIDAD")
    print("="*80)
    
    resumen = reporte['resumen']
    
    print(f"\n ESTADÍSTICAS GENERALES:")
    print(f"   Total archivos analizados: {resumen['total_archivos']}")
    print(f"    Archivos OK: {resumen['archivos_ok']}")
    print(f"    Archivos con error: {resumen['archivos_error']}")
    print(f"    Archivos vacíos: {resumen['archivos_vacios']}")
    print(f"    Total geometrías: {resumen['total_geometrias']:,}")
    
    print(f"\n PROBLEMAS DETECTADOS:")
    print(f"    Críticos: {resumen['problemas_criticos']}")
    print(f"   🟡 Menores: {resumen['problemas_menores']}")
    
    # Mostrar problemas críticos
    if reporte['problemas_detectados']['criticos']:
        print(f"\n PROBLEMAS CRÍTICOS (requieren atención):")
        for problema in reporte['problemas_detectados']['criticos'][:10]:
            print(f"   - {problema}")
        if len(reporte['problemas_detectados']['criticos']) > 10:
            print(f"   ... y {len(reporte['problemas_detectados']['criticos'])-10} más")
    
    # Mostrar problemas menores
    if reporte['problemas_detectados']['menores']:
        print(f"\n🟡 PROBLEMAS MENORES (revisar cuando sea posible):")
        for problema in reporte['problemas_detectados']['menores'][:5]:
            print(f"   - {problema}")
        if len(reporte['problemas_detectados']['menores']) > 5:
            print(f"   ... y {len(reporte['problemas_detectados']['menores'])-5} más")
    
    # Consistencia entre archivos
    consistencia = reporte['consistencia_entre_archivos']
    print(f"\n CONSISTENCIA ENTRE ARCHIVOS:")
    print(f"   CRS consistente: {'' if consistencia['crs_consistente'] else ''}")
    print(f"   Bounds coherentes: {'' if consistencia['bounds_coherentes'] else ''}")
    
    if consistencia['problemas_detectados']:
        print(f"   Problemas de consistencia:")
        for problema in consistencia['problemas_detectados']:
            print(f"      - {problema}")

if __name__ == "__main__":
    # Configuración
    base_path = Path(__file__).parent.parent
    carpeta_datos = base_path / "datos_normalizados"
    carpeta_reportes = base_path / "reportes"
    
    carpeta_reportes.mkdir(exist_ok=True)
    
    # Validar todos los archivos
    archivos_geojson = list(carpeta_datos.glob('*.geojson'))
    
    print(f" Validando calidad de {len(archivos_geojson)} archivos...")
    
    validaciones = []
    for archivo in sorted(archivos_geojson):
        print(f"   Validando: {archivo.name}")
        validacion = validar_calidad_archivo(archivo)
        validaciones.append(validacion)
    
    # Validar consistencia entre archivos
    consistencia = validar_consistencia_entre_archivos(validaciones)
    
    # Generar reporte
    reporte = generar_reporte_calidad(validaciones, consistencia, carpeta_reportes)
    
    # Mostrar resumen
    imprimir_resumen_validacion(reporte)
    
    print(f"\n Reportes de validación guardados en: {carpeta_reportes}")
    print(f"\n Validación de calidad completada!")