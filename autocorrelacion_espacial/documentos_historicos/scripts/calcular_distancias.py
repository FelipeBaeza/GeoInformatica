#!/usr/bin/env python3
"""
Script para calcular características de distancia desde cada punto de la grilla
a diferentes tipos de servicios y amenidades.

Autor: Proyecto GeoInformática
Fecha: Octubre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
import os
import sys
from datetime import datetime

def cargar_grilla():
    """Carga la grilla de evaluación generada previamente"""
    grilla_path = "features/grilla_evaluacion_santiago.geojson"
    
    if not os.path.exists(grilla_path):
        print(f"❌ Error: Grilla no encontrada en {grilla_path}")
        print("   Ejecute primero: python generar_grilla.py")
        return None
    
    grilla = gpd.read_file(grilla_path)
    print(f"✅ Grilla cargada: {len(grilla)} puntos")
    return grilla

def cargar_servicios_por_categoria():
    """
    Carga y organiza todos los servicios por categorías para cálculo de distancias
    """
    servicios = {}
    base_path = "datos_normalizados"
    
    # Definir categorías y archivos
    categorias = {
        'educacion_basica': 'establecimientos_educacion_escolar.geojson',
        'educacion_superior': 'establecimientos_educacion_superior.geojson',
        'educacion_parvularia': 'establecimientos_parvularia_filtrados.geojson',
        'salud': 'puntos_medicos_farmacias_hospitales_filtrados.geojson',
        'salud_clinicas': 'redes_de_clinicas_filtradas.geojson',
        'transporte_metro': 'Lineas_de_metro_de_Santiago.geojson',
        'transporte_carga': 'estaciones_carga_filtradas.geojson',
        'seguridad_pdi': 'unidades_operativas_pdi_filtradas.geojson',
        'seguridad_cuarteles': 'cuarteles_filtrados.geojson',
        'seguridad_bomberos': 'cuerpos_de_bomberos_filtrados.geojson',
        'areas_verdes': 'areas_verdes_filtradas.geojson',
        'ocio': 'ocio_filtrado.geojson',
        'turismo': 'atracciones_turisticas_filtradas.geojson',
        'servicios_publicos': 'municipios_filtrados.geojson',
        'servicios_sernam': 'centros_sernam_filtrados.geojson',
        'comercio': 'tiendas_filtradas.geojson',
        'puntos_interes': 'puntos_de_interes_filtrados.geojson'
    }
    
    print("Cargando servicios por categoría...")
    
    for categoria, archivo in categorias.items():
        archivo_path = os.path.join(base_path, archivo)
        
        if os.path.exists(archivo_path):
            try:
                gdf = gpd.read_file(archivo_path)
                
                # Filtrar solo geometrías de puntos para distancias
                if not gdf.empty:
                    if gdf.geometry.geom_type.iloc[0] in ['Point']:
                        servicios[categoria] = gdf
                        print(f"  ✅ {categoria}: {len(gdf)} puntos")
                    else:
                        # Para polígonos, usar centroides
                        gdf_centroides = gdf.copy()
                        gdf_centroides['geometry'] = gdf_centroides.geometry.centroid
                        servicios[categoria] = gdf_centroides
                        print(f"  ✅ {categoria}: {len(gdf)} centroides")
                
            except Exception as e:
                print(f"  ⚠️ Error cargando {categoria}: {e}")
        else:
            print(f"  ❌ No encontrado: {archivo}")
    
    print(f"\nTotal categorías cargadas: {len(servicios)}")
    return servicios

def calcular_distancia_mas_cercana(puntos_grilla, puntos_servicio):
    """
    Calcula la distancia euclidiana al servicio más cercano usando KDTree para eficiencia
    
    Args:
        puntos_grilla: GeoDataFrame con puntos de la grilla
        puntos_servicio: GeoDataFrame con puntos de servicio
        
    Returns:
        Array con distancias en metros
    """
    if puntos_servicio.empty:
        return np.full(len(puntos_grilla), np.inf)
    
    # Extraer coordenadas
    coords_grilla = np.column_stack([puntos_grilla.geometry.x, puntos_grilla.geometry.y])
    coords_servicio = np.column_stack([puntos_servicio.geometry.x, puntos_servicio.geometry.y])
    
    # Crear KDTree para búsqueda eficiente
    tree = cKDTree(coords_servicio)
    
    # Encontrar distancias al punto más cercano
    distancias, _ = tree.query(coords_grilla)
    
    return distancias

def procesar_distancias_por_categoria(grilla, servicios):
    """
    Procesa todas las distancias por categoría y agrega columnas a la grilla
    """
    print("\n=== CALCULANDO DISTANCIAS POR CATEGORÍA ===")
    
    grilla_resultado = grilla.copy()
    
    for categoria, puntos_servicio in servicios.items():
        print(f"Procesando distancias a: {categoria}...")
        
        # Calcular distancias
        distancias = calcular_distancia_mas_cercana(grilla_resultado, puntos_servicio)
        
        # Agregar columna a la grilla
        columna_nombre = f"dist_{categoria}_m"
        grilla_resultado[columna_nombre] = distancias
        
        # Estadísticas
        if not np.all(np.isinf(distancias)):
            dist_validas = distancias[~np.isinf(distancias)]
            print(f"  ✅ Min: {dist_validas.min():.0f}m, "
                  f"Max: {dist_validas.max():.0f}m, "
                  f"Promedio: {dist_validas.mean():.0f}m")
        else:
            print(f"  ⚠️ No hay servicios de esta categoría disponibles")
    
    return grilla_resultado

def crear_distancias_agrupadas(grilla_con_distancias):
    """
    Crea distancias agrupadas por tipos de servicios principales
    """
    print("\n=== CREANDO DISTANCIAS AGRUPADAS ===")
    
    # Educación: distancia al establecimiento educativo más cercano
    columnas_educacion = [col for col in grilla_con_distancias.columns 
                         if col.startswith('dist_educacion')]
    if columnas_educacion:
        dist_educacion_array = grilla_con_distancias[columnas_educacion].values
        grilla_con_distancias['dist_educacion_min_m'] = np.min(dist_educacion_array, axis=1)
        print(f"  ✅ Distancia mínima a educación creada")
    
    # Salud: distancia al servicio de salud más cercano
    columnas_salud = [col for col in grilla_con_distancias.columns 
                     if 'salud' in col and col.startswith('dist_')]
    if columnas_salud:
        dist_salud_array = grilla_con_distancias[columnas_salud].values
        grilla_con_distancias['dist_salud_min_m'] = np.min(dist_salud_array, axis=1)
        print(f"  ✅ Distancia mínima a salud creada")
    
    # Seguridad: distancia al servicio de seguridad más cercano
    columnas_seguridad = [col for col in grilla_con_distancias.columns 
                         if 'seguridad' in col and col.startswith('dist_')]
    if columnas_seguridad:
        dist_seguridad_array = grilla_con_distancias[columnas_seguridad].values
        grilla_con_distancias['dist_seguridad_min_m'] = np.min(dist_seguridad_array, axis=1)
        print(f"  ✅ Distancia mínima a seguridad creada")
    
    # Transporte: distancia al transporte más cercano
    columnas_transporte = [col for col in grilla_con_distancias.columns 
                          if 'transporte' in col and col.startswith('dist_')]
    if columnas_transporte:
        dist_transporte_array = grilla_con_distancias[columnas_transporte].values
        grilla_con_distancias['dist_transporte_min_m'] = np.min(dist_transporte_array, axis=1)
        print(f"  ✅ Distancia mínima a transporte creada")
    
    return grilla_con_distancias

def validar_resultados_distancias(grilla_con_distancias):
    """Validaciones de los resultados de distancias calculadas"""
    print("\n=== VALIDACIÓN DE RESULTADOS ===")
    
    # Contar columnas de distancia creadas
    columnas_distancia = [col for col in grilla_con_distancias.columns 
                         if col.startswith('dist_') and col.endswith('_m')]
    print(f"Total columnas de distancia: {len(columnas_distancia)}")
    
    # Verificar rangos razonables (0-50km para Santiago)
    for col in columnas_distancia[:5]:  # Solo mostrar primeras 5
        serie = grilla_con_distancias[col]
        valores_validos = serie[~np.isinf(serie)]
        
        if len(valores_validos) > 0:
            print(f"  {col}: Min={valores_validos.min():.0f}m, Max={valores_validos.max():.0f}m")
            
            # Verificar que no hay valores negativos
            assert valores_validos.min() >= 0, f"Distancias negativas en {col}"
            
            # Verificar rango razonable para Santiago (máximo 50km)
            assert valores_validos.max() < 50000, f"Distancias muy grandes en {col}: {valores_validos.max()}"
    
    print("✅ Validación de distancias completada")

def main():
    """Función principal"""
    print("=== CÁLCULO DE CARACTERÍSTICAS DE DISTANCIA ===\n")
    
    # 1. Cargar grilla
    print("1. Cargando grilla de evaluación...")
    grilla = cargar_grilla()
    
    if grilla is None:
        return False
    
    # 2. Cargar servicios por categoría
    print("\n2. Cargando servicios por categoría...")
    servicios = cargar_servicios_por_categoria()
    
    if not servicios:
        print("❌ Error: No se pudieron cargar servicios")
        return False
    
    # 3. Calcular distancias por categoría
    print("\n3. Calculando distancias...")
    grilla_con_distancias = procesar_distancias_por_categoria(grilla, servicios)
    
    # 4. Crear distancias agrupadas
    print("\n4. Creando distancias agrupadas...")
    grilla_final = crear_distancias_agrupadas(grilla_con_distancias)
    
    # 5. Validar resultados
    print("\n5. Validando resultados...")
    validar_resultados_distancias(grilla_final)
    
    # 6. Agregar metadatos de procesamiento
    grilla_final['distancias_calculadas'] = True
    grilla_final['fecha_distancias'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 7. Guardar resultado
    print("\n6. Guardando resultado...")
    output_path = "features/grilla_con_distancias.geojson"
    grilla_final.to_file(output_path, driver='GeoJSON')
    
    print(f"   ✅ Grilla con distancias guardada: {output_path}")
    print(f"   📊 Total puntos procesados: {len(grilla_final)}")
    
    # 8. Generar reporte de distancias
    print("\n7. Generando reporte de distancias...")
    
    columnas_distancia = [col for col in grilla_final.columns 
                         if col.startswith('dist_') and col.endswith('_m')]
    
    reporte_distancias = {
        'fecha_calculo': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'puntos_procesados': len(grilla_final),
        'categorias_servicios': len(servicios),
        'columnas_distancia_creadas': len(columnas_distancia),
        'estadisticas_por_categoria': {}
    }
    
    # Estadísticas por categoría
    for col in columnas_distancia:
        serie = grilla_final[col]
        valores_validos = serie[~np.isinf(serie)]
        
        if len(valores_validos) > 0:
            reporte_distancias['estadisticas_por_categoria'][col] = {
                'puntos_con_servicio': len(valores_validos),
                'distancia_min_m': float(valores_validos.min()),
                'distancia_max_m': float(valores_validos.max()),
                'distancia_promedio_m': float(valores_validos.mean()),
                'distancia_mediana_m': float(valores_validos.median())
            }
    
    import json
    reporte_path = "reportes/caracteristicas_distancia_reporte.json"
    with open(reporte_path, 'w', encoding='utf-8') as f:
        json.dump(reporte_distancias, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Reporte guardado: {reporte_path}")
    print("\n🎉 Cálculo de distancias completado exitosamente!")
    
    return True

if __name__ == "__main__":
    exito = main()
    if not exito:
        sys.exit(1)