#!/usr/bin/env python3
"""
Script para calcular densidades de servicios en diferentes radios alrededor
de cada punto de la grilla de evaluación.

Autor: Proyecto GeoInformática
Fecha: Octubre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from shapely.geometry import Point
import os
import sys
from datetime import datetime

def cargar_grilla_con_distancias():
    """Carga la grilla con distancias ya calculadas"""
    grilla_path = "features/grilla_con_distancias.geojson"
    
    if not os.path.exists(grilla_path):
        print(f"❌ Error: Grilla con distancias no encontrada")
        print("   Ejecute primero: python calcular_distancias.py")
        return None
    
    grilla = gpd.read_file(grilla_path)
    print(f"✅ Grilla con distancias cargada: {len(grilla)} puntos")
    return grilla

def cargar_servicios_para_densidad():
    """
    Carga servicios organizados por categorías para cálculo de densidades
    """
    servicios = {}
    base_path = "datos_normalizados"
    
    # Categorías principales para densidad
    categorias = {
        'educacion': ['establecimientos_educacion_escolar.geojson',
                     'establecimientos_educacion_superior.geojson', 
                     'establecimientos_parvularia_filtrados.geojson'],
        'salud': ['puntos_medicos_farmacias_hospitales_filtrados.geojson',
                 'redes_de_clinicas_filtradas.geojson'],
        'comercio': ['tiendas_filtradas.geojson',
                    'puntos_de_interes_filtrados.geojson'],
        'seguridad': ['unidades_operativas_pdi_filtradas.geojson',
                     'cuarteles_filtrados.geojson',
                     'cuerpos_de_bomberos_filtrados.geojson'],
        'transporte': ['estaciones_carga_filtradas.geojson'],
        'recreacion': ['areas_verdes_filtradas.geojson',
                      'ocio_filtrado.geojson',
                      'atracciones_turisticas_filtradas.geojson']
    }
    
    print("Cargando servicios para cálculo de densidades...")
    
    for categoria, archivos in categorias.items():
        puntos_categoria = []
        
        for archivo in archivos:
            archivo_path = os.path.join(base_path, archivo)
            
            if os.path.exists(archivo_path):
                try:
                    gdf = gpd.read_file(archivo_path)
                    
                    if not gdf.empty:
                        # Convertir a puntos si es necesario
                        if gdf.geometry.geom_type.iloc[0] != 'Point':
                            gdf_puntos = gdf.copy()
                            gdf_puntos['geometry'] = gdf_puntos.geometry.centroid
                            puntos_categoria.append(gdf_puntos)
                        else:
                            puntos_categoria.append(gdf)
                            
                except Exception as e:
                    print(f"  ⚠️ Error cargando {archivo}: {e}")
        
        # Combinar todos los puntos de la categoría
        if puntos_categoria:
            servicios_combinados = gpd.GeoDataFrame(
                pd.concat(puntos_categoria, ignore_index=True),
                crs='EPSG:32719'
            )
            servicios[categoria] = servicios_combinados
            print(f"  ✅ {categoria}: {len(servicios_combinados)} puntos")
        else:
            print(f"  ⚠️ {categoria}: Sin datos disponibles")
    
    print(f"\nCategorías cargadas para densidad: {len(servicios)}")
    return servicios

def calcular_densidad_en_buffer(punto, servicios_gdf, radio_metros):
    """
    Calcula la densidad de servicios en un buffer circular alrededor de un punto
    
    Args:
        punto: geometría Point
        servicios_gdf: GeoDataFrame con servicios
        radio_metros: radio del buffer en metros
        
    Returns:
        densidad por km² (float)
    """
    if servicios_gdf.empty:
        return 0.0
    
    # Crear buffer circular
    buffer_circular = punto.buffer(radio_metros)
    area_buffer_km2 = buffer_circular.area / 1e6  # convertir m² a km²
    
    # Contar servicios dentro del buffer
    servicios_dentro = servicios_gdf[servicios_gdf.intersects(buffer_circular)]
    count_servicios = len(servicios_dentro)
    
    # Calcular densidad por km²
    if area_buffer_km2 > 0:
        densidad = count_servicios / area_buffer_km2
    else:
        densidad = 0.0
    
    return densidad

def procesar_densidades_por_categoria_y_radio(grilla, servicios):
    """
    Calcula densidades para todas las categorías y todos los radios
    """
    print("\n=== CALCULANDO DENSIDADES POR CATEGORÍA Y RADIO ===")
    
    grilla_resultado = grilla.copy()
    radios = [300, 600, 1000]  # metros
    
    total_calculos = len(servicios) * len(radios)
    calculo_actual = 0
    
    for categoria, servicios_gdf in servicios.items():
        print(f"\nProcesando densidades para: {categoria}")
        
        for radio in radios:
            calculo_actual += 1
            print(f"  Radio {radio}m ({calculo_actual}/{total_calculos})...")
            
            # Calcular densidad para cada punto de la grilla
            densidades = []
            
            for idx, row in grilla_resultado.iterrows():
                densidad = calcular_densidad_en_buffer(
                    row.geometry, servicios_gdf, radio
                )
                densidades.append(densidad)
            
            # Agregar columna a la grilla
            columna_nombre = f"dens_{categoria}_{radio}m_km2"
            grilla_resultado[columna_nombre] = densidades
            
            # Estadísticas
            densidades_array = np.array(densidades)
            densidades_no_cero = densidades_array[densidades_array > 0]
            
            if len(densidades_no_cero) > 0:
                print(f"    ✅ Min: {densidades_no_cero.min():.2f}, "
                      f"Max: {densidades_no_cero.max():.2f}, "
                      f"Promedio: {densidades_no_cero.mean():.2f} servicios/km²")
                print(f"    📊 {len(densidades_no_cero)}/{len(densidades_array)} puntos con servicios")
            else:
                print(f"    ⚠️ No hay servicios en radio {radio}m para esta categoría")
    
    return grilla_resultado

def crear_indices_densidad_compuesta(grilla_con_densidades):
    """
    Crea índices de densidad compuesta combinando múltiples categorías y radios
    """
    print("\n=== CREANDO ÍNDICES DE DENSIDAD COMPUESTA ===")
    
    grilla_resultado = grilla_con_densidades.copy()
    
    # Índice de densidad urbana general (combinando todas las categorías)
    for radio in [300, 600, 1000]:
        columnas_densidad_radio = [col for col in grilla_resultado.columns 
                                  if col.startswith(f'dens_') and f'{radio}m_km2' in col]
        
        if columnas_densidad_radio:
            # Sumar todas las densidades del radio
            densidad_total = grilla_resultado[columnas_densidad_radio].sum(axis=1)
            grilla_resultado[f'dens_total_{radio}m_km2'] = densidad_total
            
            print(f"  ✅ Densidad total para radio {radio}m creada")
            print(f"    Promedio: {densidad_total.mean():.2f} servicios/km²")
    
    # Índice de diversidad de servicios (número de tipos diferentes disponibles)
    for radio in [300, 600, 1000]:
        diversidad = []
        
        for idx, row in grilla_resultado.iterrows():
            tipos_disponibles = 0
            
            # Contar cuántos tipos de servicios están disponibles (densidad > 0)
            for categoria in ['educacion', 'salud', 'comercio', 'seguridad', 'transporte', 'recreacion']:
                col_nombre = f'dens_{categoria}_{radio}m_km2'
                if col_nombre in grilla_resultado.columns and row[col_nombre] > 0:
                    tipos_disponibles += 1
            
            diversidad.append(tipos_disponibles)
        
        grilla_resultado[f'diversidad_servicios_{radio}m'] = diversidad
        print(f"  ✅ Índice de diversidad para radio {radio}m creado")
        print(f"    Diversidad promedio: {np.mean(diversidad):.2f}/6 tipos de servicios")
    
    return grilla_resultado

def normalizar_indices_densidad(grilla_con_indices):
    """
    Normaliza los índices de densidad a escala 0-10 para facilitar interpretación
    """
    print("\n=== NORMALIZANDO ÍNDICES DE DENSIDAD ===")
    
    grilla_resultado = grilla_con_indices.copy()
    
    # Columnas a normalizar (densidades y diversidad)
    columnas_a_normalizar = [col for col in grilla_resultado.columns 
                            if col.startswith('dens_') or col.startswith('diversidad_')]
    
    for col in columnas_a_normalizar:
        serie = grilla_resultado[col]
        
        # Normalización min-max a escala 0-10
        min_val = serie.min()
        max_val = serie.max()
        
        if max_val > min_val:
            serie_normalizada = ((serie - min_val) / (max_val - min_val)) * 10
            col_normalizado = col.replace('dens_', 'dens_norm_').replace('diversidad_', 'div_norm_')
            grilla_resultado[col_normalizado] = serie_normalizada
            
            print(f"  ✅ {col} → {col_normalizado} (escala 0-10)")
    
    return grilla_resultado

def validar_resultados_densidad(grilla_con_densidades):
    """Validaciones de los resultados de densidad"""
    print("\n=== VALIDACIÓN DE RESULTADOS DE DENSIDAD ===")
    
    # Contar columnas de densidad
    columnas_densidad = [col for col in grilla_con_densidades.columns 
                        if col.startswith('dens_') and 'km2' in col]
    columnas_normalizadas = [col for col in grilla_con_densidades.columns 
                            if col.startswith('dens_norm_')]
    
    print(f"Columnas de densidad creadas: {len(columnas_densidad)}")
    print(f"Columnas normalizadas creadas: {len(columnas_normalizadas)}")
    
    # Verificar rangos
    for col in columnas_densidad[:3]:  # Solo primeras 3
        serie = grilla_con_densidades[col]
        print(f"  {col}: Min={serie.min():.2f}, Max={serie.max():.2f}")
        
        # Verificar que no hay valores negativos
        assert serie.min() >= 0, f"Densidades negativas en {col}"
    
    # Verificar normalización
    for col in columnas_normalizadas[:3]:  # Solo primeras 3
        serie = grilla_con_densidades[col]
        print(f"  {col}: Min={serie.min():.2f}, Max={serie.max():.2f}")
        
        # Verificar rango 0-10
        assert 0 <= serie.min() <= serie.max() <= 10, f"Normalización incorrecta en {col}"
    
    print("✅ Validación de densidades completada")

def main():
    """Función principal"""
    print("=== CÁLCULO DE CARACTERÍSTICAS DE DENSIDAD ===\n")
    
    # 1. Cargar grilla con distancias
    print("1. Cargando grilla con distancias...")
    grilla = cargar_grilla_con_distancias()
    
    if grilla is None:
        return False
    
    # 2. Cargar servicios para densidad
    print("\n2. Cargando servicios por categoría...")
    servicios = cargar_servicios_para_densidad()
    
    if not servicios:
        print("❌ Error: No se pudieron cargar servicios para densidad")
        return False
    
    # 3. Calcular densidades por categoría y radio
    print("\n3. Calculando densidades...")
    grilla_con_densidades = procesar_densidades_por_categoria_y_radio(grilla, servicios)
    
    # 4. Crear índices compuestos
    print("\n4. Creando índices de densidad compuesta...")
    grilla_con_indices = crear_indices_densidad_compuesta(grilla_con_densidades)
    
    # 5. Normalizar índices
    print("\n5. Normalizando índices a escala 0-10...")
    grilla_final = normalizar_indices_densidad(grilla_con_indices)
    
    # 6. Validar resultados
    print("\n6. Validando resultados...")
    validar_resultados_densidad(grilla_final)
    
    # 7. Agregar metadatos
    grilla_final['densidades_calculadas'] = True
    grilla_final['fecha_densidades'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 8. Guardar resultado
    print("\n7. Guardando resultado...")
    output_path = "features/grilla_con_densidades.geojson"
    grilla_final.to_file(output_path, driver='GeoJSON')
    
    print(f"   ✅ Grilla con densidades guardada: {output_path}")
    print(f"   📊 Total puntos procesados: {len(grilla_final)}")
    
    # 9. Generar reporte
    print("\n8. Generando reporte de densidades...")
    
    columnas_densidad = [col for col in grilla_final.columns 
                        if col.startswith('dens_') and 'km2' in col]
    columnas_normalizadas = [col for col in grilla_final.columns 
                            if col.startswith('dens_norm_')]
    
    reporte = {
        'fecha_calculo': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'puntos_procesados': len(grilla_final),
        'categorias_servicios': len(servicios),
        'radios_calculados': [300, 600, 1000],
        'columnas_densidad_creadas': len(columnas_densidad),
        'columnas_normalizadas_creadas': len(columnas_normalizadas),
        'resumen_por_categoria': {}
    }
    
    # Estadísticas por categoría
    for categoria in ['educacion', 'salud', 'comercio', 'seguridad', 'transporte', 'recreacion']:
        cols_categoria = [col for col in columnas_densidad if f'dens_{categoria}_' in col]
        if cols_categoria:
            densidades_categoria = grilla_final[cols_categoria].mean().mean()
            reporte['resumen_por_categoria'][categoria] = {
                'densidad_promedio_km2': float(densidades_categoria),
                'columnas_generadas': len(cols_categoria)
            }
    
    import json
    reporte_path = "reportes/caracteristicas_densidad_reporte.json"
    with open(reporte_path, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Reporte guardado: {reporte_path}")
    print("\n🎉 Cálculo de densidades completado exitosamente!")
    
    return True

if __name__ == "__main__":
    exito = main()
    if not exito:
        sys.exit(1)