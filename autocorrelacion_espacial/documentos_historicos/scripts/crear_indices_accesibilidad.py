#!/usr/bin/env python3
"""
Script para crear índices de accesibilidad compuestos combinando distancias y densidades
para generar puntuaciones más significativas para el sistema de recomendaciones.

Autor: Proyecto GeoInformática
Fecha: Octubre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime

def cargar_grilla_con_densidades():
    """Carga la grilla con distancias y densidades calculadas"""
    grilla_path = "features/grilla_con_densidades.geojson"
    
    if not os.path.exists(grilla_path):
        print(f"❌ Error: Grilla con densidades no encontrada")
        print("   Ejecute primero: python calcular_densidades.py")
        return None
    
    grilla = gpd.read_file(grilla_path)
    print(f"✅ Grilla con densidades cargada: {len(grilla)} puntos")
    return grilla

def crear_indice_accesibilidad_educativa(grilla):
    """
    Crea índice compuesto de accesibilidad educativa combinando distancia y densidad
    """
    print("Creando índice de accesibilidad educativa...")
    
    # Inversión de distancia (más cercano = mejor puntuación)
    dist_educacion = grilla['dist_educacion_min_m']
    
    # Normalizar distancia (invertida): 0m=10, >5000m=0
    dist_max = 5000  # 5km como distancia máxima razonable
    acceso_distancia = np.clip(10 * (1 - dist_educacion / dist_max), 0, 10)
    
    # Usar densidad normalizada de 600m como compromiso entre local y regional
    acceso_densidad = grilla['dens_norm_educacion_600m_km2']
    
    # Combinar distancia (60%) y densidad (40%) para accesibilidad final
    indice_educacion = (acceso_distancia * 0.6) + (acceso_densidad * 0.4)
    
    grilla['acc_educacion'] = indice_educacion
    
    print(f"  ✅ Índice educativo: Min={indice_educacion.min():.2f}, "
          f"Max={indice_educacion.max():.2f}, Promedio={indice_educacion.mean():.2f}")
    
    return grilla

def crear_indice_accesibilidad_salud(grilla):
    """
    Crea índice compuesto de accesibilidad a servicios de salud
    """
    print("Creando índice de accesibilidad a salud...")
    
    # Inversión de distancia
    dist_salud = grilla['dist_salud_min_m']
    
    # Normalizar distancia (invertida): 0m=10, >3000m=0 (salud es más crítica)
    dist_max = 3000  # 3km para servicios de salud
    acceso_distancia = np.clip(10 * (1 - dist_salud / dist_max), 0, 10)
    
    # Usar densidad normalizada de 1000m para salud (área más amplia)
    acceso_densidad = grilla['dens_norm_salud_1000m_km2']
    
    # Combinar distancia (70%) y densidad (30%) - distancia más importante para salud
    indice_salud = (acceso_distancia * 0.7) + (acceso_densidad * 0.3)
    
    grilla['acc_salud'] = indice_salud
    
    print(f"  ✅ Índice salud: Min={indice_salud.min():.2f}, "
          f"Max={indice_salud.max():.2f}, Promedio={indice_salud.mean():.2f}")
    
    return grilla

def crear_indice_conectividad_transporte(grilla):
    """
    Crea índice de conectividad basado en transporte público
    """
    print("Creando índice de conectividad de transporte...")
    
    # Combinación de metro y estaciones de carga
    dist_transporte = grilla['dist_transporte_min_m']
    
    # Normalizar distancia: 0m=10, >2000m=0 (transporte público debe estar cerca)
    dist_max = 2000  # 2km para transporte
    acceso_distancia = np.clip(10 * (1 - dist_transporte / dist_max), 0, 10)
    
    # Densidad de transporte en 600m
    acceso_densidad = grilla['dens_norm_transporte_600m_km2']
    
    # Combinar distancia (80%) y densidad (20%) - cercanía es crucial para transporte
    indice_transporte = (acceso_distancia * 0.8) + (acceso_densidad * 0.2)
    
    grilla['acc_transporte'] = indice_transporte
    
    print(f"  ✅ Índice transporte: Min={indice_transporte.min():.2f}, "
          f"Max={indice_transporte.max():.2f}, Promedio={indice_transporte.mean():.2f}")
    
    return grilla

def crear_indice_calidad_entorno(grilla):
    """
    Crea índice de calidad del entorno basado en áreas verdes y recreación
    """
    print("Creando índice de calidad del entorno...")
    
    # Usar distancia a áreas verdes
    dist_areas_verdes = grilla['dist_areas_verdes_m']
    
    # Normalizar distancia: 0m=10, >1000m=0 (áreas verdes deben estar accesibles)
    dist_max = 1000  # 1km para áreas verdes
    acceso_distancia = np.clip(10 * (1 - dist_areas_verdes / dist_max), 0, 10)
    
    # Densidad de recreación en 300m (entorno inmediato)
    acceso_densidad = grilla['dens_norm_recreacion_300m_km2']
    
    # Combinar distancia (50%) y densidad (50%) - equilibrio para calidad de vida
    indice_entorno = (acceso_distancia * 0.5) + (acceso_densidad * 0.5)
    
    grilla['acc_entorno'] = indice_entorno
    
    print(f"  ✅ Índice entorno: Min={indice_entorno.min():.2f}, "
          f"Max={indice_entorno.max():.2f}, Promedio={indice_entorno.mean():.2f}")
    
    return grilla

def crear_indice_seguridad_percibida(grilla):
    """
    Crea índice de seguridad basado en proximidad a servicios de seguridad
    """
    print("Creando índice de seguridad percibida...")
    
    # Distancia a servicios de seguridad
    dist_seguridad = grilla['dist_seguridad_min_m']
    
    # Normalizar distancia: 0m=10, >2500m=0
    dist_max = 2500  # 2.5km para seguridad
    acceso_distancia = np.clip(10 * (1 - dist_seguridad / dist_max), 0, 10)
    
    # Densidad de seguridad en 1000m
    acceso_densidad = grilla['dens_norm_seguridad_1000m_km2']
    
    # Combinar distancia (60%) y densidad (40%)
    indice_seguridad = (acceso_distancia * 0.6) + (acceso_densidad * 0.4)
    
    grilla['acc_seguridad'] = indice_seguridad
    
    print(f"  ✅ Índice seguridad: Min={indice_seguridad.min():.2f}, "
          f"Max={indice_seguridad.max():.2f}, Promedio={indice_seguridad.mean():.2f}")
    
    return grilla

def crear_indice_comercial(grilla):
    """
    Crea índice de accesibilidad comercial
    """
    print("Creando índice comercial...")
    
    # Distancia a comercio
    dist_comercio = grilla['dist_comercio_m']
    
    # Normalizar distancia: 0m=10, >1500m=0 (comercio debe estar cerca)
    dist_max = 1500  # 1.5km para comercio
    acceso_distancia = np.clip(10 * (1 - dist_comercio / dist_max), 0, 10)
    
    # Densidad comercial en 600m
    acceso_densidad = grilla['dens_norm_comercio_600m_km2']
    
    # Combinar distancia (40%) y densidad (60%) - diversidad comercial es importante
    indice_comercial = (acceso_distancia * 0.4) + (acceso_densidad * 0.6)
    
    grilla['acc_comercial'] = indice_comercial
    
    print(f"  ✅ Índice comercial: Min={indice_comercial.min():.2f}, "
          f"Max={indice_comercial.max():.2f}, Promedio={indice_comercial.mean():.2f}")
    
    return grilla

def crear_indices_compuestos_avanzados(grilla):
    """
    Crea índices compuestos de nivel superior combinando múltiples dimensiones
    """
    print("\nCreando índices compuestos avanzados...")
    
    # Índice de Vida Urbana (educación + salud + transporte)
    grilla['idx_vida_urbana'] = (
        grilla['acc_educacion'] * 0.3 + 
        grilla['acc_salud'] * 0.4 + 
        grilla['acc_transporte'] * 0.3
    )
    
    print(f"  ✅ Índice Vida Urbana: Min={grilla['idx_vida_urbana'].min():.2f}, "
          f"Max={grilla['idx_vida_urbana'].max():.2f}, Promedio={grilla['idx_vida_urbana'].mean():.2f}")
    
    # Índice de Calidad de Vida (entorno + seguridad + comercial)
    grilla['idx_calidad_vida'] = (
        grilla['acc_entorno'] * 0.4 + 
        grilla['acc_seguridad'] * 0.3 + 
        grilla['acc_comercial'] * 0.3
    )
    
    print(f"  ✅ Índice Calidad de Vida: Min={grilla['idx_calidad_vida'].min():.2f}, "
          f"Max={grilla['idx_calidad_vida'].max():.2f}, Promedio={grilla['idx_calidad_vida'].mean():.2f}")
    
    # Índice Global de Habitabilidad (todos los factores)
    grilla['idx_habitabilidad_global'] = (
        grilla['idx_vida_urbana'] * 0.6 + 
        grilla['idx_calidad_vida'] * 0.4
    )
    
    print(f"  ✅ Índice Habitabilidad Global: Min={grilla['idx_habitabilidad_global'].min():.2f}, "
          f"Max={grilla['idx_habitabilidad_global'].max():.2f}, Promedio={grilla['idx_habitabilidad_global'].mean():.2f}")
    
    return grilla

def validar_indices_accesibilidad(grilla):
    """Validaciones de los índices de accesibilidad"""
    print("\n=== VALIDACIÓN DE ÍNDICES DE ACCESIBILIDAD ===")
    
    # Columnas de índices
    columnas_indices = [col for col in grilla.columns 
                       if col.startswith('acc_') or col.startswith('idx_')]
    
    print(f"Total índices creados: {len(columnas_indices)}")
    
    for col in columnas_indices:
        serie = grilla[col]
        print(f"  {col}: Min={serie.min():.2f}, Max={serie.max():.2f}, "
              f"Promedio={serie.mean():.2f}")
        
        # Verificar rango 0-10
        assert 0 <= serie.min() <= serie.max() <= 10, f"Rango incorrecto en {col}"
    
    print("✅ Validación de índices completada")

def main():
    """Función principal"""
    print("=== CREACIÓN DE ÍNDICES DE ACCESIBILIDAD ===\n")
    
    # 1. Cargar grilla con densidades
    print("1. Cargando grilla con densidades...")
    grilla = cargar_grilla_con_densidades()
    
    if grilla is None:
        return False
    
    # 2. Crear índices individuales de accesibilidad
    print("\n2. Creando índices de accesibilidad individuales...")
    grilla = crear_indice_accesibilidad_educativa(grilla)
    grilla = crear_indice_accesibilidad_salud(grilla)
    grilla = crear_indice_conectividad_transporte(grilla)
    grilla = crear_indice_calidad_entorno(grilla)
    grilla = crear_indice_seguridad_percibida(grilla)
    grilla = crear_indice_comercial(grilla)
    
    # 3. Crear índices compuestos avanzados
    print("\n3. Creando índices compuestos avanzados...")
    grilla = crear_indices_compuestos_avanzados(grilla)
    
    # 4. Validar resultados
    print("\n4. Validando resultados...")
    validar_indices_accesibilidad(grilla)
    
    # 5. Agregar metadatos
    grilla['indices_calculados'] = True
    grilla['fecha_indices'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # 6. Guardar resultado
    print("\n5. Guardando resultado...")
    output_path = "features/grilla_con_indices.geojson"
    grilla.to_file(output_path, driver='GeoJSON')
    
    print(f"   ✅ Grilla con índices guardada: {output_path}")
    print(f"   📊 Total puntos procesados: {len(grilla)}")
    
    # 7. Generar reporte
    print("\n6. Generando reporte de índices...")
    
    columnas_indices = [col for col in grilla.columns 
                       if col.startswith('acc_') or col.startswith('idx_')]
    
    reporte = {
        'fecha_calculo': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'puntos_procesados': len(grilla),
        'indices_individuales': len([col for col in columnas_indices if col.startswith('acc_')]),
        'indices_compuestos': len([col for col in columnas_indices if col.startswith('idx_')]),
        'estadisticas_indices': {}
    }
    
    # Estadísticas por índice
    for col in columnas_indices:
        serie = grilla[col]
        reporte['estadisticas_indices'][col] = {
            'minimo': float(serie.min()),
            'maximo': float(serie.max()),
            'promedio': float(serie.mean()),
            'mediana': float(serie.median()),
            'desviacion_estandar': float(serie.std())
        }
    
    import json
    reporte_path = "reportes/indices_accesibilidad_reporte.json"
    with open(reporte_path, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Reporte guardado: {reporte_path}")
    print("\n🎉 Creación de índices de accesibilidad completada exitosamente!")
    
    return True

if __name__ == "__main__":
    exito = main()
    if not exito:
        sys.exit(1)