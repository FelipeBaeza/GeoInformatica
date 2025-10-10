#!/usr/bin/env python3
"""
Script principal para ejecutar toda la Semana 1: 
Análisis, normalización, validación y documentación
"""

import subprocess
import sys
from pathlib import Path
import time

def ejecutar_script(script_path, descripcion):
    """Ejecuta un script Python y maneja errores"""
    
    print(f"\n{'='*60}")
    print(f" EJECUTANDO: {descripcion}")
    print(f" Script: {script_path.name}")
    print('='*60)
    
    try:
        inicio = time.time()
        
        # Ejecutar script
        result = subprocess.run([
            sys.executable, str(script_path)
        ], 
        cwd=script_path.parent,
        capture_output=True, 
        text=True,
        encoding='utf-8'
        )
        
        tiempo_total = time.time() - inicio
        
        # Mostrar salida
        if result.stdout:
            print(result.stdout)
        
        if result.returncode == 0:
            print(f"\n {descripcion} completado exitosamente")
            print(f"⏱ Tiempo: {tiempo_total:.1f} segundos")
            return True
        else:
            print(f"\n Error en {descripcion}")
            if result.stderr:
                print(f"Error: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"\n Excepción ejecutando {descripcion}: {str(e)}")
        return False

def verificar_dependencias():
    """Verifica que las dependencias necesarias están instaladas"""
    
    dependencias = ['geopandas', 'pandas', 'numpy']
    faltantes = []
    
    for dep in dependencias:
        try:
            __import__(dep)
        except ImportError:
            faltantes.append(dep)
    
    if faltantes:
        print(f" Dependencias faltantes: {', '.join(faltantes)}")
        print(f" Instalar con: pip install {' '.join(faltantes)}")
        return False
    
    return True

def mostrar_resumen_final(base_path):
    """Muestra resumen final de lo completado"""
    
    print(f"\n{'='*80}")
    print(" SEMANA 1 COMPLETADA - RESUMEN FINAL")
    print('='*80)
    
    # Verificar archivos generados
    carpetas_verificar = [
        ('datos_normalizados', 'Datos normalizados a UTM 19S'),
        ('reportes', 'Reportes de análisis y validación'),
        ('features', 'Carpeta para features (creada)'),
        ('scripts', 'Scripts de procesamiento')
    ]
    
    print(f"\n ESTRUCTURA DE CARPETAS CREADA:")
    for carpeta, descripcion in carpetas_verificar:
        ruta = base_path / carpeta
        existe = "" if ruta.exists() else ""
        archivos = len(list(ruta.glob('*'))) if ruta.exists() else 0
        print(f"   {existe} {carpeta}/  ({archivos} archivos) - {descripcion}")
    
    # Reportes generados
    carpeta_reportes = base_path / "reportes"
    if carpeta_reportes.exists():
        reportes = list(carpeta_reportes.glob('*'))
        print(f"\n REPORTES GENERADOS ({len(reportes)}):")
        for reporte in sorted(reportes):
            print(f"    {reporte.name}")
    
    # Datos normalizados
    carpeta_normalizados = base_path / "datos_normalizados" 
    if carpeta_normalizados.exists():
        archivos_norm = list(carpeta_normalizados.glob('*.geojson'))
        print(f"\n DATOS NORMALIZADOS ({len(archivos_norm)}):")
        print(f"    Todos en EPSG:32719 (UTM 19S)")
        print(f"    Geometrías validadas y reparadas")
        print(f"    Áreas y distancias en metros")
    
    print(f"\n PRÓXIMOS PASOS (SEMANA 2):")
    print(f"   1. Completar estaciones de metro (OSM)")
    print(f"   2. Obtener datos de propiedades con precios")
    print(f"   3. Generar features espaciales iniciales")
    print(f"   4. Configurar ambiente para modelado")
    
    print(f"\n DOCUMENTACIÓN:")
    readme_path = carpeta_reportes / "README_datos.md"
    if readme_path.exists():
        print(f"    Guía de uso: {readme_path}")
    diccionario_path = carpeta_reportes / "diccionario_datos.json"
    if diccionario_path.exists():
        print(f"    Diccionario completo: {diccionario_path}")
    
    print(f"\n RECORDATORIO IMPORTANTE:")
    print(f"     Usar SIEMPRE archivos de datos_normalizados/")
    print(f"     NO usar archivos de datos_filtrados/ para análisis")
    print(f"    Todos los cálculos de distancia ya están en metros")

def main():
    """Función principal que ejecuta todo el pipeline de Semana 1"""
    
    print("="*80)
    print(" PIPELINE SEMANA 1 - PREPARACIÓN DE DATOS")
    print(" Normalización CRS + Validación + Documentación") 
    print("="*80)
    
    # Configurar rutas
    base_path = Path(__file__).parent.parent
    scripts_path = base_path / "scripts"
    
    print(f"\n Directorio base: {base_path}")
    print(f" Scripts: {scripts_path}")
    
    # Verificar dependencias
    print(f"\n Verificando dependencias...")
    if not verificar_dependencias():
        print(f"\n Instalar dependencias antes de continuar")
        return False
    
    print(f" Todas las dependencias están disponibles")
    
    # Scripts a ejecutar en orden
    scripts_ejecutar = [
        (scripts_path / "analizar_crs_geometrias.py", "Análisis de CRS y geometrías"),
        (scripts_path / "normalizar_crs.py", "Normalización a EPSG:32719"),
        (scripts_path / "validar_calidad.py", "Validación de calidad de datos"),
        (scripts_path / "crear_diccionario_datos.py", "Creación del diccionario de datos")
    ]
    
    # Ejecutar scripts secuencialmente
    inicio_total = time.time()
    scripts_exitosos = 0
    
    for script_path, descripcion in scripts_ejecutar:
        if not script_path.exists():
            print(f" Script no encontrado: {script_path}")
            continue
            
        exito = ejecutar_script(script_path, descripcion)
        if exito:
            scripts_exitosos += 1
        else:
            print(f" Script falló, pero continuando...")
    
    tiempo_total = time.time() - inicio_total
    
    # Resumen final
    mostrar_resumen_final(base_path)
    
    print(f"\n⏱ TIEMPO TOTAL: {tiempo_total:.1f} segundos")
    print(f" SCRIPTS EXITOSOS: {scripts_exitosos}/{len(scripts_ejecutar)}")
    
    if scripts_exitosos == len(scripts_ejecutar):
        print(f"\n SEMANA 1 COMPLETADA AL 100%!")
        return True
    else:
        print(f"\n Completada con algunos errores. Revisar logs.")
        return False

if __name__ == "__main__":
    exito = main()
    sys.exit(0 if exito else 1)