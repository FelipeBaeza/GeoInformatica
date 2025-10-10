#!/usr/bin/env python3
"""
Script maestro para ejecutar todo el pipeline de la Semana 2:
Ingeniería de Características Espaciales completo.

Este script ejecuta en secuencia:
1. Generación de grilla de evaluación
2. Cálculo de distancias a servicios
3. Cálculo de densidades por buffers
4. Creación de índices de accesibilidad

Autor: Proyecto GeoInformática
Fecha: Octubre 2025
"""

import subprocess
import sys
import os
from datetime import datetime
import json

def ejecutar_script(script_name, descripcion):
    """
    Ejecuta un script Python y maneja errores
    
    Args:
        script_name: nombre del archivo .py
        descripcion: descripción del proceso
    
    Returns:
        bool: True si exitoso, False si hubo error
    """
    print(f"\n{'='*20} {descripcion.upper()} {'='*20}")
    print(f"Ejecutando: {script_name}")
    print(f"Hora de inicio: {datetime.now().strftime('%H:%M:%S')}")
    
    try:
        result = subprocess.run(
            [sys.executable, f"scripts/{script_name}"],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Mostrar output del script
        if result.stdout:
            print(result.stdout)
        
        print(f" {descripcion} completado exitosamente")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f" Error en {descripcion}")
        print(f"Código de salida: {e.returncode}")
        
        if e.stdout:
            print("STDOUT:")
            print(e.stdout)
        
        if e.stderr:
            print("STDERR:")
            print(e.stderr)
        
        return False

def verificar_dependencias():
    """Verifica que las dependencias necesarias estén instaladas"""
    print("Verificando dependencias del entorno...")
    
    dependencias_requeridas = [
        'geopandas', 'pandas', 'numpy', 'scipy', 'shapely'
    ]
    
    dependencias_faltantes = []
    
    for dep in dependencias_requeridas:
        try:
            __import__(dep)
            print(f"   {dep}")
        except ImportError:
            dependencias_faltantes.append(dep)
            print(f"   {dep}")
    
    if dependencias_faltantes:
        print(f"\n Dependencias faltantes: {', '.join(dependencias_faltantes)}")
        print("Ejecute: pip install " + " ".join(dependencias_faltantes))
        return False
    
    print(" Todas las dependencias están disponibles")
    return True

def verificar_estructura_directorios():
    """Verifica y crea la estructura de directorios necesaria"""
    print("Verificando estructura de directorios...")
    
    directorios_necesarios = [
        'scripts',
        'features', 
        'reportes',
        'datos_normalizados'
    ]
    
    for directorio in directorios_necesarios:
        if not os.path.exists(directorio):
            print(f"   Creando directorio: {directorio}")
            os.makedirs(directorio, exist_ok=True)
        else:
            print(f"   {directorio}")
    
    return True

def generar_reporte_pipeline(resultados):
    """Genera un reporte resumido del pipeline completo"""
    print("\n" + "="*60)
    print("GENERANDO REPORTE FINAL DEL PIPELINE")
    print("="*60)
    
    # Estadísticas generales
    total_exitosos = sum(1 for r in resultados if r['exitoso'])
    total_scripts = len(resultados)
    
    reporte = {
        'fecha_ejecucion': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'pipeline_completo': total_exitosos == total_scripts,
        'scripts_ejecutados': total_scripts,
        'scripts_exitosos': total_exitosos,
        'scripts_fallidos': total_scripts - total_exitosos,
        'detalles_ejecucion': resultados
    }
    
    # Información de archivos generados
    archivos_esperados = [
        'features/grilla_evaluacion_santiago.geojson',
        'features/grilla_con_distancias.geojson', 
        'features/grilla_con_densidades.geojson',
        'features/grilla_con_indices.geojson'
    ]
    
    archivos_generados = []
    for archivo in archivos_esperados:
        if os.path.exists(archivo):
            stat_info = os.stat(archivo)
            archivos_generados.append({
                'archivo': archivo,
                'tamaño_mb': round(stat_info.st_size / (1024*1024), 2),
                'fecha_modificacion': datetime.fromtimestamp(stat_info.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            })
    
    reporte['archivos_generados'] = archivos_generados
    
    # Guardar reporte
    reporte_path = "reportes/pipeline_semana2_reporte.json"
    with open(reporte_path, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    # Mostrar resumen
    print(f"Pipeline ejecutado: {total_exitosos}/{total_scripts} scripts exitosos")
    print(f"Archivos generados: {len(archivos_generados)}/{len(archivos_esperados)}")
    print(f"Reporte completo guardado en: {reporte_path}")
    
    if reporte['pipeline_completo']:
        print("\n PIPELINE DE SEMANA 2 COMPLETADO EXITOSAMENTE")
        print("   Características espaciales listas para sistema de recomendaciones")
    else:
        print(f"\n Pipeline incompleto - {total_scripts - total_exitosos} errores")
    
    return reporte

def main():
    """Función principal del pipeline"""
    inicio_total = datetime.now()
    
    print("="*80)
    print("PIPELINE COMPLETO - SEMANA 2: INGENIERÍA DE CARACTERÍSTICAS ESPACIALES")
    print("="*80)
    print(f"Iniciado: {inicio_total.strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nObjetivo: Generar características espaciales para sistema de recomendaciones")
    print("Ubicación: Santiago Metropolitano (4 comunas)")
    
    # Verificaciones previas
    print(f"\n{'='*20} VERIFICACIONES PREVIAS {'='*20}")
    
    if not verificar_dependencias():
        print(" Faltan dependencias. Abortando pipeline.")
        return False
    
    if not verificar_estructura_directorios():
        print(" Error en estructura de directorios. Abortando pipeline.")
        return False
    
    # Definir secuencia de scripts
    pipeline_scripts = [
        {
            'script': 'generar_grilla.py',
            'descripcion': 'Generación de grilla de evaluación',
            'duracion_estimada': '30 segundos'
        },
        {
            'script': 'calcular_distancias.py', 
            'descripcion': 'Cálculo de distancias a servicios',
            'duracion_estimada': '2-3 minutos'
        },
        {
            'script': 'calcular_densidades.py',
            'descripcion': 'Cálculo de densidades por buffers', 
            'duracion_estimada': '5-10 minutos'
        },
        {
            'script': 'crear_indices_accesibilidad.py',
            'descripcion': 'Creación de índices de accesibilidad',
            'duracion_estimada': '30 segundos'
        }
    ]
    
    print(f"\nPipeline programado: {len(pipeline_scripts)} etapas")
    for i, etapa in enumerate(pipeline_scripts, 1):
        print(f"  {i}. {etapa['descripcion']} (~{etapa['duracion_estimada']})")
    
    input(f"\n¿Proceder con la ejecución del pipeline? (Presione Enter para continuar o Ctrl+C para cancelar)")
    
    # Ejecutar pipeline
    print(f"\n{'='*20} INICIANDO PIPELINE {'='*20}")
    
    resultados = []
    
    for i, etapa in enumerate(pipeline_scripts, 1):
        inicio_etapa = datetime.now()
        
        print(f"\n[ETAPA {i}/{len(pipeline_scripts)}]")
        exitoso = ejecutar_script(etapa['script'], etapa['descripcion'])
        
        fin_etapa = datetime.now()
        duracion = fin_etapa - inicio_etapa
        
        resultado = {
            'etapa': i,
            'script': etapa['script'],
            'descripcion': etapa['descripcion'],
            'exitoso': exitoso,
            'inicio': inicio_etapa.strftime('%H:%M:%S'),
            'fin': fin_etapa.strftime('%H:%M:%S'),
            'duracion_segundos': duracion.total_seconds()
        }
        
        resultados.append(resultado)
        
        print(f"Duración de etapa: {duracion.total_seconds():.1f} segundos")
        
        # Si falla una etapa, preguntar si continuar
        if not exitoso:
            respuesta = input(f"\n Etapa {i} falló. ¿Continuar con siguiente etapa? (s/N): ")
            if respuesta.lower() not in ['s', 'si', 'sí', 'y', 'yes']:
                print("Pipeline abortado por el usuario.")
                break
    
    # Generar reporte final
    fin_total = datetime.now()
    duracion_total = fin_total - inicio_total
    
    print(f"\n{'='*20} PIPELINE FINALIZADO {'='*20}")
    print(f"Duración total: {duracion_total.total_seconds():.1f} segundos")
    
    reporte_final = generar_reporte_pipeline(resultados)
    
    return reporte_final['pipeline_completo']

if __name__ == "__main__":
    try:
        exito = main()
        if not exito:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n Pipeline interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\n Error inesperado en pipeline: {e}")
        sys.exit(1)