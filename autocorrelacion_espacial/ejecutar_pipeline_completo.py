#!/usr/bin/env python3
"""
=============================================================================
PIPELINE COMPLETO - PROYECTO TERRAMATCH
=============================================================================

Sistema de Recomendación Inmobiliaria basado en Análisis Geoespacial

Este script ejecuta el pipeline completo del proyecto en 3 semanas:
    - Semana 1: Preparación y limpieza de datos
    - Semana 2: Generación de características espaciales (grilla + métricas)
    - Semana 3: Modelo LightGBM + Visualizaciones

Autor: Proyecto GeoInformática
Fecha: Diciembre 2025
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime
import time
import shutil

# =============================================================================
# CONFIGURACIÓN
# =============================================================================

# Directorio base (donde está este script)
BASE_DIR = Path(__file__).parent.resolve()

SEMANA1_DIR = BASE_DIR / 'semana1_preparacion_datos'
SEMANA2_DIR = BASE_DIR / 'semana2_caracteristicas_espaciales'
SEMANA3_DIR = BASE_DIR / 'semana3_modelo_satisfaccion'

# Carpeta de resultados (se inicializa en main)
RESULTADOS_DIR = None

# Colores para terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(texto):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}")
    print(f" {texto}")
    print(f"{'='*70}{Colors.END}\n")

def print_success(texto):
    print(f"{Colors.GREEN}✓ {texto}{Colors.END}")

def print_error(texto):
    print(f"{Colors.RED}✗ {texto}{Colors.END}")

def print_info(texto):
    print(f"{Colors.BLUE}ℹ {texto}{Colors.END}")

def print_warning(texto):
    print(f"{Colors.YELLOW}⚠ {texto}{Colors.END}")

# =============================================================================
# VERIFICACIONES
# =============================================================================

def verificar_dependencias():
    """Verifica que las dependencias necesarias estén instaladas"""
    print_info("Verificando dependencias...")
    
    dependencias = [
        'geopandas', 'pandas', 'numpy', 'scipy', 'shapely',
        'sklearn', 'matplotlib', 'seaborn', 'folium', 'matplotlib_scalebar'
    ]
    
    faltantes = []
    for dep in dependencias:
        try:
            if dep == 'sklearn':
                __import__('sklearn')
            else:
                __import__(dep)
        except ImportError:
            faltantes.append(dep)
    
    # Verificar LightGBM
    try:
        import lightgbm
    except ImportError:
        faltantes.append('lightgbm')
    
    if faltantes:
        print_warning(f"Dependencias faltantes: {', '.join(faltantes)}")
        print_info(f"Instalando: pip install {' '.join(faltantes)}")
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q'] + faltantes)
        print_success("Dependencias instaladas")
    else:
        print_success("Todas las dependencias disponibles")
    
    return True

def verificar_estructura():
    """Verifica que la estructura de directorios exista"""
    print_info("Verificando estructura de directorios...")
    
    directorios = [
        SEMANA1_DIR / 'scripts',
        SEMANA1_DIR / 'datos_normalizados',
        SEMANA1_DIR / 'reportes',
        SEMANA2_DIR / 'scripts',
        SEMANA2_DIR / 'features',
        SEMANA2_DIR / 'reportes',
        SEMANA3_DIR / 'scripts',
        SEMANA3_DIR / 'resultados',
        SEMANA3_DIR / 'graficos',
        SEMANA3_DIR / 'modelos',
    ]
    
    for d in directorios:
        d.mkdir(parents=True, exist_ok=True)
    
    print_success("Estructura de directorios verificada")
    return True

def crear_carpeta_resultados():
    """Crea la carpeta de resultados con subcarpetas por semana (sobrescribe existente)"""
    global RESULTADOS_DIR
    
    RESULTADOS_DIR = BASE_DIR / 'resultados_pipeline'
    
    # Limpiar carpeta existente si existe
    if RESULTADOS_DIR.exists():
        shutil.rmtree(RESULTADOS_DIR)
    
    # Crear estructura de subcarpetas
    subcarpetas = [
        RESULTADOS_DIR / 'semana1_preparacion_datos' / 'datos_normalizados',
        RESULTADOS_DIR / 'semana1_preparacion_datos' / 'reportes',
        RESULTADOS_DIR / 'semana2_caracteristicas_espaciales' / 'features',
        RESULTADOS_DIR / 'semana2_caracteristicas_espaciales' / 'reportes',
        RESULTADOS_DIR / 'semana3_modelo_satisfaccion' / 'resultados',
        RESULTADOS_DIR / 'semana3_modelo_satisfaccion' / 'graficos',
        RESULTADOS_DIR / 'semana3_modelo_satisfaccion' / 'modelos',
    ]
    
    for carpeta in subcarpetas:
        carpeta.mkdir(parents=True, exist_ok=True)
    
    print_success(f"Carpeta de resultados creada: {RESULTADOS_DIR.name}")
    return RESULTADOS_DIR

def copiar_resultados_semana1():
    """Copia los resultados de semana 1 a la carpeta de resultados"""
    if RESULTADOS_DIR is None:
        return
    
    destino = RESULTADOS_DIR / 'semana1_preparacion_datos'
    
    # Copiar datos normalizados
    origen_datos = SEMANA1_DIR / 'datos_normalizados'
    if origen_datos.exists():
        for archivo in origen_datos.glob('*.geojson'):
            shutil.copy2(archivo, destino / 'datos_normalizados' / archivo.name)
    
    # Copiar reportes
    origen_reportes = SEMANA1_DIR / 'reportes'
    if origen_reportes.exists():
        for archivo in origen_reportes.glob('*'):
            if archivo.is_file():
                shutil.copy2(archivo, destino / 'reportes' / archivo.name)
    
    print_info(f"Resultados Semana 1 copiados a: {destino.relative_to(BASE_DIR)}")

def copiar_resultados_semana2():
    """Copia los resultados de semana 2 a la carpeta de resultados"""
    if RESULTADOS_DIR is None:
        return
    
    destino = RESULTADOS_DIR / 'semana2_caracteristicas_espaciales'
    
    # Copiar features (grillas)
    origen_features = SEMANA2_DIR / 'features'
    if origen_features.exists():
        for archivo in origen_features.glob('*.geojson'):
            shutil.copy2(archivo, destino / 'features' / archivo.name)
    
    # Copiar reportes
    origen_reportes = SEMANA2_DIR / 'reportes'
    if origen_reportes.exists():
        for archivo in origen_reportes.glob('*'):
            if archivo.is_file():
                shutil.copy2(archivo, destino / 'reportes' / archivo.name)
    
    print_info(f"Resultados Semana 2 copiados a: {destino.relative_to(BASE_DIR)}")

def copiar_resultados_semana3():
    """Copia los resultados de semana 3 a la carpeta de resultados"""
    if RESULTADOS_DIR is None:
        return
    
    destino = RESULTADOS_DIR / 'semana3_modelo_satisfaccion'
    
    # Copiar resultados (CSVs)
    origen_resultados = SEMANA3_DIR / 'resultados'
    if origen_resultados.exists():
        for archivo in origen_resultados.rglob('*'):
            if archivo.is_file():
                # Mantener estructura de subdirectorios
                rel_path = archivo.relative_to(origen_resultados)
                dest_path = destino / 'resultados' / rel_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(archivo, dest_path)
    
    # Copiar gráficos
    origen_graficos = SEMANA3_DIR / 'graficos'
    if origen_graficos.exists():
        for archivo in origen_graficos.glob('*'):
            if archivo.is_file():
                shutil.copy2(archivo, destino / 'graficos' / archivo.name)
    
    # Copiar modelos
    origen_modelos = SEMANA3_DIR / 'modelos'
    if origen_modelos.exists():
        for archivo in origen_modelos.glob('*'):
            if archivo.is_file():
                shutil.copy2(archivo, destino / 'modelos' / archivo.name)
    
    print_info(f"Resultados Semana 3 copiados a: {destino.relative_to(BASE_DIR)}")

# =============================================================================
# EJECUCIÓN DE SCRIPTS
# =============================================================================

def ejecutar_script(script_path, descripcion, cwd=None):
    """Ejecuta un script Python y maneja errores"""
    print(f"\n   Ejecutando: {descripcion}")
    print(f"   Script: {script_path.name}")
    
    inicio = time.time()
    
    try:
        # Configurar entorno para permitir UTF-8 en scripts hijos
        env = os.environ.copy()
        env['PYTHONIOENCODING'] = 'utf-8'
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=cwd or script_path.parent,
            capture_output=True,
            text=True,
            encoding='utf-8',
            errors='replace',
            env=env
        )
        
        duracion = time.time() - inicio
        
        if result.returncode == 0:
            print_success(f"{descripcion} completado ({duracion:.1f}s)")
            return True
        else:
            print_error(f"Error en {descripcion}")
            if result.stderr:
                print(f"   {result.stderr[:500]}")
            return False
            
    except Exception as e:
        print_error(f"Excepción: {str(e)}")
        return False

# =============================================================================
# SEMANA 1: PREPARACIÓN DE DATOS
# =============================================================================

def ejecutar_semana1():
    """
    SEMANA 1: Preparación de Datos
    
    Procesos:
    - Recolección de 29 datasets
    - Filtrado por área de interés (4 comunas)
    - Normalización a UTM 19S (EPSG:32719)
    - Validación de calidad geométrica
    - Eliminación de duplicados espaciales
    """
    print_header("SEMANA 1: PREPARACIÓN DE DATOS")
    
    print("""
    Objetivos:
    ├── Recolectar y organizar 29 datasets de servicios urbanos
    ├── Filtrar datos de 4 comunas: Santiago, Ñuñoa, La Reina, Estación Central
    ├── Normalizar coordenadas a UTM 19S (EPSG:32719)
    ├── Validar calidad geométrica
    └── Eliminar duplicados espaciales
    """)
    
    scripts = [
        ('analizar_crs_geometrias.py', 'Análisis de CRS y geometrías'),
        ('normalizar_crs.py', 'Normalización a EPSG:32719'),
        ('validar_calidad.py', 'Validación de calidad de datos'),
        ('crear_diccionario_datos.py', 'Creación del diccionario de datos'),
    ]
    
    resultados = []
    for script, descripcion in scripts:
        script_path = SEMANA1_DIR / 'scripts' / script
        if script_path.exists():
            exito = ejecutar_script(script_path, descripcion)
            resultados.append(exito)
        else:
            print_warning(f"Script no encontrado: {script}")
            resultados.append(False)
    
    exitosos = sum(resultados)
    print(f"\n   Semana 1: {exitosos}/{len(scripts)} scripts completados")
    
    return exitosos == len(scripts)

# =============================================================================
# SEMANA 2: CARACTERÍSTICAS ESPACIALES
# =============================================================================

def ejecutar_semana2():
    """
    SEMANA 2: Ingeniería de Características Espaciales
    
    Procesos:
    - Generación de grilla regular de evaluación (200m espaciado)
    - Cálculo de 21 métricas de distancia euclidiana
    - Cálculo de densidades en radios 300m, 600m, 1000m
    - Creación de índices de accesibilidad compuestos
    """
    print_header("SEMANA 2: CARACTERÍSTICAS ESPACIALES")
    
    print("""
    Objetivos:
    ├── Generar grilla regular de ~3,149 puntos de evaluación
    ├── Calcular distancias euclidianas a 21 categorías de servicios
    ├── Calcular densidades en buffers de 300m, 600m, 1000m
    └── Crear índices de accesibilidad compuestos
    """)
    
    scripts = [
        ('generar_grilla.py', 'Generación de grilla de evaluación'),
        ('calcular_distancias.py', 'Cálculo de distancias a servicios'),
        ('calcular_densidades.py', 'Cálculo de densidades por buffers'),
        ('crear_indices_accesibilidad.py', 'Creación de índices de accesibilidad'),
    ]
    
    resultados = []
    for script, descripcion in scripts:
        script_path = SEMANA2_DIR / 'scripts' / script
        if script_path.exists():
            exito = ejecutar_script(script_path, descripcion, cwd=SEMANA2_DIR)
            resultados.append(exito)
        else:
            print_warning(f"Script no encontrado: {script}")
            resultados.append(False)
    
    exitosos = sum(resultados)
    print(f"\n   Semana 2: {exitosos}/{len(scripts)} scripts completados")
    
    # Verificar archivos generados
    archivos_esperados = [
        'features/grilla_evaluacion_santiago.geojson',
        'features/grilla_con_distancias.geojson',
        'features/grilla_con_densidades.geojson',
        'features/grilla_con_indices.geojson',
    ]
    
    print("\n   Archivos generados:")
    for archivo in archivos_esperados:
        path = SEMANA2_DIR / Path(archivo)
        if path.exists():
            size_mb = path.stat().st_size / (1024*1024)
            print_success(f"{archivo} ({size_mb:.2f} MB)")
        else:
            print_warning(f"{archivo} no encontrado")
    
    return exitosos == len(scripts)

# =============================================================================
# SEMANA 3: MODELO Y VISUALIZACIONES
# =============================================================================

def ejecutar_semana3():
    """
    SEMANA 3: Modelo de Satisfacción y Visualizaciones
    
    Procesos:
    - Integración de propiedades con características espaciales (matching)
    - Entrenamiento de modelo LightGBM
    - Generación de 3 mapas temáticos
    - Generación de 5 gráficos estadísticos
    - Visualización interactiva
    """
    print_header("SEMANA 3: MODELO LIGHTGBM Y VISUALIZACIONES")
    
    print("""
    Objetivos:
    ├── Integrar propiedades con características espaciales (KDTree matching)
    ├── Entrenar modelo LightGBM para predicción de satisfacción
    ├── Generar 3 mapas temáticos con elementos cartográficos
    ├── Generar 5 gráficos estadísticos
    └── Crear visualización interactiva funcional
    """)
    
    scripts = [
        ('modelo_satisfaccion.py', 'Modelo LightGBM de satisfacción'),
        ('generar_visualizaciones.py', 'Generación de visualizaciones'),
    ]
    
    resultados = []
    for script, descripcion in scripts:
        script_path = SEMANA3_DIR / 'scripts' / script
        if script_path.exists():
            exito = ejecutar_script(script_path, descripcion, cwd=SEMANA3_DIR)
            resultados.append(exito)
        else:
            print_warning(f"Script no encontrado: {script}")
            resultados.append(False)
    
    exitosos = sum(resultados)
    print(f"\n   Semana 3: {exitosos}/{len(scripts)} scripts completados")
    
    # Verificar resultados
    print("\n   Resultados generados:")
    resultados_esperados = [
        ('resultados/modelo_venta/propiedades_venta_con_satisfaccion.csv', 'Dataset con predicciones'),
        ('modelos/lightgbm_satisfaccion.pkl', 'Modelo entrenado'),
        ('graficos/mapa_interactivo.html', 'Mapa interactivo'),
    ]
    
    for archivo, desc in resultados_esperados:
        path = SEMANA3_DIR / Path(archivo)
        if path.exists():
            print_success(f"{desc}: {archivo}")
        else:
            print_warning(f"{desc} no encontrado")
    
    return exitosos == len(scripts)

# =============================================================================
# MAIN
# =============================================================================

def main():
    """Función principal del pipeline"""
    
    inicio = datetime.now()
    
    print(f"Inicio: {inicio.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Verificaciones previas
    print_header("VERIFICACIONES PREVIAS")
    verificar_dependencias()
    verificar_estructura()
    
    # Crear carpeta de resultados
    carpeta_resultados = crear_carpeta_resultados()
    
    # Ejecutar pipeline completo (3 semanas)
    print_info("Ejecutando pipeline completo (Semanas 1, 2 y 3)...")
    
    resultados = {'semana1': None, 'semana2': None, 'semana3': None}
    
    # Semana 1
    resultados['semana1'] = ejecutar_semana1()
    if resultados['semana1']:
        copiar_resultados_semana1()
    
    # Semana 2
    resultados['semana2'] = ejecutar_semana2()
    if resultados['semana2']:
        copiar_resultados_semana2()
    
    # Semana 3
    resultados['semana3'] = ejecutar_semana3()
    if resultados['semana3']:
        copiar_resultados_semana3()
    
    # Resumen final
    fin = datetime.now()
    duracion = fin - inicio
    
    print_header("RESUMEN FINAL")
    
    print(f"   Duración total: {duracion.total_seconds():.1f} segundos")
    print()
    
    for semana, exito in resultados.items():
        if exito is None:
            print(f"   {semana}: No ejecutada")
        elif exito:
            print_success(f"{semana}: Completada exitosamente")
        else:
            print_error(f"{semana}: Completada con errores")
    
    print(f"""
{Colors.BOLD}
════════════════════════════════════════════════════════════════════════════
                         PIPELINE FINALIZADO
════════════════════════════════════════════════════════════════════════════
{Colors.END}
    Resultados consolidados en:
    resultados_pipeline/
    ├── semana1_preparacion_datos/
    │   ├── datos_normalizados/
    │   └── reportes/
    ├── semana2_caracteristicas_espaciales/
    │   ├── features/
    │   └── reportes/
    └── semana3_modelo_satisfaccion/
        ├── resultados/
        ├── graficos/
        └── modelos/
    
    Para ver el mapa interactivo:
    $ start resultados_pipeline/semana3_modelo_satisfaccion/graficos/mapa_interactivo.html
    """)
    
    return all(v is None or v for v in resultados.values())

if __name__ == "__main__":
    try:
        exito = main()
        sys.exit(0 if exito else 1)
    except KeyboardInterrupt:
        print("\n\nPipeline interrumpido por el usuario.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError inesperado: {e}")
        sys.exit(1)
