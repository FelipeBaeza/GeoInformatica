#!/usr/bin/env python3
"""
Script principal para ejecutar todos los análisis de la Semana 3:
- Autocorrelación espacial global (Índice de Moran)
- Autocorrelación espacial local (LISA)
- Identificación de submercados

Autor: Proyecto GeoInformática - Semana 3
Fecha: Noviembre 2025
"""

import os
import sys
from datetime import datetime

def verificar_dependencias():
    """
    Verifica que todas las dependencias estén instaladas
    """
    print("=" * 80)
    print("VERIFICANDO DEPENDENCIAS")
    print("=" * 80 + "\n")
    
    dependencias = {
        'geopandas': 'geopandas',
        'pandas': 'pandas',
        'numpy': 'numpy',
        'libpysal': 'pysal (libpysal)',
        'esda': 'pysal (esda)',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'sklearn': 'scikit-learn',
        'scipy': 'scipy'
    }
    
    faltantes = []
    
    for modulo, nombre in dependencias.items():
        try:
            __import__(modulo)
            print(f"✓ {nombre}")
        except ImportError:
            print(f"✗ {nombre} - NO INSTALADO")
            faltantes.append(nombre)
    
    if faltantes:
        print("\n" + "=" * 80)
        print("⚠️  FALTAN DEPENDENCIAS")
        print("=" * 80)
        print("\nPara instalar las dependencias faltantes, ejecuta:")
        print("\n  pip install geopandas pandas numpy pysal matplotlib seaborn scikit-learn scipy")
        print("\nO con conda:")
        print("\n  conda install -c conda-forge geopandas pysal matplotlib seaborn scikit-learn scipy")
        print("\n" + "=" * 80 + "\n")
        return False
    
    print("\n✅ Todas las dependencias están instaladas\n")
    return True

def ejecutar_script(script_name):
    """
    Ejecuta un script y maneja errores
    """
    print("\n" + "=" * 80)
    print(f"EJECUTANDO: {script_name}")
    print("=" * 80 + "\n")
    
    try:
        # Cambiar al directorio de scripts
        os.chdir('scripts')
        
        # Ejecutar script
        exit_code = os.system(f'python3 {script_name}')
        
        # Volver al directorio original
        os.chdir('..')
        
        if exit_code == 0:
            print(f"\n✅ {script_name} ejecutado exitosamente\n")
            return True
        else:
            print(f"\n❌ Error al ejecutar {script_name} (código: {exit_code})\n")
            return False
            
    except Exception as e:
        print(f"\n❌ Excepción al ejecutar {script_name}: {e}\n")
        os.chdir('..')
        return False

def main():
    """
    Función principal
    """
    print("\n" + "=" * 80)
    print(" " * 15 + "SEMANA 3: ANÁLISIS ESPACIAL AVANZADO")
    print(" " * 10 + "Sistema de Recomendación Inmobiliaria Geoespacial")
    print("=" * 80)
    print(f"\nFecha de ejecución: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    print("=" * 80 + "\n")
    
    # 1. Verificar dependencias
    if not verificar_dependencias():
        print("❌ Por favor instala las dependencias faltantes e intenta nuevamente.")
        sys.exit(1)
    
    # 2. Confirmar ejecución
    print("Este script ejecutará los siguientes análisis:\n")
    print("  1. Autocorrelación Espacial Global (Índice de Moran)")
    print("  2. Autocorrelación Espacial Local (LISA - Clusters)")
    print("  3. Identificación de Submercados (K-Means Clustering)\n")
    print("Esto puede tardar varios minutos dependiendo de tu hardware.\n")
    
    respuesta = input("¿Deseas continuar? (s/n): ").strip().lower()
    
    if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
        print("\n❌ Ejecución cancelada por el usuario.\n")
        sys.exit(0)
    
    # 3. Ejecutar análisis secuencialmente
    scripts = [
        ('calcular_autocorrelacion_global.py', 'Autocorrelación Global'),
        ('calcular_autocorrelacion_local.py', 'Autocorrelación Local (LISA)'),
        ('identificar_submercados.py', 'Identificación de Submercados')
    ]
    
    resultados = []
    
    for script, nombre in scripts:
        print("\n" + "═" * 80)
        print(f" ANÁLISIS {len(resultados) + 1}/{len(scripts)}: {nombre}")
        print("═" * 80)
        
        exito = ejecutar_script(script)
        resultados.append((nombre, exito))
        
        if not exito:
            print(f"\n⚠️  El script {script} falló. ¿Deseas continuar con los siguientes?")
            respuesta = input("Continuar (s/n): ").strip().lower()
            
            if respuesta not in ['s', 'si', 'sí', 'y', 'yes']:
                print("\n❌ Ejecución detenida por el usuario.\n")
                break
    
    # 4. Resumen final
    print("\n" + "=" * 80)
    print(" " * 25 + "RESUMEN FINAL")
    print("=" * 80 + "\n")
    
    print("RESULTADOS DE EJECUCIÓN:\n")
    
    for i, (nombre, exito) in enumerate(resultados, 1):
        estado = "✅ EXITOSO" if exito else "❌ FALLIDO"
        print(f"  {i}. {nombre}: {estado}")
    
    exitosos = sum(1 for _, exito in resultados if exito)
    total = len(resultados)
    
    print(f"\n📊 Total: {exitosos}/{total} análisis completados exitosamente")
    
    if exitosos == total:
        print("\n" + "=" * 80)
        print("✅ TODOS LOS ANÁLISIS COMPLETADOS EXITOSAMENTE")
        print("=" * 80)
        print("\n📁 ARCHIVOS GENERADOS:\n")
        print("Reportes:")
        print("  - reportes/autocorrelacion_global_reporte.md")
        print("  - reportes/autocorrelacion_global_resultados.csv")
        print("  - reportes/autocorrelacion_local_reporte.md")
        print("  - reportes/submercados_reporte.md")
        print("  - reportes/submercados_perfiles.json")
        print("\nMapas:")
        print("  - mapas/autocorrelacion_global_resumen.png")
        print("  - mapas/lisa_clusters_*.png (múltiples)")
        print("  - mapas/moran_scatterplot_*.png (múltiples)")
        print("  - mapas/lisa_significancia_*.png (múltiples)")
        print("\nSubmercados:")
        print("  - submercados/mapa_submercados.png")
        print("  - submercados/determinacion_k_optimo.png")
        print("  - submercados/grilla_con_submercados.geojson")
        print("\n" + "=" * 80)
        print("\n🎯 PRÓXIMOS PASOS:\n")
        print("1. Revisar los reportes Markdown generados")
        print("2. Analizar los mapas de clusters LISA")
        print("3. Interpretar los perfiles de submercados")
        print("4. Proceder a Fase 4: Adquisición de datos de mercado inmobiliario")
        print("5. Luego: Fase 5: Modelado hedónico (OLS, GWR/MGWR)")
        print("\n" + "=" * 80 + "\n")
    else:
        print("\n" + "=" * 80)
        print("⚠️  ALGUNOS ANÁLISIS FALLARON")
        print("=" * 80)
        print("\nRevisa los mensajes de error arriba para identificar el problema.")
        print("Posibles causas:")
        print("  - Datos de entrada faltantes")
        print("  - Permisos de escritura")
        print("  - Memoria insuficiente")
        print("\n" + "=" * 80 + "\n")

if __name__ == "__main__":
    main()
