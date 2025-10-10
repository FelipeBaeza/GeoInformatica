#!/usr/bin/env python3
"""
SCRIPT: Visualizador Simple de Gráficos - Semana 2

PROPÓSITO:
Script simplificado para mostrar únicamente las visualizaciones de habitabilidad 
urbana de forma interactiva en pantalla. No genera archivos nuevos, solo carga 
los datos existentes y muestra los gráficos uno por uno de manera sencilla.

FUNCIONALIDAD:
- Carga datos ya procesados de habitabilidad
- Muestra 6 gráficos principales de forma secuencial
- Cada gráfico se muestra en ventana separada
- El usuario puede cerrar cada ventana para continuar
- No requiere navegador web ni URLs externas

USO:
python mostrar_graficos.py

REQUISITOS:
- Datos procesados en ../features/grilla_con_indices.geojson
- Librerías: geopandas, pandas, matplotlib, seaborn

Autor: Proyecto GeoInformática
Fecha: Octubre 2025
"""

# Importar funciones del generador principal
from generar_graficos import (
    configurar_matplotlib, 
    cargar_datos, 
    mostrar_graficos_interactivos
)

def main():
    """
    FUNCIÓN PRINCIPAL: Visualizador simple
    
    PROPÓSITO:
    Función principal que carga los datos y muestra los gráficos de forma
    simple e interactiva. No genera archivos nuevos, solo visualiza.
    
    PROCESO:
    1. Configura matplotlib para visualización óptima
    2. Carga datos procesados de habitabilidad
    3. Muestra gráficos uno por uno de forma interactiva
    
    INTERACCIÓN:
    El usuario debe cerrar cada ventana de gráfico para continuar al siguiente.
    """
    print(" VISUALIZADOR SIMPLE - GRÁFICOS SEMANA 2")
    print("="*50)
    print(" Este script muestra los gráficos de forma simple en pantalla")
    print(" Cierra cada ventana para continuar al siguiente gráfico")
    print()
    
    # Configurar matplotlib para visualización
    configurar_matplotlib()
    
    # Cargar datos procesados
    print(" Cargando datos de habitabilidad...")
    grilla = cargar_datos()
    
    if grilla is None:
        print(" No se pudieron cargar los datos. Verifica que exista:")
        print("   ../features/grilla_con_indices.geojson")
        return False
    
    # Mostrar gráficos de forma interactiva
    mostrar_graficos_interactivos(grilla)
    
    print("\n ¡Visualización completada!")
    return True

if __name__ == "__main__":
    exito = main()
    if not exito:
        exit(1)