"""
Script para convertir archivos CSV geolocalizados a GeoJSON para visualización en QGIS.
Convierte archivos con columnas latitud y longitud a formato GeoJSON.
"""

import csv
import json
import os
from datetime import datetime


def csv_a_geojson(archivo_csv, archivo_geojson=None):
    """
    Convierte un CSV con columnas latitud/longitud a GeoJSON.
    
    Args:
        archivo_csv: Ruta al archivo CSV de entrada
        archivo_geojson: Ruta al archivo GeoJSON de salida (opcional)
    """
    if not archivo_geojson:
        nombre_base = os.path.splitext(archivo_csv)[0]
        archivo_geojson = f"{nombre_base}.geojson"
    
    print("=" * 60)
    print("🗺️  CONVERSOR CSV → GeoJSON")
    print("=" * 60)
    print(f"\n📂 Archivo entrada: {archivo_csv}")
    print(f"📂 Archivo salida: {archivo_geojson}")
    
    # Leer CSV
    features = []
    propiedades_validas = 0
    propiedades_sin_coords = 0
    
    with open(archivo_csv, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f, delimiter=';')
        
        for fila in reader:
            # Verificar que tenga coordenadas válidas
            try:
                lat = float(fila.get('latitud', ''))
                lng = float(fila.get('longitud', ''))
                
                if lat and lng:
                    # Crear feature GeoJSON
                    properties = {}
                    for key, value in fila.items():
                        # Excluir las coordenadas de las propiedades (ya están en geometry)
                        if key not in ['latitud', 'longitud']:
                            properties[key] = value
                    
                    feature = {
                        "type": "Feature",
                        "geometry": {
                            "type": "Point",
                            "coordinates": [lng, lat]  # GeoJSON usa [lon, lat]
                        },
                        "properties": properties
                    }
                    
                    features.append(feature)
                    propiedades_validas += 1
                else:
                    propiedades_sin_coords += 1
            except (ValueError, TypeError):
                propiedades_sin_coords += 1
                continue
    
    # Crear estructura GeoJSON
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
            }
        },
        "features": features
    }
    
    # Guardar GeoJSON
    with open(archivo_geojson, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Conversión completada:")
    print(f"   📊 Propiedades válidas: {propiedades_validas}")
    print(f"   ⚠️  Sin coordenadas: {propiedades_sin_coords}")
    print(f"\n📁 Archivo guardado: {archivo_geojson}")
    print(f"📍 Ubicación: {os.path.abspath(archivo_geojson)}")
    print(f"\n💡 Para abrir en QGIS:")
    print(f"   1. Capa → Agregar capa → Agregar capa vectorial")
    print(f"   2. Selecciona: {archivo_geojson}")
    print(f"   3. O simplemente arrastra el archivo a QGIS")
    
    return archivo_geojson


# --- Menú Principal ---
if __name__ == "__main__":
    import sys
    
    print("\n" + "=" * 60)
    print("🗺️  CONVERSOR CSV → GeoJSON")
    print("=" * 60)
    
    # Si se proporciona un archivo como argumento, procesarlo directamente
    if len(sys.argv) > 1:
        archivo = sys.argv[1]
        if os.path.exists(archivo):
            csv_a_geojson(archivo)
        else:
            print(f"\n❌ Archivo no encontrado: {archivo}")
    else:
        # Modo interactivo
        # Listar archivos CSV geolocalizados
        archivos_csv = [f for f in os.listdir('.') 
                       if f.endswith('_geolocacion.csv')]
        
        if not archivos_csv:
            print("\n❌ No se encontraron archivos CSV geolocalizados.")
            print("   Primero ejecute geocodificar.py para generar datos.")
        else:
            # Ordenar por fecha
            archivos_csv.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            print("\n📂 Archivos CSV geolocalizados disponibles:\n")
            for i, archivo in enumerate(archivos_csv, 1):
                try:
                    with open(archivo, 'r', encoding='utf-8-sig') as f:
                        total = sum(1 for _ in f) - 1
                except:
                    total = "?"
                print(f"   [{i}] {archivo} ({total} filas)")
            
            print(f"\n   [0] Salir")
            print(f"   [99] Convertir TODOS los archivos")
            
            try:
                opcion = input("\n👉 Seleccione el archivo a convertir: ").strip()
                
                if opcion == '0':
                    print("\n👋 ¡Hasta luego!")
                elif opcion == '99':
                    print(f"\n🚀 Convirtiendo {len(archivos_csv)} archivo(s)...\n")
                    for archivo in archivos_csv:
                        csv_a_geojson(archivo)
                        print()
                    print(f"✅ ¡Todos los archivos convertidos!")
                else:
                    idx = int(opcion) - 1
                    if 0 <= idx < len(archivos_csv):
                        archivo = archivos_csv[idx]
                        csv_a_geojson(archivo)
                    else:
                        print("\n❌ Opción no válida.")
                        
            except ValueError:
                print("\n❌ Entrada inválida.")
            except KeyboardInterrupt:
                print("\n\n⚠️  Operación cancelada.")
