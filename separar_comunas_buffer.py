#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Separar comunas_buffer.geojson en archivos individuales
Genera un archivo GeoJSON por cada comuna
"""

import json
import os

def separar_comunas():
    """Separa el archivo comunas_buffer.geojson en archivos individuales"""
    
    archivo_entrada = "autocorrelacion_espacial/semana1_preparacion_datos/datos_originales/datos_filtrados/comunas_buffer.geojson"
    carpeta_salida = "datos_nuevos/comunas_individuales"
    
    # Crear carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)
    
    print("\n" + "="*60)
    print("📂 SEPARADOR DE COMUNAS")
    print("="*60)
    
    # Leer archivo GeoJSON
    with open(archivo_entrada, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"\n📥 Archivo leído: {archivo_entrada}")
    print(f"📊 Total de comunas: {len(data['features'])}")
    
    # Procesar cada comuna
    comunas_creadas = []
    for feature in data['features']:
        comuna = feature['properties']['comuna']
        
        # Limpiar nombre de comuna para archivo
        nombre_archivo = comuna.replace('ó', 'o').replace('ñ', 'n').replace('Ó', 'O').replace('Ñ', 'N')
        nombre_archivo = nombre_archivo.replace(' ', '_').lower()
        archivo_salida = os.path.join(carpeta_salida, f"buffer_{nombre_archivo}.geojson")
        
        # Crear GeoJSON individual
        geojson_individual = {
            "type": "FeatureCollection",
            "crs": data.get("crs", {
                "type": "name",
                "properties": {
                    "name": "urn:ogc:def:crs:OGC:1.3:CRS84"
                }
            }),
            "features": [feature]
        }
        
        # Guardar archivo
        with open(archivo_salida, 'w', encoding='utf-8') as f:
            json.dump(geojson_individual, f, ensure_ascii=False, indent=2)
        
        comunas_creadas.append({
            'nombre': comuna,
            'archivo': archivo_salida
        })
        
        print(f"   ✅ {comuna:20} → {os.path.basename(archivo_salida)}")
    
    print(f"\n📁 Archivos generados en: {os.path.abspath(carpeta_salida)}")
    print(f"✨ Total: {len(comunas_creadas)} archivos GeoJSON creados")
    
    return comunas_creadas

if __name__ == "__main__":
    separar_comunas()
