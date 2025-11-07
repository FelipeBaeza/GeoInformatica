
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Integración de Datos Espaciales
==========================================

Integra las propiedades con la grilla de factores espaciales (Semana 2)
y genera un CSV y GeoJSON de salida con las features espaciales unidas.

Se detectan columnas de coordenadas 'latitude'/'longitude' (o 'lat'/'lon').
"""

import sys
from pathlib import Path
import pandas as pd
import geopandas as gpd

BASE_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_ROOT = BASE_DIR.parent.parent
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = WORKSPACE_ROOT / "autocorrelacion_espacial/semana2_caracteristicas_espaciales/features"

DATA_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("INTEGRACIÓN DE DATOS ESPACIALES")
print("="*70)

# 1. Cargar propiedades (buscar en workspace root)
print("\n📂 Cargando datos de propiedades...")
archivo_propiedades = WORKSPACE_ROOT / "clean_alquiler_02_11_2023cc.csv"
if not archivo_propiedades.exists():
	archivos = list(DATA_DIR.glob("*.csv"))
	if archivos:
		archivo_propiedades = archivos[0]
	else:
		print(f"❌ No se encontró archivo de propiedades en: {archivo_propiedades}")
		sys.exit(1)

df_prop = pd.read_csv(archivo_propiedades)
print(f"✓ Propiedades cargadas: {len(df_prop)} registros")

# 2. Detectar columnas de coordenadas
lat_col = None
lon_col = None
for c in ['latitude', 'lat', 'y', 'coord_lat']:
	if c in df_prop.columns:
		lat_col = c
		break
for c in ['longitude', 'lon', 'lng', 'x', 'coord_lon']:
	if c in df_prop.columns:
		lon_col = c
		break

if lat_col is None or lon_col is None:
	print("❌ No se encontraron columnas de coordenadas (latitude/longitude).")
	print(f"Columnas disponibles: {list(df_prop.columns)[:30]}")
	sys.exit(1)

print(f"✓ Coordenadas detectadas: lat={lat_col}, lon={lon_col}")

# 3. Convertir a GeoDataFrame y reproyectar a UTM 19S (EPSG:32719)
gdf_prop = gpd.GeoDataFrame(
	df_prop,
	geometry=gpd.points_from_xy(df_prop[lon_col], df_prop[lat_col]),
	crs='EPSG:4326'
)
gdf_prop = gdf_prop.to_crs('EPSG:32719')
print(f"✓ GeoDataFrame creado y reproyectado a EPSG:32719")

# 4. Cargar grilla con factores espaciales
grilla_path = FEATURES_DIR / 'grilla_con_densidades.geojson'
if not grilla_path.exists():
	print(f"❌ No se encontró la grilla en: {grilla_path}")
	print("   Ejecuta primero los scripts de semana2: generar_grilla.py, calcular_distancias.py, calcular_densidades.py")
	sys.exit(1)

gdf_grilla = gpd.read_file(grilla_path)
print(f"✓ Grilla cargada: {len(gdf_grilla)} puntos")

# Asegurar CRS coincidente
if gdf_grilla.crs != gdf_prop.crs:
	gdf_grilla = gdf_grilla.to_crs(gdf_prop.crs)

# 5. Join espacial (nearest)
print("\n🔗 Realizando join espacial (nearest)...")
gdf_integrado = gpd.sjoin_nearest(gdf_prop, gdf_grilla, how='left', distance_col='dist_to_grid')
print(f"✓ Datos integrados: {len(gdf_integrado)} registros")

# 6. Limpieza: eliminar columnas temporales
if 'index_right' in gdf_integrado.columns:
	gdf_integrado = gdf_integrado.drop(columns=['index_right'])

# 7. Guardar CSV (sin geometría) y GeoJSON
df_salida = pd.DataFrame(gdf_integrado.drop(columns='geometry'))
csv_path = DATA_DIR / 'propiedades_con_factores_espaciales.csv'
df_salida.to_csv(csv_path, index=False)
print(f"✓ CSV guardado: {csv_path}")

geojson_path = DATA_DIR / 'propiedades_con_factores_espaciales.geojson'
gdf_integrado.to_file(geojson_path, driver='GeoJSON')
print(f"✓ GeoJSON guardado: {geojson_path}")

print("\n✅ INTEGRACIÓN COMPLETADA")
print("="*70)
