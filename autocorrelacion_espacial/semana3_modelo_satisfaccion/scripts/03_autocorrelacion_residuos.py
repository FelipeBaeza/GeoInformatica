#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Análisis de Autocorrelación Espacial de Residuos
==========================================================

Este script analiza la autocorrelación espacial de los residuos del modelo
de Random Forest utilizando el Índice de Moran.

Autor: Equipo GeoInformática
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from scipy.spatial import distance_matrix
import pickle

# Configuración
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTADOS_DIR = BASE_DIR / "resultados"
GRAFICOS_DIR = BASE_DIR / "graficos"

RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("ANÁLISIS DE AUTOCORRELACIÓN ESPACIAL DE RESIDUOS")
print("="*70)

def calcular_moran_i(residuos, coords, k_vecinos=8):
    """
    Calcula el Índice de Moran I para los residuos.
    
    Parameters:
    -----------
    residuos : array
        Residuos del modelo
    coords : array
        Coordenadas (x, y) de las observaciones
    k_vecinos : int
        Número de vecinos más cercanos
        
    Returns:
    --------
    float : Índice de Moran I
    """
    n = len(residuos)
    
    # Calcular matriz de distancias
    dist_matrix = distance_matrix(coords, coords)
    
    # Crear matriz de pesos (k-vecinos más cercanos)
    W = np.zeros((n, n))
    for i in range(n):
        # Obtener índices de los k vecinos más cercanos (excluyendo el punto mismo)
        nearest = np.argsort(dist_matrix[i])[1:k_vecinos+1]
        W[i, nearest] = 1
    
    # Normalizar por filas
    row_sums = W.sum(axis=1)
    W = W / row_sums[:, np.newaxis]
    
    # Calcular Moran's I
    residuos_centered = residuos - residuos.mean()
    numerador = np.sum(W * np.outer(residuos_centered, residuos_centered))
    denominador = np.sum(residuos_centered**2)
    
    moran_i = (n / W.sum()) * (numerador / denominador)
    
    return moran_i

# 1. Cargar datos
print("\n📂 Cargando datos...")
geojson_path = DATA_DIR / "propiedades_con_factores_espaciales.geojson"

if not geojson_path.exists():
    print(f"❌ No se encontró: {geojson_path}")
    print("   Ejecuta primero: 01_integrar_datos.py")
    exit(1)

gdf = gpd.read_file(geojson_path)
print(f"✓ Dataset cargado: {len(gdf)} registros")

# 2. Preparar variables
print("\n🔧 Preparando variables...")

# Variable objetivo
target = 'precio_m2' if 'precio_m2' in gdf.columns else 'precio'

# Features
features_espaciales = [c for c in gdf.columns if c.startswith('dens_') or c.startswith('dist_')]
features_basicas = ['sup_total', 'sup_cubierta', 'banos', 'dormitorios', 'antiguedad']
features_basicas = [f for f in features_basicas if f in gdf.columns]
features = features_basicas + features_espaciales

# Limpieza
gdf_clean = gdf[features + [target, 'geometry']].copy()
gdf_clean = gdf_clean.dropna()

# Eliminar outliers
Q1 = gdf_clean[target].quantile(0.01)
Q99 = gdf_clean[target].quantile(0.99)
gdf_clean = gdf_clean[(gdf_clean[target] >= Q1) & (gdf_clean[target] <= Q99)]

print(f"✓ Registros limpios: {len(gdf_clean)}")

# 3. Entrenar modelo o cargar existente
print("\n⏳ Cargando/entrenando modelo...")

model_path = RESULTADOS_DIR / 'random_forest_model.pkl'
if model_path.exists():
    with open(model_path, 'rb') as f:
        rf_model = pickle.load(f)
    print("✓ Modelo cargado desde archivo")
else:
    X = gdf_clean[features]
    y = gdf_clean[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    rf_model = RandomForestRegressor(
        n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    print("✓ Modelo entrenado")

# 4. Calcular residuos
print("\n📊 Calculando residuos...")
X = gdf_clean[features]
y = gdf_clean[target]

y_pred = rf_model.predict(X)
residuos = y - y_pred

gdf_clean['residuos'] = residuos
gdf_clean['prediccion'] = y_pred

print(f"✓ Residuos calculados")
print(f"  • Media: ${residuos.mean():,.2f}")
print(f"  • Std: ${residuos.std():,.2f}")
print(f"  • Min: ${residuos.min():,.2f}")
print(f"  • Max: ${residuos.max():,.2f}")

# 5. Calcular Índice de Moran
print("\n🗺️  Calculando autocorrelación espacial...")

# Extraer coordenadas
coords = np.array([[geom.x, geom.y] for geom in gdf_clean.geometry])

# Calcular Moran's I
moran_i = calcular_moran_i(residuos.values, coords, k_vecinos=8)

print(f"\n📈 ÍNDICE DE MORAN I: {moran_i:.4f}")
print("\nInterpretación:")
if moran_i > 0.3:
    print("  ⚠️  Autocorrelación positiva FUERTE")
    print("      Los residuos similares se agrupan espacialmente.")
    print("      El modelo NO captura toda la variación espacial.")
elif moran_i > 0.1:
    print("  ⚠️  Autocorrelación positiva MODERADA")
    print("      Existe cierta agrupación espacial de residuos.")
elif moran_i > -0.1:
    print("  ✅ Autocorrelación BAJA o nula")
    print("      Los residuos están distribuidos aleatoriamente.")
else:
    print("  ℹ️  Autocorrelación negativa")
    print("      Patrón de dispersión espacial.")

# 6. Visualización
print("\n📊 Generando visualizaciones...")

# Mapa de residuos
fig, ax = plt.subplots(figsize=(12, 10))
gdf_clean.plot(column='residuos', cmap='RdBu_r', legend=True, 
               ax=ax, markersize=20, alpha=0.6,
               legend_kwds={'label': 'Residuos ($)'})
ax.set_title(f'Distribución Espacial de Residuos\n(Moran I = {moran_i:.4f})', 
             fontsize=14, fontweight='bold')
ax.set_xlabel('Longitud (UTM)')
ax.set_ylabel('Latitud (UTM)')
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'mapa_residuos.png', dpi=300, bbox_inches='tight')
print("✓ Mapa guardado: mapa_residuos.png")

# Histograma de residuos
fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(residuos, bins=50, edgecolor='black', alpha=0.7)
ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Residuo = 0')
ax.set_xlabel('Residuo ($)')
ax.set_ylabel('Frecuencia')
ax.set_title('Distribución de Residuos')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'histograma_residuos.png', dpi=300, bbox_inches='tight')
print("✓ Histograma guardado: histograma_residuos.png")

plt.close('all')

# 7. Guardar resultados
print("\n💾 Guardando resultados...")

# GeoJSON con residuos
output_path = DATA_DIR / "propiedades_con_residuos.geojson"
gdf_clean.to_file(output_path, driver='GeoJSON')
print(f"✓ GeoJSON con residuos: {output_path}")

# Reporte
reporte = {
    'moran_i': moran_i,
    'n_observaciones': len(gdf_clean),
    'media_residuos': float(residuos.mean()),
    'std_residuos': float(residuos.std()),
    'min_residuos': float(residuos.min()),
    'max_residuos': float(residuos.max())
}

reporte_df = pd.DataFrame([reporte])
reporte_path = RESULTADOS_DIR / 'autocorrelacion_residuos.csv'
reporte_df.to_csv(reporte_path, index=False)
print(f"✓ Reporte guardado: {reporte_path}")

print("\n" + "="*70)
print("✅ ANÁLISIS COMPLETADO")
print("="*70)
