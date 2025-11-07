#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Validación de Autocorrelación Espacial
================================================

Este script valida si la autocorrelación espacial detectada en los residuos
es estadísticamente significativa mediante pruebas de permutación.

Autor: Equipo GeoInformática
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
from tqdm import tqdm

# Configuración
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RESULTADOS_DIR = BASE_DIR / "resultados"
GRAFICOS_DIR = BASE_DIR / "graficos"

RESULTADOS_DIR.mkdir(parents=True, exist_ok=True)
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

print("="*70)
print("VALIDACIÓN DE AUTOCORRELACIÓN ESPACIAL (Permutation Test)")
print("="*70)

def calcular_moran_i(residuos, W):
    """Calcula el Índice de Moran I."""
    n = len(residuos)
    residuos_centered = residuos - residuos.mean()
    numerador = np.sum(W * np.outer(residuos_centered, residuos_centered))
    denominador = np.sum(residuos_centered**2)
    moran_i = (n / W.sum()) * (numerador / denominador)
    return moran_i

def crear_matriz_pesos(coords, k_vecinos=8):
    """Crea matriz de pesos espaciales (k-vecinos)."""
    n = len(coords)
    dist_matrix = distance_matrix(coords, coords)
    
    W = np.zeros((n, n))
    for i in range(n):
        nearest = np.argsort(dist_matrix[i])[1:k_vecinos+1]
        W[i, nearest] = 1
    
    # Normalizar
    row_sums = W.sum(axis=1)
    W = W / row_sums[:, np.newaxis]
    
    return W

def test_permutacion(residuos, W, n_permutaciones=999):
    """
    Realiza test de permutación para Moran's I.
    
    Returns:
    --------
    dict : Resultados del test (moran_i, p_value, distribución)
    """
    # Moran's I observado
    moran_obs = calcular_moran_i(residuos, W)
    
    # Permutaciones
    moran_perm = []
    for _ in tqdm(range(n_permutaciones), desc="Permutaciones"):
        residuos_perm = np.random.permutation(residuos)
        moran_perm.append(calcular_moran_i(residuos_perm, W))
    
    moran_perm = np.array(moran_perm)
    
    # P-value (bilateral)
    p_value = np.sum(np.abs(moran_perm) >= np.abs(moran_obs)) / n_permutaciones
    
    return {
        'moran_i': moran_obs,
        'p_value': p_value,
        'distribucion': moran_perm,
        'significativo': p_value < 0.05
    }

# 1. Cargar datos con residuos
print("\n📂 Cargando datos...")
geojson_path = DATA_DIR / "propiedades_con_residuos.geojson"

if not geojson_path.exists():
    print(f"❌ No se encontró: {geojson_path}")
    print("   Ejecuta primero: 03_autocorrelacion_residuos.py")
    exit(1)

gdf = gpd.read_file(geojson_path)
print(f"✓ Dataset cargado: {len(gdf)} registros")

residuos = gdf['residuos'].values

# 2. Crear matriz de pesos
print("\n🔧 Creando matriz de pesos espaciales...")
coords = np.array([[geom.x, geom.y] for geom in gdf.geometry])
W = crear_matriz_pesos(coords, k_vecinos=8)
print("✓ Matriz de pesos creada")

# 3. Test de permutación
print("\n🔬 Ejecutando test de permutación...")
n_perm = 999
resultados = test_permutacion(residuos, W, n_permutaciones=n_perm)

print(f"\n📊 RESULTADOS DEL TEST:")
print(f"  • Moran's I observado: {resultados['moran_i']:.4f}")
print(f"  • P-value: {resultados['p_value']:.4f}")
print(f"  • Número de permutaciones: {n_perm}")

if resultados['significativo']:
    print(f"\n  ⚠️  AUTOCORRELACIÓN SIGNIFICATIVA (p < 0.05)")
    print(f"      La autocorrelación espacial es estadísticamente significativa.")
    print(f"      El modelo NO está capturando toda la estructura espacial.")
else:
    print(f"\n  ✅ Autocorrelación NO significativa (p >= 0.05)")
    print(f"      No hay evidencia estadística de autocorrelación espacial.")

# 4. Visualización
print("\n📊 Generando visualización...")

fig, ax = plt.subplots(figsize=(10, 6))

# Histograma de la distribución de permutaciones
ax.hist(resultados['distribucion'], bins=50, alpha=0.7, 
        label='Distribución bajo H₀', edgecolor='black')

# Línea del valor observado
ax.axvline(resultados['moran_i'], color='red', linestyle='--', 
          linewidth=2, label=f"Moran's I observado = {resultados['moran_i']:.4f}")

# Configuración
ax.set_xlabel("Moran's I")
ax.set_ylabel('Frecuencia')
ax.set_title(f"Test de Permutación - Autocorrelación Espacial\n(p-value = {resultados['p_value']:.4f})")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'test_permutacion.png', dpi=300, bbox_inches='tight')
print("✓ Gráfico guardado: test_permutacion.png")

plt.close('all')

# 5. Guardar resultados
print("\n💾 Guardando resultados...")

reporte = pd.DataFrame([{
    'moran_i': resultados['moran_i'],
    'p_value': resultados['p_value'],
    'significativo': resultados['significativo'],
    'n_permutaciones': n_perm,
    'n_observaciones': len(gdf)
}])

reporte_path = RESULTADOS_DIR / 'test_autocorrelacion.csv'
reporte.to_csv(reporte_path, index=False)
print(f"✓ Reporte guardado: {reporte_path}")

# Guardar distribución de permutaciones
dist_df = pd.DataFrame({'moran_i_permutado': resultados['distribucion']})
dist_path = RESULTADOS_DIR / 'distribucion_permutaciones.csv'
dist_df.to_csv(dist_path, index=False)
print(f"✓ Distribución guardada: {dist_path}")

print("\n" + "="*70)
print("✅ VALIDACIÓN COMPLETADA")
print("="*70)

# Recomendación
if resultados['significativo']:
    print("\n💡 RECOMENDACIÓN:")
    print("   Considera implementar modelos espacialmente explícitos:")
    print("   • Geographically Weighted Regression (GWR)")
    print("   • Spatial Lag Model")
    print("   • Spatial Error Model")
    print("   • Random Forest con particionamiento espacial (GWRF)")
