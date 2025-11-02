#!/usr/bin/env python3
"""
Validación simple de autocorrelación espacial de residuos
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime

# Cargar datos
BASE_DIR = Path(__file__).parent
DATOS_DIR = BASE_DIR / "datos_procesados"

archivos = list(DATOS_DIR.glob("propiedades_limpias_*.geojson"))
gdf = gpd.read_file(max(archivos, key=lambda x: x.stat().st_mtime))

print(f"✅ Cargado: {len(gdf)} propiedades")

# Cargar modelo
with open(BASE_DIR / "modelo_ols_limpio.pkl", 'rb') as f:
    modelo = pickle.load(f)

print(f"✅ Modelo cargado (R²={modelo.rsquared:.4f})")

# Agregar residuos
residuos = modelo.resid.values
gdf['residuos'] = residuos

print(f"\nEstadísticas de residuos:")
print(f"  Media: {residuos.mean():.6f}")
print(f"  Std: {residuos.std():.4f}")
print(f"  Min: {residuos.min():.4f}")
print(f"  Max: {residuos.max():.4f}")

# Calcular Moran's I manualmente con Queen contiguity
from libpysal.weights import Queen

print(f"\nConstruyendo matriz de pesos espaciales (Queen contiguity)...")
w = Queen.from_dataframe(gdf, use_index=False)
w.transform = 'r'

print(f"✅ Matriz construida: {w.n} observaciones")
print(f"  Componentes conectados: {w.n_components}")
print(f"  Promedio de vecinos: {w.mean_neighbors:.1f}")

# Calcular Moran's I
from esda.moran import Moran

print(f"\nCalculando Moran's I...")
moran = Moran(residuos, w, permutations=999)

print(f"\n{'='*60}")
print(f"RESULTADOS - MORAN'S I GLOBAL")
print(f"{'='*60}")
print(f"  Moran's I:        {moran.I:.6f}")
print(f"  Valor esperado:   {moran.EI:.6f}")
print(f"  Desviación:       {moran.I - moran.EI:.6f}")
print(f"  Z-score:          {moran.z_sim:.4f}")
print(f"  p-value:          {moran.p_sim:.6f}")

# Interpretación
print(f"\n{'='*60}")
print(f"INTERPRETACIÓN")
print(f"{'='*60}")

if moran.p_sim < 0.001:
    sig = "*** Altamente significativo (p<0.001)"
elif moran.p_sim < 0.01:
    sig = "** Significativo (p<0.01)"
elif moran.p_sim < 0.05:
    sig = "* Significativo (p<0.05)"
else:
    sig = "No significativo (p≥0.05)"

print(f"  Significancia: {sig}")

if moran.p_sim < 0.05:
    if abs(moran.I) < 0.1:
        print(f"  Intensidad: DÉBIL (|I|<0.1)")
        print(f"  ⚠️  Hay autocorrelación pero es débil")
        print(f"  → OLS probablemente suficiente, pero considerar GWR")
        decision = "OLS_CON_PRECAUCION"
    elif abs(moran.I) < 0.3:
        print(f"  Intensidad: MODERADA (0.1≤|I|<0.3)")
        print(f"  ⚠️  Se recomienda modelo espacial")
        print(f"  → Considerar GWR o SAR")
        decision = "CONSIDERAR_ESPACIAL"
    else:
        print(f"  Intensidad: FUERTE (|I|≥0.3)")
        print(f"  🚨 REQUIERE modelo espacial")
        print(f"  → GWR, SAR o CAR necesario")
        decision = "REQUIERE_ESPACIAL"
    
    print(f"\n❌ CONCLUSIÓN: Hay autocorrelación espacial en los residuos")
    print(f"   El modelo OLS NO captura toda la estructura espacial")
else:
    print(f"  ✅ NO hay autocorrelación significativa")
    print(f"  → OLS es SUFICIENTE")
    print(f"\n✅ CONCLUSIÓN: Los residuos son espacialmente independientes")
    print(f"   El modelo OLS captura adecuadamente la estructura espacial")
    decision = "OLS_SUFICIENTE"

print(f"\n{'='*60}")
print(f"DECISIÓN FINAL: {decision}")
print(f"{'='*60}")

# Guardar resultado
output_file = BASE_DIR / "decision_autocorrelacion.txt"
with open(output_file, 'w', encoding='utf-8') as f:
    f.write(f"VALIDACIÓN DE AUTOCORRELACIÓN ESPACIAL DE RESIDUOS\n")
    f.write(f"{'='*60}\n\n")
    f.write(f"Fecha: {datetime.now()}\n")
    f.write(f"Observaciones: {len(gdf)}\n")
    f.write(f"R² del modelo: {modelo.rsquared:.4f}\n\n")
    f.write(f"MORAN'S I:\n")
    f.write(f"  I = {moran.I:.6f}\n")
    f.write(f"  p-value = {moran.p_sim:.6f}\n")
    f.write(f"  Z-score = {moran.z_sim:.4f}\n\n")
    f.write(f"DECISIÓN: {decision}\n\n")
    f.write(f"Significancia: {sig}\n")
    
    if moran.p_sim < 0.05:
        f.write(f"\nRecomendación: Considerar modelo espacial (GWR/SAR)\n")
    else:
        f.write(f"\nRecomendación: Mantener modelo OLS\n")

print(f"\n✅ Resultado guardado en: {output_file.name}")
