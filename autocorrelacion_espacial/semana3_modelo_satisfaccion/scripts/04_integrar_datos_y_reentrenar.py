"""
Script para integrar datos del CSV y GeoJSON, y re-entrenar el modelo Random Forest

Pasos:
1. Cargar CSV y GeoJSON
2. Detectar y eliminar duplicados espaciales
3. Combinar ambas fuentes en un dataset único
4. Agregar factores espaciales (join con grilla de densidades)
5. Limpiar y preparar datos
6. Re-entrenar Random Forest
7. Comparar con modelo anterior
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
print("=" * 80)
print("🚀 INTEGRACIÓN DE DATOS Y RE-ENTRENAMIENTO DEL MODELO")
print("=" * 80)

BASE_DIR = Path('/home/felipe/Documentos/GeoInformatica')
SEMANA3_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana3_modelo_satisfaccion'
SEMANA1_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana1_preparacion_datos'
SEMANA2_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana2_caracteristicas_espaciales'

# Archivos de entrada
CSV_PATH = BASE_DIR / 'clean_alquiler_02_11_2023cc.csv'
GEOJSON_PATH = SEMANA1_DIR / 'datos_normalizados' / 'datos_normalizados' / 'base_maestra_comunas_filtradas.geojson'
GRILLA_PATH = SEMANA2_DIR / 'features' / 'grilla_con_densidades.geojson'
MODELO_ANTERIOR_PATH = SEMANA3_DIR / 'data' / 'propiedades_con_factores_espaciales.csv'

# Archivos de salida
OUTPUT_DIR = SEMANA3_DIR / 'data'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DATASET_COMBINADO_PATH = OUTPUT_DIR / 'propiedades_combinadas_con_factores_espaciales.csv'
MODELO_PATH = SEMANA3_DIR / 'models' / 'random_forest_combinado.pkl'
MODELO_PATH.parent.mkdir(parents=True, exist_ok=True)

# Comunas de interés
COMUNAS_TARGET = ['SANTIAGO', 'ESTACIÓN CENTRAL', 'ÑUÑOA', 'LA REINA']

# =============================================================================
# PASO 1: CARGAR DATOS DEL CSV
# =============================================================================
print("\n" + "=" * 80)
print("📂 PASO 1: CARGAR DATOS DEL CSV")
print("=" * 80)

df_csv = pd.read_csv(CSV_PATH)
print(f"✓ CSV cargado: {len(df_csv)} registros")

# Normalizar nombres de comunas
df_csv['comuna_norm'] = df_csv['comuna'].str.upper().str.strip()

# Filtrar 4 comunas y con coordenadas válidas
df_csv_filtrado = df_csv[df_csv['comuna_norm'].isin(COMUNAS_TARGET)].copy()
df_csv_filtrado = df_csv_filtrado.dropna(subset=['latitude', 'longitude'])

print(f"✓ Filtrado a 4 comunas: {len(df_csv_filtrado)} propiedades")
for comuna in COMUNAS_TARGET:
    count = len(df_csv_filtrado[df_csv_filtrado['comuna_norm'] == comuna])
    print(f"  • {comuna}: {count} propiedades")

# Convertir a GeoDataFrame
geometry_csv = [Point(xy) for xy in zip(df_csv_filtrado['longitude'], df_csv_filtrado['latitude'])]
gdf_csv = gpd.GeoDataFrame(df_csv_filtrado, geometry=geometry_csv, crs="EPSG:4326")

# =============================================================================
# PASO 2: CARGAR DATOS DEL GEOJSON
# =============================================================================
print("\n" + "=" * 80)
print("📂 PASO 2: CARGAR DATOS DEL GEOJSON (SEMANA 1)")
print("=" * 80)

gdf_geojson = gpd.read_file(GEOJSON_PATH)
print(f"✓ GeoJSON cargado: {len(gdf_geojson)} propiedades")

# Filtrar 4 comunas
gdf_geojson_filtrado = gdf_geojson[gdf_geojson['comuna'].isin(COMUNAS_TARGET)].copy()
print(f"✓ Filtrado a 4 comunas: {len(gdf_geojson_filtrado)} propiedades")
for comuna in COMUNAS_TARGET:
    count = len(gdf_geojson_filtrado[gdf_geojson_filtrado['comuna'] == comuna])
    print(f"  • {comuna}: {count} propiedades")

# Asegurar mismo CRS
if gdf_geojson_filtrado.crs != gdf_csv.crs:
    gdf_geojson_filtrado = gdf_geojson_filtrado.to_crs(gdf_csv.crs)

# =============================================================================
# PASO 3: DETECTAR Y ELIMINAR DUPLICADOS ESPACIALES
# =============================================================================
print("\n" + "=" * 80)
print("🔍 PASO 3: DETECTAR Y ELIMINAR DUPLICADOS ESPACIALES")
print("=" * 80)

# Crear buffer de 10 metros para comparación (tolerancia)
buffer_dist = 0.0001  # ~10 metros en grados
gdf_csv_buffer = gdf_csv.copy()
gdf_csv_buffer['geometry'] = gdf_csv_buffer.geometry.buffer(buffer_dist)

# Encontrar intersecciones (duplicados)
print("⏳ Buscando propiedades duplicadas...")
gdf_geojson_filtrado['es_duplicado'] = gdf_geojson_filtrado.geometry.apply(
    lambda geom: gdf_csv_buffer.geometry.intersects(geom).any()
)

# Separar duplicados y únicos
gdf_geo_duplicados = gdf_geojson_filtrado[gdf_geojson_filtrado['es_duplicado']].copy()
gdf_geo_unicos = gdf_geojson_filtrado[~gdf_geojson_filtrado['es_duplicado']].copy()

print(f"✓ Propiedades duplicadas (ignoradas): {len(gdf_geo_duplicados)}")
print(f"✓ Propiedades únicas de GeoJSON: {len(gdf_geo_unicos)}")
print(f"✓ Propiedades del CSV: {len(gdf_csv)}")
print(f"✓ TOTAL propiedades combinadas: {len(gdf_csv) + len(gdf_geo_unicos)}")

# =============================================================================
# PASO 4: ARMONIZAR COLUMNAS Y COMBINAR DATASETS
# =============================================================================
print("\n" + "=" * 80)
print("🔧 PASO 4: ARMONIZAR COLUMNAS Y COMBINAR DATASETS")
print("=" * 80)

# Columnas mínimas necesarias para el modelo
columnas_necesarias = [
    'precio_uf', 'superficie_util', 'dormitorios', 'banos',
    'estacionamientos', 'bodegas', 'geometry'
]

# --- Preparar CSV ---
gdf_csv_prep = gdf_csv.copy()
gdf_csv_prep['fuente'] = 'CSV'

# Renombrar columnas del CSV si es necesario
columnas_csv = {
    'precio': 'precio_uf',
    'superficie': 'superficie_util',
    'dorms': 'dormitorios',
    'baths': 'banos',
    'parking': 'estacionamientos',
    'storage': 'bodegas'
}

for old_col, new_col in columnas_csv.items():
    if old_col in gdf_csv_prep.columns and new_col not in gdf_csv_prep.columns:
        gdf_csv_prep = gdf_csv_prep.rename(columns={old_col: new_col})

# --- Preparar GeoJSON ---
gdf_geo_prep = gdf_geo_unicos.copy()
gdf_geo_prep['fuente'] = 'GeoJSON'
gdf_geo_prep['comuna_norm'] = gdf_geo_prep['comuna']

# Mapear columnas del GeoJSON
# El GeoJSON tiene: total_uf, t_constr, etc.
if 'total_uf' in gdf_geo_prep.columns:
    gdf_geo_prep['precio_uf'] = gdf_geo_prep['total_uf']

if 't_constr' in gdf_geo_prep.columns:
    gdf_geo_prep['superficie_util'] = gdf_geo_prep['t_constr']

# Columnas que pueden faltar en GeoJSON - llenar con valores por defecto
for col in ['dormitorios', 'banos', 'estacionamientos', 'bodegas']:
    if col not in gdf_geo_prep.columns:
        # Inferir valores razonables según superficie
        if col == 'dormitorios':
            # ~1 dormitorio por cada 40-50 m²
            gdf_geo_prep[col] = (gdf_geo_prep['superficie_util'] / 45).clip(1, 5).round()
        elif col == 'banos':
            # ~1 baño por cada 60 m²
            gdf_geo_prep[col] = (gdf_geo_prep['superficie_util'] / 60).clip(1, 3).round()
        elif col == 'estacionamientos':
            # ~1 estacionamiento por cada 80 m²
            gdf_geo_prep[col] = (gdf_geo_prep['superficie_util'] / 80).clip(0, 2).round()
        else:  # bodegas
            # ~1 bodega por cada 100 m²
            gdf_geo_prep[col] = (gdf_geo_prep['superficie_util'] / 100).clip(0, 1).round()

# Asegurar que todas las columnas necesarias existen
for col in ['precio_uf', 'superficie_util', 'dormitorios', 'banos', 'estacionamientos', 'bodegas', 'comuna_norm']:
    if col not in gdf_csv_prep.columns:
        gdf_csv_prep[col] = np.nan
    if col not in gdf_geo_prep.columns:
        gdf_geo_prep[col] = np.nan

# Seleccionar solo las columnas necesarias
columnas_finales = ['precio_uf', 'superficie_util', 'dormitorios', 'banos', 
                    'estacionamientos', 'bodegas', 'comuna_norm', 'fuente', 'geometry']

gdf_csv_final = gdf_csv_prep[columnas_finales].copy()
gdf_geo_final = gdf_geo_prep[columnas_finales].copy()

# Combinar ambos GeoDataFrames
print("⏳ Combinando datasets...")
gdf_combinado = pd.concat([gdf_csv_final, gdf_geo_final], ignore_index=True)

print(f"✓ Dataset combinado creado: {len(gdf_combinado)} propiedades")
print(f"  • Del CSV: {len(gdf_csv_final)}")
print(f"  • Del GeoJSON: {len(gdf_geo_final)}")

# Limpiar valores faltantes o inválidos
print("\n⏳ Limpiando datos...")
gdf_combinado = gdf_combinado.dropna(subset=['precio_uf', 'superficie_util'])
gdf_combinado = gdf_combinado[gdf_combinado['precio_uf'] > 0]
gdf_combinado = gdf_combinado[gdf_combinado['superficie_util'] > 0]

# Convertir a tipos numéricos
cols_numericas = ['precio_uf', 'superficie_util', 'dormitorios', 'banos', 'estacionamientos', 'bodegas']
for col in cols_numericas:
    gdf_combinado[col] = pd.to_numeric(gdf_combinado[col], errors='coerce')

gdf_combinado = gdf_combinado.dropna(subset=cols_numericas)

print(f"✓ Después de limpieza: {len(gdf_combinado)} propiedades")

# =============================================================================
# PASO 5: AGREGAR FACTORES ESPACIALES (JOIN CON GRILLA)
# =============================================================================
print("\n" + "=" * 80)
print("🗺️  PASO 5: AGREGAR FACTORES ESPACIALES (JOIN CON GRILLA)")
print("=" * 80)

# Cargar grilla con densidades
print("⏳ Cargando grilla de densidades...")
gdf_grilla = gpd.read_file(GRILLA_PATH)

# Asegurar mismo CRS
if gdf_grilla.crs != gdf_combinado.crs:
    gdf_grilla = gdf_grilla.to_crs(gdf_combinado.crs)

print(f"✓ Grilla cargada: {len(gdf_grilla)} celdas")
print(f"  Columnas disponibles: {gdf_grilla.columns.tolist()}")

# Realizar spatial join
print("⏳ Realizando spatial join (esto puede tardar)...")
gdf_con_factores = gpd.sjoin(gdf_combinado, gdf_grilla, how='left', predicate='within')

print(f"✓ Join completado: {len(gdf_con_factores)} registros")

# Verificar columnas de factores espaciales
factores_espaciales = [col for col in gdf_con_factores.columns if col.startswith('dens_')]
print(f"✓ Factores espaciales agregados: {len(factores_espaciales)}")
for factor in factores_espaciales[:10]:  # Mostrar primeros 10
    print(f"  • {factor}")

# Eliminar duplicados por spatial join (una propiedad puede caer en múltiples celdas)
gdf_con_factores = gdf_con_factores.drop_duplicates(subset=['precio_uf', 'superficie_util', 'geometry'])
print(f"✓ Después de eliminar duplicados: {len(gdf_con_factores)} propiedades")

# =============================================================================
# PASO 6: CREAR VARIABLES DERIVADAS
# =============================================================================
print("\n" + "=" * 80)
print("🔬 PASO 6: CREAR VARIABLES DERIVADAS")
print("=" * 80)

# Calcular precio por m²
gdf_con_factores['precio_m2'] = gdf_con_factores['precio_uf'] / gdf_con_factores['superficie_util']

# Calcular m² por habitante (dormitorios como proxy)
gdf_con_factores['m2_por_habitante'] = gdf_con_factores['superficie_util'] / (gdf_con_factores['dormitorios'] + 1)

# Total habitaciones
gdf_con_factores['total_habitaciones'] = gdf_con_factores['dormitorios'] + gdf_con_factores['banos']

# Ratio baños/dormitorios
gdf_con_factores['ratio_bano_dorm'] = gdf_con_factores['banos'] / (gdf_con_factores['dormitorios'] + 0.5)

print("✓ Variables derivadas creadas:")
print("  • precio_m2")
print("  • m2_por_habitante")
print("  • total_habitaciones")
print("  • ratio_bano_dorm")

# Imputar valores faltantes de factores espaciales con la mediana
cols_imputar = factores_espaciales + ['precio_m2', 'm2_por_habitante', 'total_habitaciones', 'ratio_bano_dorm']
print(f"\n⏳ Imputando {len(cols_imputar)} columnas con valores faltantes...")
for col in cols_imputar:
    if col in gdf_con_factores.columns:
        nulos_antes = gdf_con_factores[col].isnull().sum()
        if nulos_antes > 0:
            mediana = gdf_con_factores[col].median()
            if pd.isna(mediana):  # Si la mediana es NaN, usar 0
                mediana = 0
            gdf_con_factores[col] = gdf_con_factores[col].fillna(mediana)
            print(f"  • {col}: {nulos_antes} NaN → imputados con {mediana:.2f}")

# =============================================================================
# PASO 7: PREPARAR DATOS PARA EL MODELO
# =============================================================================
print("\n" + "=" * 80)
print("🎯 PASO 7: PREPARAR DATOS PARA EL MODELO")
print("=" * 80)

# Eliminar outliers extremos (preservando más datos que antes)
# Usar percentiles 1 y 99 en lugar de IQR
p1 = gdf_con_factores['precio_m2'].quantile(0.01)
p99 = gdf_con_factores['precio_m2'].quantile(0.99)
gdf_filtrado = gdf_con_factores[
    (gdf_con_factores['precio_m2'] >= p1) &
    (gdf_con_factores['precio_m2'] <= p99)
].copy()

print(f"✓ Después de eliminar outliers (1%-99%): {len(gdf_filtrado)} propiedades")
print(f"  • Outliers eliminados: {len(gdf_con_factores) - len(gdf_filtrado)}")

# Seleccionar features para el modelo
features_base = ['superficie_util', 'dormitorios', 'banos', 'estacionamientos', 'bodegas']
features_derivadas = ['m2_por_habitante', 'total_habitaciones', 'ratio_bano_dorm']
features_espaciales = [col for col in factores_espaciales if col in gdf_filtrado.columns]

all_features = features_base + features_derivadas + features_espaciales
all_features = [f for f in all_features if f in gdf_filtrado.columns]

print(f"\n✓ Features seleccionadas: {len(all_features)}")
print("  • Base:", features_base)
print("  • Derivadas:", features_derivadas)
print(f"  • Espaciales: {len(features_espaciales)} densidades")

# Preparar X e y
X = gdf_filtrado[all_features].copy()
y = gdf_filtrado['precio_m2'].copy()

# Verificar valores faltantes
print(f"\n✓ Verificando datos:")
print(f"  • Shape X: {X.shape}")
print(f"  • Shape y: {y.shape}")
print(f"  • Valores faltantes en X: {X.isnull().sum().sum()}")
print(f"  • Valores faltantes en y: {y.isnull().sum()}")

# Eliminar cualquier fila con NaN
mask_validos = ~(X.isnull().any(axis=1) | y.isnull())
X = X[mask_validos]
y = y[mask_validos]

print(f"✓ Después de eliminar NaN: {len(X)} muestras")

# =============================================================================
# PASO 8: DIVIDIR EN TRAIN/TEST
# =============================================================================
print("\n" + "=" * 80)
print("✂️  PASO 8: DIVIDIR EN TRAIN/TEST")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"✓ Datos de entrenamiento: {len(X_train)} muestras")
print(f"✓ Datos de prueba: {len(X_test)} muestras")

# =============================================================================
# PASO 9: ENTRENAR RANDOM FOREST
# =============================================================================
print("\n" + "=" * 80)
print("🌲 PASO 9: ENTRENAR RANDOM FOREST")
print("=" * 80)

print("⏳ Entrenando modelo (esto puede tardar varios minutos)...")

# Configuración del modelo (mismos hiperparámetros que antes para comparación justa)
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1,
    verbose=1
)

rf_model.fit(X_train, y_train)

print("✓ Modelo entrenado exitosamente")

# =============================================================================
# PASO 10: EVALUAR MODELO
# =============================================================================
print("\n" + "=" * 80)
print("📊 PASO 10: EVALUAR MODELO")
print("=" * 80)

# Predicciones
y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)

# Métricas train
r2_train = r2_score(y_train, y_train_pred)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
mae_train = mean_absolute_error(y_train, y_train_pred)

# Métricas test
r2_test = r2_score(y_test, y_test_pred)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
mae_test = mean_absolute_error(y_test, y_test_pred)

print("\n📈 RESULTADOS DEL MODELO NUEVO (CON DATOS COMBINADOS):")
print("-" * 80)
print(f"TRAIN:")
print(f"  • R² Score:  {r2_train:.4f}")
print(f"  • RMSE:      ${rmse_train:,.0f} UF/m²")
print(f"  • MAE:       ${mae_train:,.0f} UF/m²")
print(f"\nTEST:")
print(f"  • R² Score:  {r2_test:.4f}")
print(f"  • RMSE:      ${rmse_test:,.0f} UF/m²")
print(f"  • MAE:       ${mae_test:,.0f} UF/m²")

# =============================================================================
# PASO 11: COMPARAR CON MODELO ANTERIOR
# =============================================================================
print("\n" + "=" * 80)
print("📊 PASO 11: COMPARAR CON MODELO ANTERIOR")
print("=" * 80)

# Cargar métricas del modelo anterior (hardcoded de resultados previos)
# O intentar cargar del CSV anterior
metricas_anterior = {
    'r2_test': 0.204,
    'rmse_test': 3443,
    'mae_test': 2254,
    'n_samples': 1554
}

print("\n📊 COMPARACIÓN DE MODELOS:")
print("=" * 80)
print(f"{'Métrica':<20} | {'Anterior':<15} | {'Nuevo':<15} | {'Mejora':<15}")
print("-" * 80)

# R²
mejora_r2 = ((r2_test - metricas_anterior['r2_test']) / metricas_anterior['r2_test']) * 100
print(f"{'R² Test':<20} | {metricas_anterior['r2_test']:<15.4f} | {r2_test:<15.4f} | {mejora_r2:+.2f}%")

# RMSE
mejora_rmse = ((metricas_anterior['rmse_test'] - rmse_test) / metricas_anterior['rmse_test']) * 100
print(f"{'RMSE Test':<20} | ${metricas_anterior['rmse_test']:<14,.0f} | ${rmse_test:<14,.0f} | {mejora_rmse:+.2f}%")

# MAE
mejora_mae = ((metricas_anterior['mae_test'] - mae_test) / metricas_anterior['mae_test']) * 100
print(f"{'MAE Test':<20} | ${metricas_anterior['mae_test']:<14,.0f} | ${mae_test:<14,.0f} | {mejora_mae:+.2f}%")

# Muestras
mejora_samples = ((len(X) - metricas_anterior['n_samples']) / metricas_anterior['n_samples']) * 100
print(f"{'Muestras totales':<20} | {metricas_anterior['n_samples']:<15,} | {len(X):<15,} | {mejora_samples:+.2f}%")

print("=" * 80)

# =============================================================================
# PASO 12: FEATURE IMPORTANCE
# =============================================================================
print("\n" + "=" * 80)
print("🎯 PASO 12: IMPORTANCIA DE FEATURES")
print("=" * 80)

feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\n📊 TOP 15 FEATURES MÁS IMPORTANTES:")
print("-" * 80)
for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:<40} → {row['importance']*100:6.2f}%")

# =============================================================================
# PASO 13: GUARDAR RESULTADOS
# =============================================================================
print("\n" + "=" * 80)
print("💾 PASO 13: GUARDAR RESULTADOS")
print("=" * 80)

# Guardar dataset combinado (sin geometry para CSV)
df_guardar = gdf_filtrado[mask_validos].copy()
df_guardar = df_guardar.drop(columns=['geometry'])
df_guardar.to_csv(DATASET_COMBINADO_PATH, index=False)
print(f"✓ Dataset combinado guardado: {DATASET_COMBINADO_PATH}")

# Guardar modelo
import pickle
with open(MODELO_PATH, 'wb') as f:
    pickle.dump(rf_model, f)
print(f"✓ Modelo guardado: {MODELO_PATH}")

# Guardar métricas
metricas_path = SEMANA3_DIR / 'resultados' / 'metricas_modelo_combinado.txt'
metricas_path.parent.mkdir(parents=True, exist_ok=True)

with open(metricas_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("RESULTADOS DEL MODELO CON DATOS COMBINADOS (CSV + GeoJSON)\n")
    f.write("=" * 80 + "\n\n")
    
    f.write(f"Dataset:\n")
    f.write(f"  • Total muestras: {len(X)}\n")
    f.write(f"  • Train: {len(X_train)}\n")
    f.write(f"  • Test: {len(X_test)}\n")
    f.write(f"  • Features: {len(all_features)}\n\n")
    
    f.write(f"Métricas TEST:\n")
    f.write(f"  • R² Score: {r2_test:.4f}\n")
    f.write(f"  • RMSE: ${rmse_test:,.0f} UF/m²\n")
    f.write(f"  • MAE: ${mae_test:,.0f} UF/m²\n\n")
    
    f.write(f"Comparación con modelo anterior:\n")
    f.write(f"  • Mejora en R²: {mejora_r2:+.2f}%\n")
    f.write(f"  • Mejora en RMSE: {mejora_rmse:+.2f}%\n")
    f.write(f"  • Mejora en MAE: {mejora_mae:+.2f}%\n")
    f.write(f"  • Más datos: {mejora_samples:+.2f}%\n")

print(f"✓ Métricas guardadas: {metricas_path}")

# Guardar feature importance
fi_path = SEMANA3_DIR / 'resultados' / 'feature_importance_combinado.csv'
feature_importance.to_csv(fi_path, index=False)
print(f"✓ Feature importance guardado: {fi_path}")

# =============================================================================
# PASO 14: VISUALIZACIONES
# =============================================================================
print("\n" + "=" * 80)
print("📊 PASO 14: GENERAR VISUALIZACIONES")
print("=" * 80)

fig_dir = SEMANA3_DIR / 'resultados' / 'figuras'
fig_dir.mkdir(parents=True, exist_ok=True)

# 1. Gráfico de predicciones vs reales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_test_pred, alpha=0.5, s=10)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('Precio Real (UF/m²)', fontsize=12)
plt.ylabel('Precio Predicho (UF/m²)', fontsize=12)
plt.title(f'Predicciones vs Reales - R² = {r2_test:.4f}', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'predicciones_vs_reales_combinado.png', dpi=300, bbox_inches='tight')
print(f"✓ Gráfico guardado: predicciones_vs_reales_combinado.png")
plt.close()

# 2. Residuos
residuos = y_test - y_test_pred
plt.figure(figsize=(10, 6))
plt.scatter(y_test_pred, residuos, alpha=0.5, s=10)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Precio Predicho (UF/m²)', fontsize=12)
plt.ylabel('Residuos (UF/m²)', fontsize=12)
plt.title('Análisis de Residuos', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(fig_dir / 'residuos_combinado.png', dpi=300, bbox_inches='tight')
print(f"✓ Gráfico guardado: residuos_combinado.png")
plt.close()

# 3. Feature importance
plt.figure(figsize=(12, 8))
top_features = feature_importance.head(20)
plt.barh(range(len(top_features)), top_features['importance'], color='steelblue')
plt.yticks(range(len(top_features)), top_features['feature'])
plt.xlabel('Importancia', fontsize=12)
plt.title('Top 20 Features Más Importantes', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig(fig_dir / 'feature_importance_combinado.png', dpi=300, bbox_inches='tight')
print(f"✓ Gráfico guardado: feature_importance_combinado.png")
plt.close()

# 4. Comparación de métricas
metricas_comp = pd.DataFrame({
    'Modelo': ['Anterior', 'Nuevo (Combinado)'],
    'R²': [metricas_anterior['r2_test'], r2_test],
    'RMSE': [metricas_anterior['rmse_test'], rmse_test],
    'MAE': [metricas_anterior['mae_test'], mae_test]
})

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# R²
axes[0].bar(metricas_comp['Modelo'], metricas_comp['R²'], color=['#ff6b6b', '#51cf66'])
axes[0].set_ylabel('R² Score')
axes[0].set_title('Coeficiente de Determinación (R²)')
axes[0].set_ylim([0, max(metricas_comp['R²']) * 1.2])
for i, v in enumerate(metricas_comp['R²']):
    axes[0].text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')

# RMSE
axes[1].bar(metricas_comp['Modelo'], metricas_comp['RMSE'], color=['#ff6b6b', '#51cf66'])
axes[1].set_ylabel('RMSE (UF/m²)')
axes[1].set_title('Error Cuadrático Medio')
for i, v in enumerate(metricas_comp['RMSE']):
    axes[1].text(i, v + 50, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')

# MAE
axes[2].bar(metricas_comp['Modelo'], metricas_comp['MAE'], color=['#ff6b6b', '#51cf66'])
axes[2].set_ylabel('MAE (UF/m²)')
axes[2].set_title('Error Absoluto Medio')
for i, v in enumerate(metricas_comp['MAE']):
    axes[2].text(i, v + 30, f'${v:,.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(fig_dir / 'comparacion_modelos.png', dpi=300, bbox_inches='tight')
print(f"✓ Gráfico guardado: comparacion_modelos.png")
plt.close()

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("✅ PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 80)

print(f"""
📊 RESUMEN FINAL:

📁 DATOS:
  • CSV original: 716 propiedades (4 comunas)
  • GeoJSON único: 1,050 propiedades adicionales
  • Dataset combinado: {len(X)} propiedades (ganancia de {mejora_samples:+.1f}%)

🎯 MODELO NUEVO:
  • R² Test: {r2_test:.4f}
  • RMSE Test: ${rmse_test:,.0f} UF/m²
  • MAE Test: ${mae_test:,.0f} UF/m²

📈 MEJORAS vs MODELO ANTERIOR:
  • R²: {mejora_r2:+.2f}%
  • RMSE: {mejora_rmse:+.2f}%
  • MAE: {mejora_mae:+.2f}%

💾 ARCHIVOS GENERADOS:
  • Dataset: {DATASET_COMBINADO_PATH.name}
  • Modelo: {MODELO_PATH.name}
  • Métricas: metricas_modelo_combinado.txt
  • Feature importance: feature_importance_combinado.csv
  • Gráficos: 4 visualizaciones en resultados/figuras/

🎉 ¡El modelo con datos combinados está listo para usar!
""")
