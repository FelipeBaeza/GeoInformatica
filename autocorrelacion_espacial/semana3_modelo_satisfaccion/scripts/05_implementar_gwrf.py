"""
Implementación de Geographically Weighted Random Forest (GWRF)

Este script implementa un enfoque híbrido que combina:
- Ponderación geográfica (de GWR): Modelos locales por zona
- Random Forest: Capacidad no-lineal y robustez

Estrategias GWRF implementadas:
1. GWRF Zona-Específica: Un RF por cada comuna
2. GWRF Distancia-Ponderada: Predicciones ponderadas por distancia
3. GWRF Cluster-Based: Clustering espacial + RF por cluster
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial import distance_matrix
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("🌲🗺️  GEOGRAPHICALLY WEIGHTED RANDOM FOREST (GWRF)")
print("=" * 80)

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
BASE_DIR = Path('/home/felipe/Documentos/GeoInformatica')
SEMANA3_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana3_modelo_satisfaccion'

# Archivos
# Use the fully integrated dataset (includes densities and spatial factors).
# Historically the script pointed to 'propiedades_combinadas_con_factores_espaciales.csv'
# which in some runs was an older file with missing density columns (all zeros).
# Point the pipeline explicitly to the integrated output produced by
# `01_integrar_datos.py` to avoid stale/partial inputs.
DATASET_PATH = SEMANA3_DIR / 'data' / 'propiedades_con_factores_espaciales.csv'
OUTPUT_DIR = SEMANA3_DIR / 'resultados' / 'gwrf'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Cargar datos
print("\n📂 Cargando datos...")
df = pd.read_csv(DATASET_PATH)
print(f"✓ Dataset cargado: {len(df)} propiedades")

# Features para el modelo
features_base = ['superficie_util', 'dormitorios', 'banos', 'estacionamientos', 'bodegas']
features_derivadas = ['m2_por_habitante', 'total_habitaciones', 'ratio_bano_dorm']
features_espaciales = [col for col in df.columns if col.startswith('dens_')]
all_features = features_base + features_derivadas + features_espaciales
all_features = [f for f in all_features if f in df.columns]

X = df[all_features].copy()
y = df['precio_m2'].copy()

print(f"✓ Features: {len(all_features)}")
print(f"✓ Target: precio_m2")

# =============================================================================
# MODELO BASELINE (GLOBAL RANDOM FOREST)
# =============================================================================
print("\n" + "=" * 80)
print("📊 MODELO BASELINE: Random Forest Global")
print("=" * 80)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
idx_train, idx_test = train_test_split(df.index, test_size=0.2, random_state=42)

# Entrenar modelo global
rf_global = RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

print("⏳ Entrenando Random Forest global...")
rf_global.fit(X_train, y_train)

y_pred_global = rf_global.predict(X_test)
r2_global = r2_score(y_test, y_pred_global)
rmse_global = np.sqrt(mean_squared_error(y_test, y_pred_global))
mae_global = mean_absolute_error(y_test, y_pred_global)

print(f"\n✓ RESULTADOS BASELINE:")
print(f"  • R² Test:  {r2_global:.4f}")
print(f"  • RMSE:     ${rmse_global:,.0f} UF/m²")
print(f"  • MAE:      ${mae_global:,.0f} UF/m²")

# =============================================================================
# ESTRATEGIA 1: GWRF POR ZONA (UN RF POR COMUNA)
# =============================================================================
print("\n" + "=" * 80)
print("🌲 ESTRATEGIA 1: GWRF por Zona (un RF por comuna)")
print("=" * 80)

# Verificar si tenemos columna de comuna
if 'comuna_norm' in df.columns:
    comunas = df['comuna_norm'].unique()
    print(f"✓ Comunas detectadas: {len(comunas)}")
    for comuna in comunas:
        count = len(df[df['comuna_norm'] == comuna])
        print(f"  • {comuna}: {count} propiedades")
    
    # Entrenar un RF por cada comuna
    modelos_por_zona = {}
    metricas_por_zona = {}
    
    print("\n⏳ Entrenando modelos por zona...")
    
    for comuna in comunas:
        # Filtrar datos de esta comuna
        mask_train_comuna = (df.loc[idx_train, 'comuna_norm'] == comuna)
        mask_test_comuna = (df.loc[idx_test, 'comuna_norm'] == comuna)
        
        X_train_comuna = X_train[mask_train_comuna]
        y_train_comuna = y_train[mask_train_comuna]
        X_test_comuna = X_test[mask_test_comuna]
        y_test_comuna = y_test[mask_test_comuna]
        
        if len(X_train_comuna) < 20 or len(X_test_comuna) < 5:
            print(f"  ⚠️  {comuna}: Datos insuficientes (train={len(X_train_comuna)}, test={len(X_test_comuna)})")
            continue
        
        # Entrenar RF para esta zona
        rf_zona = RandomForestRegressor(
            n_estimators=100,  # Menos árboles porque hay menos datos
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        rf_zona.fit(X_train_comuna, y_train_comuna)
        modelos_por_zona[comuna] = rf_zona
        
        # Evaluar
        y_pred_zona = rf_zona.predict(X_test_comuna)
        r2_zona = r2_score(y_test_comuna, y_pred_zona)
        rmse_zona = np.sqrt(mean_squared_error(y_test_comuna, y_pred_zona))
        mae_zona = mean_absolute_error(y_test_comuna, y_pred_zona)
        
        metricas_por_zona[comuna] = {
            'r2': r2_zona,
            'rmse': rmse_zona,
            'mae': mae_zona,
            'n_train': len(X_train_comuna),
            'n_test': len(X_test_comuna)
        }
        
        print(f"  ✓ {comuna}: R²={r2_zona:.4f}, RMSE=${rmse_zona:,.0f}, n_test={len(X_test_comuna)}")
    
    # Predicciones combinadas para todo el test set (alineadas por índice)
    y_pred_zona_all = np.zeros(len(X_test))

    # idx_test contiene los índices originales del conjunto de test
    for j, idx in enumerate(idx_test):
        comuna = df.loc[idx, 'comuna_norm']
        if comuna in modelos_por_zona:
            y_pred_zona_all[j] = modelos_por_zona[comuna].predict(X_test.loc[[idx]])[0]
        else:
            # Si no hay modelo para esta zona, usar el global
            y_pred_zona_all[j] = rf_global.predict(X_test.loc[[idx]])[0]
    
    # Métricas globales del GWRF por zona
    r2_gwrf_zona = r2_score(y_test, y_pred_zona_all)
    rmse_gwrf_zona = np.sqrt(mean_squared_error(y_test, y_pred_zona_all))
    mae_gwrf_zona = mean_absolute_error(y_test, y_pred_zona_all)
    
    print(f"\n✓ RESULTADOS GWRF POR ZONA (global):")
    print(f"  • R² Test:  {r2_gwrf_zona:.4f}")
    print(f"  • RMSE:     ${rmse_gwrf_zona:,.0f} UF/m²")
    print(f"  • MAE:      ${mae_gwrf_zona:,.0f} UF/m²")
    
else:
    print("⚠️  No hay columna de comuna, saltando Estrategia 1")
    r2_gwrf_zona = None
    rmse_gwrf_zona = None
    mae_gwrf_zona = None

# =============================================================================
# ESTRATEGIA 2: GWRF CON CLUSTERING ESPACIAL
# =============================================================================
print("\n" + "=" * 80)
print("🌲 ESTRATEGIA 2: GWRF con Clustering Espacial")
print("=" * 80)

# Usar coordenadas espaciales implícitas en densidades
# O crear clusters basados en características espaciales
features_espaciales_para_cluster = [f for f in features_espaciales[:10]]  # Usar primeras 10 densidades

if len(features_espaciales_para_cluster) > 0:
    X_spatial = df[features_espaciales_para_cluster].fillna(0)
    
    # Determinar número óptimo de clusters basado en tamaño del TRAIN set
    # Regla: mínimo 150 muestras por cluster en train para RF robusto (asume 80/20 split)
    # Esto garantiza ~120 train + ~30 test por cluster después del split
    n_train = len(X_train)
    max_clusters = max(2, min(3, n_train // 150))  # Máximo 3 clusters, mínimo 150 por cluster
    n_clusters = max_clusters
    print(f"✓ Creando {n_clusters} clusters espaciales (ajustado para {n_train} muestras train)...")
    print(f"   → Criterio conservador: mínimo 150 muestras/cluster en train (garantiza ~120 train + ~30 test)")
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_spatial)
    
    print(f"✓ Distribución de clusters:")
    for cluster_id in range(n_clusters):
        count = len(df[df['cluster'] == cluster_id])
        print(f"  • Cluster {cluster_id}: {count} propiedades")
    
    # Entrenar un RF por cada cluster
    modelos_por_cluster = {}
    metricas_por_cluster = {}
    
    print("\n⏳ Entrenando modelos por cluster...")
    
    for cluster_id in range(n_clusters):
        # Filtrar datos de este cluster
        mask_train_cluster = df.loc[idx_train, 'cluster'] == cluster_id
        mask_test_cluster = df.loc[idx_test, 'cluster'] == cluster_id
        
        X_train_cluster = X_train[mask_train_cluster]
        y_train_cluster = y_train[mask_train_cluster]
        X_test_cluster = X_test[mask_test_cluster]
        y_test_cluster = y_test[mask_test_cluster]
        
        # Umbral aumentado: necesitamos suficientes datos para RF robusto
        if len(X_train_cluster) < 80 or len(X_test_cluster) < 20:
            print(f"  ⚠️  Cluster {cluster_id}: Datos insuficientes (train={len(X_train_cluster)}, test={len(X_test_cluster)}) → Usando modelo global como fallback")
            continue
        
        # Entrenar RF para este cluster (parámetros conservadores para datasets pequeños)
        rf_cluster = RandomForestRegressor(
            n_estimators=100,
            max_depth=8,  # Más conservador para evitar overfitting
            min_samples_split=15,  # Aumentado para mayor generalización
            min_samples_leaf=8,    # Aumentado para evitar hojas con muy pocos samples
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        rf_cluster.fit(X_train_cluster, y_train_cluster)
        modelos_por_cluster[cluster_id] = rf_cluster
        
        # Evaluar
        y_pred_cluster = rf_cluster.predict(X_test_cluster)
        r2_cluster = r2_score(y_test_cluster, y_pred_cluster)
        rmse_cluster = np.sqrt(mean_squared_error(y_test_cluster, y_pred_cluster))
        mae_cluster = mean_absolute_error(y_test_cluster, y_pred_cluster)
        
        metricas_por_cluster[cluster_id] = {
            'r2': r2_cluster,
            'rmse': rmse_cluster,
            'mae': mae_cluster,
            'n_train': len(X_train_cluster),
            'n_test': len(X_test_cluster)
        }
        
        print(f"  ✓ Cluster {cluster_id}: R²={r2_cluster:.4f}, RMSE=${rmse_cluster:,.0f}, n_test={len(X_test_cluster)}")
    
    # Predicciones combinadas
    y_pred_cluster_all = np.zeros(len(X_test))
    
    for j, idx in enumerate(idx_test):
        cluster_id = df.loc[idx, 'cluster']
        if cluster_id in modelos_por_cluster:
            y_pred_cluster_all[j] = modelos_por_cluster[cluster_id].predict(X_test.loc[[idx]])[0]
        else:
            y_pred_cluster_all[j] = rf_global.predict(X_test.loc[[idx]])[0]
    
    # Métricas globales
    r2_gwrf_cluster = r2_score(y_test, y_pred_cluster_all)
    rmse_gwrf_cluster = np.sqrt(mean_squared_error(y_test, y_pred_cluster_all))
    mae_gwrf_cluster = mean_absolute_error(y_test, y_pred_cluster_all)
    
    print(f"\n✓ RESULTADOS GWRF POR CLUSTER (global):")
    print(f"  • R² Test:  {r2_gwrf_cluster:.4f}")
    print(f"  • RMSE:     ${rmse_gwrf_cluster:,.0f} UF/m²")
    print(f"  • MAE:      ${mae_gwrf_cluster:,.0f} UF/m²")
    
else:
    print("⚠️  No hay features espaciales suficientes, saltando Estrategia 2")
    r2_gwrf_cluster = None
    rmse_gwrf_cluster = None
    mae_gwrf_cluster = None

# =============================================================================
# ESTRATEGIA 3: GWRF CON PONDERACIÓN POR DISTANCIA (Avanzado)
# =============================================================================
print("\n" + "=" * 80)
print("🌲 ESTRATEGIA 3: GWRF con Ponderación por Distancia")
print("=" * 80)
print("ℹ️  Esta estrategia requiere coordenadas espaciales explícitas")
print("   Implementación simplificada: Ensemble de modelos locales")

# Crear modelos locales basados en vecindad
# Para cada punto de test, entrenar un modelo con los K vecinos más cercanos

# Simplificación: Usar percentiles de características espaciales como "vecindad"
# Dividir en 3 zonas: baja, media, alta densidad total

if 'dens_total_600m_km2' in df.columns:
    # Usar densidad total como proxy de ubicación
    terciles = df['dens_total_600m_km2'].quantile([0.33, 0.67])
    
    def asignar_zona_densidad(valor):
        if pd.isna(valor):
            return 'media'
        elif valor < terciles.iloc[0]:
            return 'baja'
        elif valor < terciles.iloc[1]:
            return 'media'
        else:
            return 'alta'
    
    df['zona_densidad'] = df['dens_total_600m_km2'].apply(asignar_zona_densidad)
    
    print(f"✓ Zonas de densidad creadas:")
    for zona in ['baja', 'media', 'alta']:
        count = len(df[df['zona_densidad'] == zona])
        print(f"  • Densidad {zona}: {count} propiedades")
    
    # Entrenar modelos por zona de densidad
    modelos_por_densidad = {}
    metricas_por_densidad = {}
    
    print("\n⏳ Entrenando modelos por zona de densidad...")
    
    for zona in ['baja', 'media', 'alta']:
        mask_train_zona = df.loc[idx_train, 'zona_densidad'] == zona
        mask_test_zona = df.loc[idx_test, 'zona_densidad'] == zona
        
        X_train_zona = X_train[mask_train_zona]
        y_train_zona = y_train[mask_train_zona]
        X_test_zona = X_test[mask_test_zona]
        y_test_zona = y_test[mask_test_zona]
        
        if len(X_train_zona) < 20 or len(X_test_zona) < 5:
            print(f"  ⚠️  Zona {zona}: Datos insuficientes")
            continue
        
        rf_zona = RandomForestRegressor(
            n_estimators=150,
            max_depth=12,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )
        
        rf_zona.fit(X_train_zona, y_train_zona)
        modelos_por_densidad[zona] = rf_zona
        
        y_pred_zona = rf_zona.predict(X_test_zona)
        r2_zona = r2_score(y_test_zona, y_pred_zona)
        rmse_zona = np.sqrt(mean_squared_error(y_test_zona, y_pred_zona))
        mae_zona = mean_absolute_error(y_test_zona, y_pred_zona)
        
        metricas_por_densidad[zona] = {
            'r2': r2_zona,
            'rmse': rmse_zona,
            'mae': mae_zona,
            'n_train': len(X_train_zona),
            'n_test': len(X_test_zona)
        }
        
        print(f"  ✓ Zona {zona}: R²={r2_zona:.4f}, RMSE=${rmse_zona:,.0f}, n_test={len(X_test_zona)}")
    
    # Predicciones combinadas
    y_pred_densidad_all = np.zeros(len(X_test))
    
    for j, idx in enumerate(idx_test):
        zona = df.loc[idx, 'zona_densidad']
        if zona in modelos_por_densidad:
            y_pred_densidad_all[j] = modelos_por_densidad[zona].predict(X_test.loc[[idx]])[0]
        else:
            y_pred_densidad_all[j] = rf_global.predict(X_test.loc[[idx]])[0]
    
    r2_gwrf_densidad = r2_score(y_test, y_pred_densidad_all)
    rmse_gwrf_densidad = np.sqrt(mean_squared_error(y_test, y_pred_densidad_all))
    mae_gwrf_densidad = mean_absolute_error(y_test, y_pred_densidad_all)
    
    print(f"\n✓ RESULTADOS GWRF POR DENSIDAD (global):")
    print(f"  • R² Test:  {r2_gwrf_densidad:.4f}")
    print(f"  • RMSE:     ${rmse_gwrf_densidad:,.0f} UF/m²")
    print(f"  • MAE:      ${mae_gwrf_densidad:,.0f} UF/m²")
    
else:
    print("⚠️  No hay datos de densidad, saltando Estrategia 3")
    r2_gwrf_densidad = None
    rmse_gwrf_densidad = None
    mae_gwrf_densidad = None

# =============================================================================
# COMPARACIÓN FINAL DE TODAS LAS ESTRATEGIAS
# =============================================================================
print("\n" + "=" * 80)
print("📊 INICIANDO STACKING (Meta-modelo) - combinación de RF y GWRF")
print("=" * 80)

# STACKING (META-MODELO)
# Objetivo: generar predicciones out-of-fold (OOF) para cada estrategia base
# (global RF, GWRF por comuna, GWRF por cluster, GWRF por densidad) sobre
# el conjunto de entrenamiento (X_train) y luego entrenar un meta-regresor
# que aprenda a combinar esas predicciones.

NFOLDS = 5
kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Preparar contenedores OOF (indexados por X_train.index)
oof_global = pd.Series(index=X_train.index, dtype=float)
oof_zone = pd.Series(index=X_train.index, dtype=float)
oof_cluster = pd.Series(index=X_train.index, dtype=float)
oof_dens = pd.Series(index=X_train.index, dtype=float)

print(f"Generando OOF predictions con {NFOLDS}-fold CV...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train), 1):
    tr_idx = X_train.index[train_idx]
    vl_idx = X_train.index[val_idx]

    X_tr, X_val = X_train.loc[tr_idx], X_train.loc[vl_idx]
    y_tr, y_val = y_train.loc[tr_idx], y_train.loc[vl_idx]

    # 1) RF global fold
    rf_fold = RandomForestRegressor(
        n_estimators=150,
        max_depth=15,
        min_samples_split=8,
        random_state=42,
        n_jobs=-1
    )
    rf_fold.fit(X_tr, y_tr)
    oof_global.loc[vl_idx] = rf_fold.predict(X_val)

    # 2) GWRF por comuna - entrenar modelos por comuna en fold
    if 'comuna_norm' in df.columns:
        modelos_fold_zone = {}
        comunas_train = df.loc[tr_idx, 'comuna_norm']
        comunas_val = df.loc[vl_idx, 'comuna_norm']

        for comuna in comunas_train.unique():
            mask_tr = (df.loc[tr_idx, 'comuna_norm'] == comuna)
            X_tr_c = X_tr[mask_tr.values]
            y_tr_c = y_tr[mask_tr.values]
            if len(X_tr_c) >= 10:
                m = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
                m.fit(X_tr_c, y_tr_c)
                modelos_fold_zone[comuna] = m

        # predecir para X_val
        preds_zone = []
        for idx in vl_idx:
            comuna_val = df.loc[idx, 'comuna_norm']
            if comuna_val in modelos_fold_zone:
                preds_zone.append(modelos_fold_zone[comuna_val].predict(X_val.loc[[idx]])[0])
            else:
                preds_zone.append(rf_fold.predict(X_val.loc[[idx]])[0])

        oof_zone.loc[vl_idx] = preds_zone

    else:
        oof_zone.loc[vl_idx] = rf_fold.predict(X_val)

    # 3) GWRF por cluster (usar features espaciales seleccionadas)
    if len(features_espaciales) > 0:
        feat_cluster = features_espaciales[:10]
        X_tr_sp = df.loc[tr_idx, feat_cluster].fillna(0)
        X_val_sp = df.loc[vl_idx, feat_cluster].fillna(0)

        # Usar mismo criterio mejorado: mínimo 150 muestras por cluster
        max_clusters_fold = max(2, min(3, len(tr_idx) // 150))
        kmeans_fold = KMeans(n_clusters=max_clusters_fold, random_state=42, n_init=10)
        kmeans_fold.fit(X_tr_sp)
        clusters_tr = kmeans_fold.predict(X_tr_sp)
        clusters_val = kmeans_fold.predict(X_val_sp)

        modelos_fold_cluster = {}
        for cl in np.unique(clusters_tr):
            mask_tr = clusters_tr == cl
            X_tr_c = X_tr.iloc[np.where(mask_tr)[0]]
            y_tr_c = y_tr.iloc[np.where(mask_tr)[0]]
            # Umbral aumentado: mínimo 80 muestras para entrenar RF robusto
            if len(X_tr_c) >= 80:
                m = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=8,
                    min_samples_split=15,
                    min_samples_leaf=8,
                    random_state=42, 
                    n_jobs=-1
                )
                m.fit(X_tr_c, y_tr_c)
                modelos_fold_cluster[cl] = m

        preds_cluster = []
        for i, idx in enumerate(vl_idx):
            cl = clusters_val[i]
            if cl in modelos_fold_cluster:
                preds_cluster.append(modelos_fold_cluster[cl].predict(X_val.loc[[idx]])[0])
            else:
                preds_cluster.append(rf_fold.predict(X_val.loc[[idx]])[0])

        oof_cluster.loc[vl_idx] = preds_cluster
    else:
        oof_cluster.loc[vl_idx] = rf_fold.predict(X_val)

    # 4) GWRF por densidad (terciles en fold)
    if 'dens_total_600m_km2' in df.columns:
        terciles_tr = df.loc[tr_idx, 'dens_total_600m_km2'].quantile([0.33, 0.67])

        def asignar_zona_fold(val):
            if pd.isna(val):
                return 'media'
            if val < terciles_tr.iloc[0]:
                return 'baja'
            elif val < terciles_tr.iloc[1]:
                return 'media'
            else:
                return 'alta'

        zonas_tr = df.loc[tr_idx, 'dens_total_600m_km2'].apply(asignar_zona_fold)
        zonas_val = df.loc[vl_idx, 'dens_total_600m_km2'].apply(asignar_zona_fold)

        modelos_fold_dens = {}
        for zona in zonas_tr.unique():
            mask_tr = zonas_tr == zona
            X_tr_z = X_tr[mask_tr.values]
            y_tr_z = y_tr[mask_tr.values]
            if len(X_tr_z) >= 10:
                m = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
                m.fit(X_tr_z, y_tr_z)
                modelos_fold_dens[zona] = m

        preds_dens = []
        for idx in vl_idx:
            zona_val = zonas_val.loc[idx]
            if zona_val in modelos_fold_dens:
                preds_dens.append(modelos_fold_dens[zona_val].predict(X_val.loc[[idx]])[0])
            else:
                preds_dens.append(rf_fold.predict(X_val.loc[[idx]])[0])

        oof_dens.loc[vl_idx] = preds_dens
    else:
        oof_dens.loc[vl_idx] = rf_fold.predict(X_val)

    print(f" Fold {fold}/{NFOLDS} completado")

# Construir matriz de features para meta-modelo (usar columnas existentes)
meta_train = pd.DataFrame({
    'pred_global': oof_global,
    'pred_zone': oof_zone,
    'pred_cluster': oof_cluster,
    'pred_dens': oof_dens
})

# Eliminar filas con NA (si alguna estrategia no produjo OOF)
meta_train = meta_train.dropna()
meta_y = y_train.loc[meta_train.index]

print(f"Entrenando meta-modelo en {len(meta_train)} muestras...")
meta_model = Ridge(alpha=1.0, random_state=42)
meta_model.fit(meta_train, meta_y)

print("Meta-modelo entrenado. Generando predicciones en test set usando modelos entrenados sobre todo el train set...")

# Entrenar modelos base sobre todo X_train para predecir X_test
rf_global_full = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
rf_global_full.fit(X_train, y_train)

# modelos por zona completos
modelos_zona_full = {}
if 'comuna_norm' in df.columns:
    comunas_train_full = df.loc[idx_train, 'comuna_norm']
    for comuna in comunas_train_full.unique():
        mask_tr = (df.loc[idx_train, 'comuna_norm'] == comuna)
        X_tr_c = X_train[mask_tr.values]
        y_tr_c = y_train[mask_tr.values]
        if len(X_tr_c) >= 10:
            m = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
            m.fit(X_tr_c, y_tr_c)
            modelos_zona_full[comuna] = m

# cluster full
modelos_cluster_full = {}
if len(features_espaciales) > 0:
    feat_cluster = features_espaciales[:10]
    X_train_sp_full = df.loc[idx_train, feat_cluster].fillna(0)
    # Usar mismo criterio mejorado: mínimo 150 muestras por cluster
    max_clusters_full = max(2, min(3, len(idx_train) // 150))
    kmeans_full = KMeans(n_clusters=max_clusters_full, random_state=42, n_init=10)
    kmeans_full.fit(X_train_sp_full)
    clusters_train_full = kmeans_full.predict(X_train_sp_full)
    for cl in np.unique(clusters_train_full):
        mask_tr = clusters_train_full == cl
        X_tr_c = X_train.iloc[np.where(mask_tr)[0]]
        y_tr_c = y_train.iloc[np.where(mask_tr)[0]]
        # Umbral aumentado: mínimo 80 muestras
        if len(X_tr_c) >= 80:
            m = RandomForestRegressor(
                n_estimators=100, 
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=8,
                random_state=42, 
                n_jobs=-1
            )
            m.fit(X_tr_c, y_tr_c)
            modelos_cluster_full[cl] = m

# densidad full
modelos_dens_full = {}
if 'dens_total_600m_km2' in df.columns:
    terciles_full = df.loc[idx_train, 'dens_total_600m_km2'].quantile([0.33, 0.67])
    def asignar_zona_full(val):
        if pd.isna(val):
            return 'media'
        if val < terciles_full.iloc[0]:
            return 'baja'
        elif val < terciles_full.iloc[1]:
            return 'media'
        else:
            return 'alta'

    zonas_train_full = df.loc[idx_train, 'dens_total_600m_km2'].apply(asignar_zona_full)
    for zona in zonas_train_full.unique():
        mask_tr = zonas_train_full == zona
        X_tr_z = X_train[mask_tr.values]
        y_tr_z = y_train[mask_tr.values]
        if len(X_tr_z) >= 10:
            m = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
            m.fit(X_tr_z, y_tr_z)
            modelos_dens_full[zona] = m

# Generar predicciones base sobre X_test (alineadas con X_test.index)
pred_global_test = rf_global_full.predict(X_test)

pred_zone_test = np.zeros(len(X_test))
pred_cluster_test = np.zeros(len(X_test))
pred_dens_test = np.zeros(len(X_test))

for i, idx in enumerate(X_test.index):
    # zona
    if 'comuna_norm' in df.columns:
        comuna = df.loc[idx, 'comuna_norm']
        if comuna in modelos_zona_full:
            pred_zone_test[i] = modelos_zona_full[comuna].predict(X_test.loc[[idx]])[0]
        else:
            pred_zone_test[i] = pred_global_test[i]
    else:
        pred_zone_test[i] = pred_global_test[i]

    # cluster
    if len(modelos_cluster_full) > 0:
        x_sp = df.loc[idx, feat_cluster].fillna(0).values.reshape(1, -1)
        cl = kmeans_full.predict(x_sp)[0]
        if cl in modelos_cluster_full:
            pred_cluster_test[i] = modelos_cluster_full[cl].predict(X_test.loc[[idx]])[0]
        else:
            pred_cluster_test[i] = pred_global_test[i]
    else:
        pred_cluster_test[i] = pred_global_test[i]

    # densidad
    if modelos_dens_full:
        zona = asignar_zona_full(df.loc[idx, 'dens_total_600m_km2'])
        if zona in modelos_dens_full:
            pred_dens_test[i] = modelos_dens_full[zona].predict(X_test.loc[[idx]])[0]
        else:
            pred_dens_test[i] = pred_global_test[i]
    else:
        pred_dens_test[i] = pred_global_test[i]

# Montar DataFrame meta para test
meta_test = pd.DataFrame({
    'pred_global': pred_global_test,
    'pred_zone': pred_zone_test,
    'pred_cluster': pred_cluster_test,
    'pred_dens': pred_dens_test
}, index=X_test.index)

# Predecir con meta-modelo
meta_pred_test = meta_model.predict(meta_test)

# Métricas stacking
r2_stacking = r2_score(y_test, meta_pred_test)
rmse_stacking = np.sqrt(mean_squared_error(y_test, meta_pred_test))
mae_stacking = mean_absolute_error(y_test, meta_pred_test)

print(f"\n✅ RESULTADOS STACKING (Meta-modelo): R²={r2_stacking:.4f}, RMSE=${rmse_stacking:,.0f}, MAE=${mae_stacking:,.0f}")

# Guardar meta-modelo y predicciones
with open(OUTPUT_DIR / 'meta_model_stack.pkl', 'wb') as f:
    pickle.dump({'meta_model': meta_model, 'kmeans': kmeans_full if 'kmeans_full' in locals() else None}, f)
meta_test.to_csv(OUTPUT_DIR / 'meta_features_test.csv', index=True)

print("Meta-modelo y predicciones base guardadas en resultados/gwrf/")

print("\n" + "=" * 80)
print("📊 COMPARACIÓN FINAL DE MODELOS")
print("=" * 80)

resultados = {
    'Modelo': [],
    'R²': [],
    'RMSE': [],
    'MAE': []
}

# Baseline
resultados['Modelo'].append('RF Global (Baseline)')
resultados['R²'].append(r2_global)
resultados['RMSE'].append(rmse_global)
resultados['MAE'].append(mae_global)

# GWRF por zona
if r2_gwrf_zona is not None:
    resultados['Modelo'].append('GWRF por Comuna')
    resultados['R²'].append(r2_gwrf_zona)
    resultados['RMSE'].append(rmse_gwrf_zona)
    resultados['MAE'].append(mae_gwrf_zona)

# GWRF por cluster
if r2_gwrf_cluster is not None:
    resultados['Modelo'].append('GWRF por Cluster')
    resultados['R²'].append(r2_gwrf_cluster)
    resultados['RMSE'].append(rmse_gwrf_cluster)
    resultados['MAE'].append(mae_gwrf_cluster)

# GWRF por densidad
if r2_gwrf_densidad is not None:
    resultados['Modelo'].append('GWRF por Densidad')
    resultados['R²'].append(r2_gwrf_densidad)
    resultados['RMSE'].append(rmse_gwrf_densidad)
    resultados['MAE'].append(mae_gwrf_densidad)

df_resultados = pd.DataFrame(resultados)

print("\n" + "=" * 80)
print(f"{'Modelo':<25} | {'R²':<10} | {'RMSE':<15} | {'MAE':<15}")
print("-" * 80)
for _, row in df_resultados.iterrows():
    mejora_r2 = ((row['R²'] - r2_global) / r2_global * 100) if row['Modelo'] != 'RF Global (Baseline)' else 0
    mejora_str = f"({mejora_r2:+.1f}%)" if row['Modelo'] != 'RF Global (Baseline)' else ""
    print(f"{row['Modelo']:<25} | {row['R²']:<10.4f} | ${row['RMSE']:<14,.0f} | ${row['MAE']:<14,.0f} {mejora_str}")
print("=" * 80)

# Identificar mejor modelo
idx_mejor = df_resultados['R²'].idxmax()
mejor_modelo = df_resultados.loc[idx_mejor, 'Modelo']
mejor_r2 = df_resultados.loc[idx_mejor, 'R²']

print(f"\n🏆 MEJOR MODELO: {mejor_modelo}")
print(f"   R² = {mejor_r2:.4f}")

if mejor_r2 > r2_global:
    mejora = ((mejor_r2 - r2_global) / r2_global * 100)
    print(f"   Mejora vs Baseline: +{mejora:.2f}%")
else:
    print(f"   ⚠️  El modelo global es superior a los GWRF")

# =============================================================================
# VISUALIZACIONES
# =============================================================================
print("\n" + "=" * 80)
print("📊 GENERANDO VISUALIZACIONES")
print("=" * 80)

# 1. Comparación de R²
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# R²
axes[0].barh(df_resultados['Modelo'], df_resultados['R²'], color='steelblue')
axes[0].set_xlabel('R² Score', fontsize=12)
axes[0].set_title('Coeficiente de Determinación (R²)', fontsize=14, fontweight='bold')
axes[0].axvline(r2_global, color='red', linestyle='--', linewidth=2, label='Baseline')
axes[0].legend()
for i, v in enumerate(df_resultados['R²']):
    axes[0].text(v + 0.01, i, f'{v:.4f}', va='center', fontweight='bold')

# RMSE
axes[1].barh(df_resultados['Modelo'], df_resultados['RMSE'], color='coral')
axes[1].set_xlabel('RMSE (UF/m²)', fontsize=12)
axes[1].set_title('Error Cuadrático Medio (RMSE)', fontsize=14, fontweight='bold')
axes[1].axvline(rmse_global, color='red', linestyle='--', linewidth=2, label='Baseline')
axes[1].legend()
for i, v in enumerate(df_resultados['RMSE']):
    axes[1].text(v + 20, i, f'${v:,.0f}', va='center', fontweight='bold')

# MAE
axes[2].barh(df_resultados['Modelo'], df_resultados['MAE'], color='lightgreen')
axes[2].set_xlabel('MAE (UF/m²)', fontsize=12)
axes[2].set_title('Error Absoluto Medio (MAE)', fontsize=14, fontweight='bold')
axes[2].axvline(mae_global, color='red', linestyle='--', linewidth=2, label='Baseline')
axes[2].legend()
for i, v in enumerate(df_resultados['MAE']):
    axes[2].text(v + 10, i, f'${v:,.0f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'comparacion_gwrf_vs_baseline.png', dpi=300, bbox_inches='tight')
print(f"✓ Gráfico guardado: comparacion_gwrf_vs_baseline.png")
plt.close()

# 2. Mejoras porcentuales
if len(df_resultados) > 1:
    mejoras_r2 = []
    mejoras_rmse = []
    mejoras_mae = []
    modelos_nombres = []
    
    for _, row in df_resultados.iterrows():
        if row['Modelo'] != 'RF Global (Baseline)':
            mejoras_r2.append(((row['R²'] - r2_global) / r2_global * 100))
            mejoras_rmse.append(((rmse_global - row['RMSE']) / rmse_global * 100))
            mejoras_mae.append(((mae_global - row['MAE']) / mae_global * 100))
            modelos_nombres.append(row['Modelo'])
    
    if len(modelos_nombres) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(modelos_nombres))
        width = 0.25
        
        ax.bar(x - width, mejoras_r2, width, label='R²', color='steelblue')
        ax.bar(x, mejoras_rmse, width, label='RMSE', color='coral')
        ax.bar(x + width, mejoras_mae, width, label='MAE', color='lightgreen')
        
        ax.set_ylabel('Mejora (%)', fontsize=12)
        ax.set_title('Mejoras Porcentuales vs Baseline', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(modelos_nombres, rotation=15, ha='right')
        ax.legend()
        ax.axhline(0, color='black', linewidth=0.8)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / 'mejoras_porcentuales_gwrf.png', dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico guardado: mejoras_porcentuales_gwrf.png")
        plt.close()

# =============================================================================
# GUARDAR RESULTADOS
# =============================================================================
print("\n" + "=" * 80)
print("💾 GUARDANDO RESULTADOS")
print("=" * 80)

# Guardar tabla de comparación
df_resultados.to_csv(OUTPUT_DIR / 'comparacion_modelos_gwrf.csv', index=False)
print(f"✓ Tabla guardada: comparacion_modelos_gwrf.csv")

# Guardar mejor modelo
if mejor_modelo != 'RF Global (Baseline)':
    if mejor_modelo == 'GWRF por Comuna' and 'comuna_norm' in df.columns:
        with open(OUTPUT_DIR / 'gwrf_por_comuna.pkl', 'wb') as f:
            pickle.dump(modelos_por_zona, f)
        print(f"✓ Modelos GWRF por comuna guardados: gwrf_por_comuna.pkl")
    
    elif mejor_modelo == 'GWRF por Cluster':
        with open(OUTPUT_DIR / 'gwrf_por_cluster.pkl', 'wb') as f:
            pickle.dump({'modelos': modelos_por_cluster, 'kmeans': kmeans}, f)
        print(f"✓ Modelos GWRF por cluster guardados: gwrf_por_cluster.pkl")
    
    elif mejor_modelo == 'GWRF por Densidad':
        with open(OUTPUT_DIR / 'gwrf_por_densidad.pkl', 'wb') as f:
            pickle.dump(modelos_por_densidad, f)
        print(f"✓ Modelos GWRF por densidad guardados: gwrf_por_densidad.pkl")

# Guardar reporte detallado
reporte_path = OUTPUT_DIR / 'reporte_gwrf.txt'
with open(reporte_path, 'w') as f:
    f.write("=" * 80 + "\n")
    f.write("REPORTE: GEOGRAPHICALLY WEIGHTED RANDOM FOREST (GWRF)\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("1. MODELO BASELINE (Random Forest Global)\n")
    f.write("-" * 80 + "\n")
    f.write(f"R² Test:  {r2_global:.4f}\n")
    f.write(f"RMSE:     ${rmse_global:,.0f} UF/m²\n")
    f.write(f"MAE:      ${mae_global:,.0f} UF/m²\n\n")
    
    if r2_gwrf_zona is not None:
        f.write("2. GWRF POR ZONA (un modelo por comuna)\n")
        f.write("-" * 80 + "\n")
        f.write(f"R² Test:  {r2_gwrf_zona:.4f}\n")
        f.write(f"RMSE:     ${rmse_gwrf_zona:,.0f} UF/m²\n")
        f.write(f"MAE:      ${mae_gwrf_zona:,.0f} UF/m²\n")
        mejora = ((r2_gwrf_zona - r2_global) / r2_global * 100)
        f.write(f"Mejora:   {mejora:+.2f}%\n\n")
        
        f.write("Métricas por comuna:\n")
        for comuna, metricas in metricas_por_zona.items():
            f.write(f"  • {comuna}: R²={metricas['r2']:.4f}, ")
            f.write(f"RMSE=${metricas['rmse']:,.0f}, ")
            f.write(f"n_test={metricas['n_test']}\n")
        f.write("\n")
    
    if r2_gwrf_cluster is not None:
        f.write("3. GWRF POR CLUSTER (clustering espacial)\n")
        f.write("-" * 80 + "\n")
        f.write(f"R² Test:  {r2_gwrf_cluster:.4f}\n")
        f.write(f"RMSE:     ${rmse_gwrf_cluster:,.0f} UF/m²\n")
        f.write(f"MAE:      ${mae_gwrf_cluster:,.0f} UF/m²\n")
        mejora = ((r2_gwrf_cluster - r2_global) / r2_global * 100)
        f.write(f"Mejora:   {mejora:+.2f}%\n")
        f.write(f"N° clusters: {n_clusters}\n\n")
        
        f.write("Métricas por cluster:\n")
        for cluster_id, metricas in metricas_por_cluster.items():
            f.write(f"  • Cluster {cluster_id}: R²={metricas['r2']:.4f}, ")
            f.write(f"RMSE=${metricas['rmse']:,.0f}, ")
            f.write(f"n_test={metricas['n_test']}\n")
        f.write("\n")
    
    if r2_gwrf_densidad is not None:
        f.write("4. GWRF POR DENSIDAD (zonas de densidad)\n")
        f.write("-" * 80 + "\n")
        f.write(f"R² Test:  {r2_gwrf_densidad:.4f}\n")
        f.write(f"RMSE:     ${rmse_gwrf_densidad:,.0f} UF/m²\n")
        f.write(f"MAE:      ${mae_gwrf_densidad:,.0f} UF/m²\n")
        mejora = ((r2_gwrf_densidad - r2_global) / r2_global * 100)
        f.write(f"Mejora:   {mejora:+.2f}%\n\n")
        
        f.write("Métricas por zona de densidad:\n")
        for zona, metricas in metricas_por_densidad.items():
            f.write(f"  • Zona {zona}: R²={metricas['r2']:.4f}, ")
            f.write(f"RMSE=${metricas['rmse']:,.0f}, ")
            f.write(f"n_test={metricas['n_test']}\n")
        f.write("\n")
    
    f.write("=" * 80 + "\n")
    f.write(f"CONCLUSIÓN:\n")
    f.write(f"Mejor modelo: {mejor_modelo}\n")
    f.write(f"R² = {mejor_r2:.4f}\n")
    
    if mejor_r2 > r2_global:
        mejora_final = ((mejor_r2 - r2_global) / r2_global * 100)
        f.write(f"Mejora vs Baseline: +{mejora_final:.2f}%\n")
        f.write(f"\n✅ Los enfoques GWRF mejoran el modelo global!\n")
    else:
        f.write(f"\n⚠️  El modelo global RF sigue siendo superior.\n")
        f.write(f"   Razón: Dataset puede ser muy heterogéneo o modelos locales\n")
        f.write(f"          tienen insuficientes datos para entrenar bien.\n")
    
    f.write("=" * 80 + "\n")

print(f"✓ Reporte guardado: reporte_gwrf.txt")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("✅ PROCESO COMPLETADO")
print("=" * 80)

print(f"""
📊 RESUMEN DE RESULTADOS:

📁 Archivos generados en: {OUTPUT_DIR}
   • comparacion_modelos_gwrf.csv
   • comparacion_gwrf_vs_baseline.png
   • mejoras_porcentuales_gwrf.png
   • reporte_gwrf.txt
   • Modelos entrenados (*.pkl)

🏆 MEJOR MODELO: {mejor_modelo}
   • R² = {mejor_r2:.4f}
   
{'✅ GWRF mejora el modelo baseline!' if mejor_r2 > r2_global else '⚠️  El modelo global RF sigue siendo superior'}

💡 INTERPRETACIÓN:
""")

if mejor_r2 > r2_global:
    mejora_final = ((mejor_r2 - r2_global) / r2_global * 100)
    print(f"""   Los enfoques GWRF lograron una mejora de {mejora_final:.2f}% sobre el modelo
   global, indicando que hay HETEROGENEIDAD ESPACIAL en los datos que 
   los modelos locales capturan mejor.
""")
else:
    print("""   El modelo global RF es más robusto. Posibles razones:
   • Los datos ya son relativamente homogéneos espacialmente
   • Los modelos locales tienen pocos datos para entrenar bien
   • Las features espaciales ya capturan la variabilidad geográfica
   • El dataset es pequeño para dividir en zonas
""")

print("=" * 80)
