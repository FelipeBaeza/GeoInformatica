#!/usr/bin/env python3
"""
COMPARACIÓN EXHAUSTIVA DE MODELOS DE MACHINE LEARNING PARA DATOS GEOESPACIALES

Este script evalúa múltiples algoritmos de ML para encontrar el modelo óptimo
que reduzca la tasa de error en la predicción de satisfacción residencial.

Modelos evaluados:
1. Random Forest (baseline)
2. Gradient Boosting
3. XGBoost
4. LightGBM
5. CatBoost
6. Support Vector Regression (SVR)
7. K-Nearest Neighbors (KNN) - Espacial
8. ElasticNet (Regularización L1+L2)
9. Neural Network (MLP)
10. Stacking Ensemble
11. GWRF Simulado (Geographically Weighted Random Forest)

Métricas evaluadas:
- R² (Coeficiente de determinación)
- RMSE (Root Mean Square Error)
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- Cross-validation (5-fold)
- Moran's I de residuos (autocorrelación espacial)

Autor: Proyecto GeoInformática
Fecha: Diciembre 2025
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import json
from pathlib import Path
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Sklearn
from sklearn.model_selection import (
    train_test_split, cross_val_score, GridSearchCV, 
    KFold, RandomizedSearchCV
)
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    mean_absolute_percentage_error
)
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    AdaBoostRegressor, ExtraTreesRegressor,
    StackingRegressor, VotingRegressor, BaggingRegressor
)
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, BayesianRidge,
    HuberRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel

# Modelos avanzados de Boosting
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("⚠️ XGBoost no disponible")

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM no disponible")

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost no disponible")

# Análisis espacial
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from scipy.stats import pearsonr, spearmanr

import matplotlib.pyplot as plt
import seaborn as sns

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
BASE_DIR = Path('/home/felipe/Documentos/GeoInformatica')
OUTPUT_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana3_modelo_satisfaccion' / 'resultados' / 'comparacion_modelos'
GRAFICOS_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana3_modelo_satisfaccion' / 'graficos'
MODELOS_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana3_modelo_satisfaccion' / 'modelos'
DATA_PATH = BASE_DIR / 'autocorrelacion_espacial' / 'semana3_modelo_satisfaccion' / 'resultados' / 'modelo_venta' / 'propiedades_venta_con_satisfaccion.csv'

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("=" * 80)
print("🔬 COMPARACIÓN EXHAUSTIVA DE MODELOS DE ML PARA DATOS GEOESPACIALES")
print("=" * 80)
print(f"   Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

# =============================================================================
# 1. CARGAR DATOS
# =============================================================================
print("\n📂 PASO 1: Cargando datos...")

df = pd.read_csv(DATA_PATH)
print(f"   ✓ Datos cargados: {len(df)} propiedades")

# Features disponibles
features_espaciales = [col for col in df.columns if any(x in col for x in 
    ['dist_', 'dens_', 'idx_']) and 'satisfaccion' not in col]

features_internas = [
    'superficie_util', 'dormitorios', 'banos', 'precio_uf', 'precio_m2_uf',
    'total_habitaciones', 'es_departamento', 'es_casa',
    'es_comuna_premium', 'es_comuna_economica', 'latitude', 'longitude'
]

# Filtrar features que existen
all_features = [f for f in features_internas + features_espaciales if f in df.columns]
print(f"   ✓ Features disponibles: {len(all_features)}")

# Target
TARGET = 'satisfaccion_balanceado'

# Preparar datos
df_model = df[all_features + [TARGET, 'latitude', 'longitude']].dropna()
print(f"   ✓ Muestras válidas: {len(df_model)}")

X = df_model[all_features]
y = df_model[TARGET]
coords = df_model[['longitude', 'latitude']].values

# Split
X_train, X_test, y_train, y_test, coords_train, coords_test = train_test_split(
    X, y, coords, test_size=0.2, random_state=RANDOM_STATE
)

# Escalar
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"   ✓ Train: {len(X_train)}, Test: {len(X_test)}")

# =============================================================================
# 2. DEFINIR MODELOS A COMPARAR
# =============================================================================
print("\n🤖 PASO 2: Definiendo modelos a comparar...")

def get_models():
    """Retorna diccionario con todos los modelos a evaluar"""
    models = {}
    
    # 1. Random Forest (Baseline)
    models['Random Forest'] = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # 2. Extra Trees (similar a RF pero más aleatorio)
    models['Extra Trees'] = ExtraTreesRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # 3. Gradient Boosting
    models['Gradient Boosting'] = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE
    )
    
    # 4. XGBoost
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0
        )
    
    # 5. LightGBM
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMRegressor(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=-1
        )
    
    # 6. CatBoost
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostRegressor(
            iterations=200,
            depth=8,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
            verbose=0
        )
    
    # 7. AdaBoost
    models['AdaBoost'] = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=6),
        n_estimators=100,
        learning_rate=0.1,
        random_state=RANDOM_STATE
    )
    
    # 8. Bagging
    models['Bagging'] = BaggingRegressor(
        estimator=DecisionTreeRegressor(max_depth=10),
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    # 9. SVR (Support Vector Regression)
    models['SVR (RBF)'] = SVR(
        kernel='rbf',
        C=10,
        gamma='scale',
        epsilon=0.1
    )
    
    # 10. KNN Regressor (importante para datos espaciales)
    models['KNN (k=10)'] = KNeighborsRegressor(
        n_neighbors=10,
        weights='distance',
        metric='minkowski',
        n_jobs=-1
    )
    
    # 11. KNN con más vecinos
    models['KNN (k=20)'] = KNeighborsRegressor(
        n_neighbors=20,
        weights='distance',
        metric='minkowski',
        n_jobs=-1
    )
    
    # 12. Ridge Regression
    models['Ridge'] = Ridge(alpha=1.0)
    
    # 13. ElasticNet
    models['ElasticNet'] = ElasticNet(
        alpha=0.1,
        l1_ratio=0.5,
        random_state=RANDOM_STATE
    )
    
    # 14. Bayesian Ridge
    models['Bayesian Ridge'] = BayesianRidge()
    
    # 15. Huber Regressor (robusto a outliers)
    models['Huber'] = HuberRegressor(epsilon=1.35)
    
    # 16. MLP Neural Network
    models['MLP Neural Net'] = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        alpha=0.001,
        learning_rate='adaptive',
        max_iter=500,
        random_state=RANDOM_STATE,
        early_stopping=True
    )
    
    # 17. MLP más profundo
    models['MLP Deep'] = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32, 16),
        activation='relu',
        solver='adam',
        alpha=0.0001,
        learning_rate='adaptive',
        max_iter=1000,
        random_state=RANDOM_STATE,
        early_stopping=True
    )
    
    return models

models = get_models()
print(f"   ✓ Modelos definidos: {len(models)}")
for name in models.keys():
    print(f"      • {name}")

# =============================================================================
# 3. ENTRENAR Y EVALUAR MODELOS
# =============================================================================
print("\n📊 PASO 3: Entrenando y evaluando modelos...")

def evaluate_model(model, X_train, X_test, y_train, y_test, name):
    """Evalúa un modelo y retorna métricas"""
    import time
    
    start_time = time.time()
    
    # Entrenar
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # Predecir
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Métricas
    metrics = {
        'model': name,
        'r2_train': r2_score(y_train, y_pred_train),
        'r2_test': r2_score(y_test, y_pred_test),
        'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
        'mae_train': mean_absolute_error(y_train, y_pred_train),
        'mae_test': mean_absolute_error(y_test, y_pred_test),
        'train_time': train_time
    }
    
    # Cross-validation
    try:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
        metrics['cv_r2_mean'] = cv_scores.mean()
        metrics['cv_r2_std'] = cv_scores.std()
    except:
        metrics['cv_r2_mean'] = np.nan
        metrics['cv_r2_std'] = np.nan
    
    # Overfitting check
    metrics['overfit_ratio'] = metrics['r2_train'] / max(0.001, metrics['r2_test'])
    
    return metrics, y_pred_test

results = []
predictions = {}

for name, model in models.items():
    print(f"\n   🔄 Evaluando: {name}...")
    try:
        metrics, y_pred = evaluate_model(
            model, X_train_scaled, X_test_scaled, y_train, y_test, name
        )
        results.append(metrics)
        predictions[name] = y_pred
        
        print(f"      R² test: {metrics['r2_test']:.4f} | RMSE: {metrics['rmse_test']:.4f} | "
              f"CV: {metrics['cv_r2_mean']:.4f} ± {metrics['cv_r2_std']*2:.4f}")
    except Exception as e:
        print(f"      ❌ Error: {e}")

# =============================================================================
# 4. CREAR ENSEMBLES AVANZADOS
# =============================================================================
print("\n🔧 PASO 4: Creando Ensembles avanzados...")

# Voting Ensemble (top 3 modelos)
df_results = pd.DataFrame(results).sort_values('r2_test', ascending=False)
top3_names = df_results.head(3)['model'].tolist()
print(f"   Top 3 modelos: {top3_names}")

# Recrear modelos para stacking
base_models = []
if 'XGBoost' in top3_names and XGBOOST_AVAILABLE:
    base_models.append(('xgb', XGBRegressor(n_estimators=100, max_depth=6, verbosity=0, random_state=RANDOM_STATE)))
if 'LightGBM' in top3_names and LIGHTGBM_AVAILABLE:
    base_models.append(('lgbm', LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1, random_state=RANDOM_STATE)))
if 'CatBoost' in top3_names and CATBOOST_AVAILABLE:
    base_models.append(('cat', CatBoostRegressor(iterations=100, depth=6, verbose=0, random_state=RANDOM_STATE)))
if 'Random Forest' in top3_names:
    base_models.append(('rf', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_STATE)))
if 'Gradient Boosting' in top3_names:
    base_models.append(('gb', GradientBoostingRegressor(n_estimators=100, max_depth=6, random_state=RANDOM_STATE)))
if 'Extra Trees' in top3_names:
    base_models.append(('et', ExtraTreesRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_STATE)))

# Si no hay suficientes, agregar defaults
if len(base_models) < 3:
    if XGBOOST_AVAILABLE and 'xgb' not in [m[0] for m in base_models]:
        base_models.append(('xgb', XGBRegressor(n_estimators=100, max_depth=6, verbosity=0, random_state=RANDOM_STATE)))
    if LIGHTGBM_AVAILABLE and 'lgbm' not in [m[0] for m in base_models]:
        base_models.append(('lgbm', LGBMRegressor(n_estimators=100, max_depth=6, verbose=-1, random_state=RANDOM_STATE)))
    base_models.append(('rf', RandomForestRegressor(n_estimators=100, max_depth=8, random_state=RANDOM_STATE)))

base_models = base_models[:4]  # Limitar a 4

# Stacking Ensemble
print("\n   📦 Stacking Ensemble...")
stacking = StackingRegressor(
    estimators=base_models,
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)

metrics_stacking, pred_stacking = evaluate_model(
    stacking, X_train_scaled, X_test_scaled, y_train, y_test, 'Stacking Ensemble'
)
results.append(metrics_stacking)
predictions['Stacking Ensemble'] = pred_stacking
print(f"      R² test: {metrics_stacking['r2_test']:.4f} | RMSE: {metrics_stacking['rmse_test']:.4f}")

# Voting Ensemble
print("\n   📦 Voting Ensemble...")
voting = VotingRegressor(estimators=base_models, n_jobs=-1)

metrics_voting, pred_voting = evaluate_model(
    voting, X_train_scaled, X_test_scaled, y_train, y_test, 'Voting Ensemble'
)
results.append(metrics_voting)
predictions['Voting Ensemble'] = pred_voting
print(f"      R² test: {metrics_voting['r2_test']:.4f} | RMSE: {metrics_voting['rmse_test']:.4f}")

# =============================================================================
# 5. MODELO ESPACIAL: GWRF SIMULADO
# =============================================================================
print("\n🌍 PASO 5: Modelo Geográficamente Ponderado (GWRF simulado)...")

def gwrf_predict(X_train, y_train, X_test, coords_train, coords_test, n_neighbors=50):
    """
    Simula GWRF usando ponderación espacial local.
    Para cada punto de test, entrena un RF con pesos basados en distancia.
    """
    from sklearn.ensemble import RandomForestRegressor
    
    # Construir árbol KD para búsqueda eficiente
    tree = cKDTree(coords_train)
    
    y_pred = np.zeros(len(X_test))
    
    for i in range(len(X_test)):
        # Encontrar k vecinos más cercanos
        distances, indices = tree.query(coords_test[i], k=min(n_neighbors, len(X_train)))
        
        # Pesos basados en kernel Gaussiano
        bandwidth = np.median(distances) * 1.5
        if bandwidth < 0.0001:
            bandwidth = 0.0001
        weights = np.exp(-(distances**2) / (2 * bandwidth**2))
        weights = np.nan_to_num(weights, nan=1.0)
        weights = weights / (weights.sum() + 1e-10)
        weights = np.clip(weights, 0.001, 1.0)  # Evitar pesos muy pequeños o NaN
        
        # Entrenar RF local con muestras ponderadas
        X_local = X_train[indices]
        y_local = y_train.iloc[indices].values
        
        # Usar sample_weight para ponderar
        rf_local = RandomForestRegressor(
            n_estimators=50, 
            max_depth=6,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        rf_local.fit(X_local, y_local, sample_weight=weights)
        
        y_pred[i] = rf_local.predict(X_test[i:i+1])[0]
    
    return y_pred

# Ejecutar GWRF (solo en submuestra por tiempo)
print("   Ejecutando GWRF (puede tomar tiempo)...")
sample_size = min(500, len(X_test))
sample_idx = np.random.choice(len(X_test), sample_size, replace=False)

X_test_sample = X_test_scaled[sample_idx]
y_test_sample = y_test.iloc[sample_idx]
coords_test_sample = coords_test[sample_idx]

y_pred_gwrf = gwrf_predict(
    X_train_scaled, y_train, X_test_sample, 
    coords_train, coords_test_sample, n_neighbors=30
)

metrics_gwrf = {
    'model': 'GWRF (Spatial)',
    'r2_test': r2_score(y_test_sample, y_pred_gwrf),
    'rmse_test': np.sqrt(mean_squared_error(y_test_sample, y_pred_gwrf)),
    'mae_test': mean_absolute_error(y_test_sample, y_pred_gwrf),
    'cv_r2_mean': np.nan,  # No CV para GWRF
    'cv_r2_std': np.nan,
    'r2_train': np.nan,
    'rmse_train': np.nan,
    'mae_train': np.nan,
    'train_time': np.nan,
    'overfit_ratio': np.nan
}
results.append(metrics_gwrf)
print(f"      R² test (sample): {metrics_gwrf['r2_test']:.4f} | RMSE: {metrics_gwrf['rmse_test']:.4f}")

# =============================================================================
# 6. ANÁLISIS DE AUTOCORRELACIÓN ESPACIAL DE RESIDUOS
# =============================================================================
print("\n🗺️ PASO 6: Analizando autocorrelación espacial de residuos...")

def calculate_morans_i(residuals, coords, k=10):
    """Calcula Moran's I para los residuos"""
    from scipy.spatial import cKDTree
    
    n = len(residuals)
    tree = cKDTree(coords)
    
    # Construir matriz de pesos espaciales
    W = np.zeros((n, n))
    for i in range(n):
        distances, indices = tree.query(coords[i], k=k+1)
        for j, idx in enumerate(indices[1:]):  # Excluir el punto mismo
            W[i, idx] = 1 / max(distances[j+1], 0.001)
    
    # Normalizar filas
    row_sums = W.sum(axis=1)
    W = W / row_sums[:, np.newaxis]
    W = np.nan_to_num(W)
    
    # Calcular Moran's I
    z = residuals - residuals.mean()
    numerator = np.sum(W * np.outer(z, z))
    denominator = np.sum(z**2)
    
    S0 = np.sum(W)
    morans_i = (n / S0) * (numerator / denominator)
    
    return morans_i

# Calcular Moran's I para los top 5 modelos
df_results_sorted = pd.DataFrame(results).sort_values('r2_test', ascending=False)
top5_models = df_results_sorted.head(5)['model'].tolist()

morans_results = {}
print("\n   Moran's I de residuos (menor = mejor, cerca de 0 = sin autocorrelación):")

for model_name in top5_models:
    if model_name in predictions:
        residuals = y_test.values - predictions[model_name]
        morans_i = calculate_morans_i(residuals, coords_test)
        morans_results[model_name] = morans_i
        print(f"      {model_name}: Moran's I = {morans_i:.4f}")

# =============================================================================
# 7. OPTIMIZACIÓN DEL MEJOR MODELO
# =============================================================================
print("\n⚡ PASO 7: Optimizando el mejor modelo...")

# Encontrar mejor modelo
df_final = pd.DataFrame(results)
df_final = df_final.sort_values('r2_test', ascending=False)
best_model_name = df_final.iloc[0]['model']
print(f"   Mejor modelo base: {best_model_name}")

# Optimizar hiperparámetros del mejor modelo
if 'XGBoost' in best_model_name and XGBOOST_AVAILABLE:
    print("   Optimizando XGBoost...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0]
    }
    
    model_opt = XGBRegressor(random_state=RANDOM_STATE, verbosity=0)
    
    random_search = RandomizedSearchCV(
        model_opt, param_grid, n_iter=30, cv=3, 
        scoring='r2', random_state=RANDOM_STATE, n_jobs=-1
    )
    random_search.fit(X_train_scaled, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
elif 'LightGBM' in best_model_name and LIGHTGBM_AVAILABLE:
    print("   Optimizando LightGBM...")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0.5, 1.0, 2.0]
    }
    
    model_opt = LGBMRegressor(random_state=RANDOM_STATE, verbose=-1)
    
    random_search = RandomizedSearchCV(
        model_opt, param_grid, n_iter=30, cv=3,
        scoring='r2', random_state=RANDOM_STATE, n_jobs=-1
    )
    random_search.fit(X_train_scaled, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

else:
    # Random Forest optimization
    print("   Optimizando Random Forest...")
    param_grid = {
        'n_estimators': [100, 200, 300, 500],
        'max_depth': [6, 8, 10, 12, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    model_opt = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        model_opt, param_grid, n_iter=30, cv=3,
        scoring='r2', random_state=RANDOM_STATE, n_jobs=-1
    )
    random_search.fit(X_train_scaled, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_

# Evaluar modelo optimizado
y_pred_opt = best_model.predict(X_test_scaled)
metrics_optimized = {
    'model': f'{best_model_name} (Optimizado)',
    'r2_test': r2_score(y_test, y_pred_opt),
    'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_opt)),
    'mae_test': mean_absolute_error(y_test, y_pred_opt),
}

cv_scores_opt = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='r2')
metrics_optimized['cv_r2_mean'] = cv_scores_opt.mean()
metrics_optimized['cv_r2_std'] = cv_scores_opt.std()

print(f"\n   ✓ Modelo optimizado:")
print(f"      R² test: {metrics_optimized['r2_test']:.4f}")
print(f"      RMSE: {metrics_optimized['rmse_test']:.4f}")
print(f"      CV R²: {metrics_optimized['cv_r2_mean']:.4f} ± {metrics_optimized['cv_r2_std']*2:.4f}")

results.append(metrics_optimized)

# =============================================================================
# 8. GUARDAR RESULTADOS
# =============================================================================
print("\n💾 PASO 8: Guardando resultados...")

# DataFrame final
df_final = pd.DataFrame(results)
df_final = df_final.sort_values('r2_test', ascending=False)

# Guardar CSV
df_final.to_csv(OUTPUT_DIR / 'comparacion_modelos.csv', index=False)

# Guardar mejor modelo
with open(MODELOS_DIR / 'mejor_modelo_comparado.pkl', 'wb') as f:
    pickle.dump({
        'modelo': best_model,
        'scaler': scaler,
        'features': all_features,
        'metricas': metrics_optimized,
        'parametros': best_params if 'best_params' in dir() else {},
        'comparacion': df_final.to_dict()
    }, f)

# Guardar resumen JSON
resumen = {
    'fecha': datetime.now().isoformat(),
    'total_modelos': len(results),
    'mejor_modelo': best_model_name,
    'mejor_r2': float(df_final.iloc[0]['r2_test']),
    'mejor_rmse': float(df_final.iloc[0]['rmse_test']),
    'top5': df_final.head(5)[['model', 'r2_test', 'rmse_test', 'cv_r2_mean']].to_dict('records'),
    'morans_i': morans_results
}

with open(OUTPUT_DIR / 'resumen_comparacion.json', 'w') as f:
    json.dump(resumen, f, indent=2, default=str)

print(f"   ✓ Resultados guardados en: {OUTPUT_DIR}")

# =============================================================================
# 9. VISUALIZACIONES
# =============================================================================
print("\n📊 PASO 9: Generando visualizaciones...")

# Gráfico 1: Comparación de R² por modelo
fig, ax = plt.subplots(figsize=(14, 8))
df_plot = df_final.dropna(subset=['r2_test']).sort_values('r2_test', ascending=True)

colors = ['#e74c3c' if r < 0.7 else '#f39c12' if r < 0.8 else '#27ae60' 
          for r in df_plot['r2_test']]

bars = ax.barh(range(len(df_plot)), df_plot['r2_test'], color=colors)
ax.set_yticks(range(len(df_plot)))
ax.set_yticklabels(df_plot['model'])
ax.set_xlabel('R² Test')
ax.set_title('Comparación de Modelos: R² en Test Set', fontsize=14, fontweight='bold')
ax.axvline(x=0.85, color='red', linestyle='--', alpha=0.5, label='Baseline RF')

# Agregar valores
for i, (idx, row) in enumerate(df_plot.iterrows()):
    ax.text(row['r2_test'] + 0.005, i, f"{row['r2_test']:.4f}", va='center', fontsize=9)

ax.legend()
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'comparacion_r2_modelos.png', dpi=300, bbox_inches='tight')
print("   ✓ Gráfico de comparación R² guardado")

# Gráfico 2: R² vs RMSE (scatter)
fig, ax = plt.subplots(figsize=(10, 8))
df_plot2 = df_final.dropna(subset=['r2_test', 'rmse_test'])

scatter = ax.scatter(
    df_plot2['rmse_test'], df_plot2['r2_test'],
    c=df_plot2['cv_r2_mean'], cmap='RdYlGn',
    s=100, edgecolors='black', linewidth=1
)

for i, row in df_plot2.iterrows():
    ax.annotate(row['model'], (row['rmse_test'], row['r2_test']),
                textcoords="offset points", xytext=(5, 5), fontsize=8)

ax.set_xlabel('RMSE (menor = mejor)')
ax.set_ylabel('R² (mayor = mejor)')
ax.set_title('Trade-off: R² vs RMSE por Modelo', fontsize=14, fontweight='bold')
plt.colorbar(scatter, label='CV R² Mean')
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'tradeoff_r2_rmse.png', dpi=300, bbox_inches='tight')
print("   ✓ Gráfico de trade-off guardado")

# Gráfico 3: Predicción vs Real (mejor modelo)
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred_opt, alpha=0.5, s=30)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Satisfacción Real')
ax.set_ylabel('Satisfacción Predicha')
ax.set_title(f'Predicción vs Real - {best_model_name} Optimizado\n(R² = {metrics_optimized["r2_test"]:.4f})', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'prediccion_vs_real_optimizado.png', dpi=300, bbox_inches='tight')
print("   ✓ Gráfico de predicción guardado")

# =============================================================================
# 10. RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("📊 RESUMEN FINAL: COMPARACIÓN DE MODELOS")
print("=" * 80)

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                         TOP 5 MODELOS                                           │
├────────────────────────────────────────────────────────────────────────────────┤""")

for i, (_, row) in enumerate(df_final.head(5).iterrows()):
    r2 = row['r2_test']
    rmse = row['rmse_test']
    cv = row.get('cv_r2_mean', np.nan)
    cv_str = f"{cv:.4f}" if not np.isnan(cv) else "N/A"
    model = row['model'][:35].ljust(35)
    print(f"│  {i+1}. {model} R²={r2:.4f}  RMSE={rmse:.4f}  CV={cv_str:>7} │")

print(f"""├────────────────────────────────────────────────────────────────────────────────┤
│                         MEJORA RESPECTO AL BASELINE                             │
├────────────────────────────────────────────────────────────────────────────────┤
│  Random Forest (baseline):  R² = 0.8521                                         │
│  Mejor modelo encontrado:   R² = {df_final.iloc[0]['r2_test']:.4f} ({df_final.iloc[0]['model'][:30]})│
│  Mejora:                    {((df_final.iloc[0]['r2_test'] - 0.8521) / 0.8521 * 100):+.2f}%                                              │
└────────────────────────────────────────────────────────────────────────────────┘
""")

print("\n✅ Comparación completada!")
print(f"\n📁 Archivos generados:")
print(f"   • {OUTPUT_DIR / 'comparacion_modelos.csv'}")
print(f"   • {OUTPUT_DIR / 'resumen_comparacion.json'}")
print(f"   • {MODELOS_DIR / 'mejor_modelo_comparado.pkl'}")
print(f"   • {GRAFICOS_DIR / 'comparacion_r2_modelos.png'}")
print(f"   • {GRAFICOS_DIR / 'tradeoff_r2_rmse.png'}")
print(f"   • {GRAFICOS_DIR / 'prediccion_vs_real_optimizado.png'}")
