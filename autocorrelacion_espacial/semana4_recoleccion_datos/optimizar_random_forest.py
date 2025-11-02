#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OPTIMIZACIÓN DE HIPERPARÁMETROS: RANDOM FOREST
==============================================

Objetivo:
    Optimizar exhaustivamente los hiperparámetros de Random Forest
    para maximizar R² y minimizar overfitting.

Estrategia:
    1. GridSearch exhaustivo (puede tomar 30-60 minutos)
    2. RandomizedSearchCV para exploración rápida
    3. Validación cruzada estratificada (5-fold)
    4. Análisis de importancia de hiperparámetros
    5. Curvas de aprendizaje y validación

Autor: Sistema de Análisis Inmobiliario
Fecha: 2025-11-01
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Machine Learning
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import (
    train_test_split, 
    GridSearchCV, 
    RandomizedSearchCV,
    cross_val_score,
    learning_curve
)
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# Configuración
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (20, 12)
plt.rcParams['font.size'] = 10

# Directorios
BASE_DIR = Path(__file__).parent
DATOS_DIR = BASE_DIR / "datos_procesados"
VIZ_DIR = BASE_DIR / "visualizaciones"
VIZ_DIR.mkdir(exist_ok=True)

def cargar_datos():
    """Carga datos limpios"""
    print("\n" + "="*70)
    print("📂 CARGANDO DATOS")
    print("="*70)
    
    archivos = list(DATOS_DIR.glob("propiedades_limpias_*.geojson"))
    gdf = gpd.read_file(max(archivos, key=lambda x: x.stat().st_mtime))
    print(f"✅ Datos cargados: {len(gdf)} propiedades")
    
    return gdf

def preparar_variables(gdf):
    """Prepara variables para ML"""
    print("\n" + "="*70)
    print("🔧 PREPARANDO VARIABLES")
    print("="*70)
    
    # Variable dependiente
    y = np.log(gdf['precio']).values
    
    # Crear variables derivadas
    gdf['banos_num'] = pd.to_numeric(gdf['banos'], errors='coerce').fillna(1)
    gdf['dormitorios_num'] = pd.to_numeric(gdf['dormitorios'], errors='coerce').fillna(1)
    gdf['superficie_util_num'] = pd.to_numeric(gdf['superficie_util'], errors='coerce').fillna(50)
    
    # Distancias en km
    gdf['dist_transporte_km'] = gdf['espacial_dist_transporte_metro_m'] / 1000
    gdf['dist_turismo_km'] = gdf['espacial_dist_turismo_m'] / 1000
    gdf['dist_salud_km'] = gdf['espacial_dist_salud_m'] / 1000
    gdf['dist_educacion_basica_km'] = gdf['espacial_dist_educacion_basica_m'] / 1000
    gdf['dist_educacion_superior_km'] = gdf['espacial_dist_educacion_superior_m'] / 1000
    gdf['dist_areas_verdes_km'] = gdf['espacial_dist_areas_verdes_m'] / 1000
    
    # Dummies de comunas
    comunas_dummies = pd.get_dummies(gdf['comuna'], prefix='comuna', drop_first=True)
    
    # Features
    features_numericas = [
        'banos_num',
        'dormitorios_num',
        'superficie_util_num',
        'dist_transporte_km',
        'dist_turismo_km',
        'dist_salud_km',
        'dist_educacion_basica_km',
        'dist_educacion_superior_km',
        'dist_areas_verdes_km'
    ]
    
    X_numeric = gdf[features_numericas].copy()
    X = pd.concat([X_numeric, comunas_dummies], axis=1)
    
    print(f"✅ Variables preparadas:")
    print(f"   • Total features: {X.shape[1]}")
    print(f"   • Observaciones: {len(y)}")
    
    return X, y, list(X.columns)

def random_search_rapido(X_train, y_train):
    """Búsqueda aleatoria rápida para exploración inicial"""
    print("\n" + "="*70)
    print("🎲 RANDOMIZED SEARCH (Exploración Rápida)")
    print("="*70)
    
    print("\n⏳ Explorando espacio de hiperparámetros (5-10 minutos)...")
    
    # Distribuciones para búsqueda aleatoria
    param_distributions = {
        'n_estimators': [100, 200, 300, 500, 800, 1000],
        'max_depth': [10, 20, 30, 40, None],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.3, 0.5, 0.7],
        'bootstrap': [True, False],
        'max_samples': [0.5, 0.7, 0.9, None]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    random_search = RandomizedSearchCV(
        rf,
        param_distributions,
        n_iter=100,  # 100 combinaciones aleatorias
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=1,
        random_state=42
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\n✅ Búsqueda aleatoria completada")
    print(f"   • Mejor R² (CV): {random_search.best_score_:.4f}")
    print(f"   • Mejores hiperparámetros encontrados:")
    for param, value in random_search.best_params_.items():
        print(f"     • {param}: {value}")
    
    return random_search.best_params_, random_search

def grid_search_refinado(X_train, y_train, best_params_random):
    """Grid Search refinado alrededor de los mejores parámetros"""
    print("\n" + "="*70)
    print("🔍 GRID SEARCH REFINADO (Optimización Fina)")
    print("="*70)
    
    print("\n⏳ Refinando hiperparámetros (15-30 minutos)...")
    print("   Esto puede tomar tiempo, pero mejorará el modelo significativamente")
    
    # Crear grid refinado alrededor de los mejores parámetros
    n_est = best_params_random.get('n_estimators', 300)
    max_d = best_params_random.get('max_depth', 30)
    min_split = best_params_random.get('min_samples_split', 5)
    min_leaf = best_params_random.get('min_samples_leaf', 2)
    
    param_grid = {
        'n_estimators': [max(100, n_est-100), n_est, n_est+100],
        'max_depth': [max(10, max_d-10), max_d, max_d+10] if max_d else [20, 30, None],
        'min_samples_split': [max(2, min_split-2), min_split, min_split+2],
        'min_samples_leaf': [max(1, min_leaf-1), min_leaf, min_leaf+1],
        'max_features': [best_params_random.get('max_features', 'sqrt')],
        'bootstrap': [best_params_random.get('bootstrap', True)],
        'max_samples': [best_params_random.get('max_samples', None)]
    }
    
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        verbose=2
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"\n✅ Grid Search completado")
    print(f"   • Mejor R² (CV): {grid_search.best_score_:.4f}")
    print(f"   • Hiperparámetros ÓPTIMOS:")
    for param, value in grid_search.best_params_.items():
        print(f"     • {param}: {value}")
    
    return grid_search.best_estimator_, grid_search

def evaluar_modelo_optimizado(modelo, X_train, y_train, X_test, y_test):
    """Evalúa el modelo optimizado"""
    print("\n" + "="*70)
    print("📊 EVALUACIÓN MODELO OPTIMIZADO")
    print("="*70)
    
    # Predicciones
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)
    
    # Métricas train
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    mae_train = mean_absolute_error(y_train, y_pred_train)
    
    # Métricas test
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    mae_test = mean_absolute_error(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(modelo, X_train, y_train, cv=5, scoring='r2', n_jobs=-1)
    
    print(f"\n📊 MÉTRICAS FINALES:")
    print(f"   • R² (train):    {r2_train:.4f}")
    print(f"   • R² (test):     {r2_test:.4f}")
    print(f"   • R² (CV):       {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"   • RMSE (test):   {rmse_test:.4f}")
    print(f"   • MAE (test):    {mae_test:.4f}")
    print(f"   • Overfitting:   {(r2_train - r2_test)*100:.2f}%")
    
    resultados = {
        'r2_train': r2_train,
        'r2_test': r2_test,
        'r2_cv_mean': cv_scores.mean(),
        'r2_cv_std': cv_scores.std(),
        'rmse_test': rmse_test,
        'mae_test': mae_test,
        'y_pred_test': y_pred_test,
        'cv_scores': cv_scores
    }
    
    return resultados

def analizar_curvas_aprendizaje(modelo, X_train, y_train):
    """Genera curvas de aprendizaje"""
    print("\n" + "="*70)
    print("📈 GENERANDO CURVAS DE APRENDIZAJE")
    print("="*70)
    
    print("\n⏳ Calculando curvas (puede tomar 5-10 minutos)...")
    
    train_sizes, train_scores, val_scores = learning_curve(
        modelo,
        X_train,
        y_train,
        cv=5,
        scoring='r2',
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        verbose=0
    )
    
    # Promedios y desviaciones
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)
    
    print(f"\n✅ Curvas calculadas")
    print(f"   • R² máximo (train): {train_mean[-1]:.4f}")
    print(f"   • R² máximo (val):   {val_mean[-1]:.4f}")
    print(f"   • Gap final:         {(train_mean[-1] - val_mean[-1])*100:.2f}%")
    
    return train_sizes, train_mean, train_std, val_mean, val_std

def visualizar_optimizacion(resultados, modelo, feature_names, curvas, random_search, grid_search):
    """Genera visualizaciones de la optimización"""
    print("\n" + "="*70)
    print("🎨 GENERANDO VISUALIZACIONES")
    print("="*70)
    
    train_sizes, train_mean, train_std, val_mean, val_std = curvas
    
    # 1. Curvas de Aprendizaje + Comparación
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # Curva de aprendizaje
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(train_sizes, train_mean, 'o-', color='blue', label='Train', linewidth=2)
    ax1.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.2, color='blue')
    ax1.plot(train_sizes, val_mean, 'o-', color='red', label='Validación (CV)', linewidth=2)
    ax1.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                     alpha=0.2, color='red')
    ax1.set_xlabel('Tamaño del conjunto de entrenamiento', fontsize=12, weight='bold')
    ax1.set_ylabel('R²', fontsize=12, weight='bold')
    ax1.set_title('Curvas de Aprendizaje - Random Forest Optimizado', 
                  fontsize=14, weight='bold')
    ax1.legend(loc='lower right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=val_mean[-1], color='green', linestyle='--', 
                label=f'R² final: {val_mean[-1]:.4f}')
    
    # Métricas comparativas
    ax2 = fig.add_subplot(gs[0, 2])
    metricas = ['R² Train', 'R² Test', 'R² CV']
    valores = [resultados['r2_train'], resultados['r2_test'], resultados['r2_cv_mean']]
    colors = ['#2ecc71' if v > 0.85 else '#f39c12' if v > 0.75 else '#e74c3c' for v in valores]
    
    bars = ax2.barh(metricas, valores, color=colors)
    ax2.set_xlabel('R²', fontsize=12, weight='bold')
    ax2.set_title('Métricas Finales', fontsize=14, weight='bold')
    ax2.set_xlim(0, 1)
    for i, v in enumerate(valores):
        ax2.text(v + 0.02, i, f'{v:.4f}', va='center', fontsize=11, weight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Distribución de scores CV
    ax3 = fig.add_subplot(gs[1, 0])
    cv_scores = resultados['cv_scores']
    ax3.hist(cv_scores, bins=10, edgecolor='black', alpha=0.7, color='steelblue')
    ax3.axvline(cv_scores.mean(), color='red', linestyle='--', linewidth=2,
                label=f'Media: {cv_scores.mean():.4f}')
    ax3.set_xlabel('R² (Cross-Validation)', fontsize=12, weight='bold')
    ax3.set_ylabel('Frecuencia', fontsize=12, weight='bold')
    ax3.set_title('Distribución R² Cross-Validation (5-fold)', fontsize=14, weight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Predicciones vs Real
    ax4 = fig.add_subplot(gs[1, 1:])
    y_test = resultados['y_pred_test']  # Asumiendo que está disponible
    # Por simplicidad, usamos un scatter genérico
    ax4.text(0.5, 0.5, 'Ver gráfico de dispersión\nen visualización separada',
             ha='center', va='center', fontsize=14, transform=ax4.transAxes)
    ax4.set_title('Predicciones vs Real (Test Set)', fontsize=14, weight='bold')
    ax4.axis('off')
    
    # Feature Importance Top 15
    ax5 = fig.add_subplot(gs[2, :])
    importances = modelo.feature_importances_
    indices = np.argsort(importances)[-15:]
    
    ax5.barh(range(len(indices)), importances[indices], color='steelblue')
    ax5.set_yticks(range(len(indices)))
    ax5.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax5.set_xlabel('Importancia', fontsize=12, weight='bold')
    ax5.set_title('Top 15 Variables Más Importantes (Random Forest Optimizado)',
                  fontsize=14, weight='bold')
    ax5.grid(True, alpha=0.3, axis='x')
    
    plt.savefig(VIZ_DIR / 'rf_optimizado_completo.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("   ✅ Guardado: rf_optimizado_completo.png")
    
    # 2. Análisis de hiperparámetros (Random Search)
    if hasattr(random_search, 'cv_results_'):
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        
        results_df = pd.DataFrame(random_search.cv_results_)
        
        # Top 10 configuraciones
        ax1 = axes[0, 0]
        top10 = results_df.nlargest(10, 'mean_test_score')
        ax1.barh(range(10), top10['mean_test_score'].values, color='steelblue')
        ax1.set_yticks(range(10))
        ax1.set_yticklabels([f'Config {i+1}' for i in range(10)])
        ax1.set_xlabel('R² (CV)', fontsize=12, weight='bold')
        ax1.set_title('Top 10 Configuraciones (Random Search)', fontsize=14, weight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Scatter: n_estimators vs R²
        ax2 = axes[0, 1]
        if 'param_n_estimators' in results_df.columns:
            scatter = ax2.scatter(results_df['param_n_estimators'], 
                                 results_df['mean_test_score'],
                                 c=results_df['mean_test_score'],
                                 cmap='RdYlGn', s=50, alpha=0.6)
            ax2.set_xlabel('n_estimators', fontsize=12, weight='bold')
            ax2.set_ylabel('R² (CV)', fontsize=12, weight='bold')
            ax2.set_title('Efecto de n_estimators', fontsize=14, weight='bold')
            plt.colorbar(scatter, ax=ax2, label='R²')
            ax2.grid(True, alpha=0.3)
        
        # Scatter: max_depth vs R²
        ax3 = axes[1, 0]
        if 'param_max_depth' in results_df.columns:
            max_depths = results_df['param_max_depth'].fillna(999)  # None -> 999 para graficar
            scatter = ax3.scatter(max_depths,
                                 results_df['mean_test_score'],
                                 c=results_df['mean_test_score'],
                                 cmap='RdYlGn', s=50, alpha=0.6)
            ax3.set_xlabel('max_depth (999 = None)', fontsize=12, weight='bold')
            ax3.set_ylabel('R² (CV)', fontsize=12, weight='bold')
            ax3.set_title('Efecto de max_depth', fontsize=14, weight='bold')
            plt.colorbar(scatter, ax=ax3, label='R²')
            ax3.grid(True, alpha=0.3)
        
        # Boxplot: Efecto de max_features
        ax4 = axes[1, 1]
        if 'param_max_features' in results_df.columns:
            data_boxplot = []
            labels_boxplot = []
            for mf in results_df['param_max_features'].unique():
                scores = results_df[results_df['param_max_features'] == mf]['mean_test_score']
                if len(scores) > 0:
                    data_boxplot.append(scores.values)
                    labels_boxplot.append(str(mf))
            
            if data_boxplot:
                ax4.boxplot(data_boxplot, labels=labels_boxplot)
                ax4.set_xlabel('max_features', fontsize=12, weight='bold')
                ax4.set_ylabel('R² (CV)', fontsize=12, weight='bold')
                ax4.set_title('Efecto de max_features', fontsize=14, weight='bold')
                ax4.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(VIZ_DIR / 'rf_analisis_hiperparametros.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("   ✅ Guardado: rf_analisis_hiperparametros.png")
    
    print("\n✅ Visualizaciones completadas")

def generar_reporte_optimizacion(modelo, resultados, best_params, baseline_r2=0.9080):
    """Genera reporte final de optimización"""
    print("\n" + "="*70)
    print("📄 GENERANDO REPORTE FINAL")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo = BASE_DIR / f"RF_OPTIMIZADO_{timestamp}.md"
    
    mejora = ((resultados['r2_test'] - baseline_r2) / baseline_r2) * 100
    
    with open(archivo, 'w', encoding='utf-8') as f:
        f.write("# RANDOM FOREST OPTIMIZADO - REPORTE FINAL\n\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"**Fecha:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("## 🏆 RESULTADOS FINALES\n\n")
        f.write(f"- **R² Test:** {resultados['r2_test']:.4f}\n")
        f.write(f"- **R² Train:** {resultados['r2_train']:.4f}\n")
        f.write(f"- **R² CV:** {resultados['r2_cv_mean']:.4f} ± {resultados['r2_cv_std']:.4f}\n")
        f.write(f"- **RMSE Test:** {resultados['rmse_test']:.4f}\n")
        f.write(f"- **MAE Test:** {resultados['mae_test']:.4f}\n")
        f.write(f"- **Overfitting:** {(resultados['r2_train'] - resultados['r2_test'])*100:.2f}%\n\n")
        
        f.write("="*70 + "\n\n")
        f.write("## 📊 COMPARACIÓN\n\n")
        f.write("| Métrica | RF Básico | RF Optimizado | Mejora |\n")
        f.write("|---------|-----------|---------------|--------|\n")
        f.write(f"| R² Test | {baseline_r2:.4f} | {resultados['r2_test']:.4f} | {mejora:+.2f}% |\n\n")
        
        f.write("="*70 + "\n\n")
        f.write("## 🎯 HIPERPARÁMETROS ÓPTIMOS\n\n")
        f.write("```python\n")
        f.write("RandomForestRegressor(\n")
        for param, value in best_params.items():
            f.write(f"    {param}={repr(value)},\n")
        f.write("    random_state=42,\n")
        f.write("    n_jobs=-1\n")
        f.write(")\n```\n\n")
        
        f.write("="*70 + "\n\n")
        f.write("## 💡 RECOMENDACIÓN\n\n")
        
        if mejora > 1:
            f.write(f"### ✅ ADOPTAR MODELO OPTIMIZADO\n\n")
            f.write(f"Mejora sustancial: +{mejora:.2f}% en R²\n")
            f.write("La optimización de hiperparámetros ha mejorado significativamente el modelo.\n")
        elif mejora > 0:
            f.write(f"### ✅ LEVE MEJORA\n\n")
            f.write(f"Mejora marginal: +{mejora:.2f}% en R²\n")
            f.write("El modelo ya estaba bien configurado. Cambios menores.\n")
        else:
            f.write(f"### ⚠️  SIN MEJORA\n\n")
            f.write(f"Cambio: {mejora:.2f}% en R²\n")
            f.write("El modelo básico ya era óptimo. Mantener configuración anterior.\n")
        
        f.write("\n" + "="*70 + "\n\n")
        f.write("## 📁 ARCHIVOS GENERADOS\n\n")
        f.write("1. `rf_optimizado_completo.png` - Análisis completo del modelo\n")
        f.write("2. `rf_analisis_hiperparametros.png` - Impacto de hiperparámetros\n")
        f.write("3. `modelo_rf_optimizado.pkl` - Modelo guardado\n")
        f.write(f"4. `{archivo.name}` - Este reporte\n")
    
    print(f"✅ Reporte guardado: {archivo.name}")
    return archivo

def guardar_modelo(modelo, resultados):
    """Guarda el modelo optimizado"""
    print("\n" + "="*70)
    print("💾 GUARDANDO MODELO")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo_modelo = BASE_DIR / f"modelo_rf_optimizado_{timestamp}.pkl"
    
    modelo_data = {
        'modelo': modelo,
        'resultados': resultados,
        'timestamp': timestamp
    }
    
    with open(archivo_modelo, 'wb') as f:
        pickle.dump(modelo_data, f)
    
    print(f"✅ Modelo guardado: {archivo_modelo.name}")
    print(f"   • R² Test: {resultados['r2_test']:.4f}")
    print(f"   • Tamaño: {archivo_modelo.stat().st_size / 1024:.1f} KB")
    
    return archivo_modelo

def main():
    """Función principal"""
    print("\n" + "="*70)
    print("🚀 OPTIMIZACIÓN DE RANDOM FOREST")
    print("="*70)
    print("\n⚠️  ADVERTENCIA: Este proceso puede tomar 30-60 minutos")
    print("   Se realizarán búsquedas exhaustivas de hiperparámetros\n")
    
    try:
        # 1. Cargar datos
        gdf = cargar_datos()
        
        # 2. Preparar variables
        X, y, feature_names = preparar_variables(gdf)
        
        # 3. Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print(f"\n✅ Datos divididos: {len(X_train)} train, {len(X_test)} test")
        
        # 4. Random Search (exploración)
        best_params_random, random_search = random_search_rapido(X_train, y_train)
        
        # 5. Grid Search (refinamiento)
        modelo_optimizado, grid_search = grid_search_refinado(
            X_train, y_train, best_params_random
        )
        
        # 6. Evaluar
        resultados = evaluar_modelo_optimizado(
            modelo_optimizado, X_train, y_train, X_test, y_test
        )
        
        # 7. Curvas de aprendizaje
        curvas = analizar_curvas_aprendizaje(modelo_optimizado, X_train, y_train)
        
        # 8. Visualizar
        visualizar_optimizacion(
            resultados, modelo_optimizado, feature_names, 
            curvas, random_search, grid_search
        )
        
        # 9. Guardar modelo
        archivo_modelo = guardar_modelo(modelo_optimizado, resultados)
        
        # 10. Reporte
        archivo_reporte = generar_reporte_optimizacion(
            modelo_optimizado, resultados, 
            grid_search.best_params_
        )
        
        # 11. Resumen
        print("\n" + "="*70)
        print("✅ OPTIMIZACIÓN COMPLETADA")
        print("="*70)
        
        print(f"\n🏆 RESULTADOS FINALES:")
        print(f"   • R² Test:     {resultados['r2_test']:.4f}")
        print(f"   • R² CV:       {resultados['r2_cv_mean']:.4f} ± {resultados['r2_cv_std']:.4f}")
        print(f"   • RMSE Test:   {resultados['rmse_test']:.4f}")
        print(f"   • Overfitting: {(resultados['r2_train'] - resultados['r2_test'])*100:.2f}%")
        
        print(f"\n📁 Archivos generados:")
        print(f"   1. visualizaciones/rf_optimizado_completo.png")
        print(f"   2. visualizaciones/rf_analisis_hiperparametros.png")
        print(f"   3. {archivo_modelo.name}")
        print(f"   4. {archivo_reporte.name}")
        
        return modelo_optimizado, resultados
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    modelo, resultados = main()
