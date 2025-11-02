#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análisis de Outliers del Modelo Hedónico OLS

Identifica observaciones influyentes mediante:
- Cook's Distance
- Leverage (hat values)
- Studentized Residuals
- DFFITS

Autor: Proyecto GeoInformática
Fecha: 1 de noviembre de 2025
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import OLSInfluence
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def cargar_datos():
    """Carga el dataset procesado"""
    
    print("\n" + "=" * 80)
    print("📊 ANÁLISIS DE OUTLIERS DEL MODELO HEDÓNICO")
    print("=" * 80)
    
    # Buscar archivo más reciente
    import glob
    archivos = glob.glob('/home/felipe/Documentos/GeoInformatica/datos_procesados/propiedades_kaggle_*.geojson')
    
    if not archivos:
        raise FileNotFoundError("No se encontró archivo de propiedades procesadas")
    
    archivo = sorted(archivos)[-1]
    print(f"\n📂 Cargando: {archivo.split('/')[-1]}")
    
    gdf = gpd.read_file(archivo)
    print(f"✅ Cargado: {len(gdf):,} propiedades")
    
    return gdf


def preparar_variables(df):
    """Prepara las variables para el modelo (igual que modelo_hedonico_ols.py)"""
    
    print("\n" + "=" * 80)
    print("🔧 PREPARACIÓN DE VARIABLES")
    print("=" * 80)
    
    df_model = df.copy()
    
    # 1. Variable dependiente
    print("\n1️⃣  Variable dependiente: log(precio)")
    df_model['log_precio'] = np.log(df_model['precio'])
    print("   ✓ Transformación logarítmica aplicada")
    
    # 2. Variables intrínsecas
    print("\n2️⃣  Variables intrínsecas:")
    df_model['log_superficie'] = np.log(df_model['superficie_util'])
    df_model['dormitorios_num'] = pd.to_numeric(df_model['dormitorios'], errors='coerce')
    df_model['banos_num'] = pd.to_numeric(df_model['banos'], errors='coerce')
    print("   ✓ log_superficie, dormitorios, baños")
    
    # 3. Variables de comuna
    print("\n3️⃣  Variables de ubicación (dummies de comuna):")
    comunas_principales = ['Santiago', 'Estación Central', 'Las Condes', 
                          'Ñuñoa', 'Providencia', 'Vitacura']
    
    df_model['comuna_norm'] = df_model['comuna'].str.title().str.strip()
    
    for comuna in comunas_principales:
        col_name = f'comuna_{comuna.replace(" ", "_").lower()}'
        df_model[col_name] = (df_model['comuna_norm'] == comuna).astype(int)
        print(f"   ✓ {col_name}")
    
    # 4. Variables espaciales
    print("\n4️⃣  Variables espaciales del entorno:")
    variables_espaciales = [
        'espacial_dist_transporte_metro_m',
        'espacial_dist_turismo_m',
        'espacial_dist_salud_m',
        'espacial_dist_educacion_basica_m'
    ]
    
    vars_espaciales = []
    for var in variables_espaciales:
        if var in df_model.columns:
            nueva_var = var.replace('espacial_', '').replace('_m', '_km')
            df_model[nueva_var] = df_model[var] / 1000  # Convertir a km
            vars_espaciales.append(nueva_var)
            print(f"   ✓ {nueva_var}")
    
    print(f"\n   Total variables espaciales: {len(vars_espaciales)}")
    
    # 5. Limpieza de datos faltantes
    print("\n5️⃣  Limpieza de datos faltantes:")
    inicial = len(df_model)
    
    variables_clave = ['log_precio', 'banos_num'] + vars_espaciales
    variables_clave = [v for v in variables_clave if v in df_model.columns]
    
    df_model = df_model.dropna(subset=variables_clave)
    
    print(f"   ✓ Datos iniciales: {inicial:,}")
    print(f"   ✓ Datos finales: {len(df_model):,}")
    print(f"   ✓ Eliminados: {inicial - len(df_model):,}")
    
    return df_model, vars_espaciales


def ajustar_modelo(df_model, vars_espaciales):
    """Ajusta el modelo OLS (versión simplificada sin VIF)"""
    
    print("\n" + "=" * 80)
    print("📈 AJUSTE DEL MODELO OLS")
    print("=" * 80)
    
    # Variables independientes (sin VIF para mantener consistencia)
    variables_independientes = ['banos_num']
    
    # Agregar comunas (excluir Vitacura como referencia)
    comunas_vars = [c for c in df_model.columns if c.startswith('comuna_')]
    variables_independientes.extend([c for c in comunas_vars if 'vitacura' not in c])
    
    # Agregar variables espaciales
    variables_independientes.extend(vars_espaciales)
    
    # Filtrar solo variables numéricas presentes
    variables_independientes = [v for v in variables_independientes 
                                if v in df_model.columns and pd.api.types.is_numeric_dtype(df_model[v])]
    
    print(f"\n📊 Variables en el modelo: {len(variables_independientes)}")
    
    # Preparar X e y
    X = df_model[variables_independientes]
    y = df_model['log_precio']
    
    # Agregar constante
    X_const = sm.add_constant(X)
    
    # Ajustar modelo
    print(f"\n⏳ Ajustando modelo OLS...")
    modelo = sm.OLS(y, X_const).fit()
    print(f"✅ Modelo ajustado")
    print(f"   R² = {modelo.rsquared:.4f}")
    print(f"   R² ajustado = {modelo.rsquared_adj:.4f}")
    
    return modelo, X_const, y, df_model


def calcular_metricas_influencia(modelo, X, y, df):
    """Calcula métricas de influencia de observaciones"""
    
    print("\n" + "=" * 80)
    print("🔍 CALCULANDO MÉTRICAS DE INFLUENCIA")
    print("=" * 80)
    
    # Crear objeto de influencia
    influence = OLSInfluence(modelo)
    
    # 1. Cook's Distance
    print("\n1️⃣  Cook's Distance")
    cooks_d = influence.cooks_distance[0]
    threshold_cooks = 4 / len(df)
    outliers_cooks = cooks_d > threshold_cooks
    n_outliers_cooks = outliers_cooks.sum()
    
    print(f"   Threshold: {threshold_cooks:.6f}")
    print(f"   Outliers detectados: {n_outliers_cooks} ({n_outliers_cooks/len(df)*100:.2f}%)")
    print(f"   Max Cook's D: {cooks_d.max():.6f}")
    
    # 2. Leverage (hat values)
    print("\n2️⃣  Leverage (Hat Values)")
    leverage = influence.hat_matrix_diag
    threshold_leverage = 2 * X.shape[1] / len(df)
    outliers_leverage = leverage > threshold_leverage
    n_outliers_leverage = outliers_leverage.sum()
    
    print(f"   Threshold: {threshold_leverage:.6f}")
    print(f"   High leverage detectados: {n_outliers_leverage} ({n_outliers_leverage/len(df)*100:.2f}%)")
    print(f"   Max leverage: {leverage.max():.6f}")
    
    # 3. Studentized Residuals
    print("\n3️⃣  Studentized Residuals")
    student_resid = influence.resid_studentized_internal
    outliers_resid = np.abs(student_resid) > 3
    n_outliers_resid = outliers_resid.sum()
    
    print(f"   Threshold: ±3")
    print(f"   Outliers detectados: {n_outliers_resid} ({n_outliers_resid/len(df)*100:.2f}%)")
    print(f"   Max |residuo|: {np.abs(student_resid).max():.4f}")
    
    # 4. DFFITS
    print("\n4️⃣  DFFITS")
    dffits = influence.dffits[0]
    threshold_dffits = 2 * np.sqrt(X.shape[1] / len(df))
    outliers_dffits = np.abs(dffits) > threshold_dffits
    n_outliers_dffits = outliers_dffits.sum()
    
    print(f"   Threshold: ±{threshold_dffits:.6f}")
    print(f"   Outliers detectados: {n_outliers_dffits} ({n_outliers_dffits/len(df)*100:.2f}%)")
    print(f"   Max |DFFITS|: {np.abs(dffits).max():.6f}")
    
    # Crear DataFrame con métricas
    metricas_df = df.copy()
    metricas_df['cooks_d'] = cooks_d
    metricas_df['leverage'] = leverage
    metricas_df['student_resid'] = student_resid
    metricas_df['dffits'] = dffits
    metricas_df['residuos'] = modelo.resid
    metricas_df['fitted'] = modelo.fittedvalues
    
    # Identificar outliers combinados
    metricas_df['is_outlier_cooks'] = outliers_cooks
    metricas_df['is_outlier_leverage'] = outliers_leverage
    metricas_df['is_outlier_resid'] = outliers_resid
    metricas_df['is_outlier_dffits'] = outliers_dffits
    
    metricas_df['is_outlier_any'] = (outliers_cooks | outliers_leverage | 
                                      outliers_resid | outliers_dffits)
    
    # Resumen
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE OUTLIERS")
    print("=" * 80)
    
    n_outliers_total = metricas_df['is_outlier_any'].sum()
    print(f"\n🎯 Outliers únicos (cualquier criterio): {n_outliers_total} ({n_outliers_total/len(df)*100:.2f}%)")
    print(f"   • Por Cook's D: {n_outliers_cooks}")
    print(f"   • Por Leverage: {n_outliers_leverage}")
    print(f"   • Por Residuos: {n_outliers_resid}")
    print(f"   • Por DFFITS: {n_outliers_dffits}")
    
    return metricas_df, {
        'threshold_cooks': threshold_cooks,
        'threshold_leverage': threshold_leverage,
        'threshold_dffits': threshold_dffits
    }


def analizar_outliers_detalle(metricas_df):
    """Analiza las propiedades outliers en detalle"""
    
    print("\n" + "=" * 80)
    print("🔎 ANÁLISIS DETALLADO DE OUTLIERS")
    print("=" * 80)
    
    outliers = metricas_df[metricas_df['is_outlier_any']].copy()
    
    print(f"\n📋 Top 10 Outliers por Cook's Distance:")
    print("=" * 80)
    
    top_outliers = outliers.nlargest(10, 'cooks_d')
    
    for idx, row in top_outliers.iterrows():
        print(f"\n🏠 Propiedad #{idx}")
        print(f"   • Precio: ${row['precio']:,.0f}/mes")
        print(f"   • Comuna: {row['comuna']}")
        print(f"   • Superficie: {row['superficie_util']:.0f} m²")
        print(f"   • Dormitorios: {row['dormitorios']}")
        print(f"   • Baños: {row['banos']}")
        print(f"   • Cook's D: {row['cooks_d']:.6f}")
        print(f"   • Leverage: {row['leverage']:.6f}")
        print(f"   • Residuo Studentizado: {row['student_resid']:.4f}")
        print(f"   • Residuo: {row['residuos']:.4f} (log)")
        print(f"   • Predicho: ${np.exp(row['fitted']):,.0f}/mes")
        print(f"   • Error: ${row['precio'] - np.exp(row['fitted']):,.0f}/mes")
    
    # Estadísticas de outliers
    print("\n" + "=" * 80)
    print("📊 ESTADÍSTICAS DE OUTLIERS")
    print("=" * 80)
    
    print(f"\n💰 Precio:")
    print(f"   • Promedio outliers: ${outliers['precio'].mean():,.0f}")
    print(f"   • Promedio no-outliers: ${metricas_df[~metricas_df['is_outlier_any']]['precio'].mean():,.0f}")
    print(f"   • Mediana outliers: ${outliers['precio'].median():,.0f}")
    print(f"   • Mediana no-outliers: ${metricas_df[~metricas_df['is_outlier_any']]['precio'].median():,.0f}")
    
    print(f"\n🏘️  Comuna (outliers):")
    comuna_counts = outliers['comuna'].value_counts()
    for comuna, count in comuna_counts.items():
        print(f"   • {comuna}: {count} ({count/len(outliers)*100:.1f}%)")
    
    return outliers


def visualizar_outliers(metricas_df, thresholds):
    """Genera visualizaciones de outliers"""
    
    print("\n" + "=" * 80)
    print("🎨 GENERANDO VISUALIZACIONES")
    print("=" * 80)
    
    import os
    os.makedirs('visualizaciones', exist_ok=True)
    
    # 1. Panel de 4 gráficos de diagnóstico
    print("\n1️⃣  Panel de diagnóstico de outliers...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # A) Cook's Distance
    ax = axes[0, 0]
    indices = range(len(metricas_df))
    colors = ['red' if x else 'blue' for x in metricas_df['is_outlier_cooks']]
    ax.stem(indices, metricas_df['cooks_d'], linefmt='b-', markerfmt='bo', basefmt=' ')
    for i, (idx, val, color) in enumerate(zip(indices, metricas_df['cooks_d'], colors)):
        if color == 'red':
            ax.plot(idx, val, 'ro', markersize=4)
    ax.axhline(thresholds['threshold_cooks'], color='r', linestyle='--', 
               linewidth=2, label=f"Threshold = {thresholds['threshold_cooks']:.6f}")
    ax.set_xlabel('Índice de Observación', fontsize=11, fontweight='bold')
    ax.set_ylabel("Cook's Distance", fontsize=11, fontweight='bold')
    ax.set_title("A) Cook's Distance (Observaciones Influyentes)", fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # B) Leverage vs Studentized Residuals
    ax = axes[0, 1]
    colors = metricas_df['is_outlier_any'].map({True: 'red', False: 'blue'})
    ax.scatter(metricas_df['leverage'], metricas_df['student_resid'], 
               c=colors, alpha=0.6, s=30)
    ax.axhline(3, color='orange', linestyle='--', linewidth=2, label='|Resid| = 3')
    ax.axhline(-3, color='orange', linestyle='--', linewidth=2)
    ax.axvline(thresholds['threshold_leverage'], color='purple', linestyle='--', 
               linewidth=2, label=f'Leverage = {thresholds["threshold_leverage"]:.4f}')
    ax.set_xlabel('Leverage (Hat Values)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Studentized Residuals', fontsize=11, fontweight='bold')
    ax.set_title('B) Leverage vs Residuos Studentizados', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # C) DFFITS
    ax = axes[1, 0]
    indices = range(len(metricas_df))
    colors = ['red' if x else 'blue' for x in metricas_df['is_outlier_dffits']]
    ax.stem(indices, metricas_df['dffits'], linefmt='b-', markerfmt='bo', basefmt=' ')
    for i, (idx, val, color) in enumerate(zip(indices, metricas_df['dffits'], colors)):
        if color == 'red':
            ax.plot(idx, val, 'ro', markersize=4)
    ax.axhline(thresholds['threshold_dffits'], color='r', linestyle='--', 
               linewidth=2, label=f"Threshold = ±{thresholds['threshold_dffits']:.4f}")
    ax.axhline(-thresholds['threshold_dffits'], color='r', linestyle='--', linewidth=2)
    ax.set_xlabel('Índice de Observación', fontsize=11, fontweight='bold')
    ax.set_ylabel('DFFITS', fontsize=11, fontweight='bold')
    ax.set_title('C) DFFITS (Influencia en Predicción)', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # D) Residuos vs Fitted (con outliers marcados)
    ax = axes[1, 1]
    colors = metricas_df['is_outlier_any'].map({True: 'red', False: 'blue'})
    ax.scatter(metricas_df['fitted'], metricas_df['residuos'], c=colors, alpha=0.6, s=30)
    ax.axhline(0, color='black', linestyle='-', linewidth=2)
    ax.set_xlabel('Valores Ajustados (log precio)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Residuos', fontsize=11, fontweight='bold')
    ax.set_title('D) Residuos vs Valores Ajustados (Outliers en Rojo)', fontsize=12, fontweight='bold')
    ax.grid(alpha=0.3)
    
    plt.suptitle('Diagnóstico de Outliers del Modelo OLS\n931 Propiedades de Alquiler', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = 'visualizaciones/analisis_outliers_diagnostico.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Guardado: {output_file}")
    plt.close()
    
    # 2. Mapa de outliers
    if 'geometry' in metricas_df.columns:
        print("\n2️⃣  Mapa de outliers...")
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Propiedades normales
        metricas_df[~metricas_df['is_outlier_any']].plot(
            ax=ax, color='lightblue', markersize=20, alpha=0.4, label='Normal'
        )
        
        # Outliers
        metricas_df[metricas_df['is_outlier_any']].plot(
            ax=ax, color='red', markersize=60, alpha=0.8, label='Outlier', marker='X'
        )
        
        ax.set_title('Distribución Espacial de Outliers\n931 Propiedades de Alquiler', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Longitud', fontsize=11, fontweight='bold')
        ax.set_ylabel('Latitud', fontsize=11, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        output_file = 'visualizaciones/mapa_outliers.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"   ✅ Guardado: {output_file}")
        plt.close()
    
    # 3. Histograma de Cook's Distance
    print("\n3️⃣  Histograma de Cook's Distance...")
    
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.hist(metricas_df['cooks_d'], bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(thresholds['threshold_cooks'], color='red', linestyle='--', 
               linewidth=2, label=f"Threshold = {thresholds['threshold_cooks']:.6f}")
    ax.set_xlabel("Cook's Distance", fontsize=11, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax.set_title("Distribución de Cook's Distance\n931 Propiedades", 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3, axis='y')
    
    output_file = 'visualizaciones/histograma_cooks_distance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Guardado: {output_file}")
    plt.close()
    
    print("\n✅ Visualizaciones completadas")


def reajustar_sin_outliers(df_original, metricas_df, vars_espaciales):
    """Reajusta el modelo eliminando outliers"""
    
    print("\n" + "=" * 80)
    print("🔄 REAJUSTE DEL MODELO SIN OUTLIERS")
    print("=" * 80)
    
    # Filtrar outliers
    df_sin_outliers = metricas_df[~metricas_df['is_outlier_any']].copy()
    
    print(f"\n📊 Comparación de datasets:")
    print(f"   • Observaciones originales: {len(metricas_df):,}")
    print(f"   • Outliers eliminados: {metricas_df['is_outlier_any'].sum():,}")
    print(f"   • Observaciones finales: {len(df_sin_outliers):,}")
    print(f"   • % eliminado: {metricas_df['is_outlier_any'].sum()/len(metricas_df)*100:.2f}%")
    
    # Reajustar modelo
    print(f"\n⏳ Reajustando modelo OLS sin outliers...")
    
    # Variables independientes
    variables_independientes = ['banos_num']
    comunas_vars = [c for c in df_sin_outliers.columns if c.startswith('comuna_')]
    variables_independientes.extend([c for c in comunas_vars if 'vitacura' not in c])
    variables_independientes.extend(vars_espaciales)
    variables_independientes = [v for v in variables_independientes 
                                if v in df_sin_outliers.columns and 
                                pd.api.types.is_numeric_dtype(df_sin_outliers[v])]
    
    X = df_sin_outliers[variables_independientes]
    y = df_sin_outliers['log_precio']
    X_const = sm.add_constant(X)
    
    modelo_sin_outliers = sm.OLS(y, X_const).fit()
    
    print(f"✅ Modelo reajustado")
    
    # Comparar métricas
    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN DE MODELOS")
    print("=" * 80)
    
    # Necesitamos el modelo original para comparar
    # Reajustar modelo original
    X_original = metricas_df[variables_independientes]
    y_original = metricas_df['log_precio']
    X_original_const = sm.add_constant(X_original)
    modelo_original = sm.OLS(y_original, X_original_const).fit()
    
    print(f"\n🔍 Modelo ORIGINAL (con outliers):")
    print(f"   • R²: {modelo_original.rsquared:.4f}")
    print(f"   • R² ajustado: {modelo_original.rsquared_adj:.4f}")
    print(f"   • AIC: {modelo_original.aic:.2f}")
    print(f"   • BIC: {modelo_original.bic:.2f}")
    print(f"   • RMSE: {np.sqrt(np.mean(modelo_original.resid**2)):.4f}")
    
    print(f"\n✨ Modelo SIN OUTLIERS:")
    print(f"   • R²: {modelo_sin_outliers.rsquared:.4f}")
    print(f"   • R² ajustado: {modelo_sin_outliers.rsquared_adj:.4f}")
    print(f"   • AIC: {modelo_sin_outliers.aic:.2f}")
    print(f"   • BIC: {modelo_sin_outliers.bic:.2f}")
    print(f"   • RMSE: {np.sqrt(np.mean(modelo_sin_outliers.resid**2)):.4f}")
    
    print(f"\n📈 MEJORA:")
    mejora_r2 = modelo_sin_outliers.rsquared - modelo_original.rsquared
    mejora_r2_adj = modelo_sin_outliers.rsquared_adj - modelo_original.rsquared_adj
    mejora_rmse = (np.sqrt(np.mean(modelo_original.resid**2)) - 
                   np.sqrt(np.mean(modelo_sin_outliers.resid**2)))
    
    print(f"   • ΔR²: {mejora_r2:+.4f} ({mejora_r2/modelo_original.rsquared*100:+.2f}%)")
    print(f"   • ΔR² ajustado: {mejora_r2_adj:+.4f} ({mejora_r2_adj/modelo_original.rsquared_adj*100:+.2f}%)")
    print(f"   • ΔRMSE: {mejora_rmse:+.4f} ({mejora_rmse/np.sqrt(np.mean(modelo_original.resid**2))*100:+.2f}%)")
    
    if mejora_r2_adj > 0.01:
        print(f"\n✅ RECOMENDACIÓN: Eliminar outliers MEJORA el modelo significativamente")
    else:
        print(f"\n⚠️  RECOMENDACIÓN: Eliminar outliers NO mejora el modelo significativamente")
        print(f"    → Considerar usar modelo robusto (Huber Regression) en lugar de eliminar datos")
    
    return modelo_sin_outliers, df_sin_outliers


def main():
    """Función principal"""
    
    # 1. Cargar datos
    gdf = cargar_datos()
    
    # 2. Preparar variables
    df_model, vars_espaciales = preparar_variables(gdf)
    
    # 3. Ajustar modelo
    modelo, X, y, df_final = ajustar_modelo(df_model, vars_espaciales)
    
    # 4. Calcular métricas de influencia
    metricas_df, thresholds = calcular_metricas_influencia(modelo, X, y, df_final)
    
    # 5. Analizar outliers en detalle
    outliers = analizar_outliers_detalle(metricas_df)
    
    # 6. Visualizar outliers
    visualizar_outliers(metricas_df, thresholds)
    
    # 7. Reajustar sin outliers
    modelo_sin_outliers, df_sin_outliers = reajustar_sin_outliers(gdf, metricas_df, vars_espaciales)
    
    # 8. Guardar resultados
    print("\n" + "=" * 80)
    print("💾 GUARDANDO RESULTADOS")
    print("=" * 80)
    
    # Guardar dataset con métricas
    output_file = 'datos_procesados/propiedades_con_metricas_outliers.csv'
    metricas_df[[
        'precio', 'comuna', 'superficie_util', 'dormitorios', 'banos',
        'cooks_d', 'leverage', 'student_resid', 'dffits', 'residuos', 'fitted',
        'is_outlier_cooks', 'is_outlier_leverage', 'is_outlier_resid', 
        'is_outlier_dffits', 'is_outlier_any'
    ]].to_csv(output_file, index=False)
    print(f"✅ Dataset con métricas: {output_file}")
    
    # Guardar resumen de outliers
    output_file = 'analisis_outliers_resumen.txt'
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("RESUMEN DEL ANÁLISIS DE OUTLIERS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Observaciones totales: {len(metricas_df):,}\n")
        f.write(f"Outliers detectados: {metricas_df['is_outlier_any'].sum():,} ({metricas_df['is_outlier_any'].sum()/len(metricas_df)*100:.2f}%)\n\n")
        
        f.write("Criterios de detección:\n")
        f.write(f"  • Cook's Distance > {thresholds['threshold_cooks']:.6f}: {metricas_df['is_outlier_cooks'].sum():,}\n")
        f.write(f"  • Leverage > {thresholds['threshold_leverage']:.6f}: {metricas_df['is_outlier_leverage'].sum():,}\n")
        f.write(f"  • |Studentized Residual| > 3: {metricas_df['is_outlier_resid'].sum():,}\n")
        f.write(f"  • |DFFITS| > {thresholds['threshold_dffits']:.6f}: {metricas_df['is_outlier_dffits'].sum():,}\n\n")
        
        f.write("Comparación de modelos:\n")
        f.write(f"  • R² original: {modelo.rsquared:.4f}\n")
        f.write(f"  • R² sin outliers: {modelo_sin_outliers.rsquared:.4f}\n")
        f.write(f"  • Mejora: {(modelo_sin_outliers.rsquared - modelo.rsquared):+.4f}\n")
    
    print(f"✅ Resumen de outliers: {output_file}")
    
    print("\n" + "=" * 80)
    print("✅ ANÁLISIS DE OUTLIERS COMPLETADO")
    print("=" * 80)
    
    print("\n📁 Archivos generados:")
    print("   1. visualizaciones/analisis_outliers_diagnostico.png")
    print("   2. visualizaciones/mapa_outliers.png")
    print("   3. visualizaciones/histograma_cooks_distance.png")
    print("   4. datos_procesados/propiedades_con_metricas_outliers.csv")
    print("   5. analisis_outliers_resumen.txt")


if __name__ == "__main__":
    main()
