#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
IMPLEMENTACIÓN DE GEOGRAPHICALLY WEIGHTED REGRESSION (GWR)
===========================================================

Objetivo:
    Implementar GWR para comparar con modelo OLS y determinar si
    la heterogeneidad espacial justifica el modelo más complejo.

Comparación:
    - OLS: Coeficientes globales (mismos en toda la región)
    - GWR: Coeficientes locales (varían por ubicación)

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

# GWR y selección de bandwidth
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

# Configuración
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10

# Directorios
BASE_DIR = Path(__file__).parent
DATOS_DIR = BASE_DIR / "datos_procesados"
VIZ_DIR = BASE_DIR / "visualizaciones"
VIZ_DIR.mkdir(exist_ok=True)

def cargar_datos_y_modelo_ols():
    """Carga datos limpios y modelo OLS para comparación"""
    print("\n" + "="*70)
    print("📂 CARGANDO DATOS Y MODELO OLS")
    print("="*70)
    
    # Cargar GeoJSON
    archivos = list(DATOS_DIR.glob("propiedades_limpias_*.geojson"))
    gdf = gpd.read_file(max(archivos, key=lambda x: x.stat().st_mtime))
    print(f"✅ Datos cargados: {len(gdf)} propiedades")
    
    # Cargar modelo OLS
    with open(BASE_DIR / "modelo_ols_limpio.pkl", 'rb') as f:
        modelo_ols = pickle.load(f)
    
    print(f"✅ Modelo OLS cargado")
    print(f"   • R² ajustado: {modelo_ols.rsquared_adj:.4f}")
    print(f"   • RMSE: {np.sqrt(modelo_ols.mse_resid):.4f}")
    
    return gdf, modelo_ols

def preparar_variables_gwr(gdf):
    """Prepara matrices X, y y coordenadas para GWR"""
    print("\n" + "="*70)
    print("🔧 PREPARANDO VARIABLES PARA GWR")
    print("="*70)
    
    # Variable dependiente
    y = np.log(gdf['precio']).values.reshape(-1, 1)
    
    # Crear variables derivadas
    gdf['banos_num'] = pd.to_numeric(gdf['banos'], errors='coerce').fillna(1)
    gdf['dist_transporte_km'] = gdf['espacial_dist_transporte_metro_m'] / 1000
    gdf['dist_turismo_km'] = gdf['espacial_dist_turismo_m'] / 1000
    gdf['dist_salud_km'] = gdf['espacial_dist_salud_m'] / 1000
    gdf['dist_educacion_basica_km'] = gdf['espacial_dist_educacion_basica_m'] / 1000
    
    # Crear dummies de comunas (eliminamos una categoría para evitar multicolinealidad)
    gdf['comuna_santiago'] = (gdf['comuna'] == 'Santiago').astype(int)
    gdf['comuna_estacion_central'] = (gdf['comuna'] == 'Estación Central').astype(int)
    gdf['comuna_las_condes'] = (gdf['comuna'] == 'Las Condes').astype(int)
    gdf['comuna_nunoa'] = (gdf['comuna'] == 'Ñuñoa').astype(int)
    # Vitacura será la categoría de referencia (no incluimos dummy)
    
    # Variables independientes (reducidas para evitar singularidad)
    variables = [
        'banos_num',
        'comuna_santiago',
        'comuna_estacion_central', 
        'comuna_las_condes',
        'comuna_nunoa',
        'dist_turismo_km',
        'dist_educacion_basica_km'
    ]
    
    # Crear matriz X con intercepto
    X_sin_intercepto = gdf[variables].values
    X = np.column_stack([np.ones(len(gdf)), X_sin_intercepto])
    
    # Coordenadas (necesarias para GWR)
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    
    print(f"✅ Variables preparadas:")
    print(f"   • Variable dependiente: log(precio)")
    print(f"   • Variables independientes: {len(variables)} + intercepto")
    print(f"   • Observaciones: {len(y)}")
    print(f"   • Coordenadas: {coords.shape}")
    print(f"   • Forma de X: {X.shape}")
    
    return y, X, coords, variables

def seleccionar_bandwidth_optimo(coords, y, X):
    """Selecciona bandwidth óptimo usando validación cruzada"""
    print("\n" + "="*70)
    print("🔍 SELECCIÓN DE BANDWIDTH ÓPTIMO")
    print("="*70)
    
    print("\n⏳ Buscando bandwidth óptimo (puede tomar varios minutos)...")
    print("   Método: AICc")
    print("   Kernel: Bisquare (adaptive)")
    
    # Selector de bandwidth (AICc está implícito por defecto)
    selector = Sel_BW(coords, y, X, kernel='bisquare')
    
    # Buscar bandwidth óptimo con límites más amplios
    try:
        bw = selector.search(bw_min=50, bw_max=len(y)-1)
    except Exception as e:
        # Si falla, usar bandwidth fijo conservador
        print(f"\n⚠️  Búsqueda automática falló ({str(e)}), usando bandwidth conservador...")
        bw = int(len(y) * 0.3)  # 30% de los datos
    
    print(f"\n✅ Bandwidth óptimo encontrado: {bw}")
    print(f"   • Tipo: Adaptive (número de vecinos)")
    print(f"   • Valor: {int(bw)} vecinos más cercanos")
    print(f"   • % del total: {(bw/len(y))*100:.1f}%")
    
    return bw

def ajustar_modelo_gwr(coords, y, X, bw):
    """Ajusta modelo GWR con bandwidth óptimo"""
    print("\n" + "="*70)
    print("📈 AJUSTANDO MODELO GWR")
    print("="*70)
    
    print("\n⏳ Entrenando GWR (puede tomar varios minutos)...")
    
    # Crear y ajustar modelo
    modelo_gwr = GWR(coords, y, X, bw)
    resultados_gwr = modelo_gwr.fit()
    
    print(f"\n✅ Modelo GWR ajustado")
    
    return resultados_gwr

def comparar_modelos(modelo_ols, resultados_gwr, y):
    """Compara métricas entre OLS y GWR"""
    print("\n" + "="*70)
    print("📊 COMPARACIÓN: OLS vs GWR")
    print("="*70)
    
    # Métricas OLS
    y_pred_ols = modelo_ols.fittedvalues.values.reshape(-1, 1)
    rmse_ols = np.sqrt(modelo_ols.mse_resid)
    r2_ols = modelo_ols.rsquared
    r2_adj_ols = modelo_ols.rsquared_adj
    aic_ols = modelo_ols.aic
    
    # Métricas GWR
    y_pred_gwr = resultados_gwr.predy
    residuos_gwr = y - y_pred_gwr
    rmse_gwr = np.sqrt((residuos_gwr**2).mean())
    
    # R² local promedio
    r2_local_mean = resultados_gwr.localR2.mean()
    
    # AIC
    aic_gwr = resultados_gwr.aicc
    
    print("\n📊 MÉTRICAS DE BONDAD DE AJUSTE:")
    print("="*70)
    print(f"{'Métrica':<30} {'OLS':>15} {'GWR':>15} {'Mejora':>15}")
    print("-"*70)
    
    # R²
    mejora_r2 = r2_local_mean - r2_ols
    mejora_r2_pct = (mejora_r2 / r2_ols) * 100
    print(f"{'R²':<30} {r2_ols:>15.4f} {r2_local_mean:>15.4f} {mejora_r2_pct:>14.1f}%")
    
    # R² ajustado
    mejora_r2_adj = r2_local_mean - r2_adj_ols
    mejora_r2_adj_pct = (mejora_r2_adj / r2_adj_ols) * 100
    print(f"{'R² ajustado (promedio)':<30} {r2_adj_ols:>15.4f} {r2_local_mean:>15.4f} {mejora_r2_adj_pct:>14.1f}%")
    
    # RMSE
    mejora_rmse = rmse_ols - rmse_gwr
    mejora_rmse_pct = (mejora_rmse / rmse_ols) * 100
    print(f"{'RMSE':<30} {rmse_ols:>15.4f} {rmse_gwr:>15.4f} {mejora_rmse_pct:>14.1f}%")
    
    # AIC
    mejora_aic = aic_ols - aic_gwr
    print(f"{'AIC':<30} {aic_ols:>15.2f} {aic_gwr:>15.2f} {mejora_aic:>15.2f}")
    
    print("\n" + "="*70)
    print("🔍 INTERPRETACIÓN:")
    print("="*70)
    
    if mejora_r2_pct > 5:
        print(f"✅ GWR ofrece mejora SIGNIFICATIVA (+{mejora_r2_pct:.1f}% en R²)")
        print(f"   → La heterogeneidad espacial es importante")
        print(f"   → GWR captura mejor las variaciones locales")
        decision = "GWR_RECOMENDADO"
    elif mejora_r2_pct > 2:
        print(f"⚠️  GWR ofrece mejora MODERADA (+{mejora_r2_pct:.1f}% en R²)")
        print(f"   → Hay heterogeneidad espacial leve")
        print(f"   → Considerar trade-off complejidad vs mejora")
        decision = "GWR_CONSIDERAR"
    else:
        print(f"❌ GWR ofrece mejora MARGINAL (+{mejora_r2_pct:.1f}% en R²)")
        print(f"   → La heterogeneidad espacial es mínima")
        print(f"   → OLS es suficiente (más simple)")
        decision = "OLS_SUFICIENTE"
    
    # Análisis de variación espacial de coeficientes
    print(f"\n📍 VARIACIÓN ESPACIAL DE COEFICIENTES:")
    print("="*70)
    
    for i, coef_name in enumerate(['Intercepto'] + [f'β{j}' for j in range(1, resultados_gwr.params.shape[1])]):
        coef_values = resultados_gwr.params[:, i]
        cv = (coef_values.std() / abs(coef_values.mean())) * 100 if coef_values.mean() != 0 else 0
        print(f"{coef_name:<20} Min:{coef_values.min():>8.3f}  Max:{coef_values.max():>8.3f}  CV:{cv:>6.1f}%")
    
    resultados = {
        'decision': decision,
        'mejora_r2_pct': mejora_r2_pct,
        'mejora_rmse_pct': mejora_rmse_pct,
        'mejora_aic': mejora_aic,
        'ols': {
            'r2': r2_ols,
            'r2_adj': r2_adj_ols,
            'rmse': rmse_ols,
            'aic': aic_ols
        },
        'gwr': {
            'r2_local_mean': r2_local_mean,
            'rmse': rmse_gwr,
            'aic': aic_gwr
        }
    }
    
    return resultados

def visualizar_comparacion(gdf, modelo_ols, resultados_gwr, y, variables):
    """Genera visualizaciones comparativas"""
    print("\n" + "="*70)
    print("🎨 GENERANDO VISUALIZACIONES")
    print("="*70)
    
    # 1. Comparación de R² local
    print("\n1️⃣  Mapa de R² local (GWR)...")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # R² local GWR
    gdf_plot = gdf.copy()
    gdf_plot['r2_local'] = resultados_gwr.localR2
    
    ax1 = axes[0]
    gdf_plot.plot(column='r2_local',
                  cmap='RdYlGn',
                  legend=True,
                  ax=ax1,
                  edgecolor='black',
                  linewidth=0.2,
                  vmin=0,
                  vmax=1)
    ax1.set_title(f'R² Local (GWR)\nPromedio: {resultados_gwr.localR2.mean():.4f}',
                  fontsize=14, weight='bold')
    ax1.axis('off')
    
    # Distribución de R² local
    ax2 = axes[1]
    ax2.hist(resultados_gwr.localR2, bins=30, edgecolor='black', alpha=0.7)
    ax2.axvline(resultados_gwr.localR2.mean(), color='red', linestyle='--', 
                linewidth=2, label=f'Media: {resultados_gwr.localR2.mean():.4f}')
    ax2.axvline(modelo_ols.rsquared, color='blue', linestyle='--',
                linewidth=2, label=f'OLS Global: {modelo_ols.rsquared:.4f}')
    ax2.set_xlabel('R² Local', fontsize=12)
    ax2.set_ylabel('Frecuencia', fontsize=12)
    ax2.set_title('Distribución de R² Local (GWR)', fontsize=14, weight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'gwr_r2_local.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Guardado: gwr_r2_local.png")
    
    # 2. Comparación de residuos
    print("\n2️⃣  Comparación de residuos...")
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    
    residuos_ols = modelo_ols.resid.values
    residuos_gwr = (y - resultados_gwr.predy).flatten()
    
    # Mapa residuos OLS
    ax1 = axes[0, 0]
    gdf_plot['residuos_ols'] = residuos_ols
    gdf_plot.plot(column='residuos_ols',
                  cmap='RdBu_r',
                  legend=True,
                  ax=ax1,
                  edgecolor='black',
                  linewidth=0.2,
                  vmin=-residuos_ols.std()*2,
                  vmax=residuos_ols.std()*2)
    ax1.set_title(f'Residuos OLS\nRMSE: {residuos_ols.std():.4f}',
                  fontsize=14, weight='bold')
    ax1.axis('off')
    
    # Mapa residuos GWR
    ax2 = axes[0, 1]
    gdf_plot['residuos_gwr'] = residuos_gwr
    gdf_plot.plot(column='residuos_gwr',
                  cmap='RdBu_r',
                  legend=True,
                  ax=ax2,
                  edgecolor='black',
                  linewidth=0.2,
                  vmin=-residuos_gwr.std()*2,
                  vmax=residuos_gwr.std()*2)
    ax2.set_title(f'Residuos GWR\nRMSE: {residuos_gwr.std():.4f}',
                  fontsize=14, weight='bold')
    ax2.axis('off')
    
    # Histograma comparativo
    ax3 = axes[1, 0]
    ax3.hist(residuos_ols, bins=30, alpha=0.6, label='OLS', edgecolor='black')
    ax3.hist(residuos_gwr, bins=30, alpha=0.6, label='GWR', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2)
    ax3.set_xlabel('Residuos', fontsize=12)
    ax3.set_ylabel('Frecuencia', fontsize=12)
    ax3.set_title('Distribución de Residuos', fontsize=14, weight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Q-Q plot comparativo
    ax4 = axes[1, 1]
    from scipy import stats
    stats.probplot(residuos_ols, dist="norm", plot=ax4)
    ax4.get_lines()[0].set_marker('o')
    ax4.get_lines()[0].set_markerfacecolor('blue')
    ax4.get_lines()[0].set_alpha(0.6)
    ax4.get_lines()[0].set_label('OLS')
    
    stats.probplot(residuos_gwr, dist="norm", plot=ax4)
    ax4.get_lines()[2].set_marker('s')
    ax4.get_lines()[2].set_markerfacecolor('green')
    ax4.get_lines()[2].set_alpha(0.6)
    ax4.get_lines()[2].set_label('GWR')
    
    ax4.set_title('Q-Q Plot: OLS vs GWR', fontsize=14, weight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'gwr_comparacion_residuos.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Guardado: gwr_comparacion_residuos.png")
    
    # 3. Mapas de coeficientes variables (top 3)
    print("\n3️⃣  Mapas de coeficientes locales...")
    
    # Calcular coeficiente de variación para cada parámetro
    cv_params = []
    for i in range(resultados_gwr.params.shape[1]):
        coef = resultados_gwr.params[:, i]
        cv = (coef.std() / abs(coef.mean())) * 100 if coef.mean() != 0 else 0
        cv_params.append((i, cv))
    
    # Top 3 coeficientes más variables
    top3 = sorted(cv_params, key=lambda x: x[1], reverse=True)[:3]
    
    fig, axes = plt.subplots(1, 3, figsize=(25, 8))
    
    for idx, (param_idx, cv) in enumerate(top3):
        ax = axes[idx]
        
        if param_idx == 0:
            param_name = 'Intercepto'
        else:
            param_name = variables[param_idx - 1]
        
        gdf_plot[f'coef_{param_idx}'] = resultados_gwr.params[:, param_idx]
        
        gdf_plot.plot(column=f'coef_{param_idx}',
                      cmap='RdBu_r',
                      legend=True,
                      ax=ax,
                      edgecolor='black',
                      linewidth=0.2)
        
        ax.set_title(f'{param_name}\nCV: {cv:.1f}%',
                     fontsize=14, weight='bold')
        ax.axis('off')
    
    plt.suptitle('Top 3 Coeficientes con Mayor Variación Espacial (GWR)',
                 fontsize=16, weight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(VIZ_DIR / 'gwr_coeficientes_locales.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✅ Guardado: gwr_coeficientes_locales.png")
    
    print("\n✅ Visualizaciones completadas")

def generar_reporte_decision(resultados, resultados_gwr, bw):
    """Genera reporte final con decisión"""
    print("\n" + "="*70)
    print("📄 GENERANDO REPORTE FINAL")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo = BASE_DIR / f"decision_gwr_vs_ols_{timestamp}.txt"
    
    with open(archivo, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("COMPARACIÓN GWR vs OLS - DECISIÓN FINAL\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Fecha: {datetime.now()}\n")
        f.write(f"Bandwidth GWR: {bw} vecinos\n\n")
        
        f.write("="*70 + "\n")
        f.write("MÉTRICAS COMPARATIVAS\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"OLS:\n")
        f.write(f"  R²:           {resultados['ols']['r2']:.4f}\n")
        f.write(f"  R² ajustado:  {resultados['ols']['r2_adj']:.4f}\n")
        f.write(f"  RMSE:         {resultados['ols']['rmse']:.4f}\n")
        f.write(f"  AIC:          {resultados['ols']['aic']:.2f}\n\n")
        
        f.write(f"GWR:\n")
        f.write(f"  R² local (promedio): {resultados['gwr']['r2_local_mean']:.4f}\n")
        f.write(f"  RMSE:                {resultados['gwr']['rmse']:.4f}\n")
        f.write(f"  AIC:                 {resultados['gwr']['aic']:.2f}\n\n")
        
        f.write(f"MEJORA:\n")
        f.write(f"  ΔR²:    {resultados['mejora_r2_pct']:+.2f}%\n")
        f.write(f"  ΔRMSE:  {resultados['mejora_rmse_pct']:+.2f}%\n")
        f.write(f"  ΔAIC:   {resultados['mejora_aic']:+.2f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("DECISIÓN FINAL\n")
        f.write("="*70 + "\n\n")
        
        if resultados['decision'] == "GWR_RECOMENDADO":
            f.write("✅ RECOMENDACIÓN: ADOPTAR GWR\n\n")
            f.write("Justificación:\n")
            f.write(f"  • Mejora significativa en R² (+{resultados['mejora_r2_pct']:.1f}%)\n")
            f.write(f"  • Reducción en RMSE ({resultados['mejora_rmse_pct']:.1f}%)\n")
            f.write(f"  • Heterogeneidad espacial importante detectada\n")
            f.write(f"  • Coeficientes varían sustancialmente por ubicación\n\n")
            f.write("Próximos pasos:\n")
            f.write("  1. Validar GWR con datos out-of-sample\n")
            f.write("  2. Analizar coeficientes locales por comuna\n")
            f.write("  3. Documentar variación espacial\n")
            f.write("  4. Actualizar sistema de predicción\n")
        
        elif resultados['decision'] == "GWR_CONSIDERAR":
            f.write("⚠️  RECOMENDACIÓN: CONSIDERAR GWR\n\n")
            f.write("Justificación:\n")
            f.write(f"  • Mejora moderada en R² (+{resultados['mejora_r2_pct']:.1f}%)\n")
            f.write(f"  • Hay heterogeneidad espacial pero no extrema\n")
            f.write(f"  • Trade-off entre mejora y complejidad\n\n")
            f.write("Evaluación:\n")
            f.write("  • Si prioridad es PRECISIÓN → GWR\n")
            f.write("  • Si prioridad es SIMPLICIDAD → OLS\n\n")
            f.write("Próximos pasos:\n")
            f.write("  1. Validar mejora con cross-validation\n")
            f.write("  2. Evaluar costo computacional\n")
            f.write("  3. Decidir según aplicación final\n")
        
        else:  # OLS_SUFICIENTE
            f.write("✅ RECOMENDACIÓN: MANTENER OLS\n\n")
            f.write("Justificación:\n")
            f.write(f"  • Mejora marginal con GWR (+{resultados['mejora_r2_pct']:.1f}%)\n")
            f.write(f"  • Heterogeneidad espacial mínima\n")
            f.write(f"  • OLS más simple e interpretable\n")
            f.write(f"  • Relación costo-beneficio favorece OLS\n\n")
            f.write("Conclusión:\n")
            f.write("  El modelo OLS captura adecuadamente la estructura espacial.\n")
            f.write("  Los coeficientes globales son suficientes para esta aplicación.\n")
    
    print(f"✅ Reporte guardado: {archivo.name}")
    
    return archivo

def main():
    """Función principal"""
    print("\n" + "="*70)
    print("🌍 IMPLEMENTACIÓN DE GWR vs OLS")
    print("="*70)
    print("\nObjetivo: Determinar si la heterogeneidad espacial justifica")
    print("          el uso de GWR sobre OLS más simple")
    
    try:
        # 1. Cargar datos y modelo OLS
        gdf, modelo_ols = cargar_datos_y_modelo_ols()
        
        # 2. Preparar variables para GWR
        y, X, coords, variables = preparar_variables_gwr(gdf)
        
        # 3. Seleccionar bandwidth óptimo
        bw = seleccionar_bandwidth_optimo(coords, y, X)
        
        # 4. Ajustar modelo GWR
        resultados_gwr = ajustar_modelo_gwr(coords, y, X, bw)
        
        # 5. Comparar modelos
        resultados = comparar_modelos(modelo_ols, resultados_gwr, y)
        
        # 6. Visualizar
        visualizar_comparacion(gdf, modelo_ols, resultados_gwr, y, variables)
        
        # 7. Generar reporte
        archivo_reporte = generar_reporte_decision(resultados, resultados_gwr, bw)
        
        # 8. Resumen final
        print("\n" + "="*70)
        print("✅ ANÁLISIS GWR COMPLETADO")
        print("="*70)
        
        print(f"\n📊 RESULTADOS:")
        print(f"   • Mejora R²: {resultados['mejora_r2_pct']:+.2f}%")
        print(f"   • Mejora RMSE: {resultados['mejora_rmse_pct']:+.2f}%")
        print(f"   • Decisión: {resultados['decision'].replace('_', ' ')}")
        
        print(f"\n📁 Archivos generados:")
        print(f"   1. visualizaciones/gwr_r2_local.png")
        print(f"   2. visualizaciones/gwr_comparacion_residuos.png")
        print(f"   3. visualizaciones/gwr_coeficientes_locales.png")
        print(f"   4. {archivo_reporte.name}")
        
        return resultados
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    resultados = main()
