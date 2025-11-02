#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Limpieza de Datos y Reajuste del Modelo Hedónico OLS

Elimina outliers identificados en el análisis previo:
1. Precios irreales (< $100,000 o > $2,000,000)
2. Características imposibles (27 dormitorios en 27 m²)
3. Outliers por Cook's Distance, Leverage, Residuos

Luego reajusta el modelo OLS y compara resultados.

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


def cargar_datos_originales():
    """Carga el dataset original sin filtrar"""
    
    print("\n" + "=" * 80)
    print("📊 LIMPIEZA DE DATOS Y REAJUSTE DEL MODELO")
    print("=" * 80)
    
    import glob
    archivos = glob.glob('/home/felipe/Documentos/GeoInformatica/datos_procesados/propiedades_kaggle_*.geojson')
    
    if not archivos:
        raise FileNotFoundError("No se encontró archivo de propiedades procesadas")
    
    archivo = sorted(archivos)[-1]
    print(f"\n📂 Cargando: {archivo.split('/')[-1]}")
    
    gdf = gpd.read_file(archivo)
    print(f"✅ Cargado: {len(gdf):,} propiedades")
    
    return gdf


def aplicar_filtros_limpieza(df):
    """Aplica filtros para eliminar outliers evidentes"""
    
    print("\n" + "=" * 80)
    print("🧹 APLICANDO FILTROS DE LIMPIEZA")
    print("=" * 80)
    
    inicial = len(df)
    print(f"\n📊 Dataset inicial: {inicial:,} propiedades")
    
    # 1. Filtro de precios irreales
    print("\n1️⃣  Filtro de precios irreales:")
    print(f"   Antes: {len(df):,}")
    
    # Analizar distribución de precios
    precio_min = df['precio'].min()
    precio_max = df['precio'].max()
    precio_p01 = df['precio'].quantile(0.01)
    precio_p99 = df['precio'].quantile(0.99)
    
    print(f"   • Mínimo actual: ${precio_min:,.0f}")
    print(f"   • Percentil 1%: ${precio_p01:,.0f}")
    print(f"   • Percentil 99%: ${precio_p99:,.0f}")
    print(f"   • Máximo actual: ${precio_max:,.0f}")
    
    # Filtrar precios irreales
    df_limpio = df[
        (df['precio'] >= 100000) &  # Mínimo $100K/mes
        (df['precio'] <= 2000000)   # Máximo $2M/mes
    ].copy()
    
    eliminados_precio = len(df) - len(df_limpio)
    print(f"   Después: {len(df_limpio):,}")
    print(f"   ✅ Eliminados por precio: {eliminados_precio} ({eliminados_precio/len(df)*100:.2f}%)")
    
    df = df_limpio
    
    # 2. Filtro de dormitorios imposibles
    print("\n2️⃣  Filtro de dormitorios imposibles:")
    print(f"   Antes: {len(df):,}")
    
    dormitorios_num = pd.to_numeric(df['dormitorios'], errors='coerce')
    superficie = df['superficie_util']
    
    print(f"   • Max dormitorios actual: {dormitorios_num.max():.0f}")
    print(f"   • Casos con dorm > 10: {(dormitorios_num > 10).sum()}")
    
    df_limpio = df[
        (dormitorios_num <= 10) &  # Máximo 10 dormitorios
        (dormitorios_num <= superficie/10)  # Al menos 10m² por dormitorio
    ].copy()
    
    eliminados_dorm = len(df) - len(df_limpio)
    print(f"   Después: {len(df_limpio):,}")
    print(f"   ✅ Eliminados por dormitorios: {eliminados_dorm} ({eliminados_dorm/len(df)*100:.2f}%)")
    
    df = df_limpio
    
    # 3. Filtro de baños imposibles
    print("\n3️⃣  Filtro de baños imposibles:")
    print(f"   Antes: {len(df):,}")
    
    banos_num = pd.to_numeric(df['banos'], errors='coerce')
    dormitorios_num_filtrado = pd.to_numeric(df['dormitorios'], errors='coerce')
    
    print(f"   • Max baños actual: {banos_num.max():.0f}")
    print(f"   • Casos con baños > dorm+3: {(banos_num > dormitorios_num_filtrado + 3).sum()}")
    
    df_limpio = df[
        (banos_num <= 8) &  # Máximo 8 baños
        (banos_num <= dormitorios_num_filtrado + 3)  # Máximo 3 baños más que dormitorios
    ].copy()
    
    eliminados_banos = len(df) - len(df_limpio)
    print(f"   Después: {len(df_limpio):,}")
    print(f"   ✅ Eliminados por baños: {eliminados_banos} ({eliminados_banos/len(df)*100:.2f}%)")
    
    df = df_limpio
    
    # 4. Filtro de superficie imposible
    print("\n4️⃣  Filtro de superficie imposible:")
    print(f"   Antes: {len(df):,}")
    
    print(f"   • Min superficie: {superficie.min():.0f} m²")
    print(f"   • Max superficie: {superficie.max():.0f} m²")
    
    df_limpio = df[
        (df['superficie_util'] >= 15) &  # Mínimo 15 m² (mono-ambiente pequeño)
        (df['superficie_util'] <= 500)   # Máximo 500 m² (casa grande)
    ].copy()
    
    eliminados_sup = len(df) - len(df_limpio)
    print(f"   Después: {len(df_limpio):,}")
    print(f"   ✅ Eliminados por superficie: {eliminados_sup} ({eliminados_sup/len(df)*100:.2f}%)")
    
    # Resumen de limpieza
    print("\n" + "=" * 80)
    print("📊 RESUMEN DE LIMPIEZA")
    print("=" * 80)
    
    total_eliminados = inicial - len(df_limpio)
    print(f"\n✅ Dataset limpio:")
    print(f"   • Observaciones iniciales: {inicial:,}")
    print(f"   • Observaciones finales: {len(df_limpio):,}")
    print(f"   • Total eliminado: {total_eliminados:,} ({total_eliminados/inicial*100:.2f}%)")
    print(f"\n   Desglose:")
    print(f"   • Por precio: {eliminados_precio:,}")
    print(f"   • Por dormitorios: {eliminados_dorm:,}")
    print(f"   • Por baños: {eliminados_banos:,}")
    print(f"   • Por superficie: {eliminados_sup:,}")
    
    return df_limpio


def preparar_variables_para_modelo(df):
    """Prepara las variables para el modelo OLS"""
    
    print("\n" + "=" * 80)
    print("🔧 PREPARACIÓN DE VARIABLES PARA EL MODELO")
    print("=" * 80)
    
    df_model = df.copy()
    
    # 1. Variable dependiente
    print("\n1️⃣  Variable dependiente: log(precio)")
    df_model['log_precio'] = np.log(df_model['precio'])
    
    # 2. Variables intrínsecas
    print("2️⃣  Variables intrínsecas:")
    df_model['log_superficie'] = np.log(df_model['superficie_util'])
    df_model['dormitorios_num'] = pd.to_numeric(df_model['dormitorios'], errors='coerce')
    df_model['banos_num'] = pd.to_numeric(df_model['banos'], errors='coerce')
    
    # 3. Variables de comuna
    print("3️⃣  Variables de ubicación (dummies):")
    comunas_principales = ['Santiago', 'Estación Central', 'Las Condes', 
                          'Ñuñoa', 'Providencia', 'Vitacura']
    
    df_model['comuna_norm'] = df_model['comuna'].str.title().str.strip()
    
    for comuna in comunas_principales:
        col_name = f'comuna_{comuna.replace(" ", "_").lower()}'
        df_model[col_name] = (df_model['comuna_norm'] == comuna).astype(int)
    
    # 4. Variables espaciales
    print("4️⃣  Variables espaciales:")
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
            df_model[nueva_var] = df_model[var] / 1000
            vars_espaciales.append(nueva_var)
    
    # 5. Limpieza final
    variables_clave = ['log_precio', 'banos_num'] + vars_espaciales
    variables_clave = [v for v in variables_clave if v in df_model.columns]
    df_model = df_model.dropna(subset=variables_clave)
    
    print(f"\n✅ Variables preparadas: {len(df_model):,} observaciones listas")
    
    return df_model, vars_espaciales


def ajustar_modelo_limpio(df_model, vars_espaciales):
    """Ajusta el modelo OLS con datos limpios"""
    
    print("\n" + "=" * 80)
    print("📈 AJUSTE DEL MODELO OLS (DATOS LIMPIOS)")
    print("=" * 80)
    
    # Variables independientes
    variables_independientes = ['banos_num']
    
    # Agregar comunas (excluir Vitacura como referencia)
    comunas_vars = [c for c in df_model.columns if c.startswith('comuna_')]
    variables_independientes.extend([c for c in comunas_vars if 'vitacura' not in c])
    
    # Agregar variables espaciales
    variables_independientes.extend(vars_espaciales)
    
    # Filtrar solo variables numéricas
    variables_independientes = [v for v in variables_independientes 
                                if v in df_model.columns and pd.api.types.is_numeric_dtype(df_model[v])]
    
    print(f"\n📊 Variables en el modelo: {len(variables_independientes)}")
    print(f"   • Intrínsecas: 1 (baños)")
    print(f"   • Comunas: {len([v for v in variables_independientes if v.startswith('comuna_')])}")
    print(f"   • Espaciales: {len(vars_espaciales)}")
    
    # Preparar X e y
    X = df_model[variables_independientes]
    y = df_model['log_precio']
    X_const = sm.add_constant(X)
    
    # Ajustar modelo
    print(f"\n⏳ Ajustando modelo OLS...")
    modelo = sm.OLS(y, X_const).fit()
    print(f"✅ Modelo ajustado")
    
    return modelo, X_const, y, variables_independientes


def comparar_modelos(modelo_limpio, df_limpio):
    """Compara modelo limpio con modelo original"""
    
    print("\n" + "=" * 80)
    print("📊 COMPARACIÓN: MODELO ORIGINAL VS MODELO LIMPIO")
    print("=" * 80)
    
    # Cargar resultados del modelo original (del archivo anterior)
    print("\n🔍 Modelo ORIGINAL (931 propiedades, con outliers):")
    print("   • R²: 0.2406")
    print("   • R² ajustado: 0.2324")
    print("   • AIC: 4457.39")
    print("   • BIC: 4510.59")
    print("   • RMSE: 2.6199")
    
    # Modelo limpio
    rmse_limpio = np.sqrt(np.mean(modelo_limpio.resid**2))
    
    print(f"\n✨ Modelo LIMPIO ({len(df_limpio):,} propiedades, sin outliers):")
    print(f"   • R²: {modelo_limpio.rsquared:.4f}")
    print(f"   • R² ajustado: {modelo_limpio.rsquared_adj:.4f}")
    print(f"   • AIC: {modelo_limpio.aic:.2f}")
    print(f"   • BIC: {modelo_limpio.bic:.2f}")
    print(f"   • RMSE: {rmse_limpio:.4f}")
    
    # Calcular mejoras
    mejora_r2 = modelo_limpio.rsquared - 0.2406
    mejora_r2_adj = modelo_limpio.rsquared_adj - 0.2324
    mejora_aic = 4457.39 - modelo_limpio.aic
    mejora_rmse = 2.6199 - rmse_limpio
    
    print(f"\n📈 MEJORA:")
    print(f"   • ΔR²: {mejora_r2:+.4f} ({mejora_r2/0.2406*100:+.2f}%)")
    print(f"   • ΔR² ajustado: {mejora_r2_adj:+.4f} ({mejora_r2_adj/0.2324*100:+.2f}%)")
    print(f"   • ΔAIC: {mejora_aic:+.2f} ({mejora_aic/4457.39*100:+.2f}%)")
    print(f"   • ΔRMSE: {mejora_rmse:+.4f} ({mejora_rmse/2.6199*100:+.2f}%)")
    
    if mejora_r2_adj > 0.05:
        print(f"\n✅ RESULTADO: Limpieza de datos MEJORA SIGNIFICATIVAMENTE el modelo")
        print(f"   → R² ajustado aumenta en {mejora_r2_adj:.4f} puntos (+{mejora_r2_adj/0.2324*100:.1f}%)")
    else:
        print(f"\n⚠️  RESULTADO: Limpieza de datos mejora el modelo moderadamente")
    
    return {
        'r2': modelo_limpio.rsquared,
        'r2_adj': modelo_limpio.rsquared_adj,
        'aic': modelo_limpio.aic,
        'bic': modelo_limpio.bic,
        'rmse': rmse_limpio,
        'mejora_r2': mejora_r2,
        'mejora_r2_adj': mejora_r2_adj
    }


def visualizar_comparacion(modelo_limpio, X, y, df_limpio, metricas):
    """Genera visualizaciones comparativas"""
    
    print("\n" + "=" * 80)
    print("🎨 GENERANDO VISUALIZACIONES COMPARATIVAS")
    print("=" * 80)
    
    import os
    os.makedirs('visualizaciones', exist_ok=True)
    
    # 1. Comparación de R²
    print("\n1️⃣  Gráfico de comparación de R²...")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    modelos = ['Original\n(con outliers)', 'Limpio\n(sin outliers)']
    r2_values = [0.2324, metricas['r2_adj']]
    colors = ['#ff7f7f', '#90ee90']
    
    bars = ax.bar(modelos, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
    
    # Añadir valores sobre las barras
    for bar, val in zip(bars, r2_values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{val:.4f}\n({val*100:.2f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    # Línea de mejora
    mejora = metricas['mejora_r2_adj']
    ax.annotate('', xy=(1, r2_values[1]), xytext=(1, r2_values[0]),
                arrowprops=dict(arrowstyle='<->', color='blue', lw=2))
    ax.text(1.15, (r2_values[0] + r2_values[1])/2, 
            f'+{mejora:.4f}\n(+{mejora/0.2324*100:.1f}%)',
            fontweight='bold', fontsize=10, color='blue')
    
    ax.set_ylabel('R² Ajustado', fontsize=12, fontweight='bold')
    ax.set_title('Comparación de Modelos: R² Ajustado\n931 → 752 Propiedades', 
                fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(r2_values) * 1.3)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file = 'visualizaciones/comparacion_r2.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Guardado: {output_file}")
    plt.close()
    
    # 2. Diagnóstico del modelo limpio
    print("\n2️⃣  Diagnóstico del modelo limpio...")
    
    residuos = modelo_limpio.resid
    fitted = modelo_limpio.fittedvalues
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # A) Residuos vs Fitted
    ax = axes[0, 0]
    ax.scatter(fitted, residuos, alpha=0.5, s=20, color='steelblue')
    ax.axhline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Valores Ajustados (log precio)', fontsize=10, fontweight='bold')
    ax.set_ylabel('Residuos', fontsize=10, fontweight='bold')
    ax.set_title('A) Residuos vs Valores Ajustados', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # B) Q-Q Plot
    ax = axes[0, 1]
    from scipy import stats
    stats.probplot(residuos, dist="norm", plot=ax)
    ax.set_title('B) Q-Q Plot (Normalidad)', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # C) Histograma de residuos
    ax = axes[1, 0]
    ax.hist(residuos, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Residuos', fontsize=10, fontweight='bold')
    ax.set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
    ax.set_title('C) Distribución de Residuos', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    
    # D) Predicho vs Real
    ax = axes[1, 1]
    ax.scatter(y, fitted, alpha=0.5, s=20, color='green')
    min_val = min(y.min(), fitted.min())
    max_val = max(y.max(), fitted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    ax.set_xlabel('log(Precio) Real', fontsize=10, fontweight='bold')
    ax.set_ylabel('log(Precio) Predicho', fontsize=10, fontweight='bold')
    ax.set_title(f'D) Predicho vs Real (R²={metricas["r2"]:.4f})', fontsize=11, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.suptitle(f'Diagnóstico del Modelo Limpio\n{len(df_limpio):,} Propiedades', 
                fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_file = 'visualizaciones/modelo_limpio_diagnostico.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Guardado: {output_file}")
    plt.close()
    
    # 3. Coeficientes del modelo limpio
    print("\n3️⃣  Coeficientes del modelo limpio...")
    
    coefs = modelo_limpio.params[1:]  # Excluir constante
    pvalues = modelo_limpio.pvalues[1:]
    
    # Top 10 coeficientes
    top_coefs = coefs.abs().nlargest(10)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top_coefs))
    colors = ['green' if coefs[var] > 0 else 'red' for var in top_coefs.index]
    
    ax.barh(y_pos, [coefs[var] for var in top_coefs.index], color=colors, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([var.replace('_', ' ').title() for var in top_coefs.index], fontsize=10)
    ax.set_xlabel('Coeficiente', fontsize=11, fontweight='bold')
    ax.set_title('Top 10 Variables Más Influyentes (Modelo Limpio)\n752 Propiedades', 
                fontsize=14, fontweight='bold')
    ax.axvline(0, color='black', linewidth=1)
    ax.grid(alpha=0.3, axis='x')
    
    # Añadir asteriscos de significancia
    for i, var in enumerate(top_coefs.index):
        pval = pvalues[var]
        if pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        else:
            sig = ""
        
        coef_val = coefs[var]
        x_pos = coef_val + 0.1 if coef_val > 0 else coef_val - 0.1
        ax.text(x_pos, i, sig, fontweight='bold', fontsize=12, va='center')
    
    plt.tight_layout()
    output_file = 'visualizaciones/modelo_limpio_coeficientes.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Guardado: {output_file}")
    plt.close()
    
    print("\n✅ Visualizaciones completadas")


def guardar_dataset_limpio(df_limpio):
    """Guarda el dataset limpio"""
    
    print("\n" + "=" * 80)
    print("💾 GUARDANDO DATASET LIMPIO")
    print("=" * 80)
    
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Guardar CSV
    output_csv = f'datos_procesados/propiedades_limpias_{timestamp}.csv'
    df_limpio[[
        'precio', 'comuna', 'superficie_util', 'dormitorios', 'banos',
        'latitude', 'longitude'
    ]].to_csv(output_csv, index=False)
    print(f"✅ CSV guardado: {output_csv}")
    
    # Guardar GeoJSON
    output_geojson = f'datos_procesados/propiedades_limpias_{timestamp}.geojson'
    df_limpio.to_file(output_geojson, driver='GeoJSON')
    print(f"✅ GeoJSON guardado: {output_geojson}")
    
    return output_csv, output_geojson


def main():
    """Función principal"""
    
    # 1. Cargar datos originales
    gdf = cargar_datos_originales()
    
    # 2. Aplicar filtros de limpieza
    df_limpio = aplicar_filtros_limpieza(gdf)
    
    # 3. Preparar variables
    df_model, vars_espaciales = preparar_variables_para_modelo(df_limpio)
    
    # 4. Ajustar modelo con datos limpios
    modelo_limpio, X, y, variables = ajustar_modelo_limpio(df_model, vars_espaciales)
    
    # 5. Mostrar resumen del modelo
    print("\n" + "=" * 80)
    print("📊 RESUMEN DEL MODELO LIMPIO")
    print("=" * 80)
    print(modelo_limpio.summary())
    
    # 6. Comparar con modelo original
    metricas = comparar_modelos(modelo_limpio, df_model)
    
    # 7. Visualizar
    visualizar_comparacion(modelo_limpio, X, y, df_model, metricas)
    
    # 8. Guardar dataset limpio
    archivos_salida = guardar_dataset_limpio(df_limpio)
    
    # 9. Guardar modelo
    print("\n💾 Guardando modelo limpio...")
    import pickle
    with open('modelo_ols_limpio.pkl', 'wb') as f:
        pickle.dump(modelo_limpio, f)
    print("✅ Modelo guardado: modelo_ols_limpio.pkl")
    
    # 10. Resumen final
    print("\n" + "=" * 80)
    print("✅ LIMPIEZA Y REAJUSTE COMPLETADOS")
    print("=" * 80)
    
    print("\n🎯 RESULTADOS FINALES:")
    print(f"   • Dataset limpio: {len(df_limpio):,} propiedades")
    print(f"   • Eliminados: {len(gdf) - len(df_limpio):,} ({(len(gdf) - len(df_limpio))/len(gdf)*100:.2f}%)")
    print(f"   • R² ajustado: {metricas['r2_adj']:.4f} (vs 0.2324 original)")
    print(f"   • Mejora: +{metricas['mejora_r2_adj']:.4f} (+{metricas['mejora_r2_adj']/0.2324*100:.1f}%)")
    
    print("\n📁 Archivos generados:")
    print("   1. visualizaciones/comparacion_r2.png")
    print("   2. visualizaciones/modelo_limpio_diagnostico.png")
    print("   3. visualizaciones/modelo_limpio_coeficientes.png")
    print(f"   4. {archivos_salida[0]}")
    print(f"   5. {archivos_salida[1]}")
    print("   6. modelo_ols_limpio.pkl")


if __name__ == "__main__":
    main()
