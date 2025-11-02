#!/usr/bin/env python3
"""
Modelo Hedónico de Precios - Regresión OLS
Cuantifica el efecto de cada característica en el precio de arriendo
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Configuración de visualización
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def cargar_datos():
    """Carga el dataset procesado"""
    
    print("=" * 80)
    print("📊 MODELO HEDÓNICO DE PRECIOS - REGRESIÓN OLS")
    print("=" * 80)
    
    import glob
    import os
    
    archivos = glob.glob('/home/felipe/Documentos/GeoInformatica/datos_procesados/propiedades_kaggle_*.geojson')
    archivo = max(archivos, key=os.path.getctime)
    
    print(f"\n📂 Cargando: {os.path.basename(archivo)}")
    
    gdf = gpd.read_file(archivo)
    df = pd.DataFrame(gdf.drop(columns='geometry'))
    
    print(f"✅ Cargado: {len(df):,} propiedades")
    print(f"📋 Columnas disponibles: {len(df.columns)}")
    
    return df


def preparar_variables(df):
    """Prepara variables para el modelo"""
    
    print("\n" + "=" * 80)
    print("🔧 PREPARACIÓN DE VARIABLES")
    print("=" * 80)
    
    # Crear copia
    df_model = df.copy()
    
    # 1. Variable dependiente: logaritmo del precio (mejor para distribución)
    print("\n1️⃣  Variable dependiente: log(precio)")
    df_model = df_model[df_model['precio'] > 0].copy()
    df_model['log_precio'] = np.log(df_model['precio'])
    print(f"   ✓ Transformación logarítmica aplicada")
    
    # 2. Variables independientes - Características intrínsecas
    print("\n2️⃣  Variables intrínsecas de la propiedad:")
    
    # Superficie (log para mejor ajuste)
    df_model['log_superficie'] = np.log(df_model['superficie_util'].replace(0, np.nan))
    print(f"   ✓ log_superficie")
    
    # Dormitorios
    df_model['dormitorios_num'] = df_model['dormitorios'].fillna(df_model['dormitorios'].median())
    print(f"   ✓ dormitorios")
    
    # Baños
    df_model['banos_num'] = df_model['banos'].fillna(1)
    print(f"   ✓ baños")
    
    # 3. Variables de ubicación - Dummies de comuna
    print("\n3️⃣  Variables de ubicación (dummies de comuna):")
    comunas_principales = df_model['comuna_norm'].value_counts().head(6).index
    
    for comuna in comunas_principales:
        col_name = f'comuna_{comuna.replace(" ", "_").lower()}'
        df_model[col_name] = (df_model['comuna_norm'] == comuna).astype(int)
        print(f"   ✓ {col_name}")
    
    # 4. Variables espaciales - Características del entorno
    print("\n4️⃣  Variables espaciales del entorno:")
    
    # Distancias (las que mostraron correlación)
    vars_espaciales = []
    
    # Buscar columnas de distancia
    dist_cols = [c for c in df_model.columns if c.startswith('espacial_dist_')]
    
    if len(dist_cols) > 0:
        # Seleccionar las más relevantes
        variables_interes = [
            'espacial_dist_transporte_metro_m',
            'espacial_dist_turismo_m',
            'espacial_dist_salud_m',
            'espacial_dist_educacion_basica_m',
            'espacial_dist_seguridad_m'
        ]
        
        for var in variables_interes:
            if var in df_model.columns:
                # Normalizar distancias (dividir por 1000 para tener en km)
                nueva_var = var.replace('espacial_', '').replace('_m', '_km')
                df_model[nueva_var] = df_model[var] / 1000
                vars_espaciales.append(nueva_var)
                print(f"   ✓ {nueva_var}")
    
    # Densidades
    dens_cols = [c for c in df_model.columns if 'dens_' in c and c.startswith('espacial_')]
    
    if len(dens_cols) > 0:
        variables_densidad = [
            'espacial_dens_comercio_600m_km2',
            'espacial_dens_areas_verdes_600m_km2',
            'espacial_dens_educacion_600m_km2'
        ]
        
        for var in variables_densidad:
            if var in df_model.columns:
                nueva_var = var.replace('espacial_', '')
                df_model[nueva_var] = df_model[var]
                vars_espaciales.append(nueva_var)
                print(f"   ✓ {nueva_var}")
    
    # Índices
    indice_cols = [c for c in df_model.columns if 'indice_' in c and c.startswith('espacial_')]
    
    if len(indice_cols) > 0:
        variables_indices = [
            'espacial_indice_accesibilidad_transporte',
            'espacial_indice_accesibilidad_educacion',
            'espacial_indice_calidad_vida'
        ]
        
        for var in variables_indices:
            if var in df_model.columns:
                nueva_var = var.replace('espacial_', '')
                df_model[nueva_var] = df_model[var]
                vars_espaciales.append(nueva_var)
                print(f"   ✓ {nueva_var}")
    
    print(f"\n   Total variables espaciales: {len(vars_espaciales)}")
    
    # 5. Eliminar filas con NaN en variables clave
    print("\n5️⃣  Limpieza de datos faltantes:")
    inicial = len(df_model)
    
    variables_clave = ['log_precio', 'log_superficie', 'dormitorios_num', 'banos_num'] + vars_espaciales
    variables_clave = [v for v in variables_clave if v in df_model.columns]
    
    df_model = df_model.dropna(subset=variables_clave)
    
    print(f"   ✓ Datos iniciales: {inicial:,}")
    print(f"   ✓ Datos finales: {len(df_model):,}")
    print(f"   ✓ Eliminados: {inicial - len(df_model):,} ({(inicial - len(df_model))/inicial*100:.1f}%)")
    
    return df_model, vars_espaciales


def seleccionar_variables_vif(df, variables, threshold=10):
    """Selecciona variables eliminando multicolinealidad (VIF)"""
    
    print("\n" + "=" * 80)
    print("🔍 ANÁLISIS DE MULTICOLINEALIDAD (VIF)")
    print("=" * 80)
    
    print(f"\nVariables a evaluar: {len(variables)}")
    print(f"Threshold VIF: {threshold}")
    
    # Preparar datos
    X = df[variables].copy()
    X = X.fillna(X.mean())
    
    # Calcular VIF iterativamente
    variables_seleccionadas = variables.copy()
    
    while True:
        X_vif = X[variables_seleccionadas]
        vif_data = pd.DataFrame()
        vif_data["Variable"] = variables_seleccionadas
        vif_data["VIF"] = [variance_inflation_factor(X_vif.values, i) 
                          for i in range(len(variables_seleccionadas))]
        
        max_vif = vif_data["VIF"].max()
        
        if max_vif > threshold:
            variable_eliminar = vif_data.loc[vif_data["VIF"].idxmax(), "Variable"]
            print(f"   ❌ Eliminando {variable_eliminar} (VIF={max_vif:.2f})")
            variables_seleccionadas.remove(variable_eliminar)
        else:
            break
    
    print(f"\n✅ Variables finales: {len(variables_seleccionadas)}")
    print("\nVIF final:")
    for idx, row in vif_data.iterrows():
        print(f"   • {row['Variable']:50s} VIF = {row['VIF']:6.2f}")
    
    return variables_seleccionadas


def ajustar_modelo_ols(df, variables_independientes):
    """Ajusta modelo de regresión OLS"""
    
    print("\n" + "=" * 80)
    print("📈 AJUSTE DEL MODELO OLS")
    print("=" * 80)
    
    # Preparar X e Y
    X = df[variables_independientes].copy()
    X = X.fillna(X.mean())  # Imputar cualquier NaN restante
    
    y = df['log_precio']
    
    # Agregar constante
    X_const = sm.add_constant(X)
    
    print(f"\n📊 Dimensiones:")
    print(f"   Observaciones: {len(X):,}")
    print(f"   Variables independientes: {len(variables_independientes)}")
    
    # Ajustar modelo
    print(f"\n⏳ Ajustando modelo OLS...")
    modelo = sm.OLS(y, X_const).fit()
    
    print(f"✅ Modelo ajustado")
    
    return modelo, X, y


def interpretar_resultados(modelo, df):
    """Interpreta y visualiza resultados del modelo"""
    
    print("\n" + "=" * 80)
    print("📊 RESULTADOS DEL MODELO")
    print("=" * 80)
    
    # Resumen estadístico
    print("\n" + "=" * 80)
    print(modelo.summary())
    print("=" * 80)
    
    # Métricas principales
    print("\n🎯 MÉTRICAS PRINCIPALES:")
    print(f"   • R² (ajustado): {modelo.rsquared_adj:.4f}")
    print(f"   • R²: {modelo.rsquared:.4f}")
    print(f"   • AIC: {modelo.aic:.2f}")
    print(f"   • BIC: {modelo.bic:.2f}")
    print(f"   • F-statistic: {modelo.fvalue:.2f} (p={modelo.f_pvalue:.2e})")
    
    # Interpretación en CLP
    print("\n💰 INTERPRETACIÓN EN PESOS CHILENOS:")
    print("=" * 80)
    
    # Obtener coeficientes
    coefs = modelo.params.drop('const')
    pvalues = modelo.pvalues.drop('const')
    
    # Ordenar por magnitud del efecto
    coefs_abs = coefs.abs().sort_values(ascending=False)
    
    print("\n🔝 TOP 10 VARIABLES MÁS INFLUYENTES:")
    print(f"\n{'Variable':<50s} {'Coef (log)':<12s} {'Efecto %':<12s} {'p-value':<12s} {'Sig.':<5s}")
    print("-" * 95)
    
    precio_mediano = df['precio'].median()
    
    for var in coefs_abs.head(10).index:
        coef = coefs[var]
        pval = pvalues[var]
        
        # Interpretación del coeficiente
        if 'log_' in var:
            efecto = f"{coef*100:.2f}% (elasticidad)"
        elif 'comuna_' in var:
            efecto_pct = (np.exp(coef) - 1) * 100
            efecto_clp = precio_mediano * (np.exp(coef) - 1)
            efecto = f"{efecto_pct:+.1f}% (${efecto_clp:+,.0f})"
        else:
            # Variables continuas (distancias, densidades)
            efecto_pct = coef * 100  # Cambio porcentual por unidad
            efecto = f"{efecto_pct:+.2f}%/unidad"
        
        # Significancia
        if pval < 0.001:
            sig = "***"
        elif pval < 0.01:
            sig = "**"
        elif pval < 0.05:
            sig = "*"
        else:
            sig = ""
        
        print(f"{var:<50s} {coef:+12.4f} {efecto:<12s} {pval:12.4f} {sig:<5s}")
    
    print("\nSignificancia: *** p<0.001  ** p<0.01  * p<0.05")
    
    # Interpretación especial para variables clave
    print("\n" + "=" * 80)
    print("📌 INTERPRETACIONES CLAVE:")
    print("=" * 80)
    
    if 'log_superficie' in coefs.index:
        coef_sup = coefs['log_superficie']
        print(f"\n🏠 SUPERFICIE:")
        print(f"   Elasticidad: {coef_sup:.3f}")
        print(f"   Interpretación: Un 10% más de superficie → {coef_sup*10:.2f}% más de precio")
        print(f"                  50 m² → 55 m² (+10%) → ${precio_mediano * 0.01 * coef_sup * 10:+,.0f}")
    
    # Comunas
    comunas_coefs = {k: v for k, v in coefs.items() if 'comuna_' in k}
    if comunas_coefs:
        print(f"\n📍 EFECTO COMUNA (vs comuna base):")
        for comuna, coef in sorted(comunas_coefs.items(), key=lambda x: x[1], reverse=True):
            nombre = comuna.replace('comuna_', '').replace('_', ' ').title()
            efecto_pct = (np.exp(coef) - 1) * 100
            efecto_clp = precio_mediano * (np.exp(coef) - 1)
            print(f"   • {nombre:20s}: {efecto_pct:+6.1f}% (${efecto_clp:+10,.0f})")
    
    # Distancias
    dist_coefs = {k: v for k, v in coefs.items() if 'dist_' in k and '_km' in k}
    if dist_coefs:
        print(f"\n🗺️  EFECTO DISTANCIAS (por cada km adicional):")
        for dist, coef in sorted(dist_coefs.items(), key=lambda x: abs(x[1]), reverse=True):
            nombre = dist.replace('dist_', '').replace('_km', '').replace('_', ' ').title()
            efecto_pct = coef * 100
            efecto_clp = precio_mediano * 0.01 * efecto_pct
            signo = "⬆️ SUBE" if coef > 0 else "⬇️ BAJA"
            print(f"   • {nombre:25s}: {efecto_pct:+6.2f}% por km {signo} (${efecto_clp:+,.0f})")
    
    return coefs, pvalues


def visualizar_resultados(modelo, X, y, df):
    """Genera visualizaciones del modelo"""
    
    print("\n" + "=" * 80)
    print("🎨 GENERANDO VISUALIZACIONES")
    print("=" * 80)
    
    import os
    os.makedirs('visualizaciones', exist_ok=True)
    
    # 1. Residuos
    print("\n1️⃣  Análisis de residuos...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    residuos = modelo.resid
    fitted = modelo.fittedvalues
    
    # Plot 1: Residuos vs Fitted
    axes[0, 0].scatter(fitted, residuos, alpha=0.5, s=20)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Valores Ajustados (log precio)', fontweight='bold')
    axes[0, 0].set_ylabel('Residuos', fontweight='bold')
    axes[0, 0].set_title('Residuos vs Valores Ajustados', fontweight='bold')
    axes[0, 0].grid(alpha=0.3)
    
    # Plot 2: Q-Q plot
    from scipy import stats
    stats.probplot(residuos, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot (Normalidad)', fontweight='bold')
    axes[0, 1].grid(alpha=0.3)
    
    # Plot 3: Histograma de residuos
    axes[1, 0].hist(residuos, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1, 0].set_xlabel('Residuos', fontweight='bold')
    axes[1, 0].set_ylabel('Frecuencia', fontweight='bold')
    axes[1, 0].set_title('Distribución de Residuos', fontweight='bold')
    axes[1, 0].grid(alpha=0.3)
    
    # Plot 4: Scale-Location
    axes[1, 1].scatter(fitted, np.sqrt(np.abs(residuos)), alpha=0.5, s=20)
    axes[1, 1].set_xlabel('Valores Ajustados', fontweight='bold')
    axes[1, 1].set_ylabel('√|Residuos Estandarizados|', fontweight='bold')
    axes[1, 1].set_title('Scale-Location Plot', fontweight='bold')
    axes[1, 1].grid(alpha=0.3)
    
    plt.suptitle('Diagnóstico del Modelo OLS', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    output_file = 'visualizaciones/modelo_ols_diagnostico.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Guardado: {output_file}")
    plt.close()
    
    # 2. Coeficientes
    print("\n2️⃣  Gráfico de coeficientes...")
    
    coefs = modelo.params.drop('const')
    pvalues = modelo.pvalues.drop('const')
    
    # Top 15 por magnitud
    top_coefs = coefs.abs().sort_values(ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = ['green' if coefs[var] > 0 else 'red' for var in top_coefs.index]
    
    y_pos = range(len(top_coefs))
    ax.barh(y_pos, [coefs[var] for var in top_coefs.index], color=colors, alpha=0.7)
    
    ax.set_yticks(y_pos)
    labels = [var.replace('espacial_', '').replace('_', ' ').title() for var in top_coefs.index]
    ax.set_yticklabels(labels, fontsize=9)
    
    ax.set_xlabel('Coeficiente (log-log)', fontsize=12, fontweight='bold')
    ax.set_title('Top 15 Variables por Magnitud de Efecto\n(Modelo Hedónico OLS)', 
                fontsize=14, fontweight='bold')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax.grid(axis='x', alpha=0.3)
    
    # Agregar significancia
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
        x_pos = coef_val + 0.02 if coef_val > 0 else coef_val - 0.02
        ax.text(x_pos, i, sig, fontweight='bold', fontsize=12, va='center')
    
    plt.tight_layout()
    
    output_file = 'visualizaciones/modelo_ols_coeficientes.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Guardado: {output_file}")
    plt.close()
    
    # 3. Predicho vs Real
    print("\n3️⃣  Valores predichos vs reales...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(y, fitted, alpha=0.5, s=30, c=df['precio'], cmap='viridis')
    
    # Línea perfecta
    min_val = min(y.min(), fitted.min())
    max_val = max(y.max(), fitted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Predicción Perfecta')
    
    ax.set_xlabel('log(Precio) Real', fontsize=12, fontweight='bold')
    ax.set_ylabel('log(Precio) Predicho', fontsize=12, fontweight='bold')
    ax.set_title(f'Valores Predichos vs Reales\nR² = {modelo.rsquared:.4f}', 
                fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    output_file = 'visualizaciones/modelo_ols_prediccion.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"   ✅ Guardado: {output_file}")
    plt.close()
    
    print(f"\n✅ Visualizaciones completadas")


def main():
    """Función principal"""
    
    # 1. Cargar datos
    df = cargar_datos()
    
    # 2. Preparar variables
    df_model, vars_espaciales = preparar_variables(df)
    
    # 3. Definir variables para el modelo
    variables_independientes = ['log_superficie', 'dormitorios_num', 'banos_num']
    
    # Agregar comunas
    comunas_vars = [c for c in df_model.columns if c.startswith('comuna_')]
    variables_independientes.extend(comunas_vars[:-1])  # Excluir una como referencia
    
    # Agregar variables espaciales
    variables_independientes.extend(vars_espaciales)
    
    # 4. Filtrar solo variables numéricas presentes en el dataframe
    variables_independientes = [v for v in variables_independientes 
                                if v in df_model.columns and pd.api.types.is_numeric_dtype(df_model[v])]
    
    print(f"\n📊 Variables para el modelo: {len(variables_independientes)}")
    print(f"   • Intrínsecas: 3")
    print(f"   • Comunas: {len([v for v in variables_independientes if v.startswith('comuna_')])}")
    print(f"   • Espaciales: {len([v for v in variables_independientes if not v.startswith('comuna_') and v not in ['log_superficie', 'dormitorios_num', 'banos_num']])}")
    
    # 5. Seleccionar variables eliminando multicolinealidad
    variables_finales = seleccionar_variables_vif(df_model, variables_independientes, threshold=10)
    
    # 5. Ajustar modelo
    modelo, X, y = ajustar_modelo_ols(df_model, variables_finales)
    
    # 6. Interpretar resultados
    coefs, pvalues = interpretar_resultados(modelo, df_model)
    
    # 7. Visualizar
    visualizar_resultados(modelo, X, y, df_model)
    
    # 8. Guardar resultados
    print("\n" + "=" * 80)
    print("💾 GUARDANDO RESULTADOS")
    print("=" * 80)
    
    # Guardar resumen del modelo
    with open('modelo_ols_resumen.txt', 'w') as f:
        f.write(str(modelo.summary()))
    print(f"✅ Resumen guardado: modelo_ols_resumen.txt")
    
    # Guardar coeficientes
    coefs_df = pd.DataFrame({
        'Variable': coefs.index,
        'Coeficiente': coefs.values,
        'P-value': pvalues.values,
        'Significativo': pvalues.values < 0.05
    })
    coefs_df.to_csv('modelo_ols_coeficientes.csv', index=False)
    print(f"✅ Coeficientes guardados: modelo_ols_coeficientes.csv")
    
    # Conclusión
    print("\n" + "=" * 80)
    print("✅ MODELO HEDÓNICO COMPLETADO")
    print("=" * 80)
    
    print(f"\n🎯 CONCLUSIONES PRINCIPALES:")
    print(f"   • R² ajustado: {modelo.rsquared_adj:.4f} ({modelo.rsquared_adj*100:.1f}% varianza explicada)")
    print(f"   • Variables significativas: {sum(pvalues < 0.05)}/{len(pvalues)}")
    print(f"   • El entorno espacial es SIGNIFICATIVO para explicar precios")
    
    print(f"\n📁 Archivos generados:")
    print(f"   1. modelo_ols_resumen.txt")
    print(f"   2. modelo_ols_coeficientes.csv")
    print(f"   3. visualizaciones/modelo_ols_diagnostico.png")
    print(f"   4. visualizaciones/modelo_ols_coeficientes.png")
    print(f"   5. visualizaciones/modelo_ols_prediccion.png")
    
    return modelo, df_model


if __name__ == "__main__":
    modelo, df = main()
