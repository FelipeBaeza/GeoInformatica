#!/usr/bin/env python3
"""
Script para calcular Índice de Moran Global (autocorrelación espacial)
para todas las características espaciales de la grilla

Autor: Proyecto GeoInformática - Semana 3
Fecha: Noviembre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from libpysal import weights
from esda.moran import Moran
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def cargar_grilla_completa():
    """
    Carga la grilla con todas las características calculadas
    """
    print("=" * 80)
    print("CARGANDO DATOS DE LA GRILLA")
    print("=" * 80)
    
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.dirname(script_dir)
    grilla_path = os.path.join(base_dir, "semana2_caracteristicas_espaciales", "features", "grilla_con_densidades.geojson")
    
    try:
        grilla = gpd.read_file(grilla_path)
        print(f"✓ Grilla cargada exitosamente")
        print(f"  - Puntos: {len(grilla)}")
        print(f"  - Columnas: {len(grilla.columns)}")
        print(f"  - CRS: {grilla.crs}")
        return grilla
    except Exception as e:
        print(f"✗ Error al cargar grilla: {e}")
        return None

def crear_matriz_pesos_espaciales(grilla, tipo='queen', k=8):
    """
    Crea matriz de pesos espaciales W
    
    Parámetros:
    - tipo: 'queen', 'rook', 'knn', 'distance'
    - k: número de vecinos para KNN
    """
    print("\n" + "=" * 80)
    print(f"CREANDO MATRIZ DE PESOS ESPACIALES ({tipo.upper()})")
    print("=" * 80)
    
    try:
        if tipo == 'queen':
            # Contigüidad Queen (8 vecinos)
            w = weights.Queen.from_dataframe(grilla, use_index=True)
            print(f"✓ Matriz Queen creada")
            
        elif tipo == 'rook':
            # Contigüidad Rook (4 vecinos)
            w = weights.Rook.from_dataframe(grilla, use_index=True)
            print(f"✓ Matriz Rook creada")
            
        elif tipo == 'knn':
            # K-vecinos más cercanos
            w = weights.KNN.from_dataframe(grilla, k=k)
            print(f"✓ Matriz KNN creada (k={k})")
            
        elif tipo == 'distance':
            # Distancia umbral (800 metros)
            w = weights.DistanceBand.from_dataframe(grilla, threshold=800, binary=True)
            print(f"✓ Matriz Distance Band creada (800m)")
        
        # Normalizar por filas
        w.transform = 'r'
        
        # Estadísticas
        print(f"  - Observaciones: {w.n}")
        print(f"  - Conexiones promedio: {w.mean_neighbors:.2f}")
        print(f"  - Conexiones mín/máx: {w.min_neighbors}/{w.max_neighbors}")
        
        return w
        
    except Exception as e:
        print(f"✗ Error al crear matriz de pesos: {e}")
        return None

def calcular_moran_global(grilla, variable, w):
    """
    Calcula el Índice de Moran Global para una variable
    
    Retorna:
    - I: Índice de Moran
    - p: p-valor
    - z: z-score
    - interpretación
    """
    try:
        # Eliminar NaN
        valores = grilla[variable].dropna()
        indices_validos = valores.index
        
        if len(valores) < 30:
            return None
        
        # Filtrar matriz de pesos
        w_filtrado = weights.w_subset(w, indices_validos)
        
        # Calcular Moran I
        moran = Moran(valores.values, w_filtrado, permutations=999)
        
        # Interpretación
        if moran.p_sim < 0.01:
            significancia = "Altamente significativo (p < 0.01)"
        elif moran.p_sim < 0.05:
            significancia = "Significativo (p < 0.05)"
        elif moran.p_sim < 0.10:
            significancia = "Marginalmente significativo (p < 0.10)"
        else:
            significancia = "No significativo (p >= 0.10)"
        
        if moran.I > 0:
            patron = "Clustering positivo (valores similares se agrupan)"
        elif moran.I < 0:
            patron = "Dispersión (valores diferentes se agrupan)"
        else:
            patron = "Aleatorio"
        
        return {
            'variable': variable,
            'I': float(moran.I),
            'EI': float(moran.EI),  # Valor esperado bajo H0
            'VI': float(moran.VI_norm) if hasattr(moran, 'VI_norm') else float(moran.VI_rand) if hasattr(moran, 'VI_rand') else np.nan,  # Varianza
            'z': float(moran.z_sim),
            'p': float(moran.p_sim),
            'significancia': significancia,
            'patron': patron,
            'n_observaciones': len(valores)
        }
        
    except Exception as e:
        print(f"  ✗ Error calculando Moran para {variable}: {e}")
        return None

def analizar_todas_caracteristicas(grilla, w):
    """
    Calcula Índice de Moran para todas las características numéricas
    """
    print("\n" + "=" * 80)
    print("ANÁLISIS DE AUTOCORRELACIÓN GLOBAL - TODAS LAS CARACTERÍSTICAS")
    print("=" * 80)
    
    # Identificar columnas numéricas (excluir geometría y metadatos)
    columnas_excluir = ['geometry', 'punto_id', 'comuna', 'x', 'y']
    columnas_numericas = [col for col in grilla.columns 
                         if col not in columnas_excluir 
                         and grilla[col].dtype in ['float64', 'int64']]
    
    print(f"\n📊 Analizando {len(columnas_numericas)} características numéricas...")
    print("-" * 80)
    
    resultados = []
    
    for i, variable in enumerate(columnas_numericas, 1):
        print(f"\n[{i}/{len(columnas_numericas)}] {variable}")
        
        resultado = calcular_moran_global(grilla, variable, w)
        
        if resultado:
            resultados.append(resultado)
            print(f"  I = {resultado['I']:.4f} | p = {resultado['p']:.4f} | {resultado['significancia']}")
            print(f"  → {resultado['patron']}")
    
    return pd.DataFrame(resultados)

def generar_moran_scatterplot(grilla, variable, w, output_path):
    """
    Genera Moran Scatterplot para una variable
    """
    from esda.moran import Moran
    import matplotlib.pyplot as plt
    
    valores = grilla[variable].dropna()
    indices_validos = valores.index
    w_filtrado = weights.w_subset(w, indices_validos)
    
    moran = Moran(valores.values, w_filtrado)
    
    # Crear figura
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Subplot 1: Distribución simulada de I
    ax1.hist(moran.sim, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(moran.I, color='red', linestyle='--', linewidth=2, label=f'I observado = {moran.I:.4f}')
    ax1.axvline(moran.EI, color='green', linestyle='--', linewidth=2, label=f'E[I] = {moran.EI:.4f}')
    ax1.set_xlabel('Índice de Moran I', fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.set_title(f'Distribución de I bajo H₀\n(999 permutaciones)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # Subplot 2: Moran Scatterplot
    lag = weights.lag_spatial(w_filtrado, valores.values)
    
    # Estandarizar
    valores_std = (valores.values - valores.mean()) / valores.std()
    lag_std = (lag - lag.mean()) / lag.std()
    
    # Scatter
    ax2.scatter(valores_std, lag_std, alpha=0.5, s=30, color='steelblue')
    
    # Línea de regresión
    from scipy.stats import linregress
    slope, intercept, r_value, p_value, std_err = linregress(valores_std, lag_std)
    x_line = np.linspace(valores_std.min(), valores_std.max(), 100)
    y_line = slope * x_line + intercept
    ax2.plot(x_line, y_line, 'r-', linewidth=2, label=f'Pendiente = {slope:.4f}')
    
    # Líneas de cuadrante
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    
    # Etiquetas de cuadrantes
    ax2.text(0.7, 0.7, 'HH\n(Hot spot)', transform=ax2.transAxes, 
             fontsize=10, alpha=0.6, ha='center')
    ax2.text(0.3, 0.3, 'LL\n(Cold spot)', transform=ax2.transAxes, 
             fontsize=10, alpha=0.6, ha='center')
    ax2.text(0.3, 0.7, 'LH\n(Outlier)', transform=ax2.transAxes, 
             fontsize=10, alpha=0.6, ha='center')
    ax2.text(0.7, 0.3, 'HL\n(Outlier)', transform=ax2.transAxes, 
             fontsize=10, alpha=0.6, ha='center')
    
    ax2.set_xlabel('Valores estandarizados', fontsize=12)
    ax2.set_ylabel('Lag espacial (vecinos)', fontsize=12)
    ax2.set_title(f'Moran Scatterplot\nI = {moran.I:.4f} (p = {moran.p_sim:.4f})', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Moran Scatterplot guardado: {output_path}")

def generar_resumen_visual(resultados_df, output_path):
    """
    Genera visualización resumen de todos los resultados
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribución de valores I
    ax1 = axes[0, 0]
    resultados_df['I'].hist(bins=30, ax=ax1, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='I = 0 (aleatorio)')
    ax1.set_xlabel('Índice de Moran I', fontsize=12)
    ax1.set_ylabel('Frecuencia', fontsize=12)
    ax1.set_title('Distribución de Índice de Moran I\n(Todas las características)', 
                  fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. I vs p-valor
    ax2 = axes[0, 1]
    colores = ['red' if p < 0.05 else 'gray' for p in resultados_df['p']]
    ax2.scatter(resultados_df['I'], resultados_df['p'], c=colores, alpha=0.6, s=50)
    ax2.axhline(0.05, color='red', linestyle='--', linewidth=2, label='p = 0.05')
    ax2.axvline(0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Índice de Moran I', fontsize=12)
    ax2.set_ylabel('p-valor', fontsize=12)
    ax2.set_title('Índice de Moran I vs Significancia Estadística', 
                  fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')
    
    # 3. Top 10 características con mayor autocorrelación positiva
    ax3 = axes[1, 0]
    top_positivo = resultados_df.nlargest(10, 'I')
    colores_p = ['darkred' if p < 0.01 else 'red' if p < 0.05 else 'orange' 
                 for p in top_positivo['p']]
    y_pos = np.arange(len(top_positivo))
    ax3.barh(y_pos, top_positivo['I'], color=colores_p, alpha=0.7, edgecolor='black')
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels([var[:30] + '...' if len(var) > 30 else var 
                         for var in top_positivo['variable']], fontsize=9)
    ax3.set_xlabel('Índice de Moran I', fontsize=12)
    ax3.set_title('Top 10: Mayor Clustering Espacial Positivo', 
                  fontsize=14, fontweight='bold')
    ax3.grid(alpha=0.3, axis='x')
    
    # 4. Significancia estadística
    ax4 = axes[1, 1]
    significancia_counts = resultados_df['significancia'].value_counts()
    colores_sig = {'Altamente significativo (p < 0.01)': 'darkred',
                   'Significativo (p < 0.05)': 'red',
                   'Marginalmente significativo (p < 0.10)': 'orange',
                   'No significativo (p >= 0.10)': 'gray'}
    colors = [colores_sig.get(cat, 'gray') for cat in significancia_counts.index]
    ax4.pie(significancia_counts.values, labels=significancia_counts.index, 
            autopct='%1.1f%%', startangle=90, colors=colors)
    ax4.set_title('Distribución de Significancia Estadística', 
                  fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\n✓ Resumen visual guardado: {output_path}")

def generar_reporte_json(resultados_df, w, output_path):
    """
    Genera reporte JSON con todos los resultados
    """
    reporte = {
        'metadata': {
            'fecha_analisis': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'descripcion': 'Análisis de Autocorrelación Espacial Global (Índice de Moran)',
            'tipo_matriz_pesos': 'Queen',
            'n_observaciones': w.n,
            'conexiones_promedio': float(w.mean_neighbors)
        },
        'resumen_estadistico': {
            'total_caracteristicas': len(resultados_df),
            'altamente_significativas': len(resultados_df[resultados_df['p'] < 0.01]),
            'significativas': len(resultados_df[resultados_df['p'] < 0.05]),
            'marginalmente_significativas': len(resultados_df[resultados_df['p'] < 0.10]),
            'no_significativas': len(resultados_df[resultados_df['p'] >= 0.10]),
            'clustering_positivo': len(resultados_df[resultados_df['I'] > 0]),
            'dispersion_negativa': len(resultados_df[resultados_df['I'] < 0]),
            'I_promedio': float(resultados_df['I'].mean()),
            'I_mediana': float(resultados_df['I'].median()),
            'I_max': float(resultados_df['I'].max()),
            'I_min': float(resultados_df['I'].min())
        },
        'top_clustering_positivo': resultados_df.nlargest(15, 'I').to_dict('records'),
        'top_dispersion_negativa': resultados_df.nsmallest(5, 'I').to_dict('records'),
        'resultados_completos': resultados_df.to_dict('records')
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Reporte JSON guardado: {output_path}")

def generar_reporte_markdown(resultados_df, w, output_path):
    """
    Genera reporte en formato Markdown legible
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Autocorrelación Espacial Global\n")
        f.write("## Índice de Moran - Análisis Completo\n\n")
        
        f.write(f"**Fecha de análisis:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("---\n\n")
        f.write("## 📊 Resumen Ejecutivo\n\n")
        
        sig = resultados_df[resultados_df['p'] < 0.05]
        f.write(f"- **Total de características analizadas:** {len(resultados_df)}\n")
        f.write(f"- **Características con autocorrelación significativa (p < 0.05):** {len(sig)} ({len(sig)/len(resultados_df)*100:.1f}%)\n")
        f.write(f"- **Índice de Moran promedio:** {resultados_df['I'].mean():.4f}\n")
        f.write(f"- **Rango de I:** [{resultados_df['I'].min():.4f}, {resultados_df['I'].max():.4f}]\n\n")
        
        if len(sig) > 0:
            f.write("### 🔴 **CONCLUSIÓN PRINCIPAL:**\n")
            f.write(f"**{len(sig)} de {len(resultados_df)} características muestran AUTOCORRELACIÓN ESPACIAL SIGNIFICATIVA.**\n\n")
            f.write("Esto confirma que:\n")
            f.write("- ✅ Existe dependencia espacial en los datos\n")
            f.write("- ✅ Valores similares se agrupan geográficamente\n")
            f.write("- ✅ Se requieren modelos espaciales (GWR/MGWR)\n")
            f.write("- ✅ La ubicación SÍ importa para las preferencias\n\n")
        
        f.write("---\n\n")
        f.write("## 🏆 Top 15: Mayor Clustering Espacial Positivo\n\n")
        f.write("Características donde valores altos tienden a estar cerca de otros valores altos:\n\n")
        f.write("| # | Variable | I de Moran | p-valor | Significancia |\n")
        f.write("|---|----------|-----------|---------|---------------|\n")
        
        for i, row in enumerate(resultados_df.nlargest(15, 'I').itertuples(), 1):
            f.write(f"| {i} | {row.variable} | {row.I:.4f} | {row.p:.4f} | {row.significancia} |\n")
        
        f.write("\n---\n\n")
        f.write("## 📋 Resultados Completos\n\n")
        f.write("| Variable | I | p-valor | Patrón | Significancia |\n")
        f.write("|----------|---|---------|--------|---------------|\n")
        
        for row in resultados_df.sort_values('I', ascending=False).itertuples():
            f.write(f"| {row.variable} | {row.I:.4f} | {row.p:.4f} | {row.patron} | {row.significancia} |\n")
        
        f.write("\n---\n\n")
        f.write("## 📚 Interpretación del Índice de Moran (I)\n\n")
        f.write("- **I > 0:** Autocorrelación positiva (clustering) - valores similares se agrupan\n")
        f.write("- **I ≈ 0:** Patrón aleatorio - sin autocorrelación espacial\n")
        f.write("- **I < 0:** Autocorrelación negativa (dispersión) - valores diferentes se agrupan\n\n")
        f.write("**Significancia estadística:**\n")
        f.write("- p < 0.01: Altamente significativo ⭐⭐⭐\n")
        f.write("- p < 0.05: Significativo ⭐⭐\n")
        f.write("- p < 0.10: Marginalmente significativo ⭐\n")
        f.write("- p ≥ 0.10: No significativo\n\n")
    
    print(f"✓ Reporte Markdown guardado: {output_path}")

def main():
    """
    Función principal
    """
    import os
    
    # Determinar rutas absolutas
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reportes_dir = os.path.join(script_dir, "reportes")
    mapas_dir = os.path.join(script_dir, "mapas")
    
    # Crear directorios si no existen
    os.makedirs(reportes_dir, exist_ok=True)
    os.makedirs(mapas_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(" " * 20 + "ANÁLISIS DE AUTOCORRELACIÓN ESPACIAL GLOBAL")
    print(" " * 25 + "Índice de Moran - Semana 3")
    print("=" * 80 + "\n")
    
    # 1. Cargar datos
    grilla = cargar_grilla_completa()
    if grilla is None:
        return
    
    # 2. Crear matriz de pesos espaciales
    w = crear_matriz_pesos_espaciales(grilla, tipo='queen')
    if w is None:
        return
    
    # 3. Analizar todas las características
    resultados_df = analizar_todas_caracteristicas(grilla, w)
    
    # 4. Guardar resultados
    print("\n" + "=" * 80)
    print("GENERANDO REPORTES Y VISUALIZACIONES")
    print("=" * 80)
    
    # CSV
    csv_path = os.path.join(reportes_dir, "autocorrelacion_global_resultados.csv")
    resultados_df.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✓ CSV guardado: {csv_path}")
    
    # JSON
    json_path = os.path.join(reportes_dir, "autocorrelacion_global_reporte.json")
    generar_reporte_json(resultados_df, w, json_path)
    
    # Markdown
    md_path = os.path.join(reportes_dir, "autocorrelacion_global_reporte.md")
    generar_reporte_markdown(resultados_df, w, md_path)
    
    # Visualización resumen
    vis_path = os.path.join(mapas_dir, "autocorrelacion_global_resumen.png")
    generar_resumen_visual(resultados_df, vis_path)
    
    # 5. Generar Moran Scatterplots para top 5 características
    print("\n" + "=" * 80)
    print("GENERANDO MORAN SCATTERPLOTS PARA TOP 5 CARACTERÍSTICAS")
    print("=" * 80)
    
    top5 = resultados_df.nlargest(5, 'I')
    
    for i, row in enumerate(top5.itertuples(), 1):
        print(f"\n[{i}/5] Generando scatterplot para: {row.variable}")
        output_path = os.path.join(mapas_dir, f"moran_scatterplot_{row.variable.replace('/', '_')}.png")
        generar_moran_scatterplot(grilla, row.variable, w, output_path)
    
    # 6. Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    
    sig_05 = resultados_df[resultados_df['p'] < 0.05]
    sig_01 = resultados_df[resultados_df['p'] < 0.01]
    
    print(f"\n📊 RESULTADOS:")
    print(f"  - Total características analizadas: {len(resultados_df)}")
    print(f"  - Altamente significativas (p < 0.01): {len(sig_01)} ({len(sig_01)/len(resultados_df)*100:.1f}%)")
    print(f"  - Significativas (p < 0.05): {len(sig_05)} ({len(sig_05)/len(resultados_df)*100:.1f}%)")
    print(f"  - Clustering positivo: {len(resultados_df[resultados_df['I'] > 0])}")
    print(f"  - I promedio: {resultados_df['I'].mean():.4f}")
    print(f"  - I máximo: {resultados_df['I'].max():.4f}")
    
    if len(sig_05) > 0:
        print(f"\n✅ CONCLUSIÓN:")
        print(f"   {len(sig_05)} características muestran AUTOCORRELACIÓN ESPACIAL SIGNIFICATIVA.")
        print(f"   Esto confirma que la ubicación SÍ afecta las características del entorno.")
        print(f"   Se recomienda usar modelos espaciales (GWR/MGWR) en lugar de OLS simple.")
    else:
        print(f"\n⚠️  NOTA:")
        print(f"   No se encontró autocorrelación espacial significativa.")
        print(f"   Los datos parecen distribuirse aleatoriamente en el espacio.")
    
    print("\n" + "=" * 80)
    print("ANÁLISIS COMPLETADO EXITOSAMENTE")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
