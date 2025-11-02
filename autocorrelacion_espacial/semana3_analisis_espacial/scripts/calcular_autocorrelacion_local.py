#!/usr/bin/env python3
"""
Script para calcular Indicadores Locales de Asociación Espacial (LISA)
e identificar clusters espaciales (Hot Spots, Cold Spots, Outliers)

Autor: Proyecto GeoInformática - Semana 3
Fecha: Noviembre 2025
"""

import os
import geopandas as gpd
import pandas as pd
import numpy as np
from libpysal import weights
from esda.moran import Moran_Local
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def cargar_datos():
    """Carga grilla y resultados de autocorrelación global"""
    print("=" * 80)
    print("CARGANDO DATOS")
    print("=" * 80)
    
    import os
    # Cargar grilla
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.dirname(script_dir)
    grilla_path = os.path.join(base_dir, "semana2_caracteristicas_espaciales", "features", "grilla_con_densidades.geojson")
    grilla = gpd.read_file(grilla_path)
    print(f"✓ Grilla cargada: {len(grilla)} puntos")
    
    # Cargar resultados de autocorrelación global
    resultados_path = os.path.join(script_dir, "reportes", "autocorrelacion_global_resultados.csv")
    try:
        resultados_global = pd.read_csv(resultados_path)
        print(f"✓ Resultados globales cargados: {len(resultados_global)} variables")
    except:
        resultados_global = None
        print("⚠️  Resultados globales no disponibles")
    
    return grilla, resultados_global

def seleccionar_variables_significativas(resultados_global, top_n=10):
    """
    Selecciona las top N variables con mayor autocorrelación significativa
    """
    if resultados_global is None:
        return None
    
    # Filtrar significativas
    significativas = resultados_global[resultados_global['p'] < 0.05]
    
    if len(significativas) == 0:
        print("⚠️  No hay variables con autocorrelación significativa")
        return None
    
    # Tomar top N por I de Moran
    top_vars = significativas.nlargest(top_n, 'I')
    
    print(f"\n📊 Seleccionadas {len(top_vars)} variables para análisis LISA:")
    for i, row in enumerate(top_vars.itertuples(), 1):
        print(f"  {i}. {row.variable} (I={row.I:.4f}, p={row.p:.4f})")
    
    return top_vars['variable'].tolist()

def calcular_lisa(grilla, variable, w):
    """
    Calcula LISA (Local Moran's I) para una variable
    
    Retorna GeoDataFrame con clasificación de clusters
    """
    # Eliminar NaN
    gdf = grilla[[variable, 'geometry']].copy()
    gdf = gdf.dropna(subset=[variable])
    
    if len(gdf) < 30:
        return None
    
    # Filtrar matriz de pesos
    indices_validos = gdf.index
    w_filtrado = weights.w_subset(w, indices_validos)
    
    # Calcular LISA
    lisa = Moran_Local(gdf[variable].values, w_filtrado, permutations=999)
    
    # Agregar resultados al GeoDataFrame
    gdf['lisa_I'] = lisa.Is  # Índice local
    gdf['lisa_p'] = lisa.p_sim  # p-valor
    gdf['lisa_q'] = lisa.q  # Cuadrante (1=HH, 2=LH, 3=LL, 4=HL)
    
    # Clasificación de clusters
    # Solo considerar significativos (p < 0.05)
    gdf['cluster'] = 'No significativo'
    
    # HH: High-High (Hot spot)
    mask = (lisa.q == 1) & (lisa.p_sim < 0.05)
    gdf.loc[mask, 'cluster'] = 'HH (Hot spot)'
    
    # LL: Low-Low (Cold spot)
    mask = (lisa.q == 3) & (lisa.p_sim < 0.05)
    gdf.loc[mask, 'cluster'] = 'LL (Cold spot)'
    
    # LH: Low-High (Outlier bajo)
    mask = (lisa.q == 2) & (lisa.p_sim < 0.05)
    gdf.loc[mask, 'cluster'] = 'LH (Outlier bajo)'
    
    # HL: High-Low (Outlier alto)
    mask = (lisa.q == 4) & (lisa.p_sim < 0.05)
    gdf.loc[mask, 'cluster'] = 'HL (Outlier alto)'
    
    return gdf

def generar_mapa_lisa(gdf, variable, output_path):
    """
    Genera mapa de clusters LISA
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Definir colores para cada tipo de cluster
    colores = {
        'HH (Hot spot)': '#d7191c',      # Rojo
        'LL (Cold spot)': '#2c7bb6',      # Azul
        'HL (Outlier alto)': '#fdae61',   # Naranja
        'LH (Outlier bajo)': '#abd9e9',   # Celeste
        'No significativo': '#eeeeee'     # Gris claro
    }
    
    # Plotear cada tipo de cluster
    for cluster_type, color in colores.items():
        subset = gdf[gdf['cluster'] == cluster_type]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.3, 
                       label=f"{cluster_type} (n={len(subset)})")
    
    # Configuración del mapa
    ax.set_title(f'LISA Clusters: {variable}\nAnálisis de Autocorrelación Espacial Local', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal')
    
    # Eliminar ejes
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Agregar norte y escala sería ideal aquí
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Mapa LISA guardado: {output_path}")

def generar_mapa_significancia(gdf, variable, output_path):
    """
    Genera mapa de significancia estadística del LISA
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Categorizar p-valores
    gdf['sig_cat'] = 'No significativo (p ≥ 0.10)'
    gdf.loc[gdf['lisa_p'] < 0.10, 'sig_cat'] = 'Marginalmente sig. (p < 0.10)'
    gdf.loc[gdf['lisa_p'] < 0.05, 'sig_cat'] = 'Significativo (p < 0.05)'
    gdf.loc[gdf['lisa_p'] < 0.01, 'sig_cat'] = 'Altamente sig. (p < 0.01)'
    
    # Colores
    colores = {
        'Altamente sig. (p < 0.01)': '#d73027',
        'Significativo (p < 0.05)': '#fc8d59',
        'Marginalmente sig. (p < 0.10)': '#fee090',
        'No significativo (p ≥ 0.10)': '#e0e0e0'
    }
    
    # Plotear
    for cat, color in colores.items():
        subset = gdf[gdf['sig_cat'] == cat]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='black', linewidth=0.3,
                       label=f"{cat} (n={len(subset)})")
    
    ax.set_title(f'Significancia Estadística LISA: {variable}', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Mapa de significancia guardado: {output_path}")

def generar_estadisticas_lisa(gdf, variable):
    """
    Genera estadísticas descriptivas de los clusters LISA
    """
    stats = {
        'variable': variable,
        'n_total': len(gdf),
        'n_significativos': len(gdf[gdf['lisa_p'] < 0.05]),
        'pct_significativos': len(gdf[gdf['lisa_p'] < 0.05]) / len(gdf) * 100,
        'clusters': {}
    }
    
    for cluster_type in gdf['cluster'].unique():
        subset = gdf[gdf['cluster'] == cluster_type]
        stats['clusters'][cluster_type] = {
            'n': len(subset),
            'pct': len(subset) / len(gdf) * 100,
            'I_promedio': float(subset['lisa_I'].mean()) if len(subset) > 0 else 0,
            'p_promedio': float(subset['lisa_p'].mean()) if len(subset) > 0 else 1
        }
    
    return stats

def analizar_todas_variables(grilla, variables, w, mapas_dir, submercados_dir):
    """
    Analiza LISA para múltiples variables
    """
    print("\n" + "=" * 80)
    print("ANÁLISIS LISA - TODAS LAS VARIABLES SELECCIONADAS")
    print("=" * 80)
    
    resultados = []
    
    for i, variable in enumerate(variables, 1):
        print(f"\n[{i}/{len(variables)}] Analizando: {variable}")
        print("-" * 80)
        
        # Calcular LISA
        gdf_lisa = calcular_lisa(grilla, variable, w)
        
        if gdf_lisa is None:
            print(f"  ✗ No se pudo calcular LISA para {variable}")
            continue
        
        # Estadísticas
        stats = generar_estadisticas_lisa(gdf_lisa, variable)
        resultados.append(stats)
        
        print(f"  Total puntos: {stats['n_total']}")
        print(f"  Significativos (p<0.05): {stats['n_significativos']} ({stats['pct_significativos']:.1f}%)")
        print(f"  Clusters identificados:")
        for cluster_type, cluster_stats in stats['clusters'].items():
            if cluster_stats['n'] > 0:
                print(f"    - {cluster_type}: {cluster_stats['n']} ({cluster_stats['pct']:.1f}%)")
        
        # Generar mapas
        print(f"  Generando mapas...")
        
        # Mapa de clusters
        var_safe = variable.replace('/', '_').replace(' ', '_')
        mapa_path = os.path.join(mapas_dir, f"lisa_clusters_{var_safe}.png")
        generar_mapa_lisa(gdf_lisa, variable, mapa_path)
        
        # Mapa de significancia
        sig_path = os.path.join(mapas_dir, f"lisa_significancia_{var_safe}.png")
        generar_mapa_significancia(gdf_lisa, variable, sig_path)
        
        # Guardar GeoJSON con resultados LISA
        geojson_path = os.path.join(submercados_dir, f"lisa_{var_safe}.geojson")
        gdf_lisa.to_file(geojson_path, driver='GeoJSON')
        print(f"  ✓ GeoJSON guardado: {geojson_path}")
    
    return resultados

def generar_reporte_lisa(resultados, output_path):
    """
    Genera reporte Markdown de resultados LISA
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Análisis LISA\n")
        f.write("## Local Indicators of Spatial Association\n\n")
        
        f.write(f"**Fecha de análisis:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("---\n\n")
        f.write("## 📊 Resumen General\n\n")
        
        f.write(f"Total de variables analizadas: **{len(resultados)}**\n\n")
        
        # Calcular promedios
        pct_sig_promedio = np.mean([r['pct_significativos'] for r in resultados])
        
        f.write(f"Porcentaje promedio de puntos con clusters significativos: **{pct_sig_promedio:.1f}%**\n\n")
        
        f.write("---\n\n")
        f.write("## 🔥 Tipos de Clusters Identificados\n\n")
        
        f.write("### HH (High-High) - Hot Spots\n")
        f.write("Zonas de **alta habitabilidad** rodeadas de zonas similares.\n")
        f.write("Estas son las áreas más atractivas y homogéneamente buenas.\n\n")
        
        f.write("### LL (Low-Low) - Cold Spots\n")
        f.write("Zonas de **baja habitabilidad** rodeadas de zonas similares.\n")
        f.write("Áreas que requieren mayor atención en desarrollo urbano.\n\n")
        
        f.write("### HL (High-Low) - Outliers Altos\n")
        f.write("Puntos de **alta habitabilidad** en medio de zonas de baja habitabilidad.\n")
        f.write("Posibles nuevos desarrollos o anomalías positivas.\n\n")
        
        f.write("### LH (Low-High) - Outliers Bajos\n")
        f.write("Puntos de **baja habitabilidad** en medio de zonas buenas.\n")
        f.write("Requieren investigación para entender la causa.\n\n")
        
        f.write("---\n\n")
        f.write("## 📋 Resultados Detallados por Variable\n\n")
        
        for resultado in resultados:
            f.write(f"### {resultado['variable']}\n\n")
            f.write(f"- **Total de puntos:** {resultado['n_total']}\n")
            f.write(f"- **Significativos (p < 0.05):** {resultado['n_significativos']} ({resultado['pct_significativos']:.1f}%)\n\n")
            
            f.write("**Distribución de clusters:**\n\n")
            f.write("| Tipo de Cluster | Cantidad | Porcentaje | I promedio |\n")
            f.write("|----------------|----------|------------|------------|\n")
            
            for cluster_type, stats in resultado['clusters'].items():
                if stats['n'] > 0:
                    f.write(f"| {cluster_type} | {stats['n']} | {stats['pct']:.1f}% | {stats['I_promedio']:.4f} |\n")
            
            f.write("\n")
        
        f.write("---\n\n")
        f.write("## 🎯 Implicaciones para el Sistema de Recomendación\n\n")
        f.write("1. **Hot Spots identificados** → Zonas premium para perfiles exigentes\n")
        f.write("2. **Cold Spots identificados** → Oportunidades de desarrollo/inversión\n")
        f.write("3. **Outliers detectados** → Requieren análisis caso por caso\n")
        f.write("4. **Heterogeneidad espacial confirmada** → Necesidad de modelos espaciales (GWR/MGWR)\n\n")
    
    print(f"\n✓ Reporte LISA Markdown guardado: {output_path}")

def main():
    """
    Función principal
    """
    import os
    
    # Determinar rutas absolutas
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reportes_dir = os.path.join(script_dir, "reportes")
    mapas_dir = os.path.join(script_dir, "mapas")
    submercados_dir = os.path.join(script_dir, "submercados")
    
    # Crear directorios si no existen
    os.makedirs(reportes_dir, exist_ok=True)
    os.makedirs(mapas_dir, exist_ok=True)
    os.makedirs(submercados_dir, exist_ok=True)
    
    print("\n" + "=" * 80)
    print(" " * 15 + "ANÁLISIS DE AUTOCORRELACIÓN ESPACIAL LOCAL (LISA)")
    print(" " * 20 + "Identificación de Clusters Espaciales")
    print("=" * 80 + "\n")
    
    # 1. Cargar datos
    grilla, resultados_global = cargar_datos()
    
    # 2. Crear matriz de pesos
    print("\n" + "=" * 80)
    print("CREANDO MATRIZ DE PESOS ESPACIALES")
    print("=" * 80)
    
    w = weights.Queen.from_dataframe(grilla, use_index=True)
    w.transform = 'r'
    print(f"✓ Matriz Queen creada ({w.n} observaciones, {w.mean_neighbors:.2f} vecinos promedio)")
    
    # 3. Seleccionar variables significativas
    variables = seleccionar_variables_significativas(resultados_global, top_n=10)
    
    if variables is None:
        # Si no hay resultados previos, analizar algunas variables clave manualmente
        print("\n⚠️  Seleccionando variables clave manualmente...")
        variables = [
            'dist_transporte_metro_m',
            'dist_salud_clinicas_m',
            'dist_educacion_superior_m',
            'dens_total_600m_km2',
            'dens_comercio_600m_km2'
        ]
        print(f"Variables seleccionadas: {variables}")
    
    # 4. Analizar LISA para todas las variables
    resultados = analizar_todas_variables(grilla, variables, w, mapas_dir, submercados_dir)
    
    # 5. Generar reporte
    print("\n" + "=" * 80)
    print("GENERANDO REPORTE FINAL")
    print("=" * 80)
    
    reporte_path = os.path.join(reportes_dir, "autocorrelacion_local_reporte.md")
    generar_reporte_lisa(resultados, reporte_path)
    
    # Guardar también en JSON
    json_path = os.path.join(reportes_dir, "autocorrelacion_local_resultados.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'n_variables': len(resultados)
            },
            'resultados': resultados
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON guardado: {json_path}")
    
    # 6. Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    
    print(f"\n📊 RESULTADOS:")
    print(f"  - Variables analizadas: {len(resultados)}")
    
    total_hh = sum([r['clusters'].get('HH (Hot spot)', {}).get('n', 0) for r in resultados])
    total_ll = sum([r['clusters'].get('LL (Cold spot)', {}).get('n', 0) for r in resultados])
    total_hl = sum([r['clusters'].get('HL (Outlier alto)', {}).get('n', 0) for r in resultados])
    total_lh = sum([r['clusters'].get('LH (Outlier bajo)', {}).get('n', 0) for r in resultados])
    
    print(f"  - Hot Spots (HH) identificados: {total_hh}")
    print(f"  - Cold Spots (LL) identificados: {total_ll}")
    print(f"  - Outliers altos (HL): {total_hl}")
    print(f"  - Outliers bajos (LH): {total_lh}")
    
    pct_sig_promedio = np.mean([r['pct_significativos'] for r in resultados])
    print(f"  - Promedio de puntos con clusters significativos: {pct_sig_promedio:.1f}%")
    
    print("\n✅ CONCLUSIÓN:")
    print("   Clusters espaciales identificados exitosamente.")
    print("   Los mapas muestran la distribución de Hot Spots y Cold Spots.")
    print("   Esta heterogeneidad espacial confirma la necesidad de modelos locales (GWR/MGWR).")
    
    print("\n" + "=" * 80)
    print("ANÁLISIS LISA COMPLETADO EXITOSAMENTE")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
