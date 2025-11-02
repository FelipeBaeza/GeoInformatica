#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VALIDACIÓN DE AUTOCORRELACIÓN ESPACIAL DE RESIDUOS
===================================================

Objetivo:
    Verificar si los residuos del modelo OLS limpio presentan 
    autocorrelación espacial. Si hay autocorrelación significativa,
    indica que el modelo OLS no captura toda la estructura espacial
    y se requeriría un modelo espacial (GWR, SAR, CAR).

Análisis:
    1. Calcular Moran's I global en los residuos
    2. Generar Moran scatterplot
    3. Mapear distribución espacial de residuos
    4. Identificar clusters de residuos (LISA)
    5. Decidir: ¿OLS suficiente o necesitamos GWR?

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

# PySAL para análisis espacial
from libpysal.weights import KNN
from esda.moran import Moran, Moran_Local
from splot.esda import plot_moran, moran_scatterplot, lisa_cluster

# Configuración de visualización
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 12)
plt.rcParams['font.size'] = 10

# Directorios
BASE_DIR = Path(__file__).parent
DATOS_DIR = BASE_DIR / "datos_procesados"
VIZ_DIR = BASE_DIR / "visualizaciones"
VIZ_DIR.mkdir(exist_ok=True)

def cargar_datos_y_modelo():
    """
    Carga el dataset limpio y el modelo OLS ajustado
    """
    print("\n" + "="*70)
    print("📂 CARGANDO DATOS Y MODELO")
    print("="*70)
    
    # Buscar el archivo GeoJSON más reciente
    archivos_geojson = list(DATOS_DIR.glob("propiedades_limpias_*.geojson"))
    
    if not archivos_geojson:
        raise FileNotFoundError("❌ No se encontró dataset limpio")
    
    archivo_mas_reciente = max(archivos_geojson, key=lambda x: x.stat().st_mtime)
    
    print(f"📂 Cargando: {archivo_mas_reciente.name}")
    gdf = gpd.read_file(archivo_mas_reciente)
    print(f"✅ Cargado: {len(gdf):,} propiedades")
    
    # Cargar modelo
    modelo_path = BASE_DIR / "modelo_ols_limpio.pkl"
    print(f"\n📂 Cargando modelo: {modelo_path.name}")
    
    with open(modelo_path, 'rb') as f:
        modelo = pickle.load(f)
    
    print(f"✅ Modelo cargado")
    print(f"   • R²: {modelo.rsquared:.4f}")
    print(f"   • Variables: {len(modelo.params)}")
    
    return gdf, modelo

def calcular_residuos(gdf, modelo):
    """
    Calcula residuos del modelo y los agrega al GeoDataFrame
    """
    print("\n" + "="*70)
    print("🔧 CALCULANDO RESIDUOS DEL MODELO")
    print("="*70)
    
    # Los residuos ya están en el modelo
    gdf['residuos'] = modelo.resid
    gdf['residuos_studentizados'] = modelo.get_influence().resid_studentized_internal
    
    print(f"✅ Residuos calculados")
    print(f"   • Media: {gdf['residuos'].mean():.6f} (debe ser ≈0)")
    print(f"   • Desviación estándar: {gdf['residuos'].std():.4f}")
    print(f"   • Mínimo: {gdf['residuos'].min():.4f}")
    print(f"   • Máximo: {gdf['residuos'].max():.4f}")
    
    return gdf

def calcular_moran_global(gdf):
    """
    Calcula Moran's I global para los residuos
    """
    print("\n" + "="*70)
    print("📊 AUTOCORRELACIÓN GLOBAL (MORAN'S I)")
    print("="*70)
    
    # Crear matriz de pesos espaciales usando distancia threshold
    print("\n1️⃣  Construyendo matriz de pesos espaciales (Distance Band)...")
    
    from libpysal.weights import DistanceBand
    
    coords = np.column_stack([gdf.geometry.x, gdf.geometry.y])
    
    # Usar umbral adaptativo basado en percentil 10 de distancias
    from scipy.spatial.distance import pdist
    distances = pdist(coords)
    threshold = np.percentile(distances, 5)  # 5% de distancias más cortas
    
    print(f"   • Umbral de distancia: {threshold:.2f} metros")
    
    w = DistanceBand(coords, threshold=threshold, silence_warnings=True)
    w.transform = 'r'  # Row-standardized
    
    # Verificar conectividad
    if not w.n_components == 1:
        print(f"   ⚠️  Advertencia: {w.n_components} componentes desconectados")
        # Intentar con un umbral más grande
        threshold = np.percentile(distances, 10)
        print(f"   • Intentando con umbral mayor: {threshold:.2f} metros")
        w = DistanceBand(coords, threshold=threshold, silence_warnings=True)
        w.transform = 'r'
    
    print(f"✅ Matriz construida: {w.n} observaciones")
    print(f"   • Promedio de vecinos: {w.mean_neighbors:.1f}")
    
    # Calcular Moran's I
    print("\n2️⃣  Calculando Moran's I en residuos...")
    moran = Moran(gdf['residuos'], w, permutations=999)  # Reducir permutaciones para velocidad
    
    print(f"\n📊 RESULTADOS:")
    print(f"   • Moran's I: {moran.I:.4f}")
    print(f"   • Valor esperado: {moran.EI:.4f}")
    
    # Usar z_sim en lugar de calcular desde VI
    try:
        print(f"   • Z-score: {moran.z_sim:.4f}")
    except:
        print(f"   • Z-score: {moran.z_norm:.4f}")
    
    print(f"   • p-value: {moran.p_sim:.4f}")
    
    # Interpretación
    print(f"\n🔍 INTERPRETACIÓN:")
    
    if moran.p_sim < 0.001:
        significancia = "*** (p<0.001)"
    elif moran.p_sim < 0.01:
        significancia = "** (p<0.01)"
    elif moran.p_sim < 0.05:
        significancia = "* (p<0.05)"
    else:
        significancia = "no significativo (p≥0.05)"
    
    print(f"   • Significancia: {significancia}")
    
    if moran.p_sim < 0.05:
        if abs(moran.I) < 0.1:
            intensidad = "DÉBIL"
            accion = "⚠️  Considerar modelo espacial"
        elif abs(moran.I) < 0.3:
            intensidad = "MODERADA"
            accion = "⚠️  Se recomienda modelo espacial (GWR/SAR)"
        else:
            intensidad = "FUERTE"
            accion = "🚨 REQUIERE modelo espacial (GWR/SAR/CAR)"
        
        print(f"   • Intensidad: {intensidad}")
        print(f"   • Acción: {accion}")
        print(f"\n❌ CONCLUSIÓN: Los residuos muestran autocorrelación espacial")
        print(f"              El modelo OLS NO captura toda la estructura espacial")
        
        decision = "REQUIERE_MODELO_ESPACIAL"
    else:
        print(f"   • Intensidad: No hay autocorrelación")
        print(f"   • Acción: ✅ OLS es suficiente")
        print(f"\n✅ CONCLUSIÓN: Los residuos NO muestran autocorrelación espacial")
        print(f"              El modelo OLS captura adecuadamente la estructura espacial")
        
        decision = "OLS_SUFICIENTE"
    
    return moran, w, decision

def calcular_lisa(gdf, w):
    """
    Identifica clusters locales de residuos (LISA)
    """
    print("\n" + "="*70)
    print("📍 AUTOCORRELACIÓN LOCAL (LISA)")
    print("="*70)
    
    print("\n⏳ Calculando LISA en residuos...")
    lisa = Moran_Local(gdf['residuos'], w, permutations=9999)
    
    # Agregar resultados al GeoDataFrame
    gdf['lisa_cluster'] = lisa.q
    gdf['lisa_pvalue'] = lisa.p_sim
    gdf['lisa_significativo'] = lisa.p_sim < 0.05
    
    # Contar clusters significativos
    n_significativos = gdf['lisa_significativo'].sum()
    pct_significativos = (n_significativos / len(gdf)) * 100
    
    print(f"\n📊 RESULTADOS:")
    print(f"   • Observaciones significativas: {n_significativos} ({pct_significativos:.1f}%)")
    
    # Desglosar por tipo de cluster
    clusters = {
        1: "High-High (residuos altos rodeados de altos)",
        2: "Low-High (residuos bajos rodeados de altos)",
        3: "Low-Low (residuos bajos rodeados de bajos)",
        4: "High-Low (residuos altos rodeados de bajos)"
    }
    
    print(f"\n   Desglose por tipo de cluster:")
    for tipo, descripcion in clusters.items():
        n = ((gdf['lisa_cluster'] == tipo) & gdf['lisa_significativo']).sum()
        pct = (n / len(gdf)) * 100
        print(f"   • {descripcion}: {n} ({pct:.1f}%)")
    
    if pct_significativos > 10:
        print(f"\n⚠️  Más del 10% de observaciones muestran clustering local")
        print(f"    → Indica estructura espacial no capturada por OLS")
    elif pct_significativos > 5:
        print(f"\n⚠️  Entre 5-10% de observaciones muestran clustering")
        print(f"    → Considerar modelo espacial")
    else:
        print(f"\n✅ Menos del 5% de observaciones muestran clustering")
        print(f"    → OLS probablemente suficiente")
    
    return lisa, gdf

def visualizar_moran_scatterplot(gdf, moran, w):
    """
    Genera Moran scatterplot de residuos
    """
    print("\n1️⃣  Generando Moran scatterplot...")
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Calcular lag espacial
    lag_residuos = w.sparse.dot(gdf['residuos'])
    
    # Scatter plot
    ax.scatter(gdf['residuos'], lag_residuos, alpha=0.5, s=30)
    
    # Línea de regresión
    m, b = np.polyfit(gdf['residuos'], lag_residuos, 1)
    x_line = np.array([gdf['residuos'].min(), gdf['residuos'].max()])
    y_line = m * x_line + b
    ax.plot(x_line, y_line, 'r--', linewidth=2, label=f"Pendiente = {m:.3f}")
    
    # Líneas de cuadrantes
    ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
    ax.axvline(0, color='k', linestyle='-', linewidth=0.5)
    
    # Etiquetas de cuadrantes
    ax.text(0.95, 0.95, 'HH', transform=ax.transAxes, fontsize=14, 
            verticalalignment='top', horizontalalignment='right', color='red', weight='bold')
    ax.text(0.05, 0.95, 'LH', transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='left', color='blue', weight='bold')
    ax.text(0.05, 0.05, 'LL', transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='left', color='red', weight='bold')
    ax.text(0.95, 0.05, 'HL', transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='right', color='blue', weight='bold')
    
    ax.set_xlabel('Residuos', fontsize=12)
    ax.set_ylabel('Residuos Espacialmente Retrasados (Lag)', fontsize=12)
    ax.set_title(f"Moran Scatterplot - Residuos del Modelo OLS\nMoran's I = {moran.I:.4f} (p = {moran.p_sim:.4f})", 
                 fontsize=14, weight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    archivo = VIZ_DIR / "moran_scatterplot_residuos.png"
    plt.savefig(archivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Guardado: {archivo.name}")

def visualizar_mapa_residuos(gdf):
    """
    Mapa de distribución espacial de residuos
    """
    print("\n2️⃣  Generando mapa de residuos...")
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Mapa 1: Residuos continuos
    ax1 = axes[0]
    gdf.plot(column='residuos', 
             cmap='RdBu_r',
             legend=True,
             ax=ax1,
             edgecolor='black',
             linewidth=0.2,
             vmin=-gdf['residuos'].abs().max(),
             vmax=gdf['residuos'].abs().max())
    
    ax1.set_title('Distribución Espacial de Residuos\n(Azul=Sobrepredicción, Rojo=Subpredicción)',
                  fontsize=14, weight='bold')
    ax1.axis('off')
    
    # Mapa 2: Residuos categorizados
    ax2 = axes[1]
    
    # Categorizar residuos
    gdf['residuo_cat'] = pd.cut(gdf['residuos'], 
                                 bins=[-np.inf, -1, -0.5, 0.5, 1, np.inf],
                                 labels=['Muy negativo\n(<-1)',
                                        'Negativo\n(-1 a -0.5)',
                                        'Neutro\n(-0.5 a 0.5)',
                                        'Positivo\n(0.5 a 1)',
                                        'Muy positivo\n(>1)'])
    
    gdf.plot(column='residuo_cat',
             categorical=True,
             legend=True,
             ax=ax2,
             edgecolor='black',
             linewidth=0.2,
             cmap='RdBu_r')
    
    ax2.set_title('Residuos Categorizados\n(5 Categorías)',
                  fontsize=14, weight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    
    archivo = VIZ_DIR / "mapa_residuos_espacial.png"
    plt.savefig(archivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Guardado: {archivo.name}")

def visualizar_lisa_clusters(gdf):
    """
    Mapa de clusters LISA de residuos
    """
    print("\n3️⃣  Generando mapa de clusters LISA...")
    
    fig, ax = plt.subplots(figsize=(15, 12))
    
    # Crear copia para visualización
    gdf_plot = gdf.copy()
    
    # Asignar colores según cluster y significancia
    gdf_plot['cluster_color'] = 0  # No significativo
    
    mask_sig = gdf_plot['lisa_significativo']
    
    gdf_plot.loc[mask_sig & (gdf_plot['lisa_cluster'] == 1), 'cluster_color'] = 1  # HH - Rojo
    gdf_plot.loc[mask_sig & (gdf_plot['lisa_cluster'] == 2), 'cluster_color'] = 2  # LH - Azul claro
    gdf_plot.loc[mask_sig & (gdf_plot['lisa_cluster'] == 3), 'cluster_color'] = 3  # LL - Azul
    gdf_plot.loc[mask_sig & (gdf_plot['lisa_cluster'] == 4), 'cluster_color'] = 4  # HL - Rosa
    
    # Definir colores
    colors = {
        0: 'lightgray',  # No significativo
        1: 'red',        # HH
        2: 'lightblue',  # LH
        3: 'blue',       # LL
        4: 'pink'        # HL
    }
    
    labels = {
        0: 'No significativo',
        1: 'High-High (HH)',
        2: 'Low-High (LH)',
        3: 'Low-Low (LL)',
        4: 'High-Low (HL)'
    }
    
    # Plotear cada categoría
    for cat, color in colors.items():
        mask = gdf_plot['cluster_color'] == cat
        if mask.any():
            gdf_plot[mask].plot(ax=ax, 
                               color=color,
                               edgecolor='black',
                               linewidth=0.2,
                               label=labels[cat])
    
    ax.set_title('Clusters LISA de Residuos del Modelo OLS\n(p<0.05)',
                 fontsize=16, weight='bold')
    ax.axis('off')
    ax.legend(loc='lower right', fontsize=12)
    
    # Estadísticas
    n_sig = gdf_plot['lisa_significativo'].sum()
    pct_sig = (n_sig / len(gdf_plot)) * 100
    
    textstr = f'Observaciones significativas: {n_sig} ({pct_sig:.1f}%)'
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    archivo = VIZ_DIR / "lisa_clusters_residuos.png"
    plt.savefig(archivo, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"   ✅ Guardado: {archivo.name}")

def visualizar_analisis_completo(gdf, moran, w):
    """
    Genera todas las visualizaciones
    """
    print("\n" + "="*70)
    print("🎨 GENERANDO VISUALIZACIONES")
    print("="*70)
    
    visualizar_moran_scatterplot(gdf, moran, w)
    visualizar_mapa_residuos(gdf)
    visualizar_lisa_clusters(gdf)
    
    print("\n✅ Visualizaciones completadas")

def generar_reporte_decision(moran, gdf, decision):
    """
    Genera reporte con la decisión final
    """
    print("\n" + "="*70)
    print("📄 GENERANDO REPORTE DE DECISIÓN")
    print("="*70)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archivo = BASE_DIR / f"decision_modelo_espacial_{timestamp}.txt"
    
    with open(archivo, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("VALIDACIÓN DE AUTOCORRELACIÓN ESPACIAL - DECISIÓN FINAL\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Observaciones analizadas: {len(gdf):,}\n\n")
        
        f.write("="*70 + "\n")
        f.write("RESULTADOS MORAN'S I GLOBAL\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"Moran's I:        {moran.I:.6f}\n")
        f.write(f"Valor esperado:   {moran.EI:.6f}\n")
        f.write(f"Z-score:          {moran.z_sim:.4f}\n")
        f.write(f"p-value:          {moran.p_sim:.6f}\n\n")
        
        f.write("="*70 + "\n")
        f.write("RESULTADOS LISA (AUTOCORRELACIÓN LOCAL)\n")
        f.write("="*70 + "\n\n")
        
        n_sig = gdf['lisa_significativo'].sum()
        pct_sig = (n_sig / len(gdf)) * 100
        
        f.write(f"Observaciones con clustering significativo: {n_sig} ({pct_sig:.1f}%)\n\n")
        
        clusters = {
            1: "High-High",
            2: "Low-High",
            3: "Low-Low",
            4: "High-Low"
        }
        
        for tipo, nombre in clusters.items():
            n = ((gdf['lisa_cluster'] == tipo) & gdf['lisa_significativo']).sum()
            pct = (n / len(gdf)) * 100
            f.write(f"  • {nombre}: {n} ({pct:.1f}%)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("DECISIÓN FINAL\n")
        f.write("="*70 + "\n\n")
        
        if decision == "OLS_SUFICIENTE":
            f.write("✅ DECISIÓN: MANTENER MODELO OLS\n\n")
            f.write("Justificación:\n")
            f.write("  • Los residuos NO muestran autocorrelación espacial significativa\n")
            f.write("  • El modelo OLS captura adecuadamente la estructura espacial\n")
            f.write("  • No se requiere modelo espacial más complejo\n\n")
            f.write("Próximos pasos:\n")
            f.write("  1. Documentar modelo final OLS\n")
            f.write("  2. Interpretar coeficientes en contexto inmobiliario\n")
            f.write("  3. Desarrollar sistema de recomendaciones\n")
        else:
            f.write("⚠️  DECISIÓN: SE REQUIERE MODELO ESPACIAL\n\n")
            f.write("Justificación:\n")
            f.write("  • Los residuos muestran autocorrelación espacial significativa\n")
            f.write("  • El modelo OLS NO captura toda la estructura espacial\n")
            f.write("  • Hay clustering de errores en el espacio\n\n")
            
            f.write("Modelos espaciales recomendados:\n\n")
            
            if abs(moran.I) < 0.3 and pct_sig < 15:
                f.write("  🔹 OPCIÓN 1 (Recomendada): Geographically Weighted Regression (GWR)\n")
                f.write("     → Permite que coeficientes varíen espacialmente\n")
                f.write("     → Captura heterogeneidad espacial local\n")
                f.write("     → Útil para entender efectos locales\n\n")
                
                f.write("  🔹 OPCIÓN 2: Spatial Lag Model (SAR)\n")
                f.write("     → Incluye variable dependiente retrasada espacialmente\n")
                f.write("     → Captura difusión espacial de precios\n")
                f.write("     → Más parsimonioso que GWR\n\n")
            else:
                f.write("  🔹 OPCIÓN 1 (Recomendada): Spatial Error Model (SEM/CAR)\n")
                f.write("     → Modela autocorrelación en los errores\n")
                f.write("     → Captura dependencia espacial residual\n")
                f.write("     → Apropiado para clustering fuerte\n\n")
                
                f.write("  🔹 OPCIÓN 2: Geographically Weighted Regression (GWR)\n")
                f.write("     → Permite coeficientes espacialmente variables\n")
                f.write("     → Más flexible pero más complejo\n\n")
            
            f.write("Próximos pasos:\n")
            f.write("  1. Implementar modelo espacial seleccionado\n")
            f.write("  2. Comparar ajuste con OLS (R², AIC, BIC)\n")
            f.write("  3. Verificar que residuos del modelo espacial no tengan autocorrelación\n")
            f.write("  4. Documentar modelo final\n")
    
    print(f"✅ Reporte guardado: {archivo.name}")
    
    return archivo

def main():
    """
    Función principal
    """
    print("\n" + "="*70)
    print("🔍 VALIDACIÓN DE AUTOCORRELACIÓN ESPACIAL DE RESIDUOS")
    print("="*70)
    print("\nObjetivo: Determinar si el modelo OLS captura toda la estructura")
    print("          espacial o si se requiere un modelo espacial (GWR/SAR/CAR)")
    
    try:
        # 1. Cargar datos y modelo
        gdf, modelo = cargar_datos_y_modelo()
        
        # 2. Calcular residuos
        gdf = calcular_residuos(gdf, modelo)
        
        # 3. Autocorrelación global
        moran, w, decision = calcular_moran_global(gdf)
        
        # 4. Autocorrelación local (LISA)
        lisa, gdf = calcular_lisa(gdf, w)
        
        # 5. Visualizaciones
        visualizar_analisis_completo(gdf, moran, w)
        
        # 6. Reporte de decisión
        archivo_reporte = generar_reporte_decision(moran, gdf, decision)
        
        # 7. Resumen final
        print("\n" + "="*70)
        print("✅ ANÁLISIS COMPLETADO")
        print("="*70)
        
        print("\n📊 RESUMEN:")
        print(f"   • Moran's I: {moran.I:.4f} (p={moran.p_sim:.4f})")
        
        n_sig = gdf['lisa_significativo'].sum()
        pct_sig = (n_sig / len(gdf)) * 100
        print(f"   • Clustering local: {n_sig} obs ({pct_sig:.1f}%)")
        
        print(f"\n🎯 DECISIÓN: {decision.replace('_', ' ')}")
        
        if decision == "OLS_SUFICIENTE":
            print("\n✅ El modelo OLS es suficiente")
            print("   → No se requiere modelo espacial")
            print("   → Proceder a documentación final")
        else:
            print("\n⚠️  Se requiere modelo espacial")
            print("   → OLS no captura toda la estructura espacial")
            print("   → Implementar GWR/SAR/CAR según reporte")
        
        print(f"\n📁 Archivos generados:")
        print(f"   1. visualizaciones/moran_scatterplot_residuos.png")
        print(f"   2. visualizaciones/mapa_residuos_espacial.png")
        print(f"   3. visualizaciones/lisa_clusters_residuos.png")
        print(f"   4. {archivo_reporte.name}")
        
        return decision
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    decision = main()
