#!/usr/bin/env python3
"""
Análisis Exploratorio Visual - Dataset Kaggle
Genera visualizaciones impactantes para presentación
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def cargar_datos():
    """Carga el dataset procesado"""
    
    print("=" * 80)
    print("📊 ANÁLISIS EXPLORATORIO VISUAL")
    print("=" * 80)
    
    # Buscar el archivo más reciente
    import glob
    import os
    
    # Buscar en el directorio raíz del proyecto
    archivos = glob.glob('/home/felipe/Documentos/GeoInformatica/datos_procesados/propiedades_kaggle_*.geojson')
    if not archivos:
        print("❌ No se encontró archivo procesado")
        return None
    
    archivo = max(archivos, key=os.path.getctime)
    print(f"\n📂 Cargando: {archivo}")
    
    gdf = gpd.read_file(archivo)
    print(f"✅ Cargado: {len(gdf):,} propiedades")
    
    return gdf


def mapa_calor_precios(gdf):
    """Mapa de calor de precios de arriendo"""
    
    print("\n🗺️  Generando mapa de calor de precios...")
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Convertir a lat/lon para visualización
    gdf_wgs84 = gdf.to_crs('EPSG:4326')
    
    # Mapa base con comunas
    gdf_wgs84.plot(ax=ax, alpha=0.3, color='lightgray', edgecolor='white', linewidth=0.5)
    
    # Scatter plot con precios
    scatter = ax.scatter(
        gdf_wgs84.geometry.x,
        gdf_wgs84.geometry.y,
        c=gdf['precio'],
        cmap='YlOrRd',
        s=50,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    
    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Precio Arriendo Mensual (CLP)', fontsize=12, fontweight='bold')
    cbar.ax.tick_params(labelsize=10)
    
    # Formato de precios en colorbar
    import matplotlib.ticker as ticker
    cbar.ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    
    # Título y etiquetas
    ax.set_title('Mapa de Calor: Precios de Arriendo en Santiago\n(Portal Inmobiliario Chile - Nov 2023)', 
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Longitud', fontsize=12)
    ax.set_ylabel('Latitud', fontsize=12)
    
    # Anotaciones de comunas principales
    comunas_coords = {
        'Santiago': (-70.65, -33.45),
        'Las Condes': (-70.57, -33.41),
        'Ñuñoa': (-70.60, -33.46),
        'Providencia': (-70.61, -33.43),
        'Estación Central': (-70.70, -33.47)
    }
    
    for comuna, (lon, lat) in comunas_coords.items():
        ax.annotate(comuna, xy=(lon, lat), fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.tight_layout()
    
    # Guardar
    output_file = 'visualizaciones/mapa_calor_precios.png'
    import os
    os.makedirs('visualizaciones', exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    
    plt.close()


def correlacion_precio_caracteristicas(gdf):
    """Matriz de correlación precio vs características espaciales"""
    
    print("\n📈 Generando matriz de correlación...")
    
    # Seleccionar características espaciales relevantes
    cols_espaciales = [c for c in gdf.columns if c.startswith('espacial_')]
    
    # Tomar las más importantes (distancias e índices)
    cols_interes = ['precio'] + [c for c in cols_espaciales if 
                                  'dist_' in c or 'indice_' in c or 'dens_' in c][:15]
    
    # Filtrar columnas que existen
    cols_interes = [c for c in cols_interes if c in gdf.columns]
    
    # Calcular correlaciones
    df_corr = gdf[cols_interes].corr()
    
    # Solo correlaciones con precio
    correlaciones_precio = df_corr['precio'].drop('precio').sort_values(ascending=False)
    
    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Subplot 1: Top correlaciones positivas y negativas
    top_n = 10
    top_correlaciones = pd.concat([
        correlaciones_precio.head(top_n//2),
        correlaciones_precio.tail(top_n//2)
    ])
    
    # Limpiar nombres para visualización
    nombres_limpios = [c.replace('espacial_', '').replace('_', ' ').title() 
                      for c in top_correlaciones.index]
    
    colors = ['green' if x > 0 else 'red' for x in top_correlaciones.values]
    
    ax1.barh(range(len(top_correlaciones)), top_correlaciones.values, color=colors, alpha=0.7)
    ax1.set_yticks(range(len(top_correlaciones)))
    ax1.set_yticklabels(nombres_limpios, fontsize=9)
    ax1.set_xlabel('Correlación con Precio', fontsize=12, fontweight='bold')
    ax1.set_title('Top 10 Características Correlacionadas con Precio\n(Positivas y Negativas)', 
                  fontsize=13, fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    ax1.grid(axis='x', alpha=0.3)
    
    # Agregar valores
    for i, v in enumerate(top_correlaciones.values):
        ax1.text(v + 0.01 if v > 0 else v - 0.01, i, f'{v:.3f}', 
                va='center', ha='left' if v > 0 else 'right', fontsize=8, fontweight='bold')
    
    # Subplot 2: Heatmap de correlaciones
    # Seleccionar subset para heatmap
    cols_heatmap = ['precio'] + list(correlaciones_precio.head(8).index)
    corr_matrix = gdf[cols_heatmap].corr()
    
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax2,
                vmin=-1, vmax=1)
    
    ax2.set_title('Matriz de Correlación\n(Precio + Top Características)', 
                  fontsize=13, fontweight='bold')
    
    # Limpiar labels del heatmap
    labels_y = [c.replace('espacial_', '').replace('_', ' ').title() 
                for c in cols_heatmap]
    ax2.set_yticklabels(labels_y, rotation=0, fontsize=8)
    ax2.set_xticklabels(labels_y, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    
    # Guardar
    output_file = 'visualizaciones/correlaciones_precio.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    
    plt.close()
    
    return correlaciones_precio


def distribucion_precios_por_comuna(gdf):
    """Box plots de precios por comuna"""
    
    print("\n📊 Generando distribución de precios por comuna...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Preparar datos
    comunas_top = gdf['comuna_norm'].value_counts().head(8).index
    df_plot = gdf[gdf['comuna_norm'].isin(comunas_top)].copy()
    
    # Subplot 1: Box plot
    df_plot_sorted = df_plot.sort_values('precio', ascending=False)
    order = df_plot_sorted.groupby('comuna_norm')['precio'].median().sort_values(ascending=False).index
    
    sns.boxplot(data=df_plot, y='comuna_norm', x='precio', order=order, 
                palette='Set2', ax=ax1)
    
    ax1.set_xlabel('Precio Arriendo Mensual (CLP)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Comuna', fontsize=12, fontweight='bold')
    ax1.set_title('Distribución de Precios por Comuna\n(Box Plot)', 
                  fontsize=13, fontweight='bold')
    
    # Formato eje X
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax1.grid(axis='x', alpha=0.3)
    
    # Subplot 2: Violin plot
    sns.violinplot(data=df_plot, y='comuna_norm', x='precio', order=order,
                   palette='Set2', ax=ax2, inner='quartile')
    
    ax2.set_xlabel('Precio Arriendo Mensual (CLP)', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Comuna', fontsize=12, fontweight='bold')
    ax2.set_title('Distribución de Precios por Comuna\n(Violin Plot)', 
                  fontsize=13, fontweight='bold')
    
    # Formato eje X
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    
    # Guardar
    output_file = 'visualizaciones/distribucion_precios_comuna.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    
    plt.close()


def scatter_precio_vs_caracteristicas(gdf):
    """Scatter plots de precio vs características clave"""
    
    print("\n📉 Generando scatter plots precio vs características...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Características a plotear
    caracteristicas = [
        ('superficie_util', 'Superficie Útil (m²)'),
        ('dormitorios', 'Número de Dormitorios'),
        ('espacial_dist_transporte_metro_m', 'Distancia al Metro (m)'),
        ('espacial_dens_comercio_600m_km2', 'Densidad Comercial 600m (por km²)'),
        ('espacial_dens_areas_verdes_600m_km2', 'Densidad Áreas Verdes 600m'),
        ('espacial_indice_accesibilidad_transporte', 'Índice Accesibilidad Transporte')
    ]
    
    for idx, (col, label) in enumerate(caracteristicas):
        if col not in gdf.columns:
            continue
        
        ax = axes[idx]
        
        # Scatter plot
        ax.scatter(gdf[col], gdf['precio'], alpha=0.5, s=30, c=gdf['precio'], 
                  cmap='viridis', edgecolors='black', linewidth=0.3)
        
        # Línea de tendencia
        mask = gdf[col].notna() & gdf['precio'].notna()
        if mask.sum() > 10:
            z = np.polyfit(gdf.loc[mask, col], gdf.loc[mask, 'precio'], 1)
            p = np.poly1d(z)
            x_line = np.linspace(gdf[col].min(), gdf[col].max(), 100)
            ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.8, label='Tendencia')
            
            # Calcular R²
            from scipy.stats import pearsonr
            r, p_val = pearsonr(gdf.loc[mask, col], gdf.loc[mask, 'precio'])
            ax.text(0.05, 0.95, f'r = {r:.3f}\np = {p_val:.3e}', 
                   transform=ax.transAxes, fontsize=9, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel(label, fontsize=10, fontweight='bold')
        ax.set_ylabel('Precio Arriendo (CLP)', fontsize=10, fontweight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
        ax.grid(alpha=0.3)
        ax.legend(loc='upper right', fontsize=8)
    
    plt.suptitle('Relación Precio vs Características Clave', 
                 fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout()
    
    # Guardar
    output_file = 'visualizaciones/scatter_precio_caracteristicas.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    
    plt.close()


def estadisticas_descriptivas(gdf):
    """Panel de estadísticas descriptivas"""
    
    print("\n📋 Generando panel de estadísticas...")
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Histograma de precios
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(gdf['precio'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
    ax1.axvline(gdf['precio'].median(), color='red', linestyle='--', linewidth=2, label='Mediana')
    ax1.axvline(gdf['precio'].mean(), color='green', linestyle='--', linewidth=2, label='Promedio')
    ax1.set_xlabel('Precio Arriendo Mensual (CLP)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Frecuencia', fontsize=11, fontweight='bold')
    ax1.set_title('Distribución de Precios de Arriendo', fontsize=12, fontweight='bold')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x/1000:.0f}K'))
    ax1.legend()
    ax1.grid(alpha=0.3)
    
    # 2. Estadísticas clave (texto)
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    
    stats_text = f"""
    📊 ESTADÍSTICAS CLAVE
    
    Total Propiedades: {len(gdf):,}
    
    💰 PRECIO:
    • Mínimo: ${gdf['precio'].min():,.0f}
    • Q1 (25%): ${gdf['precio'].quantile(0.25):,.0f}
    • Mediana: ${gdf['precio'].median():,.0f}
    • Q3 (75%): ${gdf['precio'].quantile(0.75):,.0f}
    • Máximo: ${gdf['precio'].max():,.0f}
    • Promedio: ${gdf['precio'].mean():,.0f}
    • Desv. Est.: ${gdf['precio'].std():,.0f}
    
    🏠 SUPERFICIE:
    • Promedio: {gdf['superficie_util'].mean():.1f} m²
    • Mediana: {gdf['superficie_util'].median():.1f} m²
    
    🛏️ DORMITORIOS:
    • Promedio: {gdf['dormitorios'].mean():.1f}
    • Moda: {gdf['dormitorios'].mode()[0]:.0f}
    """
    
    ax2.text(0.1, 0.5, stats_text, fontsize=9, verticalalignment='center',
            family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 3. Distribución de dormitorios
    ax3 = fig.add_subplot(gs[1, 0])
    dorms = gdf['dormitorios'].value_counts().sort_index()
    ax3.bar(dorms.index, dorms.values, color='coral', edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Número de Dormitorios', fontsize=10, fontweight='bold')
    ax3.set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
    ax3.set_title('Distribución de Dormitorios', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Distribución de superficie
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(gdf['superficie_util'], bins=30, color='lightgreen', edgecolor='black', alpha=0.7)
    ax4.axvline(gdf['superficie_util'].median(), color='red', linestyle='--', linewidth=2)
    ax4.set_xlabel('Superficie Útil (m²)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Frecuencia', fontsize=10, fontweight='bold')
    ax4.set_title('Distribución de Superficie', fontsize=11, fontweight='bold')
    ax4.grid(alpha=0.3)
    
    # 5. Top comunas
    ax5 = fig.add_subplot(gs[1, 2])
    top_comunas = gdf['comuna_norm'].value_counts().head(5)
    ax5.barh(range(len(top_comunas)), top_comunas.values, color='plum', edgecolor='black', alpha=0.7)
    ax5.set_yticks(range(len(top_comunas)))
    ax5.set_yticklabels(top_comunas.index, fontsize=9)
    ax5.set_xlabel('Número de Propiedades', fontsize=10, fontweight='bold')
    ax5.set_title('Top 5 Comunas', fontsize=11, fontweight='bold')
    ax5.grid(axis='x', alpha=0.3)
    
    # 6. Precio por m²
    ax6 = fig.add_subplot(gs[2, :])
    gdf['precio_m2'] = gdf['precio'] / gdf['superficie_util']
    gdf['precio_m2'] = gdf['precio_m2'].replace([np.inf, -np.inf], np.nan)
    
    comunas_plot = gdf['comuna_norm'].value_counts().head(8).index
    df_precio_m2 = gdf[gdf['comuna_norm'].isin(comunas_plot)]
    
    sns.boxplot(data=df_precio_m2, x='comuna_norm', y='precio_m2', palette='Set3', ax=ax6)
    ax6.set_xlabel('Comuna', fontsize=11, fontweight='bold')
    ax6.set_ylabel('Precio por m² (CLP/m²)', fontsize=11, fontweight='bold')
    ax6.set_title('Precio por Metro Cuadrado por Comuna', fontsize=12, fontweight='bold')
    ax6.tick_params(axis='x', rotation=45)
    ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    ax6.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Panel de Estadísticas Descriptivas - Dataset de Arriendos', 
                 fontsize=16, fontweight='bold', y=0.998)
    
    # Guardar
    output_file = 'visualizaciones/estadisticas_descriptivas.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ Guardado: {output_file}")
    
    plt.close()


def generar_reporte_visual():
    """Genera reporte visual completo"""
    
    print("\n" + "=" * 80)
    print("🎨 GENERANDO VISUALIZACIONES")
    print("=" * 80)
    
    # Cargar datos
    gdf = cargar_datos()
    
    if gdf is None:
        print("❌ No se pudo cargar el dataset")
        return
    
    # Generar visualizaciones
    mapa_calor_precios(gdf)
    correlaciones = correlacion_precio_caracteristicas(gdf)
    distribucion_precios_por_comuna(gdf)
    scatter_precio_vs_caracteristicas(gdf)
    estadisticas_descriptivas(gdf)
    
    # Resumen
    print("\n" + "=" * 80)
    print("✅ VISUALIZACIONES COMPLETADAS")
    print("=" * 80)
    
    print("\n📁 Archivos generados en carpeta 'visualizaciones/':")
    print("   1. mapa_calor_precios.png")
    print("   2. correlaciones_precio.png")
    print("   3. distribucion_precios_comuna.png")
    print("   4. scatter_precio_caracteristicas.png")
    print("   5. estadisticas_descriptivas.png")
    
    print("\n🔍 HALLAZGOS PRINCIPALES:")
    print("\n💡 Top 5 características más correlacionadas con precio:")
    for i, (feat, corr) in enumerate(correlaciones.head(5).items(), 1):
        feat_clean = feat.replace('espacial_', '').replace('_', ' ').title()
        print(f"   {i}. {feat_clean}: {corr:.3f}")
    
    print("\n📍 Precio promedio por comuna:")
    precios_comuna = gdf.groupby('comuna_norm')['precio'].mean().sort_values(ascending=False).head(5)
    for comuna, precio in precios_comuna.items():
        print(f"   • {comuna}: ${precio:,.0f}")
    
    print("\n" + "=" * 80)
    print("🎯 SIGUIENTE PASO: Modelo hedónico para cuantificar efectos")
    print("=" * 80)


if __name__ == "__main__":
    generar_reporte_visual()
