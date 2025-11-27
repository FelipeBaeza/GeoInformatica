#!/usr/bin/env python3
"""
Script 07: Visualizaciones Completas del Proyecto

REQUISITOS CUBIERTOS:
D. Visualizaciones (10%)

3 MAPAS TEMÁTICOS:
1. Mapa de ubicación del área de estudio
2. Mapa de datos principales (distribución de propiedades)
3. Mapa de análisis/resultado preliminar (satisfacción predicha)

5 GRÁFICOS ESTADÍSTICOS:
1. Histogramas de variables clave
2. Series temporales (distribución por fecha)
3. Correlaciones espaciales (matriz)
4. Diagramas de dispersión
5. Boxplots comparativos por comuna

1 VISUALIZACIÓN INTERACTIVA:
- Mapa interactivo con Folium

Autor: Proyecto GeoInformática
Fecha: Noviembre 2025
"""

import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuración
BASE_DIR = Path('/home/felipe/Documentos/GeoInformatica')
SEMANA2_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana2_caracteristicas_espaciales'
SEMANA3_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana3_modelo_satisfaccion'
DATA_DIR = SEMANA3_DIR / 'data'
GRAFICOS_DIR = SEMANA3_DIR / 'graficos'
RESULTADOS_DIR = SEMANA3_DIR / 'resultados' / 'modelo_mejorado'

GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

# Configurar estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

# Colores personalizados
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent': '#F18F01',
    'success': '#C73E1D',
    'neutral': '#6C757D'
}

print("=" * 80)
print("📊 GENERACIÓN DE VISUALIZACIONES COMPLETAS")
print("=" * 80)

# =============================================================================
# CARGAR DATOS
# =============================================================================
print("\n📂 Cargando datos...")

# Intentar cargar dataset mejorado, si no existe usar el original
try:
    df = pd.read_csv(RESULTADOS_DIR / 'propiedades_con_satisfaccion.csv')
    print(f"   ✓ Dataset mejorado cargado: {len(df)} propiedades")
    tiene_satisfaccion = 'satisfaccion_compuesta' in df.columns
except:
    df = pd.read_csv(DATA_DIR / 'propiedades_con_factores_espaciales.csv')
    print(f"   ✓ Dataset original cargado: {len(df)} propiedades")
    tiene_satisfaccion = False

# Calcular precio_m2 si no existe
if 'precio_m2' not in df.columns:
    df['precio_m2'] = df['precio'] / df['superficie_util'].replace(0, np.nan)

# Limpiar datos para visualización
df_viz = df[
    (df['precio_m2'] > 1000) & 
    (df['precio_m2'] < 100000) &
    (df['latitude'].notna()) &
    (df['longitude'].notna())
].copy()

print(f"   ✓ Datos para visualización: {len(df_viz)} propiedades")

# Cargar grilla
try:
    with open(SEMANA2_DIR / 'features' / 'grilla_con_indices.geojson') as f:
        grilla_data = json.load(f)
    grilla_records = []
    for feature in grilla_data['features']:
        props = feature['properties']
        if feature['geometry']:
            coords = feature['geometry']['coordinates']
            props['x'] = coords[0]
            props['y'] = coords[1]
        grilla_records.append(props)
    df_grilla = pd.DataFrame(grilla_records)
    print(f"   ✓ Grilla cargada: {len(df_grilla)} puntos")
except Exception as e:
    print(f"   ⚠️ No se pudo cargar grilla: {e}")
    df_grilla = None

# =============================================================================
# MAPA 1: UBICACIÓN DEL ÁREA DE ESTUDIO
# =============================================================================
print("\n🗺️ Generando Mapa 1: Ubicación del área de estudio...")

fig, ax = plt.subplots(figsize=(14, 10))

# Límites del área de estudio
lon_min, lon_max = df_viz['longitude'].min() - 0.02, df_viz['longitude'].max() + 0.02
lat_min, lat_max = df_viz['latitude'].min() - 0.02, df_viz['latitude'].max() + 0.02

# Fondo con grilla si existe
if df_grilla is not None and 'x' in df_grilla.columns:
    # Convertir coordenadas UTM a lat/lon aproximado (simplificado)
    ax.scatter(df_grilla['x'], df_grilla['y'], c='lightgray', s=1, alpha=0.3, label='Grilla de análisis')

# Scatter de propiedades por comuna
comunas = df_viz['comuna_left'].dropna().unique() if 'comuna_left' in df_viz.columns else []
colors_comunas = plt.cm.Set2(np.linspace(0, 1, max(len(comunas), 1)))

for i, comuna in enumerate(comunas[:8]):  # Máximo 8 comunas para legibilidad
    mask = df_viz['comuna_left'] == comuna
    ax.scatter(
        df_viz.loc[mask, 'longitude'],
        df_viz.loc[mask, 'latitude'],
        c=[colors_comunas[i]],
        s=30,
        alpha=0.7,
        label=comuna.title(),
        edgecolors='white',
        linewidths=0.3
    )

# Configuración
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel('Longitud', fontsize=12)
ax.set_ylabel('Latitud', fontsize=12)
ax.set_title('MAPA 1: Ubicación del Área de Estudio\nDistribución de Propiedades por Comuna - Santiago, Chile', 
             fontsize=14, fontweight='bold', pad=20)

# Leyenda
ax.legend(loc='upper left', fontsize=9, framealpha=0.9)

# Añadir escala aproximada
scalebar_lon = lon_min + 0.01
scalebar_lat = lat_min + 0.005
ax.plot([scalebar_lon, scalebar_lon + 0.01], [scalebar_lat, scalebar_lat], 'k-', linewidth=2)
ax.text(scalebar_lon + 0.005, scalebar_lat + 0.002, '~1 km', ha='center', fontsize=9)

# Norte
ax.annotate('N', xy=(lon_max - 0.01, lat_max - 0.01), fontsize=14, fontweight='bold',
            ha='center', va='center')
ax.annotate('↑', xy=(lon_max - 0.01, lat_max - 0.015), fontsize=16, ha='center')

# Info box
info_text = f"Área de estudio: Santiago, Chile\n"
info_text += f"Total propiedades: {len(df_viz):,}\n"
info_text += f"Comunas: {len(comunas)}\n"
info_text += f"Período: 2023"
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'mapa_01_ubicacion_area_estudio.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: mapa_01_ubicacion_area_estudio.png")

# =============================================================================
# MAPA 2: DATOS PRINCIPALES (PRECIO POR M2)
# =============================================================================
print("\n🗺️ Generando Mapa 2: Datos principales (Precio/m²)...")

fig, ax = plt.subplots(figsize=(14, 10))

# Crear colormap personalizado
cmap = LinearSegmentedColormap.from_list('precio', ['#2ecc71', '#f39c12', '#e74c3c'])

# Scatter con color por precio
scatter = ax.scatter(
    df_viz['longitude'],
    df_viz['latitude'],
    c=df_viz['precio_m2'],
    cmap=cmap,
    s=40,
    alpha=0.7,
    edgecolors='white',
    linewidths=0.3,
    vmin=df_viz['precio_m2'].quantile(0.05),
    vmax=df_viz['precio_m2'].quantile(0.95)
)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label('Precio por m² (CLP)', fontsize=11)

# Configuración
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel('Longitud', fontsize=12)
ax.set_ylabel('Latitud', fontsize=12)
ax.set_title('MAPA 2: Distribución Espacial del Precio por m²\nMercado de Alquileres - Santiago, Chile', 
             fontsize=14, fontweight='bold', pad=20)

# Estadísticas
stats_text = f"Estadísticas Precio/m²:\n"
stats_text += f"Media: ${df_viz['precio_m2'].mean():,.0f}\n"
stats_text += f"Mediana: ${df_viz['precio_m2'].median():,.0f}\n"
stats_text += f"Std: ${df_viz['precio_m2'].std():,.0f}"
props = dict(boxstyle='round', facecolor='white', alpha=0.9)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'mapa_02_precio_m2.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: mapa_02_precio_m2.png")

# =============================================================================
# MAPA 3: ANÁLISIS/RESULTADO (SATISFACCIÓN O HABITABILIDAD)
# =============================================================================
print("\n🗺️ Generando Mapa 3: Resultado de análisis...")

fig, ax = plt.subplots(figsize=(14, 10))

# Determinar variable a mapear
if tiene_satisfaccion and 'satisfaccion_compuesta' in df_viz.columns:
    map_var = 'satisfaccion_compuesta'
    map_title = 'Satisfacción Compuesta Predicha'
    cmap_name = 'RdYlGn'
elif 'idx_habitabilidad_global' in df_viz.columns:
    map_var = 'idx_habitabilidad_global'
    map_title = 'Índice de Habitabilidad Global'
    cmap_name = 'RdYlGn'
elif 'score_habitabilidad' in df_viz.columns:
    map_var = 'score_habitabilidad'
    map_title = 'Score de Habitabilidad'
    cmap_name = 'RdYlGn'
else:
    # Crear score aproximado basado en densidades
    dens_cols = [c for c in df_viz.columns if c.startswith('dens_') and '300m' in c]
    if dens_cols:
        df_viz['habitabilidad_aprox'] = df_viz[dens_cols].mean(axis=1)
        # Normalizar
        df_viz['habitabilidad_aprox'] = (df_viz['habitabilidad_aprox'] - df_viz['habitabilidad_aprox'].min()) / \
                                         (df_viz['habitabilidad_aprox'].max() - df_viz['habitabilidad_aprox'].min()) * 10
        map_var = 'habitabilidad_aprox'
        map_title = 'Habitabilidad Aproximada (basada en densidades)'
        cmap_name = 'RdYlGn'
    else:
        map_var = 'precio_m2'
        map_title = 'Precio/m² (sin índice de habitabilidad disponible)'
        cmap_name = 'viridis'

# Scatter
scatter = ax.scatter(
    df_viz['longitude'],
    df_viz['latitude'],
    c=df_viz[map_var].fillna(df_viz[map_var].median()),
    cmap=cmap_name,
    s=40,
    alpha=0.7,
    edgecolors='white',
    linewidths=0.3
)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, pad=0.02)
cbar.set_label(map_title, fontsize=11)

# Configuración
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)
ax.set_xlabel('Longitud', fontsize=12)
ax.set_ylabel('Latitud', fontsize=12)
ax.set_title(f'MAPA 3: Resultado del Análisis\n{map_title}', 
             fontsize=14, fontweight='bold', pad=20)

# Estadísticas
stats_text = f"Estadísticas {map_var}:\n"
stats_text += f"Media: {df_viz[map_var].mean():.2f}\n"
stats_text += f"Mediana: {df_viz[map_var].median():.2f}\n"
stats_text += f"Rango: [{df_viz[map_var].min():.2f}, {df_viz[map_var].max():.2f}]"
props = dict(boxstyle='round', facecolor='white', alpha=0.9)
ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'mapa_03_resultado_analisis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: mapa_03_resultado_analisis.png")

# =============================================================================
# GRÁFICO 1: HISTOGRAMAS DE VARIABLES CLAVE
# =============================================================================
print("\n📊 Generando Gráfico 1: Histogramas de variables clave...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))

variables_hist = [
    ('precio_m2', 'Precio por m² (CLP)', COLORS['primary']),
    ('superficie_util', 'Superficie Útil (m²)', COLORS['secondary']),
    ('dormitorios', 'Dormitorios', COLORS['accent']),
    ('banos', 'Baños', COLORS['success']),
    ('estacionamientos', 'Estacionamientos', COLORS['neutral']),
]

# Agregar satisfacción si existe
if tiene_satisfaccion:
    variables_hist.append(('satisfaccion_compuesta', 'Satisfacción Compuesta', '#2ecc71'))

for idx, (var, label, color) in enumerate(variables_hist[:6]):
    ax = axes.flat[idx]
    if var in df_viz.columns:
        data = df_viz[var].dropna()
        if var == 'precio_m2':
            data = data[data < data.quantile(0.95)]
        
        ax.hist(data, bins=30, color=color, edgecolor='white', alpha=0.8)
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {data.mean():.1f}')
        ax.axvline(data.median(), color='orange', linestyle=':', linewidth=2, label=f'Mediana: {data.median():.1f}')
        ax.set_xlabel(label)
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'Distribución de {label}')
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, f'{var}\nno disponible', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'Distribución de {label}')

# Si hay menos de 6 variables, ocultar axes vacíos
for idx in range(len(variables_hist), 6):
    axes.flat[idx].axis('off')

plt.suptitle('GRÁFICO 1: Histogramas de Variables Clave', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'grafico_01_histogramas.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: grafico_01_histogramas.png")

# =============================================================================
# GRÁFICO 2: SERIES TEMPORALES / DISTRIBUCIÓN POR FECHA
# =============================================================================
print("\n📊 Generando Gráfico 2: Distribución temporal...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Intentar parsear fechas
if 'fecha' in df_viz.columns or 'published_time' in df_viz.columns:
    date_col = 'fecha' if 'fecha' in df_viz.columns else 'published_time'
    try:
        df_viz['fecha_parsed'] = pd.to_datetime(df_viz[date_col], errors='coerce')
        df_viz['mes'] = df_viz['fecha_parsed'].dt.to_period('M')
        
        # Conteo por mes
        conteo_mes = df_viz.groupby('mes').size()
        
        ax1 = axes[0]
        conteo_mes.plot(kind='bar', ax=ax1, color=COLORS['primary'], edgecolor='white')
        ax1.set_xlabel('Mes')
        ax1.set_ylabel('Número de Propiedades')
        ax1.set_title('Publicaciones por Mes')
        ax1.tick_params(axis='x', rotation=45)
        
        # Precio promedio por mes
        ax2 = axes[1]
        precio_mes = df_viz.groupby('mes')['precio_m2'].mean()
        precio_mes.plot(kind='line', ax=ax2, color=COLORS['secondary'], marker='o', linewidth=2)
        ax2.set_xlabel('Mes')
        ax2.set_ylabel('Precio Promedio por m² (CLP)')
        ax2.set_title('Evolución del Precio Promedio')
        ax2.tick_params(axis='x', rotation=45)
        
    except Exception as e:
        # Si falla, mostrar distribución por comuna como alternativa
        ax1 = axes[0]
        df_viz['comuna_left'].value_counts().head(10).plot(kind='bar', ax=ax1, color=COLORS['primary'])
        ax1.set_xlabel('Comuna')
        ax1.set_ylabel('Número de Propiedades')
        ax1.set_title('Distribución por Comuna')
        ax1.tick_params(axis='x', rotation=45)
        
        ax2 = axes[1]
        df_viz.groupby('comuna_left')['precio_m2'].mean().sort_values(ascending=False).head(10).plot(
            kind='bar', ax=ax2, color=COLORS['secondary'])
        ax2.set_xlabel('Comuna')
        ax2.set_ylabel('Precio Promedio por m²')
        ax2.set_title('Precio Promedio por Comuna')
        ax2.tick_params(axis='x', rotation=45)
else:
    # Alternativa: distribución por comuna
    ax1 = axes[0]
    if 'comuna_left' in df_viz.columns:
        df_viz['comuna_left'].value_counts().head(10).plot(kind='bar', ax=ax1, color=COLORS['primary'])
        ax1.set_xlabel('Comuna')
        ax1.set_ylabel('Número de Propiedades')
        ax1.set_title('Distribución por Comuna')
        ax1.tick_params(axis='x', rotation=45)
    
    ax2 = axes[1]
    if 'comuna_left' in df_viz.columns:
        df_viz.groupby('comuna_left')['precio_m2'].mean().sort_values(ascending=False).head(10).plot(
            kind='bar', ax=ax2, color=COLORS['secondary'])
        ax2.set_xlabel('Comuna')
        ax2.set_ylabel('Precio Promedio por m²')
        ax2.set_title('Precio Promedio por Comuna')
        ax2.tick_params(axis='x', rotation=45)

plt.suptitle('GRÁFICO 2: Análisis Temporal / Por Comuna', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'grafico_02_temporal.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: grafico_02_temporal.png")

# =============================================================================
# GRÁFICO 3: MATRIZ DE CORRELACIONES
# =============================================================================
print("\n📊 Generando Gráfico 3: Matriz de correlaciones...")

fig, ax = plt.subplots(figsize=(14, 12))

# Seleccionar variables numéricas relevantes
vars_correlacion = ['precio_m2', 'superficie_util', 'dormitorios', 'banos', 'estacionamientos']

# Agregar algunas densidades
dens_cols = [c for c in df_viz.columns if c.startswith('dens_') and '300m' in c][:5]
vars_correlacion.extend(dens_cols)

# Agregar distancias
dist_cols = [c for c in df_viz.columns if c.startswith('dist_') and c.endswith('_m')][:5]
vars_correlacion.extend(dist_cols)

# Agregar satisfacción si existe
if tiene_satisfaccion:
    vars_correlacion.append('satisfaccion_compuesta')

# Filtrar solo las que existen
vars_correlacion = [v for v in vars_correlacion if v in df_viz.columns]

# Calcular matriz de correlación
corr_matrix = df_viz[vars_correlacion].corr()

# Crear heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, square=True, linewidths=0.5, ax=ax,
            annot_kws={'size': 8}, vmin=-1, vmax=1)

ax.set_title('GRÁFICO 3: Matriz de Correlaciones Espaciales\nVariables Principales y Factores Espaciales', 
             fontsize=14, fontweight='bold', pad=20)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'grafico_03_correlaciones.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: grafico_03_correlaciones.png")

# =============================================================================
# GRÁFICO 4: DIAGRAMAS DE DISPERSIÓN
# =============================================================================
print("\n📊 Generando Gráfico 4: Diagramas de dispersión...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. Precio vs Superficie
ax1 = axes[0, 0]
ax1.scatter(df_viz['superficie_util'], df_viz['precio_m2'], alpha=0.5, c=COLORS['primary'], s=20)
ax1.set_xlabel('Superficie Útil (m²)')
ax1.set_ylabel('Precio por m² (CLP)')
ax1.set_title('Precio/m² vs Superficie')
# Línea de tendencia
z = np.polyfit(df_viz['superficie_util'].dropna(), df_viz.loc[df_viz['superficie_util'].notna(), 'precio_m2'], 1)
p = np.poly1d(z)
x_line = np.linspace(df_viz['superficie_util'].min(), df_viz['superficie_util'].max(), 100)
ax1.plot(x_line, p(x_line), 'r--', linewidth=2, label='Tendencia')
ax1.legend()

# 2. Precio vs Dormitorios
ax2 = axes[0, 1]
df_viz.boxplot(column='precio_m2', by='dormitorios', ax=ax2)
ax2.set_xlabel('Número de Dormitorios')
ax2.set_ylabel('Precio por m² (CLP)')
ax2.set_title('Precio/m² por Número de Dormitorios')
plt.suptitle('')  # Quitar título automático de boxplot

# 3. Precio vs Distancia a Metro (si existe)
ax3 = axes[1, 0]
if 'dist_transporte_metro_m' in df_viz.columns:
    mask_dist = df_viz['dist_transporte_metro_m'] < 5000  # Solo < 5km
    ax3.scatter(df_viz.loc[mask_dist, 'dist_transporte_metro_m'], 
                df_viz.loc[mask_dist, 'precio_m2'], 
                alpha=0.5, c=COLORS['secondary'], s=20)
    ax3.set_xlabel('Distancia al Metro (m)')
    ax3.set_ylabel('Precio por m² (CLP)')
    ax3.set_title('Precio/m² vs Distancia al Metro')
else:
    ax3.text(0.5, 0.5, 'Distancia al metro\nno disponible', ha='center', va='center', transform=ax3.transAxes)

# 4. Satisfacción vs Precio (si existe) o Densidad vs Precio
ax4 = axes[1, 1]
if tiene_satisfaccion and 'satisfaccion_compuesta' in df_viz.columns:
    ax4.scatter(df_viz['precio_m2'], df_viz['satisfaccion_compuesta'], 
                alpha=0.5, c=COLORS['accent'], s=20)
    ax4.set_xlabel('Precio por m² (CLP)')
    ax4.set_ylabel('Satisfacción Compuesta')
    ax4.set_title('Satisfacción vs Precio/m²')
elif 'dens_total_300m_km2' in df_viz.columns:
    ax4.scatter(df_viz['dens_total_300m_km2'], df_viz['precio_m2'], 
                alpha=0.5, c=COLORS['accent'], s=20)
    ax4.set_xlabel('Densidad Total (servicios/km²)')
    ax4.set_ylabel('Precio por m² (CLP)')
    ax4.set_title('Precio/m² vs Densidad de Servicios')
else:
    ax4.scatter(df_viz['banos'], df_viz['precio_m2'], alpha=0.5, c=COLORS['accent'], s=20)
    ax4.set_xlabel('Número de Baños')
    ax4.set_ylabel('Precio por m² (CLP)')
    ax4.set_title('Precio/m² vs Baños')

plt.suptitle('GRÁFICO 4: Diagramas de Dispersión - Relaciones entre Variables', 
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'grafico_04_dispersion.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: grafico_04_dispersion.png")

# =============================================================================
# GRÁFICO 5: BOXPLOTS COMPARATIVOS POR COMUNA
# =============================================================================
print("\n📊 Generando Gráfico 5: Boxplots comparativos...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

if 'comuna_left' in df_viz.columns:
    # Top 8 comunas por cantidad
    top_comunas = df_viz['comuna_left'].value_counts().head(8).index.tolist()
    df_top = df_viz[df_viz['comuna_left'].isin(top_comunas)]
    
    # 1. Precio por comuna
    ax1 = axes[0, 0]
    df_top.boxplot(column='precio_m2', by='comuna_left', ax=ax1)
    ax1.set_xlabel('Comuna')
    ax1.set_ylabel('Precio por m² (CLP)')
    ax1.set_title('Distribución de Precio por Comuna')
    ax1.tick_params(axis='x', rotation=45)
    plt.suptitle('')
    
    # 2. Superficie por comuna
    ax2 = axes[0, 1]
    df_top.boxplot(column='superficie_util', by='comuna_left', ax=ax2)
    ax2.set_xlabel('Comuna')
    ax2.set_ylabel('Superficie Útil (m²)')
    ax2.set_title('Distribución de Superficie por Comuna')
    ax2.tick_params(axis='x', rotation=45)
    plt.suptitle('')
    
    # 3. Dormitorios por comuna
    ax3 = axes[1, 0]
    comuna_dorms = df_top.groupby('comuna_left')['dormitorios'].mean().sort_values(ascending=False)
    comuna_dorms.plot(kind='bar', ax=ax3, color=COLORS['primary'], edgecolor='white')
    ax3.set_xlabel('Comuna')
    ax3.set_ylabel('Promedio de Dormitorios')
    ax3.set_title('Dormitorios Promedio por Comuna')
    ax3.tick_params(axis='x', rotation=45)
    
    # 4. Satisfacción o Habitabilidad por comuna
    ax4 = axes[1, 1]
    if tiene_satisfaccion and 'satisfaccion_compuesta' in df_top.columns:
        df_top.boxplot(column='satisfaccion_compuesta', by='comuna_left', ax=ax4)
        ax4.set_ylabel('Satisfacción Compuesta')
        ax4.set_title('Distribución de Satisfacción por Comuna')
    elif 'idx_habitabilidad_global' in df_top.columns:
        df_top.boxplot(column='idx_habitabilidad_global', by='comuna_left', ax=ax4)
        ax4.set_ylabel('Índice de Habitabilidad')
        ax4.set_title('Distribución de Habitabilidad por Comuna')
    else:
        comuna_precio_m2 = df_top.groupby('comuna_left')['precio_m2'].median().sort_values(ascending=False)
        comuna_precio_m2.plot(kind='bar', ax=ax4, color=COLORS['secondary'], edgecolor='white')
        ax4.set_ylabel('Mediana Precio/m²')
        ax4.set_title('Mediana de Precio por Comuna')
    ax4.set_xlabel('Comuna')
    ax4.tick_params(axis='x', rotation=45)
    plt.suptitle('')

plt.suptitle('GRÁFICO 5: Análisis Comparativo por Comuna', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'grafico_05_boxplots_comuna.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: grafico_05_boxplots_comuna.png")

# =============================================================================
# VISUALIZACIÓN INTERACTIVA: MAPA FOLIUM
# =============================================================================
print("\n🌐 Generando Visualización Interactiva con Folium...")

try:
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    
    # Centro del mapa
    center_lat = df_viz['latitude'].mean()
    center_lon = df_viz['longitude'].mean()
    
    # Crear mapa base
    mapa = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='cartodbpositron'
    )
    
    # Agregar título
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 400px; 
                background-color: white; 
                border: 2px solid grey; 
                z-index: 9999; 
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;">
        <b>🏠 Mapa Interactivo de Propiedades</b><br>
        Sistema de Recomendación Inmobiliaria<br>
        <small>Haz clic en los marcadores para ver detalles</small>
    </div>
    '''
    mapa.get_root().html.add_child(folium.Element(title_html))
    
    # Crear cluster de marcadores
    marker_cluster = MarkerCluster(name='Propiedades').add_to(mapa)
    
    # Agregar marcadores (limitar a 500 para rendimiento)
    sample_df = df_viz.sample(min(500, len(df_viz)), random_state=42)
    
    for _, row in sample_df.iterrows():
        # Determinar color por precio
        precio = row['precio_m2']
        if precio < df_viz['precio_m2'].quantile(0.33):
            color = 'green'
            icon = 'home'
        elif precio < df_viz['precio_m2'].quantile(0.66):
            color = 'orange'
            icon = 'home'
        else:
            color = 'red'
            icon = 'home'
        
        # Popup con información
        popup_html = f"""
        <div style="width: 200px;">
            <h4 style="margin: 0; color: #2E86AB;">🏠 Propiedad</h4>
            <hr style="margin: 5px 0;">
            <b>Precio/m²:</b> ${row['precio_m2']:,.0f}<br>
            <b>Superficie:</b> {row.get('superficie_util', 'N/A')} m²<br>
            <b>Dormitorios:</b> {row.get('dormitorios', 'N/A')}<br>
            <b>Baños:</b> {row.get('banos', 'N/A')}<br>
            <b>Comuna:</b> {row.get('comuna_left', 'N/A')}<br>
        """
        if tiene_satisfaccion and 'satisfaccion_compuesta' in row:
            popup_html += f"<b>Satisfacción:</b> {row['satisfaccion_compuesta']:.2f}/10<br>"
        popup_html += "</div>"
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=color, icon=icon, prefix='fa')
        ).add_to(marker_cluster)
    
    # Agregar capa de calor
    heat_data = df_viz[['latitude', 'longitude', 'precio_m2']].dropna().values.tolist()
    HeatMap(heat_data, name='Mapa de Calor - Precio', radius=15, blur=10).add_to(mapa)
    
    # Agregar control de capas
    folium.LayerControl().add_to(mapa)
    
    # Agregar leyenda
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; right: 50px; 
                width: 150px; 
                background-color: white; 
                border: 2px solid grey; 
                z-index: 9999; 
                font-size: 12px;
                padding: 10px;
                border-radius: 5px;">
        <b>Leyenda Precio/m²</b><br>
        <i class="fa fa-circle" style="color:green"></i> Bajo<br>
        <i class="fa fa-circle" style="color:orange"></i> Medio<br>
        <i class="fa fa-circle" style="color:red"></i> Alto<br>
    </div>
    '''
    mapa.get_root().html.add_child(folium.Element(legend_html))
    
    # Guardar mapa
    mapa.save(str(GRAFICOS_DIR / 'mapa_interactivo.html'))
    print(f"   ✓ Guardado: mapa_interactivo.html")
    
except ImportError:
    print("   ⚠️ Folium no instalado. Generando alternativa con Plotly...")
    
    try:
        import plotly.express as px
        import plotly.graph_objects as go
        
        # Crear mapa interactivo con Plotly
        fig = px.scatter_mapbox(
            df_viz.sample(min(1000, len(df_viz)), random_state=42),
            lat='latitude',
            lon='longitude',
            color='precio_m2',
            size='superficie_util',
            color_continuous_scale='RdYlGn_r',
            hover_data=['comuna_left', 'dormitorios', 'banos'],
            title='Mapa Interactivo de Propiedades - Santiago',
            zoom=11
        )
        
        fig.update_layout(
            mapbox_style='carto-positron',
            margin={"r": 0, "t": 50, "l": 0, "b": 0}
        )
        
        fig.write_html(str(GRAFICOS_DIR / 'mapa_interactivo.html'))
        print(f"   ✓ Guardado: mapa_interactivo.html (Plotly)")
        
    except ImportError:
        print("   ⚠️ Ni Folium ni Plotly disponibles. Creando mapa estático alternativo...")
        
        # Mapa estático como alternativa
        fig, ax = plt.subplots(figsize=(14, 10))
        scatter = ax.scatter(
            df_viz['longitude'], df_viz['latitude'],
            c=df_viz['precio_m2'], cmap='RdYlGn_r',
            s=30, alpha=0.6
        )
        plt.colorbar(scatter, label='Precio/m²')
        ax.set_title('Mapa de Propiedades (versión estática)')
        ax.set_xlabel('Longitud')
        ax.set_ylabel('Latitud')
        plt.savefig(GRAFICOS_DIR / 'mapa_interactivo_estatico.png', dpi=300)
        plt.close()
        print(f"   ✓ Guardado: mapa_interactivo_estatico.png")

# =============================================================================
# RESUMEN DE VISUALIZACIONES GENERADAS
# =============================================================================
print("\n" + "=" * 80)
print("📊 RESUMEN DE VISUALIZACIONES GENERADAS")
print("=" * 80)

print("""
┌────────────────────────────────────────────────────────────────────────┐
│                    3 MAPAS TEMÁTICOS ✅                                │
├────────────────────────────────────────────────────────────────────────┤
│  1. mapa_01_ubicacion_area_estudio.png                                 │
│     → Distribución de propiedades por comuna                           │
│                                                                        │
│  2. mapa_02_precio_m2.png                                              │
│     → Distribución espacial del precio por m²                          │
│                                                                        │
│  3. mapa_03_resultado_analisis.png                                     │
│     → Resultado del análisis (satisfacción/habitabilidad)              │
├────────────────────────────────────────────────────────────────────────┤
│                    5 GRÁFICOS ESTADÍSTICOS ✅                          │
├────────────────────────────────────────────────────────────────────────┤
│  1. grafico_01_histogramas.png                                         │
│     → Histogramas de variables clave                                   │
│                                                                        │
│  2. grafico_02_temporal.png                                            │
│     → Series temporales / distribución por fecha                       │
│                                                                        │
│  3. grafico_03_correlaciones.png                                       │
│     → Matriz de correlaciones espaciales                               │
│                                                                        │
│  4. grafico_04_dispersion.png                                          │
│     → Diagramas de dispersión                                          │
│                                                                        │
│  5. grafico_05_boxplots_comuna.png                                     │
│     → Boxplots comparativos por comuna                                 │
├────────────────────────────────────────────────────────────────────────┤
│                    1 VISUALIZACIÓN INTERACTIVA ✅                      │
├────────────────────────────────────────────────────────────────────────┤
│  • mapa_interactivo.html                                               │
│    → Mapa con Folium/Plotly (marcadores + heatmap)                     │
└────────────────────────────────────────────────────────────────────────┘
""")

# Guardar índice de visualizaciones
indice = {
    'fecha_generacion': datetime.now().isoformat(),
    'mapas_tematicos': [
        'mapa_01_ubicacion_area_estudio.png',
        'mapa_02_precio_m2.png',
        'mapa_03_resultado_analisis.png'
    ],
    'graficos_estadisticos': [
        'grafico_01_histogramas.png',
        'grafico_02_temporal.png',
        'grafico_03_correlaciones.png',
        'grafico_04_dispersion.png',
        'grafico_05_boxplots_comuna.png'
    ],
    'visualizacion_interactiva': 'mapa_interactivo.html',
    'ubicacion': str(GRAFICOS_DIR)
}

with open(GRAFICOS_DIR / 'INDICE_VISUALIZACIONES.json', 'w') as f:
    json.dump(indice, f, indent=2)

print(f"\n✅ Todas las visualizaciones guardadas en: {GRAFICOS_DIR}")
print(f"✅ Índice guardado en: INDICE_VISUALIZACIONES.json")
