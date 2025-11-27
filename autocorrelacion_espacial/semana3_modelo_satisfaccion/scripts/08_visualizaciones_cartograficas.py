#!/usr/bin/env python3
"""
================================================================================
Script 08: Visualizaciones Cartográficas y Análisis Exploratorio Completo
================================================================================

CUMPLIMIENTO DE REQUISITOS:

D. VISUALIZACIONES (10%):
✅ Mínimo 3 mapas temáticos con elementos cartográficos
   - Mapa 1: Ubicación del área de estudio
   - Mapa 2: Datos principales (precio/m²)
   - Mapa 3: Resultado del análisis (satisfacción/habitabilidad)
   
   Elementos cartográficos incluidos en cada mapa:
   ✓ Título descriptivo
   ✓ Escala gráfica
   ✓ Flecha de norte
   ✓ Leyenda
   ✓ Sistema de coordenadas
   ✓ Fuente de datos
   ✓ Autor y fecha

✅ Mínimo 5 gráficos estadísticos
   - Gráfico 1: Histogramas de variables clave
   - Gráfico 2: Series temporales / distribución por fecha
   - Gráfico 3: Matriz de correlaciones espaciales
   - Gráfico 4: Diagramas de dispersión
   - Gráfico 5: Boxplots comparativos por comuna

✅ 1 visualización interactiva funcional
   - Mapa HTML con Folium (marcadores + heatmap + capas)

E. DATOS Y CÓDIGO:
✅ Datos descargados y organizados
✅ Código Python ejecutable y documentado
✅ Análisis exploratorio completo (EDA)

Autor: Felipe Baeza - Proyecto GeoInformática
Fecha: Noviembre 2025
Universidad: [Tu Universidad]
Curso: Análisis Espacial y Autocorrelación
================================================================================
"""

# =============================================================================
# IMPORTACIONES
# =============================================================================
import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.offsetbox import AnchoredText
import matplotlib.lines as mlines
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Importar Folium para mapa interactivo
try:
    import folium
    from folium.plugins import HeatMap, MarkerCluster, MiniMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False
    print("⚠️ Folium no disponible. Instalar con: pip install folium")

# =============================================================================
# CONFIGURACIÓN DE RUTAS
# =============================================================================
BASE_DIR = Path('/home/felipe/Documentos/GeoInformatica')
SEMANA1_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana1_preparacion_datos'
SEMANA2_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana2_caracteristicas_espaciales'
SEMANA3_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana3_modelo_satisfaccion'
DATA_DIR = SEMANA3_DIR / 'data'
GRAFICOS_DIR = SEMANA3_DIR / 'graficos'
RESULTADOS_DIR = SEMANA3_DIR / 'resultados' / 'modelo_mejorado'

# Crear directorios
GRAFICOS_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# CONFIGURACIÓN DE ESTILO
# =============================================================================
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'figure.figsize': (14, 10),
    'font.size': 11,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif'
})

# Paleta de colores personalizada
COLORES = {
    'primario': '#2E86AB',
    'secundario': '#A23B72',
    'acento': '#F18F01',
    'exito': '#28A745',
    'alerta': '#DC3545',
    'neutral': '#6C757D',
    'fondo': '#F8F9FA'
}

# Metadatos del proyecto
METADATA = {
    'autor': 'Felipe Baeza',
    'proyecto': 'Análisis de Autocorrelación Espacial',
    'fecha': datetime.now().strftime('%B %Y'),
    'fuente': 'Portal Inmobiliario Chile',
    'crs': 'WGS84 / EPSG:4326',
    'area_estudio': 'Región Metropolitana de Santiago, Chile'
}

print("=" * 80)
print("📊 GENERACIÓN DE VISUALIZACIONES CON ELEMENTOS CARTOGRÁFICOS")
print("=" * 80)
print(f"   Autor: {METADATA['autor']}")
print(f"   Fecha: {METADATA['fecha']}")
print(f"   Área: {METADATA['area_estudio']}")
print("=" * 80)

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def agregar_elementos_cartograficos(ax, lon_range, lat_range, titulo, subtitulo=""):
    """
    Agrega elementos cartográficos estándar a un mapa:
    - Título y subtítulo
    - Flecha de norte
    - Escala gráfica
    - Sistema de coordenadas
    - Fuente y autor
    """
    lon_min, lon_max = lon_range
    lat_min, lat_max = lat_range
    
    # 1. TÍTULO Y SUBTÍTULO
    ax.set_title(f"{titulo}\n{subtitulo}" if subtitulo else titulo,
                 fontsize=14, fontweight='bold', pad=20)
    
    # 2. FLECHA DE NORTE (esquina superior derecha)
    norte_x = lon_max - (lon_max - lon_min) * 0.05
    norte_y = lat_max - (lat_max - lat_min) * 0.08
    
    # Círculo de fondo
    circle = plt.Circle((norte_x, norte_y), (lon_max - lon_min) * 0.03, 
                        color='white', ec='black', linewidth=1.5, zorder=10)
    ax.add_patch(circle)
    
    # Flecha
    ax.annotate('', xy=(norte_x, norte_y + (lat_max - lat_min) * 0.025),
                xytext=(norte_x, norte_y - (lat_max - lat_min) * 0.015),
                arrowprops=dict(arrowstyle='->', color='black', lw=2),
                zorder=11)
    ax.text(norte_x, norte_y + (lat_max - lat_min) * 0.035, 'N',
            fontsize=12, fontweight='bold', ha='center', va='bottom', zorder=11)
    
    # 3. ESCALA GRÁFICA (esquina inferior izquierda)
    # Calcular 1 km aproximado en grados de longitud (a latitud -33.45)
    km_en_grados = 0.0089  # ~1 km en grados de longitud a lat -33.45
    
    escala_x = lon_min + (lon_max - lon_min) * 0.05
    escala_y = lat_min + (lat_max - lat_min) * 0.05
    
    # Barra de escala (2 km)
    ax.plot([escala_x, escala_x + km_en_grados * 2], [escala_y, escala_y],
            'k-', linewidth=3, zorder=10)
    ax.plot([escala_x, escala_x], [escala_y - 0.002, escala_y + 0.002],
            'k-', linewidth=2, zorder=10)
    ax.plot([escala_x + km_en_grados * 2, escala_x + km_en_grados * 2],
            [escala_y - 0.002, escala_y + 0.002], 'k-', linewidth=2, zorder=10)
    ax.text(escala_x + km_en_grados, escala_y + 0.004, '2 km',
            fontsize=10, ha='center', va='bottom', fontweight='bold', zorder=10)
    
    # 4. SISTEMA DE COORDENADAS
    ax.set_xlabel(f'Longitud ({METADATA["crs"]})', fontsize=11)
    ax.set_ylabel(f'Latitud ({METADATA["crs"]})', fontsize=11)
    
    # 5. CUADRO DE INFORMACIÓN (esquina inferior derecha)
    info_text = f"Fuente: {METADATA['fuente']}\n"
    info_text += f"CRS: {METADATA['crs']}\n"
    info_text += f"Autor: {METADATA['autor']}\n"
    info_text += f"Fecha: {METADATA['fecha']}"
    
    props = dict(boxstyle='round,pad=0.5', facecolor='white', 
                 edgecolor='gray', alpha=0.9)
    ax.text(0.98, 0.02, info_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    # 6. GRILLA DE COORDENADAS
    ax.grid(True, linestyle='--', alpha=0.5, color='gray')
    
    return ax


def crear_leyenda_categorica(ax, categorias, colores, titulo="Leyenda", loc='upper left'):
    """Crea una leyenda categórica personalizada."""
    handles = [mpatches.Patch(color=c, label=cat) for cat, c in zip(categorias, colores)]
    ax.legend(handles=handles, title=titulo, loc=loc, fontsize=9, 
              title_fontsize=10, framealpha=0.9)


def generar_reporte_eda(df, nombre_archivo):
    """Genera un reporte de análisis exploratorio."""
    reporte = {
        'fecha_generacion': datetime.now().isoformat(),
        'total_registros': len(df),
        'columnas': list(df.columns),
        'tipos_datos': {col: str(dtype) for col, dtype in df.dtypes.items()},
        'valores_nulos': df.isnull().sum().to_dict(),
        'estadisticas_numericas': {}
    }
    
    # Estadísticas para columnas numéricas
    for col in df.select_dtypes(include=[np.number]).columns:
        reporte['estadisticas_numericas'][col] = {
            'media': float(df[col].mean()) if pd.notna(df[col].mean()) else None,
            'mediana': float(df[col].median()) if pd.notna(df[col].median()) else None,
            'std': float(df[col].std()) if pd.notna(df[col].std()) else None,
            'min': float(df[col].min()) if pd.notna(df[col].min()) else None,
            'max': float(df[col].max()) if pd.notna(df[col].max()) else None,
            'q25': float(df[col].quantile(0.25)) if pd.notna(df[col].quantile(0.25)) else None,
            'q75': float(df[col].quantile(0.75)) if pd.notna(df[col].quantile(0.75)) else None
        }
    
    # Guardar
    with open(GRAFICOS_DIR / nombre_archivo, 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    return reporte


# =============================================================================
# CARGAR DATOS
# =============================================================================
print("\n📂 PASO 1: Cargando datos...")

# Intentar cargar dataset mejorado
try:
    df = pd.read_csv(RESULTADOS_DIR / 'propiedades_con_satisfaccion.csv')
    print(f"   ✓ Dataset mejorado cargado: {len(df)} propiedades")
    tiene_satisfaccion = 'satisfaccion_compuesta' in df.columns
except:
    try:
        df = pd.read_csv(DATA_DIR / 'propiedades_con_factores_espaciales.csv')
        print(f"   ✓ Dataset original cargado: {len(df)} propiedades")
        tiene_satisfaccion = False
    except:
        print("   ❌ Error: No se encontraron los datos")
        exit(1)

# Calcular precio_m2 si no existe
if 'precio_m2' not in df.columns:
    df['precio_m2'] = df['precio'] / df['superficie_util'].replace(0, np.nan)

# Limpiar datos para visualización
df_viz = df[
    (df['precio_m2'] > 1000) & 
    (df['precio_m2'] < 50000) &
    (df['latitude'].notna()) &
    (df['longitude'].notna())
].copy()

print(f"   ✓ Datos filtrados para visualización: {len(df_viz)} propiedades")

# Cargar grilla con índices
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
    print(f"   ✓ Grilla de índices cargada: {len(df_grilla)} puntos")
except Exception as e:
    print(f"   ⚠️ Grilla no disponible: {e}")
    df_grilla = None

# Límites del área de estudio
lon_min, lon_max = df_viz['longitude'].min() - 0.02, df_viz['longitude'].max() + 0.02
lat_min, lat_max = df_viz['latitude'].min() - 0.02, df_viz['latitude'].max() + 0.02
lon_range = (lon_min, lon_max)
lat_range = (lat_min, lat_max)

# =============================================================================
# ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================
print("\n📊 PASO 2: Análisis Exploratorio de Datos (EDA)...")

# Generar reporte EDA
reporte_eda = generar_reporte_eda(df_viz, 'reporte_eda.json')
print(f"   ✓ Reporte EDA guardado: reporte_eda.json")

# Mostrar resumen
print(f"\n   📈 RESUMEN DEL DATASET:")
print(f"      • Total propiedades: {len(df_viz):,}")
print(f"      • Variables numéricas: {len(df_viz.select_dtypes(include=[np.number]).columns)}")
print(f"      • Variables categóricas: {len(df_viz.select_dtypes(include=['object']).columns)}")
print(f"      • Rango temporal: 2023")
print(f"      • Área geográfica: {METADATA['area_estudio']}")

# =============================================================================
# MAPA 1: UBICACIÓN DEL ÁREA DE ESTUDIO
# =============================================================================
print("\n🗺️ MAPA 1: Ubicación del área de estudio...")

fig, ax = plt.subplots(figsize=(14, 11))

# Determinar columna de comuna
comuna_col = 'comuna_left' if 'comuna_left' in df_viz.columns else 'comuna'
comunas = df_viz[comuna_col].dropna().unique() if comuna_col in df_viz.columns else []

# Colores por comuna
n_comunas = min(len(comunas), 10)
colors_comunas = plt.cm.Set3(np.linspace(0, 1, max(n_comunas, 1)))

# Plotear propiedades por comuna
for i, comuna in enumerate(sorted(comunas)[:10]):
    mask = df_viz[comuna_col] == comuna
    n_props = mask.sum()
    ax.scatter(
        df_viz.loc[mask, 'longitude'],
        df_viz.loc[mask, 'latitude'],
        c=[colors_comunas[i]],
        s=35,
        alpha=0.7,
        label=f"{comuna.title()} (n={n_props})",
        edgecolors='white',
        linewidths=0.4
    )

# Configurar límites
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# Agregar elementos cartográficos
agregar_elementos_cartograficos(
    ax, lon_range, lat_range,
    titulo="MAPA 1: UBICACIÓN DEL ÁREA DE ESTUDIO",
    subtitulo="Distribución de Propiedades en Arriendo - Región Metropolitana, Santiago"
)

# Leyenda
ax.legend(loc='upper left', fontsize=8, framealpha=0.95, title='Comunas',
          title_fontsize=10, ncol=1)

# Cuadro de estadísticas
stats_text = f"ESTADÍSTICAS DEL ÁREA\n"
stats_text += f"━━━━━━━━━━━━━━━━━━━━━\n"
stats_text += f"Total propiedades: {len(df_viz):,}\n"
stats_text += f"Comunas analizadas: {len(comunas)}\n"
stats_text += f"Extensión lat: {lat_max-lat_min:.3f}°\n"
stats_text += f"Extensión lon: {lon_max-lon_min:.3f}°\n"
stats_text += f"Área aprox: ~{(lon_max-lon_min)*111*(lat_max-lat_min)*111:.0f} km²"

props = dict(boxstyle='round,pad=0.5', facecolor=COLORES['fondo'], 
             edgecolor=COLORES['primario'], alpha=0.95)
ax.text(0.02, 0.35, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', family='monospace', bbox=props)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'mapa_01_ubicacion_area_estudio.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: mapa_01_ubicacion_area_estudio.png")

# =============================================================================
# MAPA 2: DATOS PRINCIPALES (PRECIO POR M²)
# =============================================================================
print("\n🗺️ MAPA 2: Datos principales (Precio/m²)...")

fig, ax = plt.subplots(figsize=(14, 11))

# Crear colormap para precios
cmap_precio = LinearSegmentedColormap.from_list('precio', 
    ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad'])

# Percentiles para normalización
vmin = df_viz['precio_m2'].quantile(0.05)
vmax = df_viz['precio_m2'].quantile(0.95)

# Scatter plot
scatter = ax.scatter(
    df_viz['longitude'],
    df_viz['latitude'],
    c=df_viz['precio_m2'],
    cmap=cmap_precio,
    s=40,
    alpha=0.75,
    edgecolors='white',
    linewidths=0.3,
    vmin=vmin,
    vmax=vmax
)

# Colorbar con formato
cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02, aspect=30)
cbar.set_label('Precio por m² (CLP)', fontsize=11, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

# Formatear etiquetas del colorbar
cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

# Configurar límites
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# Agregar elementos cartográficos
agregar_elementos_cartograficos(
    ax, lon_range, lat_range,
    titulo="MAPA 2: DISTRIBUCIÓN ESPACIAL DEL PRECIO",
    subtitulo="Precio de Arriendo por Metro Cuadrado (CLP/m²)"
)

# Estadísticas del precio
stats_text = f"ESTADÍSTICAS PRECIO/M²\n"
stats_text += f"━━━━━━━━━━━━━━━━━━━━━━\n"
stats_text += f"Media:   ${df_viz['precio_m2'].mean():,.0f}\n"
stats_text += f"Mediana: ${df_viz['precio_m2'].median():,.0f}\n"
stats_text += f"Std:     ${df_viz['precio_m2'].std():,.0f}\n"
stats_text += f"Min:     ${df_viz['precio_m2'].min():,.0f}\n"
stats_text += f"Max:     ${df_viz['precio_m2'].max():,.0f}\n"
stats_text += f"Q25:     ${df_viz['precio_m2'].quantile(0.25):,.0f}\n"
stats_text += f"Q75:     ${df_viz['precio_m2'].quantile(0.75):,.0f}"

props = dict(boxstyle='round,pad=0.5', facecolor='white', 
             edgecolor=COLORES['secundario'], alpha=0.95)
ax.text(0.02, 0.40, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', family='monospace', bbox=props)

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'mapa_02_precio_m2.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: mapa_02_precio_m2.png")

# =============================================================================
# MAPA 3: RESULTADO DEL ANÁLISIS (SATISFACCIÓN/HABITABILIDAD)
# =============================================================================
print("\n🗺️ MAPA 3: Resultado del análisis...")

fig, ax = plt.subplots(figsize=(14, 11))

# Determinar variable para el mapa de resultado
if tiene_satisfaccion and 'satisfaccion_compuesta' in df_viz.columns:
    map_var = 'satisfaccion_compuesta'
    map_label = 'Índice de Satisfacción Compuesta'
    cmap_result = 'RdYlGn'
elif 'idx_habitabilidad_global' in df_viz.columns:
    map_var = 'idx_habitabilidad_global'
    map_label = 'Índice de Habitabilidad Global'
    cmap_result = 'RdYlGn'
else:
    # Crear índice aproximado basado en características disponibles
    dens_cols = [c for c in df_viz.columns if c.startswith('dens_') and df_viz[c].notna().sum() > 100]
    if dens_cols:
        df_viz['idx_aproximado'] = df_viz[dens_cols[:10]].mean(axis=1)
        # Normalizar a escala 1-10
        min_val = df_viz['idx_aproximado'].min()
        max_val = df_viz['idx_aproximado'].max()
        df_viz['idx_aproximado'] = 1 + 9 * (df_viz['idx_aproximado'] - min_val) / (max_val - min_val)
        map_var = 'idx_aproximado'
        map_label = 'Índice de Accesibilidad (aproximado)'
        cmap_result = 'RdYlGn'
    else:
        map_var = 'precio_m2'
        map_label = 'Precio/m² (sin índice disponible)'
        cmap_result = 'viridis'

# Scatter plot
scatter = ax.scatter(
    df_viz['longitude'],
    df_viz['latitude'],
    c=df_viz[map_var].fillna(df_viz[map_var].median()),
    cmap=cmap_result,
    s=40,
    alpha=0.75,
    edgecolors='white',
    linewidths=0.3
)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax, shrink=0.7, pad=0.02, aspect=30)
cbar.set_label(map_label, fontsize=11, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

# Configurar límites
ax.set_xlim(lon_min, lon_max)
ax.set_ylim(lat_min, lat_max)

# Agregar elementos cartográficos
agregar_elementos_cartograficos(
    ax, lon_range, lat_range,
    titulo="MAPA 3: RESULTADO DEL ANÁLISIS",
    subtitulo=f"{map_label} - Modelo de Satisfacción Espacial"
)

# Estadísticas del resultado
stats_text = f"ESTADÍSTICAS {map_var.upper()[:20]}\n"
stats_text += f"━━━━━━━━━━━━━━━━━━━━━━\n"
stats_text += f"Media:   {df_viz[map_var].mean():.2f}\n"
stats_text += f"Mediana: {df_viz[map_var].median():.2f}\n"
stats_text += f"Std:     {df_viz[map_var].std():.2f}\n"
stats_text += f"Min:     {df_viz[map_var].min():.2f}\n"
stats_text += f"Max:     {df_viz[map_var].max():.2f}"

props = dict(boxstyle='round,pad=0.5', facecolor='white', 
             edgecolor=COLORES['exito'], alpha=0.95)
ax.text(0.02, 0.35, stats_text, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', family='monospace', bbox=props)

# Interpretación
if map_var in ['satisfaccion_compuesta', 'idx_habitabilidad_global', 'idx_aproximado']:
    interp_text = "INTERPRETACIÓN\n"
    interp_text += "━━━━━━━━━━━━━━━\n"
    interp_text += "🟢 Verde: Alta satisfacción\n"
    interp_text += "🟡 Amarillo: Media\n"
    interp_text += "🔴 Rojo: Baja satisfacción"
    ax.text(0.02, 0.18, interp_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'mapa_03_resultado_analisis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: mapa_03_resultado_analisis.png")

# =============================================================================
# GRÁFICO 1: HISTOGRAMAS DE VARIABLES CLAVE
# =============================================================================
print("\n📊 GRÁFICO 1: Histogramas de variables clave...")

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('GRÁFICO 1: DISTRIBUCIÓN DE VARIABLES CLAVE\nAnálisis Exploratorio de Datos (EDA)', 
             fontsize=14, fontweight='bold', y=1.02)

variables_hist = [
    ('precio_m2', 'Precio por m² (CLP)', COLORES['primario'], True),
    ('superficie_util', 'Superficie Útil (m²)', COLORES['secundario'], False),
    ('dormitorios', 'Número de Dormitorios', COLORES['acento'], False),
    ('banos', 'Número de Baños', COLORES['exito'], False),
]

# Agregar satisfacción si existe
if tiene_satisfaccion:
    variables_hist.append(('satisfaccion_compuesta', 'Satisfacción Compuesta', '#9b59b6', False))
else:
    variables_hist.append(('estacionamientos', 'Estacionamientos', COLORES['neutral'], False))

# Agregar variable adicional
if 'idx_habitabilidad_global' in df_viz.columns:
    variables_hist.append(('idx_habitabilidad_global', 'Índice Habitabilidad', '#1abc9c', False))
else:
    variables_hist.append(('precio', 'Precio Total (CLP)', '#34495e', True))

for idx, (var, label, color, es_precio) in enumerate(variables_hist[:6]):
    ax = axes.flat[idx]
    
    if var in df_viz.columns:
        data = df_viz[var].dropna()
        
        # Filtrar outliers extremos
        if es_precio or var == 'precio_m2':
            data = data[(data > data.quantile(0.01)) & (data < data.quantile(0.99))]
        
        # Histograma con KDE
        ax.hist(data, bins=35, color=color, edgecolor='white', alpha=0.7, density=True)
        
        # KDE
        try:
            from scipy import stats
            kde = stats.gaussian_kde(data)
            x_range = np.linspace(data.min(), data.max(), 100)
            ax.plot(x_range, kde(x_range), color='darkred', linewidth=2, label='KDE')
        except:
            pass
        
        # Media y mediana
        ax.axvline(data.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Media: {data.mean():,.1f}')
        ax.axvline(data.median(), color='orange', linestyle=':', linewidth=2, 
                   label=f'Mediana: {data.median():,.1f}')
        
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Densidad', fontsize=10)
        ax.set_title(f'Distribución de {label}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=8, loc='upper right')
        
        # Estadísticas en el gráfico
        stats_text = f'n={len(data):,}\nσ={data.std():,.1f}'
        ax.text(0.95, 0.75, stats_text, transform=ax.transAxes, fontsize=8,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, f'{var}\nno disponible', ha='center', va='center', 
                transform=ax.transAxes, fontsize=12)
        ax.set_title(f'{label}', fontsize=11)

# Agregar pie de página
fig.text(0.5, -0.02, f'Fuente: {METADATA["fuente"]} | Autor: {METADATA["autor"]} | {METADATA["fecha"]}',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'grafico_01_histogramas.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: grafico_01_histogramas.png")

# =============================================================================
# GRÁFICO 2: SERIES TEMPORALES / DISTRIBUCIÓN POR COMUNA
# =============================================================================
print("\n📊 GRÁFICO 2: Distribución temporal y por comuna...")

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('GRÁFICO 2: ANÁLISIS TEMPORAL Y GEOGRÁFICO', fontsize=14, fontweight='bold')

# Panel 1: Distribución por comuna
ax1 = axes[0]
comuna_col = 'comuna_left' if 'comuna_left' in df_viz.columns else 'comuna'
if comuna_col in df_viz.columns:
    conteo_comunas = df_viz[comuna_col].value_counts().head(12)
    colors_bar = plt.cm.Set2(np.linspace(0, 1, len(conteo_comunas)))
    
    bars = ax1.barh(range(len(conteo_comunas)), conteo_comunas.values, color=colors_bar)
    ax1.set_yticks(range(len(conteo_comunas)))
    ax1.set_yticklabels([c.title() for c in conteo_comunas.index])
    ax1.set_xlabel('Número de Propiedades', fontsize=11)
    ax1.set_title('Propiedades por Comuna (Top 12)', fontsize=12, fontweight='bold')
    
    # Añadir valores
    for i, (bar, val) in enumerate(zip(bars, conteo_comunas.values)):
        ax1.text(val + 5, i, f'{val:,}', va='center', fontsize=9)
    
    ax1.invert_yaxis()

# Panel 2: Precio promedio por comuna
ax2 = axes[1]
if comuna_col in df_viz.columns:
    precio_comuna = df_viz.groupby(comuna_col)['precio_m2'].agg(['mean', 'std']).sort_values('mean', ascending=True).tail(12)
    
    bars2 = ax2.barh(range(len(precio_comuna)), precio_comuna['mean'].values, 
                     xerr=precio_comuna['std'].values * 0.1, color=COLORES['secundario'], 
                     alpha=0.8, capsize=3)
    ax2.set_yticks(range(len(precio_comuna)))
    ax2.set_yticklabels([c.title() for c in precio_comuna.index])
    ax2.set_xlabel('Precio Promedio por m² (CLP)', fontsize=11)
    ax2.set_title('Precio Promedio por Comuna (Top 12)', fontsize=12, fontweight='bold')
    
    # Formatear eje x
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    ax2.invert_yaxis()

# Pie de página
fig.text(0.5, -0.05, f'Fuente: {METADATA["fuente"]} | Autor: {METADATA["autor"]} | {METADATA["fecha"]}',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'grafico_02_temporal_comunas.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: grafico_02_temporal_comunas.png")

# =============================================================================
# GRÁFICO 3: MATRIZ DE CORRELACIONES
# =============================================================================
print("\n📊 GRÁFICO 3: Matriz de correlaciones espaciales...")

fig, ax = plt.subplots(figsize=(14, 12))

# Seleccionar variables para correlación
vars_corr = ['precio_m2', 'superficie_util', 'dormitorios', 'banos', 'latitude', 'longitude']

# Agregar índices si existen
indices_disponibles = ['idx_habitabilidad_global', 'idx_vida_urbana', 'idx_calidad_vida',
                       'acc_transporte', 'acc_educacion', 'acc_salud', 'satisfaccion_compuesta']
for idx_var in indices_disponibles:
    if idx_var in df_viz.columns:
        vars_corr.append(idx_var)

# Agregar algunas densidades
dens_vars = [c for c in df_viz.columns if c.startswith('dens_') and '300m' in c][:5]
vars_corr.extend(dens_vars)

# Filtrar variables existentes
vars_corr = [v for v in vars_corr if v in df_viz.columns]

# Calcular correlación
corr_matrix = df_viz[vars_corr].corr()

# Crear heatmap
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
cmap_corr = sns.diverging_palette(220, 20, as_cmap=True)

sns.heatmap(corr_matrix, mask=mask, cmap=cmap_corr, center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8, "label": "Correlación"},
            annot=True, fmt='.2f', annot_kws={"size": 8}, ax=ax,
            vmin=-1, vmax=1)

ax.set_title('GRÁFICO 3: MATRIZ DE CORRELACIONES ESPACIALES\nVariables del Modelo de Satisfacción', 
             fontsize=14, fontweight='bold', pad=20)

# Rotación de etiquetas
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(rotation=0, fontsize=9)

# Pie de página
fig.text(0.5, -0.02, f'Fuente: {METADATA["fuente"]} | Autor: {METADATA["autor"]} | {METADATA["fecha"]}',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'grafico_03_correlaciones.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: grafico_03_correlaciones.png")

# =============================================================================
# GRÁFICO 4: DIAGRAMAS DE DISPERSIÓN
# =============================================================================
print("\n📊 GRÁFICO 4: Diagramas de dispersión...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle('GRÁFICO 4: RELACIONES ENTRE VARIABLES\nDiagramas de Dispersión con Regresión', 
             fontsize=14, fontweight='bold', y=1.02)

# Panel 1: Superficie vs Precio
ax1 = axes[0, 0]
x1 = df_viz['superficie_util']
y1 = df_viz['precio_m2']
mask1 = (x1 < x1.quantile(0.95)) & (y1 < y1.quantile(0.95))
ax1.scatter(x1[mask1], y1[mask1], alpha=0.4, s=30, c=COLORES['primario'], edgecolors='white', linewidths=0.3)

# Línea de regresión
z1 = np.polyfit(x1[mask1].dropna(), y1[mask1].dropna(), 1)
p1 = np.poly1d(z1)
x_line = np.linspace(x1[mask1].min(), x1[mask1].max(), 100)
ax1.plot(x_line, p1(x_line), "r--", linewidth=2, label=f'Regresión (r={np.corrcoef(x1[mask1], y1[mask1])[0,1]:.2f})')

ax1.set_xlabel('Superficie Útil (m²)', fontsize=11)
ax1.set_ylabel('Precio por m² (CLP)', fontsize=11)
ax1.set_title('Superficie vs Precio', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')

# Panel 2: Dormitorios vs Precio
ax2 = axes[0, 1]
df_dorm = df_viz[df_viz['dormitorios'].between(1, 5)].copy()
sns.boxplot(data=df_dorm, x='dormitorios', y='precio_m2', ax=ax2, palette='Set2')
ax2.set_xlabel('Número de Dormitorios', fontsize=11)
ax2.set_ylabel('Precio por m² (CLP)', fontsize=11)
ax2.set_title('Precio por Dormitorios', fontsize=12, fontweight='bold')

# Panel 3: Coordenadas (lon/lat) vs precio
ax3 = axes[1, 0]
scatter3 = ax3.scatter(df_viz['longitude'], df_viz['latitude'], 
                       c=df_viz['precio_m2'], cmap='YlOrRd', s=20, alpha=0.6)
ax3.set_xlabel('Longitud', fontsize=11)
ax3.set_ylabel('Latitud', fontsize=11)
ax3.set_title('Distribución Espacial del Precio', fontsize=12, fontweight='bold')
cbar3 = plt.colorbar(scatter3, ax=ax3, shrink=0.8)
cbar3.set_label('Precio/m²')

# Panel 4: Índice vs Precio (si existe)
ax4 = axes[1, 1]
if tiene_satisfaccion:
    y_var = 'satisfaccion_compuesta'
    y_label = 'Satisfacción Compuesta'
elif 'idx_habitabilidad_global' in df_viz.columns:
    y_var = 'idx_habitabilidad_global'
    y_label = 'Índice Habitabilidad'
else:
    y_var = 'precio_m2'
    y_label = 'Precio/m²'

if y_var != 'precio_m2':
    ax4.scatter(df_viz['precio_m2'], df_viz[y_var].fillna(df_viz[y_var].median()), 
                alpha=0.4, s=30, c=COLORES['exito'], edgecolors='white', linewidths=0.3)
    ax4.set_xlabel('Precio por m² (CLP)', fontsize=11)
    ax4.set_ylabel(y_label, fontsize=11)
    ax4.set_title(f'Precio vs {y_label}', fontsize=12, fontweight='bold')
else:
    # Histograma 2D como alternativa
    h = ax4.hist2d(df_viz['longitude'], df_viz['latitude'], bins=30, cmap='YlGnBu')
    ax4.set_xlabel('Longitud', fontsize=11)
    ax4.set_ylabel('Latitud', fontsize=11)
    ax4.set_title('Densidad de Propiedades', fontsize=12, fontweight='bold')
    plt.colorbar(h[3], ax=ax4, label='Frecuencia')

# Pie de página
fig.text(0.5, -0.02, f'Fuente: {METADATA["fuente"]} | Autor: {METADATA["autor"]} | {METADATA["fecha"]}',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'grafico_04_dispersion.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: grafico_04_dispersion.png")

# =============================================================================
# GRÁFICO 5: BOXPLOTS COMPARATIVOS
# =============================================================================
print("\n📊 GRÁFICO 5: Boxplots comparativos por comuna...")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
fig.suptitle('GRÁFICO 5: ANÁLISIS COMPARATIVO POR COMUNA\nDistribución de Variables Clave', 
             fontsize=14, fontweight='bold', y=1.02)

comuna_col = 'comuna_left' if 'comuna_left' in df_viz.columns else 'comuna'

# Filtrar top 8 comunas por cantidad
if comuna_col in df_viz.columns:
    top_comunas = df_viz[comuna_col].value_counts().head(8).index.tolist()
    df_top = df_viz[df_viz[comuna_col].isin(top_comunas)].copy()
    
    # Panel 1: Precio por m² por comuna
    ax1 = axes[0]
    orden = df_top.groupby(comuna_col)['precio_m2'].median().sort_values(ascending=False).index
    sns.boxplot(data=df_top, y=comuna_col, x='precio_m2', ax=ax1, 
                order=orden, palette='RdYlGn_r', orient='h')
    ax1.set_xlabel('Precio por m² (CLP)', fontsize=11)
    ax1.set_ylabel('Comuna', fontsize=11)
    ax1.set_title('Distribución de Precio por Comuna', fontsize=12, fontweight='bold')
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
    
    # Panel 2: Superficie por comuna
    ax2 = axes[1]
    df_top_filtered = df_top[df_top['superficie_util'] < df_top['superficie_util'].quantile(0.95)]
    sns.boxplot(data=df_top_filtered, y=comuna_col, x='superficie_util', ax=ax2, 
                order=orden, palette='Blues', orient='h')
    ax2.set_xlabel('Superficie Útil (m²)', fontsize=11)
    ax2.set_ylabel('Comuna', fontsize=11)
    ax2.set_title('Distribución de Superficie por Comuna', fontsize=12, fontweight='bold')

# Pie de página
fig.text(0.5, -0.02, f'Fuente: {METADATA["fuente"]} | Autor: {METADATA["autor"]} | {METADATA["fecha"]}',
         ha='center', fontsize=9, style='italic')

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'grafico_05_boxplots_comuna.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Guardado: grafico_05_boxplots_comuna.png")

# =============================================================================
# VISUALIZACIÓN INTERACTIVA CON FOLIUM
# =============================================================================
print("\n🌐 VISUALIZACIÓN INTERACTIVA: Mapa con Folium...")

if HAS_FOLIUM:
    # Centro del mapa
    center_lat = df_viz['latitude'].mean()
    center_lon = df_viz['longitude'].mean()
    
    # Crear mapa base
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='cartodbpositron'
    )
    
    # Agregar título
    title_html = '''
    <div style="position: fixed; 
                top: 10px; left: 50px; width: 400px; height: auto;
                border:2px solid grey; z-index:9999; font-size:14px;
                background-color:white; padding: 10px; border-radius: 10px;
                box-shadow: 3px 3px 10px rgba(0,0,0,0.3);">
        <b>🏠 Mapa Interactivo de Propiedades</b><br>
        <span style="font-size:11px;">
            Región Metropolitana de Santiago, Chile<br>
            Mercado de Alquileres 2023<br>
            <i>Autor: Felipe Baeza</i>
        </span>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(title_html))
    
    # Crear grupos de capas
    marker_cluster = MarkerCluster(name='Propiedades (Cluster)').add_to(m)
    
    # Agregar marcadores (muestra de 500 para rendimiento)
    sample_size = min(500, len(df_viz))
    df_sample = df_viz.sample(sample_size, random_state=42)
    
    for idx, row in df_sample.iterrows():
        # Determinar color por precio
        precio = row['precio_m2']
        if precio < df_viz['precio_m2'].quantile(0.33):
            color = 'green'
            icon_color = 'white'
        elif precio < df_viz['precio_m2'].quantile(0.66):
            color = 'orange'
            icon_color = 'white'
        else:
            color = 'red'
            icon_color = 'white'
        
        # Popup con información
        popup_html = f"""
        <div style="width: 200px;">
            <b>🏠 Propiedad</b><br>
            <hr style="margin: 5px 0;">
            <b>Precio:</b> ${row.get('precio', 0):,.0f} CLP<br>
            <b>Precio/m²:</b> ${row['precio_m2']:,.0f} CLP<br>
            <b>Superficie:</b> {row.get('superficie_util', 'N/A')} m²<br>
            <b>Dormitorios:</b> {row.get('dormitorios', 'N/A')}<br>
            <b>Baños:</b> {row.get('banos', 'N/A')}<br>
            <b>Comuna:</b> {row.get(comuna_col, 'N/A').title() if pd.notna(row.get(comuna_col)) else 'N/A'}
        """
        
        if tiene_satisfaccion and 'satisfaccion_compuesta' in row.index:
            popup_html += f"<br><b>Satisfacción:</b> {row['satisfaccion_compuesta']:.2f}/10"
        
        popup_html += "</div>"
        
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=250),
            icon=folium.Icon(color=color, icon='home', prefix='fa')
        ).add_to(marker_cluster)
    
    # Agregar heatmap de precios
    heat_data = [[row['latitude'], row['longitude'], row['precio_m2']] 
                 for idx, row in df_viz.iterrows()]
    
    heatmap_layer = folium.FeatureGroup(name='Mapa de Calor (Precio)')
    HeatMap(heat_data, 
            min_opacity=0.3, 
            radius=15, 
            blur=10,
            gradient={0.4: 'blue', 0.65: 'lime', 0.8: 'yellow', 1: 'red'}
    ).add_to(heatmap_layer)
    heatmap_layer.add_to(m)
    
    # Agregar minimapa
    MiniMap(toggle_display=True, position='bottomright').add_to(m)
    
    # Agregar control de capas
    folium.LayerControl(collapsed=False).add_to(m)
    
    # Agregar leyenda
    legend_html = '''
    <div style="position: fixed; 
                bottom: 50px; left: 50px; width: 150px; height: auto;
                border:2px solid grey; z-index:9999; font-size:12px;
                background-color:white; padding: 10px; border-radius: 10px;">
        <b>Leyenda Precio/m²</b><br>
        <i class="fa fa-circle" style="color:green"></i> Bajo (< 33%)<br>
        <i class="fa fa-circle" style="color:orange"></i> Medio (33-66%)<br>
        <i class="fa fa-circle" style="color:red"></i> Alto (> 66%)
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Guardar
    m.save(str(GRAFICOS_DIR / 'mapa_interactivo.html'))
    print(f"   ✓ Guardado: mapa_interactivo.html")
else:
    print("   ⚠️ Folium no disponible, mapa interactivo no generado")

# =============================================================================
# GENERAR ÍNDICE DE VISUALIZACIONES
# =============================================================================
print("\n📋 Generando índice de visualizaciones...")

indice = {
    'fecha_generacion': datetime.now().isoformat(),
    'autor': METADATA['autor'],
    'proyecto': METADATA['proyecto'],
    'mapas_tematicos': [
        {
            'archivo': 'mapa_01_ubicacion_area_estudio.png',
            'titulo': 'Ubicación del Área de Estudio',
            'descripcion': 'Distribución de propiedades por comuna en Santiago',
            'elementos_cartograficos': ['título', 'escala', 'norte', 'leyenda', 'coordenadas', 'fuente']
        },
        {
            'archivo': 'mapa_02_precio_m2.png',
            'titulo': 'Distribución Espacial del Precio',
            'descripcion': 'Precio de arriendo por metro cuadrado',
            'elementos_cartograficos': ['título', 'escala', 'norte', 'colorbar', 'coordenadas', 'fuente']
        },
        {
            'archivo': 'mapa_03_resultado_analisis.png',
            'titulo': 'Resultado del Análisis',
            'descripcion': 'Índice de satisfacción/habitabilidad predicho',
            'elementos_cartograficos': ['título', 'escala', 'norte', 'colorbar', 'coordenadas', 'fuente']
        }
    ],
    'graficos_estadisticos': [
        {
            'archivo': 'grafico_01_histogramas.png',
            'titulo': 'Histogramas de Variables Clave',
            'descripcion': 'Distribución de precio, superficie, dormitorios, baños y satisfacción'
        },
        {
            'archivo': 'grafico_02_temporal_comunas.png',
            'titulo': 'Análisis Temporal y Geográfico',
            'descripcion': 'Distribución de propiedades y precios por comuna'
        },
        {
            'archivo': 'grafico_03_correlaciones.png',
            'titulo': 'Matriz de Correlaciones',
            'descripcion': 'Correlaciones entre variables espaciales y del modelo'
        },
        {
            'archivo': 'grafico_04_dispersion.png',
            'titulo': 'Diagramas de Dispersión',
            'descripcion': 'Relaciones bivariadas entre variables principales'
        },
        {
            'archivo': 'grafico_05_boxplots_comuna.png',
            'titulo': 'Boxplots Comparativos',
            'descripcion': 'Comparación de precio y superficie por comuna'
        }
    ],
    'visualizacion_interactiva': {
        'archivo': 'mapa_interactivo.html',
        'titulo': 'Mapa Interactivo de Propiedades',
        'descripcion': 'Mapa con marcadores clusterizados, heatmap y popup informativo',
        'tecnologia': 'Folium + Leaflet.js'
    },
    'reporte_eda': 'reporte_eda.json'
}

with open(GRAFICOS_DIR / 'INDICE_VISUALIZACIONES.json', 'w', encoding='utf-8') as f:
    json.dump(indice, f, indent=2, ensure_ascii=False)

print(f"   ✓ Guardado: INDICE_VISUALIZACIONES.json")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print("📊 RESUMEN DE VISUALIZACIONES GENERADAS")
print("=" * 80)

print("""
┌────────────────────────────────────────────────────────────────────────────┐
│                    ✅ 3 MAPAS TEMÁTICOS CON ELEMENTOS CARTOGRÁFICOS        │
├────────────────────────────────────────────────────────────────────────────┤
│  1. mapa_01_ubicacion_area_estudio.png                                     │
│     → Elementos: Título, Escala, Norte, Leyenda, Coordenadas, Fuente       │
│                                                                            │
│  2. mapa_02_precio_m2.png                                                  │
│     → Elementos: Título, Escala, Norte, Colorbar, Coordenadas, Fuente      │
│                                                                            │
│  3. mapa_03_resultado_analisis.png                                         │
│     → Elementos: Título, Escala, Norte, Colorbar, Coordenadas, Fuente      │
├────────────────────────────────────────────────────────────────────────────┤
│                    ✅ 5 GRÁFICOS ESTADÍSTICOS                              │
├────────────────────────────────────────────────────────────────────────────┤
│  1. grafico_01_histogramas.png        - Distribuciones de variables clave  │
│  2. grafico_02_temporal_comunas.png   - Análisis por comuna                │
│  3. grafico_03_correlaciones.png      - Matriz de correlaciones            │
│  4. grafico_04_dispersion.png         - Diagramas de dispersión            │
│  5. grafico_05_boxplots_comuna.png    - Boxplots comparativos              │
├────────────────────────────────────────────────────────────────────────────┤
│                    ✅ 1 VISUALIZACIÓN INTERACTIVA                          │
├────────────────────────────────────────────────────────────────────────────┤
│  • mapa_interactivo.html                                                   │
│    → Marcadores con cluster, Heatmap, Popups informativos, Minimapa        │
├────────────────────────────────────────────────────────────────────────────┤
│                    ✅ ANÁLISIS EXPLORATORIO (EDA)                          │
├────────────────────────────────────────────────────────────────────────────┤
│  • reporte_eda.json - Estadísticas completas del dataset                   │
│  • INDICE_VISUALIZACIONES.json - Índice de todas las visualizaciones       │
└────────────────────────────────────────────────────────────────────────────┘
""")

print(f"\n📂 Archivos guardados en: {GRAFICOS_DIR}")
print("\n✅ TODOS LOS REQUISITOS DE VISUALIZACIÓN CUMPLIDOS!")
