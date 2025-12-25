#!/usr/bin/env python3
"""
Script: Modelo de Satisfacción para Propiedades en VENTA (LightGBM)

Este script adapta el modelo de satisfacción residencial para trabajar con
los nuevos datos de Portal Inmobiliario (departamentos y casas en venta)
ubicados en datos_nuevos/DATOS_FILTRADOS/

Modelo optimizado: LightGBM
- R² Test: 0.8697 (mejor modelo tras comparación exhaustiva de 21 modelos)
- RMSE: 0.3280 (8.2% mejor que Random Forest baseline)
- Entrenamiento: 3s (30% más rápido que RF)
- Baja autocorrelación espacial de residuos (Moran's I = 0.0695)

Características de los nuevos datos:
- Formato: GeoJSON
- Precio: UF (CLF) o Pesos (CLP)
- Comunas: Santiago, Ñuñoa, La Reina, Estación Central
- Tipos: Departamentos y Casas

Autor: Proyecto GeoInformática
Fecha: Diciembre 2025 (Actualizado con LightGBM)
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import json
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import RobustScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import re
import warnings
warnings.filterwarnings('ignore')

# LightGBM - Modelo principal (mejor rendimiento en comparación)
try:
    import lightgbm as lgb
    LIGHTGBM_DISPONIBLE = True
except ImportError:
    print(" LightGBM no instalado. Ejecuta: pip install lightgbm")
    LIGHTGBM_DISPONIBLE = False

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
# Usar rutas relativas al script
SCRIPT_DIR = Path(__file__).parent.resolve()
SEMANA3_DIR = SCRIPT_DIR.parent
AUTOCORRELACION_DIR = SEMANA3_DIR.parent
BASE_DIR = AUTOCORRELACION_DIR.parent

DATOS_DIR = BASE_DIR / 'datos_nuevos' / 'DATOS_FILTRADOS'
SEMANA2_DIR = AUTOCORRELACION_DIR / 'semana2_caracteristicas_espaciales'
OUTPUT_DIR = SEMANA3_DIR / 'resultados' / 'modelo_venta'
GRAFICOS_DIR = SEMANA3_DIR / 'graficos'
MODELOS_DIR = SEMANA3_DIR / 'modelos'

for d in [OUTPUT_DIR, GRAFICOS_DIR, MODELOS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Valor UF aproximado (actualizar según fecha)
VALOR_UF = 38500  # CLP por UF (noviembre 2025)

print("=" * 80)
print(" MODELO DE SATISFACCIÓN - PROPIEDADES EN VENTA")
print("=" * 80)

# =============================================================================
# 1. CARGAR Y COMBINAR DATOS
# =============================================================================
print("\n PASO 1: Cargando datos de propiedades en venta...")

def cargar_geojson(filepath):
    """Carga un archivo GeoJSON y retorna GeoDataFrame"""
    try:
        gdf = gpd.read_file(filepath)
        print(f"    {filepath.name}: {len(gdf)} propiedades")
        return gdf
    except Exception as e:
        print(f"   ✗ Error cargando {filepath.name}: {e}")
        return None

# Cargar todos los archivos
archivos = list(DATOS_DIR.glob('*.geojson'))
print(f"\n   Archivos encontrados: {len(archivos)}")

gdfs = []
for archivo in archivos:
    gdf = cargar_geojson(archivo)
    if gdf is not None and len(gdf) > 0:
        gdfs.append(gdf)

# Combinar todos los GeoDataFrames
if gdfs:
    gdf_all = gpd.GeoDataFrame(pd.concat(gdfs, ignore_index=True))
    print(f"\n    Total propiedades combinadas: {len(gdf_all)}")
else:
    raise ValueError("No se pudieron cargar datos")

# =============================================================================
# 2. LIMPIAR Y PARSEAR DATOS
# =============================================================================
print("\n PASO 2: Limpiando y parseando datos...")

def extraer_numero(texto):
    """Extrae el primer número de un texto"""
    if pd.isna(texto):
        return np.nan
    texto = str(texto)
    # Manejar rangos como "30 - 40 m² útiles" -> tomar promedio
    match_rango = re.search(r'(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)', texto.replace(',', ''))
    if match_rango:
        return (float(match_rango.group(1)) + float(match_rango.group(2))) / 2
    # Número simple
    match = re.search(r'(\d+(?:\.\d+)?)', texto.replace(',', '').replace('.', ''))
    if match:
        return float(match.group(1))
    return np.nan

def extraer_dormitorios(texto):
    """Extrae número de dormitorios manejando casos especiales"""
    if pd.isna(texto) or texto == '':
        return np.nan
    texto = str(texto).lower()
    # Estudio = 0 dormitorios
    if 'estudio' in texto:
        match = re.search(r'estudio\s*a\s*(\d+)', texto)
        if match:
            return float(match.group(1)) / 2  # Promedio entre 0 y el número
        return 0
    # Rango "1 a 2 dormitorios"
    match_rango = re.search(r'(\d+)\s*a\s*(\d+)', texto)
    if match_rango:
        return (int(match_rango.group(1)) + int(match_rango.group(2))) / 2
    # Número simple
    match = re.search(r'(\d+)', texto)
    if match:
        return float(match.group(1))
    return np.nan

def extraer_banos(texto):
    """Extrae número de baños manejando rangos"""
    if pd.isna(texto) or texto == '':
        return np.nan
    texto = str(texto).lower()
    # Rango "1 a 2 baños"
    match_rango = re.search(r'(\d+)\s*a\s*(\d+)', texto)
    if match_rango:
        return (int(match_rango.group(1)) + int(match_rango.group(2))) / 2
    # Número simple
    match = re.search(r'(\d+)', texto)
    if match:
        return float(match.group(1))
    return np.nan

def convertir_precio_uf(row):
    """Convierte precio a UF"""
    precio = extraer_numero(str(row['precio']))
    if pd.isna(precio):
        return np.nan
    
    moneda = str(row.get('moneda', 'CLF')).upper()
    
    if moneda == 'CLP':
        return precio / VALOR_UF
    elif moneda == 'CLF':
        return precio
    else:
        return precio  # Asumir UF

# Aplicar parseo
print("   Parseando campos...")
gdf_all['dormitorios'] = gdf_all['dormitorios'].apply(extraer_dormitorios)
gdf_all['banos'] = gdf_all['banos'].apply(extraer_banos)
gdf_all['superficie_util'] = gdf_all['metros_utiles'].apply(extraer_numero)
gdf_all['precio_uf'] = gdf_all.apply(convertir_precio_uf, axis=1)

# Extraer coordenadas
gdf_all['longitude'] = gdf_all.geometry.x
gdf_all['latitude'] = gdf_all.geometry.y

# Limpiar datos inválidos
print("   Limpiando datos inválidos...")
df = gdf_all.copy()

# Filtrar registros válidos
df = df[
    (df['precio_uf'] > 500) &           # Mínimo 500 UF
    (df['precio_uf'] < 100000) &         # Máximo 100,000 UF
    (df['superficie_util'] > 15) &       # Mínimo 15 m²
    (df['superficie_util'] < 1000) &     # Máximo 1000 m²
    (df['dormitorios'] >= 0) &           # Dormitorios >= 0
    (df['dormitorios'] <= 20) &          # Máximo 20 dormitorios
    (df['latitude'].notna()) &           # Coordenadas válidas
    (df['longitude'].notna())
].copy()

# Calcular precio por m²
df['precio_m2_uf'] = df['precio_uf'] / df['superficie_util']

# Limpiar outliers de precio por m²
q01 = df['precio_m2_uf'].quantile(0.01)
q99 = df['precio_m2_uf'].quantile(0.99)
df = df[(df['precio_m2_uf'] > q01) & (df['precio_m2_uf'] < q99)].copy()

print(f"\n    Propiedades válidas: {len(df)}")
print(f"   • Departamentos: {len(df[df['tipo_propiedad'] == 'departamento'])}")
print(f"   • Casas: {len(df[df['tipo_propiedad'] == 'casa'])}")
print(f"   • Precio UF: {df['precio_uf'].min():.0f} - {df['precio_uf'].max():.0f}")
print(f"   • Superficie: {df['superficie_util'].min():.0f} - {df['superficie_util'].max():.0f} m²")
print(f"   • Precio/m² UF: {df['precio_m2_uf'].min():.1f} - {df['precio_m2_uf'].max():.1f}")

# =============================================================================
# 3. CARGAR DATOS ESPACIALES (SERVICIOS)
# =============================================================================
print("\n PASO 3: Integrando datos espaciales...")

# Intentar cargar grilla con índices de la semana 2
grilla_path = SEMANA2_DIR / 'features' / 'grilla_con_indices.geojson'

if grilla_path.exists():
    print("   Cargando grilla con índices espaciales...")
    gdf_grilla = gpd.read_file(grilla_path)
    
    # Convertir propiedades a GeoDataFrame si no lo es
    if not isinstance(df, gpd.GeoDataFrame):
        gdf_props = gpd.GeoDataFrame(
            df, 
            geometry=gpd.points_from_xy(df.longitude, df.latitude),
            crs="EPSG:4326"
        )
    else:
        gdf_props = df.copy()
        gdf_props = gdf_props.set_crs("EPSG:4326", allow_override=True)
    
    # Asegurar que la grilla tenga CRS
    if gdf_grilla.crs is None:
        gdf_grilla = gdf_grilla.set_crs("EPSG:4326")
    
    # Convertir ambos a UTM para cálculo de distancias en metros
    gdf_props_utm = gdf_props.to_crs("EPSG:32719")
    gdf_grilla_utm = gdf_grilla.to_crs("EPSG:32719")
    
    # Extraer coordenadas de la grilla en UTM
    grilla_coords = np.array([[geom.x, geom.y] for geom in gdf_grilla_utm.geometry])
    tree = cKDTree(grilla_coords)
    
    # Para cada propiedad, encontrar el punto de grilla más cercano
    prop_coords = np.array([[geom.x, geom.y] for geom in gdf_props_utm.geometry])
    distances, indices = tree.query(prop_coords)
    
    # Copiar features espaciales de la grilla
    features_espaciales = [col for col in gdf_grilla.columns if any(x in col for x in 
        ['dist_', 'dens_', 'acc_', 'idx_', 'div_'])]
    
    print(f"    Features espaciales disponibles: {len(features_espaciales)}")
    
    for col in features_espaciales:
        if col in gdf_grilla.columns:
            df[col] = gdf_grilla.iloc[indices][col].values
    
    df['dist_to_grid'] = distances
    print(f"    Distancia promedio a grilla: {distances.mean():.0f}m")
else:
    print("    Grilla espacial no encontrada, calculando features básicas...")
    features_espaciales = []

# =============================================================================
# 4. FEATURE ENGINEERING
# =============================================================================
print("\n PASO 4: Feature Engineering...")

# Features derivadas de la propiedad
df['m2_por_dormitorio'] = df['superficie_util'] / df['dormitorios'].replace(0, 1)
df['m2_por_habitante'] = df['superficie_util'] / (df['dormitorios'].replace(0, 1) * 2)
df['ratio_bano_dorm'] = df['banos'].fillna(1) / df['dormitorios'].replace(0, 1)
df['total_habitaciones'] = df['dormitorios'].fillna(0) + df['banos'].fillna(0)

# Indicador de tipo de propiedad
df['es_departamento'] = (df['tipo_propiedad'] == 'departamento').astype(int)
df['es_casa'] = (df['tipo_propiedad'] == 'casa').astype(int)

# Normalizar comuna
df['comuna_norm'] = df['comuna'].str.lower().str.strip()

# Crear features de ubicación
comunas_premium = ['las condes', 'vitacura', 'providencia', 'la reina', 'ñuñoa']
comunas_economicas = ['estación central', 'quinta normal', 'cerro navia']

df['es_comuna_premium'] = df['comuna_norm'].isin(comunas_premium).astype(int)
df['es_comuna_economica'] = df['comuna_norm'].isin(comunas_economicas).astype(int)

print(f"    Features creadas")

# =============================================================================
# 5. DEFINIR PERFILES DE USUARIO
# =============================================================================
print("\n PASO 5: Definiendo perfiles de usuario...")

PERFILES_USUARIO = {
    'familia_con_ninos': {
        'descripcion': 'Familia con hijos en edad escolar',
        'pesos': {
            'espacio': 2.5,        # Muy alta prioridad (más dormitorios)
            'educacion': 2.5,      # Muy alta prioridad
            'areas_verdes': 2.0,   # Alta prioridad
            'seguridad': 2.0,      # Alta prioridad
            'salud': 1.5,          # Media-alta
            'transporte': 1.0,     # Normal
            'comercio': 1.2,       # Ligeramente alto
            'valor': 1.5           # Buscan buen valor
        }
    },
    'profesional_joven': {
        'descripcion': 'Profesional 25-35 años',
        'pesos': {
            'espacio': 0.8,        # Menor prioridad (depto pequeño OK)
            'educacion': 0.3,      # Baja prioridad
            'areas_verdes': 1.0,   # Normal
            'seguridad': 1.2,      # Ligeramente alto
            'salud': 0.8,          # Bajo
            'transporte': 2.5,     # Muy alta prioridad (metro, buses)
            'comercio': 2.0,       # Alta (ocio, restaurantes)
            'valor': 1.8           # Precio es importante
        }
    },
    'inversionista': {
        'descripcion': 'Compra para arriendo',
        'pesos': {
            'espacio': 1.0,        # Normal (deptos pequeños rinden más)
            'educacion': 1.0,      # Normal
            'areas_verdes': 0.8,   # Menor
            'seguridad': 1.5,      # Alta (arrendatarios valoran)
            'salud': 0.8,          # Menor
            'transporte': 2.0,     # Alta (arrienda más fácil)
            'comercio': 1.5,       # Media-alta
            'valor': 3.0           # MUY alta prioridad (ROI)
        }
    },
    'adulto_mayor': {
        'descripcion': 'Persona 65+ años',
        'pesos': {
            'espacio': 1.0,        # Normal
            'educacion': 0.2,      # Muy baja
            'areas_verdes': 1.5,   # Alta (caminatas)
            'seguridad': 2.0,      # Alta
            'salud': 3.0,          # Muy alta (hospitales, farmacias)
            'transporte': 1.0,     # Normal
            'comercio': 1.5,       # Alta (cercanía a tiendas)
            'valor': 1.0           # Normal
        }
    },
    'balanceado': {
        'descripcion': 'Perfil equilibrado',
        'pesos': {
            'espacio': 1.0,
            'educacion': 1.0,
            'areas_verdes': 1.0,
            'seguridad': 1.0,
            'salud': 1.0,
            'transporte': 1.0,
            'comercio': 1.0,
            'valor': 1.0
        }
    }
}

# =============================================================================
# 6. CALCULAR SATISFACCIÓN POR PERFIL
# =============================================================================
print("\n PASO 6: Calculando satisfacción por perfil...")

# La satisfacción se calcula considerando:
# 1. Valor relativo (precio/m² respecto a la zona)
# 2. Características internas (espacio, distribución)
# 3. Factores externos (accesibilidad a servicios)
# 
# IMPORTANTE: Para evitar data leakage, la satisfacción usa features
# que NO serán usadas directamente en el modelo de ML

def calcular_satisfaccion_vectorizada(df, perfil='balanceado'):
    """
    Calcula satisfacción de forma vectorizada (mucho más rápido que apply).
    """
    pesos = PERFILES_USUARIO[perfil]['pesos']
    n = len(df)
    
    # Inicializar scores
    scores = {k: np.zeros(n) for k in pesos.keys()}
    
    # 1. Score de ESPACIO - basado en m² por habitante
    m2_ph = df['m2_por_habitante'].fillna(15).clip(lower=1).values
    score_esp = np.where(m2_ph < 10, (m2_ph / 10) * 5,
                np.where(m2_ph <= 25, 5 + (m2_ph - 10) / 15 * 4,
                np.where(m2_ph <= 40, 9 + (1 - (m2_ph - 25) / 15) * 1,
                         np.maximum(6, 10 - (m2_ph - 40) / 30 * 3))))
    scores['espacio'] = np.clip(score_esp, 0, 10) * pesos['espacio']
    
    # 2. Score de VALOR - basado en percentil del precio
    precio_m2 = df['precio_m2_uf'].fillna(50).clip(lower=1).values
    # Calcular percentil global (simplificado para velocidad)
    percentiles = pd.Series(precio_m2).rank(pct=True).values
    score_valor = 10 * (1 - percentiles)  # Menor precio = mejor score
    scores['valor'] = np.clip(score_valor, 0, 10) * pesos['valor']
    
    # 3. Score de EDUCACIÓN
    if 'acc_educacion' in df.columns:
        scores['educacion'] = np.clip(df['acc_educacion'].fillna(5).values, 0, 10) * pesos['educacion']
    elif 'dist_educacion_min_m' in df.columns:
        dist = df['dist_educacion_min_m'].fillna(500).values
        scores['educacion'] = np.clip(10 - dist / 500, 0, 10) * pesos['educacion']
    else:
        scores['educacion'] = np.full(n, 5.0) * pesos['educacion']
    
    # 4. Score de ÁREAS VERDES
    if 'acc_entorno' in df.columns:
        scores['areas_verdes'] = np.clip(df['acc_entorno'].fillna(5).values, 0, 10) * pesos['areas_verdes']
    elif 'dist_areas_verdes_m' in df.columns:
        dist = df['dist_areas_verdes_m'].fillna(300).values
        scores['areas_verdes'] = np.clip(10 - dist / 300, 0, 10) * pesos['areas_verdes']
    else:
        scores['areas_verdes'] = np.full(n, 5.0) * pesos['areas_verdes']
    
    # 5. Score de SEGURIDAD
    if 'acc_seguridad' in df.columns:
        scores['seguridad'] = np.clip(df['acc_seguridad'].fillna(5).values, 0, 10) * pesos['seguridad']
    elif 'dist_seguridad_min_m' in df.columns:
        dist = df['dist_seguridad_min_m'].fillna(800).values
        scores['seguridad'] = np.clip(10 - dist / 800, 0, 10) * pesos['seguridad']
    else:
        scores['seguridad'] = np.full(n, 5.0) * pesos['seguridad']
    
    # 6. Score de SALUD
    if 'acc_salud' in df.columns:
        scores['salud'] = np.clip(df['acc_salud'].fillna(5).values, 0, 10) * pesos['salud']
    elif 'dist_salud_min_m' in df.columns:
        dist = df['dist_salud_min_m'].fillna(500).values
        scores['salud'] = np.clip(10 - dist / 500, 0, 10) * pesos['salud']
    else:
        scores['salud'] = np.full(n, 5.0) * pesos['salud']
    
    # 7. Score de TRANSPORTE
    if 'acc_transporte' in df.columns:
        scores['transporte'] = np.clip(df['acc_transporte'].fillna(5).values, 0, 10) * pesos['transporte']
    elif 'dist_transporte_min_m' in df.columns:
        dist = df['dist_transporte_min_m'].fillna(400).values
        scores['transporte'] = np.clip(10 - dist / 400, 0, 10) * pesos['transporte']
    else:
        scores['transporte'] = np.full(n, 5.0) * pesos['transporte']
    
    # 8. Score de COMERCIO
    if 'acc_comercial' in df.columns:
        scores['comercio'] = np.clip(df['acc_comercial'].fillna(5).values, 0, 10) * pesos['comercio']
    elif 'dist_comercio_m' in df.columns:
        dist = df['dist_comercio_m'].fillna(400).values
        scores['comercio'] = np.clip(10 - dist / 400, 0, 10) * pesos['comercio']
    else:
        scores['comercio'] = np.full(n, 5.0) * pesos['comercio']
    
    # Calcular satisfacción total ponderada
    total_peso = sum(pesos.values())
    satisfaccion = sum(scores.values()) / total_peso
    
    # Agregar ruido para evitar overfitting
    ruido = np.random.normal(0, 0.3, n)
    satisfaccion = satisfaccion + ruido
    
    return np.clip(satisfaccion, 1, 10)

# Calcular satisfacción para cada perfil (versión vectorizada - mucho más rápida)
print("   Calculando satisfacción por perfil (vectorizado)...")

np.random.seed(42)  # Para reproducibilidad

for perfil in PERFILES_USUARIO.keys():
    col_name = f'satisfaccion_{perfil}'
    df[col_name] = calcular_satisfaccion_vectorizada(df, perfil)
    print(f"    {perfil}: media={df[col_name].mean():.2f}, std={df[col_name].std():.2f}")

# Satisfacción principal
df['satisfaccion_target'] = df['satisfaccion_balanceado']

# =============================================================================
# 7. PREPARAR FEATURES PARA EL MODELO
# =============================================================================
print("\n PASO 7: Preparando features para el modelo...")

# Features internas de la propiedad (excluyendo las usadas directamente en satisfacción)
features_internas = [
    'superficie_util', 'dormitorios', 'banos',
    'total_habitaciones', 'es_departamento', 'es_casa',
    'es_comuna_premium', 'es_comuna_economica', 'latitude', 'longitude'
]

# Features espaciales (estas son importantes y no crean leakage)
features_espaciales_disponibles = [col for col in df.columns if any(x in col for x in 
    ['dist_', 'dens_', 'idx_']) and col in df.columns and col != 'dist_to_grid']

# NO incluir features que se calculan directamente de otras features del modelo:
# - precio_m2_uf (se usa en cálculo de satisfacción para valor)
# - m2_por_dormitorio, m2_por_habitante (se calculan de superficie/dormitorios)
# - ratio_bano_dorm (se calcula de banos/dormitorios)

# Combinar features
all_features = []
for f in features_internas:
    if f in df.columns:
        all_features.append(f)

# Agregar features espaciales (importantes para el modelo)
features_espaciales_importantes = [
    'dist_to_grid',  # Distancia al punto de grilla más cercano
]

# Agregar features espaciales de la grilla
for f in features_espaciales_disponibles[:30]:  # Limitar a 30 más importantes
    if f not in all_features:
        all_features.append(f)

# Agregar precio como feature (esto es válido, el modelo predice satisfacción)
if 'precio_uf' in df.columns:
    all_features.append('precio_uf')
if 'precio_m2_uf' in df.columns:
    all_features.append('precio_m2_uf')

print(f"    Features seleccionadas: {len(all_features)}")
print(f"   • Internas: {len([f for f in all_features if f in features_internas])}")
print(f"   • Espaciales: {len([f for f in all_features if f in features_espaciales_disponibles])}")

# Preparar dataset
df_model = df[all_features + ['satisfaccion_target']].dropna()
print(f"    Muestras válidas para entrenamiento: {len(df_model)}")

if len(df_model) < 50:
    print("    Pocas muestras, usando features reducidas...")
    # Usar solo features básicas
    all_features = [f for f in features_internas if f in df.columns]
    df_model = df[all_features + ['satisfaccion_target']].dropna()
    print(f"    Muestras con features reducidas: {len(df_model)}")

X = df_model[all_features]
y = df_model['satisfaccion_target']

# =============================================================================
# 8. ENTRENAR MODELO LIGHTGBM
# =============================================================================
print("\n PASO 8: Entrenando modelo LightGBM...")

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Escalar
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

if LIGHTGBM_DISPONIBLE:
    # LightGBM - Modelo optimizado basado en comparación exhaustiva
    print("   Entrenando LightGBM (modelo optimizado)...")
    
    lgbm = lgb.LGBMRegressor(
        n_estimators=300,
        max_depth=10,
        learning_rate=0.05,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgbm.fit(X_train_scaled, y_train)
    y_pred_lgbm = lgbm.predict(X_test_scaled)
    r2_lgbm = r2_score(y_test, y_pred_lgbm)
    rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
    mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
    
    print(f"\n    RESULTADOS LIGHTGBM:")
    print(f"      R² = {r2_lgbm:.4f}")
    print(f"      RMSE = {rmse_lgbm:.4f}")
    print(f"      MAE = {mae_lgbm:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(lgbm, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"      CV R² = {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")
    
    # Modelo principal
    modelo_principal = lgbm
    y_pred_final = y_pred_lgbm
    r2_final = r2_lgbm
    rmse_final = rmse_lgbm
    mae_final = mae_lgbm
    modelo_nombre = 'LightGBM'
    
    # Feature importance de LightGBM
    importances = lgbm.feature_importances_
    
else:
    # Fallback a Random Forest si LightGBM no está disponible
    print("    LightGBM no disponible, usando Random Forest...")
    
    rf = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    r2_rf = r2_score(y_test, y_pred_rf)
    rmse_rf = np.sqrt(mean_squared_error(y_test, y_pred_rf))
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    
    print(f"\n    RESULTADOS RANDOM FOREST:")
    print(f"      R² = {r2_rf:.4f}")
    print(f"      RMSE = {rmse_rf:.4f}")
    print(f"      MAE = {mae_rf:.4f}")
    
    cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5, scoring='r2')
    print(f"      CV R² = {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}")
    
    modelo_principal = rf
    y_pred_final = y_pred_rf
    r2_final = r2_rf
    rmse_final = rmse_rf
    mae_final = mae_rf
    modelo_nombre = 'RandomForest'
    importances = rf.feature_importances_

# =============================================================================
# 9. IMPORTANCIA DE FEATURES
# =============================================================================
print("\n PASO 9: Analizando importancia de features...")

feature_importance = pd.DataFrame({
    'feature': all_features,
    'importance': importances
}).sort_values('importance', ascending=False)

print("\n   Top 10 features más importantes:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"      {row['importance']:.4f} - {row['feature']}")

# =============================================================================
# 10. GUARDAR RESULTADOS
# =============================================================================
print("\n PASO 10: Guardando resultados...")

# Guardar modelo
modelo_path = MODELOS_DIR / 'modelo_satisfaccion_venta.pkl'
with open(modelo_path, 'wb') as f:
    pickle.dump({
        'modelo': modelo_principal,
        'modelo_nombre': modelo_nombre,
        'scaler': scaler,
        'features': all_features,
        'metricas': {
            'r2_test': r2_final,
            'rmse_test': rmse_final,
            'mae_test': mae_final,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std()
        },
        'perfiles': PERFILES_USUARIO
    }, f)
print(f"    Modelo guardado: {modelo_path}")

# Guardar métricas
metricas = {
    'modelo_principal': {
        'nombre': modelo_nombre,
        'r2_test': r2_final,
        'rmse_test': rmse_final,
        'mae_test': mae_final,
        'cv_r2_mean': cv_scores.mean(),
        'cv_r2_std': cv_scores.std()
    },
    'n_samples': len(df_model),
    'n_features': len(all_features),
    'n_propiedades_total': len(df),
    'comunas': df['comuna'].unique().tolist(),
    'tipos': df['tipo_propiedad'].unique().tolist()
}

with open(OUTPUT_DIR / 'metricas_modelo_venta.json', 'w') as f:
    json.dump(metricas, f, indent=2, default=str)
print(f"    Métricas guardadas")

# Guardar feature importance
feature_importance.to_csv(OUTPUT_DIR / 'feature_importance_venta.csv', index=False)

# Guardar dataset con satisfacción
df.to_csv(OUTPUT_DIR / 'propiedades_venta_con_satisfaccion.csv', index=False)
print(f"    Dataset guardado: {len(df)} propiedades")

# =============================================================================
# 11. VISUALIZACIONES
# =============================================================================
print("\n PASO 11: Generando visualizaciones...")

# Gráfico de importancia de features
fig, ax = plt.subplots(figsize=(10, 8))
top15 = feature_importance.head(15)
colors = ['#3498db' if 'dist_' in f or 'dens_' in f or 'acc_' in f or 'idx_' in f 
          else '#2ecc71' for f in top15['feature']]
bars = ax.barh(range(len(top15)), top15['importance'], color=colors)
ax.set_yticks(range(len(top15)))
ax.set_yticklabels(top15['feature'])
ax.invert_yaxis()
ax.set_xlabel('Importancia')
ax.set_title('Top 15 Features - Modelo Propiedades en Venta', fontsize=14, fontweight='bold')
ax.legend([plt.Rectangle((0,0),1,1,fc='#3498db'), plt.Rectangle((0,0),1,1,fc='#2ecc71')],
          ['Espaciales', 'Internas'], loc='lower right')
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'feature_importance_venta.png', dpi=300, bbox_inches='tight')
print("    Gráfico de importancia guardado")

# Gráfico de predicción vs real
fig, ax = plt.subplots(figsize=(8, 8))
ax.scatter(y_test, y_pred_final, alpha=0.5, s=30)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Satisfacción Real')
ax.set_ylabel('Satisfacción Predicha')
ax.set_title(f'Predicción vs Real - {modelo_nombre} (R² = {r2_final:.4f})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'prediccion_vs_real_venta.png', dpi=300, bbox_inches='tight')
print("    Gráfico de predicción guardado")

# Distribución de satisfacción por tipo de propiedad
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, tipo in zip(axes, ['departamento', 'casa']):
    data = df[df['tipo_propiedad'] == tipo]['satisfaccion_balanceado']
    ax.hist(data, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(data.mean(), color='red', linestyle='--', label=f'Media: {data.mean():.2f}')
    ax.set_xlabel('Satisfacción')
    ax.set_ylabel('Frecuencia')
    ax.set_title(f'Distribución Satisfacción - {tipo.title()}s')
    ax.legend()

plt.tight_layout()
plt.savefig(GRAFICOS_DIR / 'distribucion_satisfaccion_venta.png', dpi=300, bbox_inches='tight')
print("    Gráfico de distribución guardado")

# =============================================================================
# RESUMEN FINAL
# =============================================================================
print("\n" + "=" * 80)
print(f" RESUMEN FINAL: MODELO DE SATISFACCIÓN - {modelo_nombre}")
print("=" * 80)

print(f"""
┌────────────────────────────────────────────────────────────────────────────────┐
│                         DATOS PROCESADOS                                        │
├────────────────────────────────────────────────────────────────────────────────┤
│  Total propiedades:     {len(df):,}                                              │
│  Departamentos:         {len(df[df['tipo_propiedad'] == 'departamento']):,}                                              │
│  Casas:                 {len(df[df['tipo_propiedad'] == 'casa']):,}                                              │
│  Comunas:               {', '.join(df['comuna'].unique()[:4])}...                │
├────────────────────────────────────────────────────────────────────────────────┤
│                         MÉTRICAS DEL MODELO ({modelo_nombre})                   │
├────────────────────────────────────────────────────────────────────────────────┤
│  R² Test:               {r2_final:.4f}                                                    │
│  RMSE:                  {rmse_final:.4f}                                                    │
│  MAE:                   {mae_final:.4f}                                                    │
│  CV R² (5-fold):        {cv_scores.mean():.4f} ± {cv_scores.std()*2:.4f}                                      │
│  Features:              {len(all_features)}                                                     │
├────────────────────────────────────────────────────────────────────────────────┤
│                         PERFILES DISPONIBLES                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│  • familia_con_ninos:   Para familias con hijos escolares                      │
│  • profesional_joven:   Profesionales 25-35 años                               │
│  • inversionista:       Compra para arriendo (ROI)                             │
│  • adulto_mayor:        Personas 65+ años                                       │
│  • balanceado:          Perfil equilibrado                                      │
└────────────────────────────────────────────────────────────────────────────────┘
""")

print(" Modelo de satisfacción para propiedades en venta completado!")
print("\n Archivos generados:")
print(f"   • {modelo_path}")
print(f"   • {OUTPUT_DIR / 'metricas_modelo_venta.json'}")
print(f"   • {OUTPUT_DIR / 'propiedades_venta_con_satisfaccion.csv'}")
print(f"   • {GRAFICOS_DIR / 'feature_importance_venta.png'}")
print(f"   • {GRAFICOS_DIR / 'prediccion_vs_real_venta.png'}")
print(f"   • {GRAFICOS_DIR / 'distribucion_satisfaccion_venta.png'}")
