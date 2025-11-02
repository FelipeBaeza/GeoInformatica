#!/usr/bin/env python3
"""
Procesar dataset de Kaggle: clean_alquiler_02_11_2023cc.csv
- Limpieza de datos
- Geocodificación (ya tiene lat/lon)
- Enriquecimiento con características espaciales
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import os
from datetime import datetime

def cargar_datos():
    """Carga el dataset de Kaggle"""
    
    print("=" * 80)
    print("📥 CARGANDO DATASET DE KAGGLE")
    print("=" * 80)
    
    # Ruta al archivo
    file_path = '/home/felipe/Documentos/GeoInformatica/clean_alquiler_02_11_2023cc.csv'
    
    print(f"\n📂 Archivo: {file_path}")
    
    # Cargar
    df = pd.read_csv(file_path, low_memory=False)
    
    print(f"✅ Cargado: {len(df):,} propiedades")
    print(f"📋 Columnas: {len(df.columns)}")
    
    return df


def analizar_dataset(df):
    """Análisis exploratorio del dataset"""
    
    print("\n" + "=" * 80)
    print("📊 ANÁLISIS DEL DATASET")
    print("=" * 80)
    
    # Columnas disponibles
    print(f"\n📋 Columnas ({len(df.columns)}):")
    for col in df.columns:
        non_null = df[col].notna().sum()
        pct = (non_null / len(df)) * 100
        print(f"   • {col:30s} → {non_null:5,} valores ({pct:5.1f}%)")
    
    # Estadísticas de precio
    if 'precio' in df.columns:
        precios = df['precio'].dropna()
        print(f"\n💰 PRECIOS (arriendo mensual):")
        print(f"   • Mínimo:   ${precios.min():,.0f}")
        print(f"   • Q1 (25%): ${precios.quantile(0.25):,.0f}")
        print(f"   • Mediana:  ${precios.median():,.0f}")
        print(f"   • Q3 (75%): ${precios.quantile(0.75):,.0f}")
        print(f"   • Máximo:   ${precios.max():,.0f}")
        print(f"   • Promedio: ${precios.mean():,.0f}")
    
    # Comunas
    if 'comuna' in df.columns:
        print(f"\n📍 COMUNAS (top 10):")
        comunas = df['comuna'].value_counts().head(10)
        for comuna, count in comunas.items():
            print(f"   • {comuna:25s}: {count:4,} ({count/len(df)*100:5.1f}%)")
    
    # Características físicas
    if 'dormitorios' in df.columns:
        print(f"\n🛏️  DORMITORIOS:")
        dorms = df['dormitorios'].value_counts().sort_index()
        for dorm, count in dorms.items():
            if pd.notna(dorm):
                print(f"   • {int(dorm)} dorm: {count:4,} ({count/len(df)*100:5.1f}%)")
    
    if 'superficie_util' in df.columns:
        sup = df['superficie_util'].dropna()
        if len(sup) > 0:
            print(f"\n📐 SUPERFICIE ÚTIL (m²):")
            print(f"   • Mínima:  {sup.min():.0f} m²")
            print(f"   • Mediana: {sup.median():.0f} m²")
            print(f"   • Máxima:  {sup.max():.0f} m²")
    
    # Geocodificación
    if 'latitude' in df.columns and 'longitude' in df.columns:
        con_coords = df[df['latitude'].notna() & df['longitude'].notna()]
        print(f"\n🌍 GEOCODIFICACIÓN:")
        print(f"   • Con coordenadas: {len(con_coords):,} ({len(con_coords)/len(df)*100:.1f}%)")
        print(f"   • Sin coordenadas:  {len(df) - len(con_coords):,}")


def limpiar_datos(df):
    """Limpieza y preparación de datos"""
    
    print("\n" + "=" * 80)
    print("🧹 LIMPIEZA DE DATOS")
    print("=" * 80)
    
    df_clean = df.copy()
    
    # 1. Filtrar propiedades con coordenadas
    print("\n1️⃣  Filtrando propiedades con coordenadas...")
    df_clean = df_clean[df_clean['latitude'].notna() & df_clean['longitude'].notna()]
    print(f"   ✓ Propiedades con coordenadas: {len(df_clean):,}")
    
    # 2. Filtrar precios válidos
    print("\n2️⃣  Filtrando precios válidos...")
    df_clean = df_clean[df_clean['precio'].notna() & (df_clean['precio'] > 0)]
    
    # Filtrar outliers extremos (fuera de rango razonable)
    Q1 = df_clean['precio'].quantile(0.01)
    Q3 = df_clean['precio'].quantile(0.99)
    df_clean = df_clean[(df_clean['precio'] >= Q1) & (df_clean['precio'] <= Q3)]
    
    print(f"   ✓ Propiedades con precio válido: {len(df_clean):,}")
    print(f"   ✓ Rango: ${df_clean['precio'].min():,.0f} - ${df_clean['precio'].max():,.0f}")
    
    # 3. Filtrar comunas de interés (opcional)
    comunas_objetivo = ['Ñuñoa', 'La Reina', 'Santiago', 'Estación Central', 
                        'Las Condes', 'Providencia', 'Vitacura', 'Lo Barnechea']
    
    if 'comuna' in df_clean.columns:
        print(f"\n3️⃣  Filtrando comunas de interés...")
        # Normalizar nombres
        df_clean['comuna_norm'] = df_clean['comuna'].str.strip().str.title()
        
        # Filtrar (flexible - incluye si contiene el nombre)
        mask = df_clean['comuna_norm'].apply(
            lambda x: any(c.lower() in str(x).lower() for c in comunas_objetivo) if pd.notna(x) else False
        )
        df_filtrado = df_clean[mask]
        
        print(f"   ✓ Propiedades en comunas objetivo: {len(df_filtrado):,}")
        
        if len(df_filtrado) > 100:  # Si hay suficientes datos
            df_clean = df_filtrado
    
    # 4. Limpiar valores nulos en características clave
    print(f"\n4️⃣  Imputando valores faltantes...")
    
    # Dormitorios: usar mediana
    if 'dormitorios' in df_clean.columns:
        median_dorm = df_clean['dormitorios'].median()
        df_clean['dormitorios'].fillna(median_dorm, inplace=True)
    
    # Superficie: usar mediana si falta
    if 'superficie_util' in df_clean.columns:
        median_sup = df_clean['superficie_util'].median()
        df_clean['superficie_util'].fillna(median_sup, inplace=True)
    
    # Baños: usar 1 si falta
    if 'banos' in df_clean.columns:
        df_clean['banos'].fillna(1, inplace=True)
    
    print(f"   ✓ Valores faltantes imputados")
    
    print(f"\n✅ Dataset limpio: {len(df_clean):,} propiedades")
    
    return df_clean


def convertir_a_geodataframe(df):
    """Convierte a GeoDataFrame con geometría"""
    
    print("\n" + "=" * 80)
    print("🗺️  CONVERSIÓN A GEODATAFRAME")
    print("=" * 80)
    
    # Crear geometría
    geometry = [Point(lon, lat) for lon, lat in zip(df['longitude'], df['latitude'])]
    
    # GeoDataFrame
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:4326')
    
    # Convertir a UTM 19S (mismo CRS que tu grilla)
    print(f"\n🌐 Convirtiendo a EPSG:32719 (UTM 19S)...")
    gdf = gdf.to_crs('EPSG:32719')
    
    print(f"✅ GeoDataFrame creado con {len(gdf):,} propiedades")
    
    return gdf


def enriquecer_con_grilla(gdf):
    """Enriquece propiedades con características espaciales de la grilla"""
    
    print("\n" + "=" * 80)
    print("🔗 ENRIQUECIMIENTO CON CARACTERÍSTICAS ESPACIALES")
    print("=" * 80)
    
    # Cargar grilla con características
    grilla_path = '/home/felipe/Documentos/GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/features/grilla_con_distancias.geojson'
    
    print(f"\n📂 Cargando grilla: {grilla_path}")
    
    try:
        grilla = gpd.read_file(grilla_path)
        print(f"✅ Grilla cargada: {len(grilla):,} puntos, {len(grilla.columns)} características")
        
        # Spatial join: encontrar el punto de grilla más cercano a cada propiedad
        print(f"\n🔍 Buscando puntos de grilla más cercanos...")
        
        # Para cada propiedad, encontrar el punto de grilla más cercano
        propiedades_enriquecidas = []
        
        for idx, propiedad in gdf.iterrows():
            # Calcular distancias a todos los puntos de grilla
            distancias = grilla.geometry.distance(propiedad.geometry)
            idx_cercano = distancias.idxmin()
            
            # Obtener características del punto más cercano
            punto_cercano = grilla.loc[idx_cercano]
            
            # Combinar datos
            prop_enriquecida = propiedad.to_dict()
            
            # Agregar características espaciales
            for col in grilla.columns:
                if col != 'geometry' and col not in gdf.columns:
                    prop_enriquecida[f'espacial_{col}'] = punto_cercano[col]
            
            # Agregar distancia al punto de grilla
            prop_enriquecida['dist_grilla_m'] = distancias.min()
            
            propiedades_enriquecidas.append(prop_enriquecida)
            
            # Progreso
            if (idx + 1) % 100 == 0:
                print(f"   ✓ Procesadas {idx + 1}/{len(gdf)} propiedades...")
        
        # Crear nuevo GeoDataFrame
        gdf_enriquecido = gpd.GeoDataFrame(propiedades_enriquecidas, crs=gdf.crs)
        
        print(f"\n✅ Enriquecimiento completado")
        print(f"   • Propiedades: {len(gdf_enriquecido):,}")
        print(f"   • Características totales: {len(gdf_enriquecido.columns)}")
        
        # Características espaciales agregadas
        cols_espaciales = [c for c in gdf_enriquecido.columns if c.startswith('espacial_')]
        print(f"   • Características espaciales agregadas: {len(cols_espaciales)}")
        
        return gdf_enriquecido
        
    except Exception as e:
        print(f"⚠️  Error al enriquecer: {e}")
        print(f"   Continuando sin enriquecimiento espacial...")
        return gdf


def guardar_resultados(gdf, output_dir='datos_procesados'):
    """Guarda el dataset procesado"""
    
    print("\n" + "=" * 80)
    print("💾 GUARDANDO RESULTADOS")
    print("=" * 80)
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # CSV (sin geometría)
    df_csv = pd.DataFrame(gdf.drop(columns='geometry'))
    csv_file = f'{output_dir}/propiedades_kaggle_{timestamp}.csv'
    df_csv.to_csv(csv_file, index=False)
    print(f"\n✅ CSV guardado: {csv_file}")
    print(f"   Tamaño: {os.path.getsize(csv_file) / 1024:.1f} KB")
    
    # GeoJSON
    geojson_file = f'{output_dir}/propiedades_kaggle_{timestamp}.geojson'
    gdf.to_file(geojson_file, driver='GeoJSON')
    print(f"✅ GeoJSON guardado: {geojson_file}")
    print(f"   Tamaño: {os.path.getsize(geojson_file) / (1024*1024):.1f} MB")
    
    # Reporte
    print("\n" + "=" * 80)
    print("📊 RESUMEN FINAL")
    print("=" * 80)
    
    print(f"\n✅ Dataset procesado exitosamente")
    print(f"   • Total propiedades: {len(gdf):,}")
    print(f"   • Total características: {len(gdf.columns)}")
    
    if 'precio' in gdf.columns:
        print(f"\n💰 Estadísticas de precio:")
        print(f"   • Rango: ${gdf['precio'].min():,.0f} - ${gdf['precio'].max():,.0f}")
        print(f"   • Promedio: ${gdf['precio'].mean():,.0f}")
        print(f"   • Mediana: ${gdf['precio'].median():,.0f}")
    
    if 'comuna_norm' in gdf.columns:
        print(f"\n📍 Distribución por comuna:")
        for comuna, count in gdf['comuna_norm'].value_counts().head(5).items():
            print(f"   • {comuna}: {count:,}")
    
    print(f"\n🎯 SIGUIENTE PASO:")
    print(f"   Modelado hedónico (OLS, GWR, ML)")
    
    return gdf


def main():
    """Función principal"""
    
    # 1. Cargar datos
    df = cargar_datos()
    
    # 2. Análisis exploratorio
    analizar_dataset(df)
    
    # 3. Limpieza
    df_clean = limpiar_datos(df)
    
    # 4. Convertir a GeoDataFrame
    gdf = convertir_a_geodataframe(df_clean)
    
    # 5. Enriquecer con características espaciales
    gdf_enriquecido = enriquecer_con_grilla(gdf)
    
    # 6. Guardar
    gdf_final = guardar_resultados(gdf_enriquecido)
    
    print("\n" + "=" * 80)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("=" * 80)
    
    return gdf_final


if __name__ == "__main__":
    gdf = main()
