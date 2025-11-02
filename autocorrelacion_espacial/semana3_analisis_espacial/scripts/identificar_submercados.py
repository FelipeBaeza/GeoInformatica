#!/usr/bin/env python3
"""
Script para identificar submercados geográficos mediante clustering espacial

Autor: Proyecto GeoInformática - Semana 3
Fecha: Noviembre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def cargar_grilla():
    """Carga grilla con características"""
    print("=" * 80)
    print("CARGANDO DATOS")
    print("=" * 80)
    
    import os
    script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    base_dir = os.path.dirname(script_dir)
    grilla_path = os.path.join(base_dir, "semana2_caracteristicas_espaciales", "features", "grilla_con_densidades.geojson")
    grilla = gpd.read_file(grilla_path)
    print(f"✓ Grilla cargada: {len(grilla)} puntos")
    
    return grilla

def preparar_datos_clustering(grilla):
    """
    Prepara datos para clustering: selecciona características y normaliza
    """
    print("\n" + "=" * 80)
    print("PREPARACIÓN DE DATOS PARA CLUSTERING")
    print("=" * 80)
    
    # Seleccionar características relevantes
    # Excluir: geometría, IDs, coordenadas, comunas
    columnas_excluir = ['geometry', 'punto_id', 'comuna', 'x', 'y']
    
    # Características numéricas
    caracteristicas = [col for col in grilla.columns 
                      if col not in columnas_excluir 
                      and grilla[col].dtype in ['float64', 'int64']]
    
    print(f"✓ {len(caracteristicas)} características seleccionadas para clustering")
    
    # Crear matriz de características
    X = grilla[caracteristicas].copy()
    
    # Eliminar filas con NaN
    X_clean = X.dropna()
    indices_validos = X_clean.index
    
    print(f"✓ {len(X_clean)} observaciones válidas (sin NaN)")
    
    # Normalizar (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)
    
    print(f"✓ Datos normalizados (media=0, std=1)")
    
    return X_scaled, indices_validos, caracteristicas

def determinar_numero_optimo_clusters(X, min_k=3, max_k=10):
    """
    Usa método del codo y silhouette score para determinar k óptimo
    """
    print("\n" + "=" * 80)
    print("DETERMINACIÓN DE NÚMERO ÓPTIMO DE CLUSTERS")
    print("=" * 80)
    
    inertias = []
    silhouettes = []
    db_scores = []
    k_range = range(min_k, max_k + 1)
    
    print("\nProbando diferentes valores de k...")
    
    for k in k_range:
        print(f"  k={k}...", end=' ')
        
        # K-Means
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        # Métricas
        inertias.append(kmeans.inertia_)
        silhouettes.append(silhouette_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))
        
        print(f"Silhouette={silhouettes[-1]:.3f}, DB={db_scores[-1]:.3f}")
    
    # Determinar k óptimo
    # Silhouette más alto = mejor
    k_optimo_silhouette = list(k_range)[np.argmax(silhouettes)]
    
    # Davies-Bouldin más bajo = mejor
    k_optimo_db = list(k_range)[np.argmin(db_scores)]
    
    print(f"\n✓ k óptimo según Silhouette: {k_optimo_silhouette}")
    print(f"✓ k óptimo según Davies-Bouldin: {k_optimo_db}")
    
    # Generar gráfico
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Método del codo
    axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Número de clusters (k)', fontsize=12)
    axes[0].set_ylabel('Inercia (Within-cluster sum of squares)', fontsize=12)
    axes[0].set_title('Método del Codo', fontsize=14, fontweight='bold')
    axes[0].grid(alpha=0.3)
    
    # Silhouette score
    axes[1].plot(k_range, silhouettes, 'ro-', linewidth=2, markersize=8)
    axes[1].axvline(k_optimo_silhouette, color='red', linestyle='--', 
                   label=f'Óptimo: k={k_optimo_silhouette}')
    axes[1].set_xlabel('Número de clusters (k)', fontsize=12)
    axes[1].set_ylabel('Silhouette Score', fontsize=12)
    axes[1].set_title('Silhouette Score (mayor = mejor)', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Davies-Bouldin index
    axes[2].plot(k_range, db_scores, 'go-', linewidth=2, markersize=8)
    axes[2].axvline(k_optimo_db, color='green', linestyle='--',
                   label=f'Óptimo: k={k_optimo_db}')
    axes[2].set_xlabel('Número de clusters (k)', fontsize=12)
    axes[2].set_ylabel('Davies-Bouldin Index', fontsize=12)
    axes[2].set_title('Davies-Bouldin Index (menor = mejor)', fontsize=14, fontweight='bold')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    # No guardar aquí, se guarda en main()
    # plt.savefig('../submercados/determinacion_k_optimo.png', dpi=300, bbox_inches='tight')
    # plt.close()
    
    # print(f"✓ Gráfico guardado: ../submercados/determinacion_k_optimo.png")
    
    # Usar promedio de ambos como k óptimo
    k_optimo = int(np.round((k_optimo_silhouette + k_optimo_db) / 2))
    print(f"\n🎯 k óptimo final (promedio): {k_optimo}")
    
    return k_optimo, {
        'k_range': list(k_range),
        'inertias': inertias,
        'silhouettes': silhouettes,
        'db_scores': db_scores,
        'k_optimo_silhouette': k_optimo_silhouette,
        'k_optimo_db': k_optimo_db,
        'k_optimo_final': k_optimo
    }

def aplicar_kmeans(X, k):
    """
    Aplica K-Means clustering
    """
    print(f"\n" + "=" * 80)
    print(f"APLICANDO K-MEANS CLUSTERING (k={k})")
    print("=" * 80)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=20, max_iter=300)
    labels = kmeans.fit_predict(X)
    
    # Métricas
    silhouette = silhouette_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    print(f"✓ Clustering completado")
    print(f"  - Silhouette Score: {silhouette:.4f}")
    print(f"  - Davies-Bouldin Index: {db_score:.4f}")
    
    # Distribución de clusters
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\n  Distribución de clusters:")
    for cluster_id, count in zip(unique, counts):
        print(f"    Cluster {cluster_id}: {count} puntos ({count/len(labels)*100:.1f}%)")
    
    return labels, kmeans, {'silhouette': silhouette, 'db_score': db_score}

def caracterizar_submercados(grilla, labels, indices_validos, caracteristicas):
    """
    Caracteriza cada submercado calculando estadísticas por cluster
    """
    print("\n" + "=" * 80)
    print("CARACTERIZACIÓN DE SUBMERCADOS")
    print("=" * 80)
    
    # Agregar labels al GeoDataFrame
    gdf = grilla.loc[indices_validos].copy()
    gdf['submercado'] = labels
    
    perfiles = {}
    
    for cluster_id in sorted(gdf['submercado'].unique()):
        print(f"\n📊 Submercado {cluster_id}:")
        
        subset = gdf[gdf['submercado'] == cluster_id]
        
        perfil = {
            'id': int(cluster_id),
            'n_puntos': len(subset),
            'pct_total': len(subset) / len(gdf) * 100,
            'caracteristicas_destacadas': {},
            'promedios': {}
        }
        
        # Calcular promedios
        for car in caracteristicas:
            valor_medio = subset[car].mean()
            perfil['promedios'][car] = float(valor_medio)
        
        # Identificar top 5 características (normalizadas vs toda la grilla)
        z_scores = {}
        for car in caracteristicas:
            media_cluster = subset[car].mean()
            media_global = gdf[car].mean()
            std_global = gdf[car].std()
            
            if std_global > 0:
                z = (media_cluster - media_global) / std_global
                z_scores[car] = z
        
        # Top 5 características más distintivas (z-score más alto en valor absoluto)
        top_features = sorted(z_scores.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
        
        print(f"  - Puntos: {perfil['n_puntos']} ({perfil['pct_total']:.1f}%)")
        print(f"  - Características distintivas:")
        
        for feat, z in top_features:
            direccion = "↑ ALTO" if z > 0 else "↓ BAJO"
            perfil['caracteristicas_destacadas'][feat] = {
                'z_score': float(z),
                'direccion': direccion,
                'valor_promedio': float(subset[feat].mean())
            }
            print(f"    • {feat}: {direccion} (z={z:.2f})")
        
        perfiles[f'submercado_{cluster_id}'] = perfil
    
    return gdf, perfiles

def generar_mapa_submercados(gdf, output_path):
    """
    Genera mapa de submercados
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 12))
    
    # Colores para cada submercado
    n_clusters = gdf['submercado'].nunique()
    colores = plt.cm.Set3(np.linspace(0, 1, n_clusters))
    
    for i, cluster_id in enumerate(sorted(gdf['submercado'].unique())):
        subset = gdf[gdf['submercado'] == cluster_id]
        subset.plot(ax=ax, color=colores[i], edgecolor='black', linewidth=0.3,
                   label=f'Submercado {cluster_id} (n={len(subset)})', alpha=0.7)
    
    ax.set_title('Submercados Identificados por Clustering Espacial', 
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
    
    print(f"✓ Mapa de submercados guardado: {output_path}")

def generar_reporte_submercados(perfiles, metricas, output_path):
    """
    Genera reporte Markdown de submercados
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# Reporte de Submercados Identificados\n")
        f.write("## Clustering Espacial K-Means\n\n")
        
        f.write(f"**Fecha de análisis:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("---\n\n")
        f.write("## 📊 Métricas de Calidad del Clustering\n\n")
        
        f.write(f"- **Silhouette Score:** {metricas['silhouette']:.4f} (rango: -1 a 1, mayor = mejor)\n")
        f.write(f"- **Davies-Bouldin Index:** {metricas['db_score']:.4f} (menor = mejor)\n\n")
        
        f.write("---\n\n")
        f.write("## 🗺️ Submercados Identificados\n\n")
        
        for nombre_sub, perfil in perfiles.items():
            f.write(f"### Submercado {perfil['id']}\n\n")
            f.write(f"**Tamaño:** {perfil['n_puntos']} puntos ({perfil['pct_total']:.1f}% del total)\n\n")
            
            f.write("**Características Distintivas:**\n\n")
            f.write("| Característica | Dirección | Z-Score | Valor Promedio |\n")
            f.write("|----------------|-----------|---------|----------------|\n")
            
            for feat, stats in perfil['caracteristicas_destacadas'].items():
                f.write(f"| {feat} | {stats['direccion']} | {stats['z_score']:.2f} | {stats['valor_promedio']:.2f} |\n")
            
            f.write("\n")
            
            # Interpretación
            f.write("**Interpretación:**\n\n")
            
            altos = [f for f, s in perfil['caracteristicas_destacadas'].items() 
                    if s['z_score'] > 0.5]
            bajos = [f for f, s in perfil['caracteristicas_destacadas'].items() 
                    if s['z_score'] < -0.5]
            
            if altos:
                f.write(f"Este submercado se caracteriza por tener valores **ALTOS** en: {', '.join(altos[:3])}.\n\n")
            
            if bajos:
                f.write(f"Y valores **BAJOS** en: {', '.join(bajos[:3])}.\n\n")
            
            f.write("---\n\n")
        
        f.write("## 🎯 Implicaciones para el Sistema de Recomendación\n\n")
        f.write("1. Cada submercado tiene un **perfil único** de características\n")
        f.write("2. Los usuarios pueden tener **preferencias diferentes** según el submercado\n")
        f.write("3. Se debe **personalizar** la valoración de características por submercado\n")
        f.write("4. Los **modelos locales** (GWR/MGWR) capturarán mejor estas diferencias\n\n")
    
    print(f"✓ Reporte de submercados guardado: {output_path}")

def main():
    """Función principal"""
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
    print(" " * 20 + "IDENTIFICACIÓN DE SUBMERCADOS GEOGRÁFICOS")
    print(" " * 25 + "Clustering Espacial - Semana 3")
    print("=" * 80 + "\n")
    
    # 1. Cargar grilla
    grilla = cargar_grilla()
    
    # 2. Preparar datos
    X_norm, indices_validos, caracteristicas = preparar_datos_clustering(grilla)
    
    # 3. Determinar k óptimo
    k_optimo, metricas_k = determinar_numero_optimo_clusters(X_norm, min_k=3, max_k=10)
    
    # Guardar gráfico de determinación de k
    plt.savefig(os.path.join(submercados_dir, 'determinacion_k_optimo.png'), dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {os.path.join(submercados_dir, 'determinacion_k_optimo.png')}")
    plt.close()
    
    # 4. Aplicar K-Means con k óptimo
    labels, kmeans_model, metricas = aplicar_kmeans(X_norm, k=k_optimo)
    
    # 5. Caracterizar submercados
    gdf_caracterizado, perfiles = caracterizar_submercados(grilla, labels, indices_validos, caracteristicas)
    
    # 6. Generar mapa de submercados
    print("\n" + "=" * 80)
    print("GENERANDO MAPA DE SUBMERCADOS")
    print("=" * 80)
    
    # Crear grilla con submercados
    gdf_submercados = grilla.loc[indices_validos].copy()
    gdf_submercados['submercado'] = labels
    
    # Generar mapa
    generar_mapa_submercados(gdf_submercados, os.path.join(mapas_dir, 'mapa_submercados.png'))
    
    # 7. Guardar GeoJSON
    geojson_path = os.path.join(submercados_dir, "grilla_con_submercados.geojson")
    gdf_submercados.to_file(geojson_path, driver='GeoJSON')
    print(f"✓ GeoJSON guardado: {geojson_path}")
    
    # 8. Generar reportes
    print("\n" + "=" * 80)
    print("GENERANDO REPORTES")
    print("=" * 80)
    
    # Markdown
    md_path = os.path.join(reportes_dir, "submercados_reporte.md")
    generar_reporte_submercados(perfiles, metricas, md_path)
    
    # JSON
    json_path = os.path.join(reportes_dir, "submercados_perfiles.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump({
            'metadata': {
                'fecha': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'k_optimo': k_optimo,
                'metricas': metricas,
                'metricas_seleccion_k': metricas_k
            },
            'submercados': perfiles
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON guardado: {json_path}")
    
    # 9. Resumen final
    print("\n" + "=" * 80)
    print("RESUMEN FINAL")
    print("=" * 80)
    
    print(f"\n📊 SUBMERCADOS IDENTIFICADOS: {k_optimo}")
    print(f"   Silhouette Score: {metricas['silhouette']:.4f}")
    print(f"   Davies-Bouldin Index: {metricas['db_score']:.4f}")
    
    print(f"\n✅ ANÁLISIS COMPLETADO")

if __name__ == "__main__":
    main()
