#!/usr/bin/env python3
"""
Análisis de Autocorrelación Espacial para Índices de Satisfacción
Aplicando el método de Moran corregido a los índices de satisfacción
"""

import json
import math
import statistics
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import numpy as np
from collections import defaultdict

class AnalisisMoranSatisfaccion:
    def __init__(self, geojson_path):
        """Cargar datos integrados de satisfacción"""
        print("Cargando datos integrados de satisfacción...")
        
        # Cargar GeoDataFrame
        self.gdf = gpd.read_file(geojson_path)
        
        # Extraer coordenadas y datos
        self.coordinates = [[geom.x, geom.y] for geom in self.gdf.geometry]
        self.n = len(self.gdf)
        
        # Índices de satisfacción a analizar
        self.indices_satisfaccion = [
            'indice_transporte',
            'indice_servicios', 
            'indice_seguridad',
            'indice_conveniencia',
            'indice_calidad_vida',
            'total_uf'  # Para comparar con precio
        ]
        
        print(f"Datos cargados: {self.n} propiedades")
        print(f"Índices a analizar: {len(self.indices_satisfaccion)}")
        
    def calcular_distancia(self, coord1, coord2):
        """Calcular distancia euclidiana entre dos puntos"""
        lon1, lat1 = coord1
        lon2, lat2 = coord2
        
        # Conversión a metros para Santiago
        lat_factor = 111320
        lon_factor = 91290
        
        dx = (lon2 - lon1) * lon_factor
        dy = (lat2 - lat1) * lat_factor
        
        return math.sqrt(dx*dx + dy*dy)
    
    def crear_matriz_pesos(self, k=8):
        """Crear matriz de pesos espaciales usando k-vecinos más cercanos"""
        print(f"Creando matriz de pesos espaciales (k={k} vecinos)...")
        
        n = self.n
        self.weights = [[0.0 for _ in range(n)] for _ in range(n)]
        
        for i in range(n):
            # Calcular distancias a todos los otros puntos
            distances = []
            for j in range(n):
                if i != j:
                    dist = self.calcular_distancia(self.coordinates[i], self.coordinates[j])
                    distances.append((j, dist))
            
            # Ordenar por distancia y tomar k vecinos más cercanos
            distances.sort(key=lambda x: x[1])
            neighbors = distances[:k]
            
            # Asignar pesos uniformes (1/k para cada vecino)
            for neighbor_idx, _ in neighbors:
                self.weights[i][neighbor_idx] = 1.0 / k
        
        print("Matriz de pesos creada")
    
    def moran_global(self, variable_name):
        """Calcular Índice de Moran Global para una variable"""
        
        if variable_name not in self.gdf.columns:
            print(f"Variable {variable_name} no encontrada")
            return None, None
        
        print(f"\n=== MORAN GLOBAL: {variable_name.upper()} ===")
        
        n = self.n
        y = self.gdf[variable_name].values
        w = self.weights
        
        # Filtrar valores válidos
        valid_mask = ~np.isnan(y) & ~np.isinf(y)
        if not valid_mask.all():
            print(f"Advertencia: {(~valid_mask).sum()} valores inválidos encontrados")
            y = y[valid_mask]
            n = len(y)
        
        # Media
        y_mean = statistics.mean(y)
        
        # Suma de pesos
        S0 = sum(sum(row) for row in w)
        
        # Numerador
        numerator = 0
        for i in range(n):
            for j in range(n):
                numerator += w[i][j] * (y[i] - y_mean) * (y[j] - y_mean)
        
        # Denominador
        denominator = sum((yi - y_mean)**2 for yi in y)
        
        # Índice de Moran
        if denominator > 0 and S0 > 0:
            I = (n / S0) * (numerator / denominator)
        else:
            I = 0
        
        # Valor esperado
        EI = -1 / (n - 1)
        
        # Varianza simplificada
        VI = 1 / (n - 1)
        std_I = math.sqrt(VI)
        z_score = (I - EI) / std_I if std_I > 0 else 0
        
        # P-valor aproximado
        p_value = 2 * (1 - self.normal_cdf(abs(z_score)))
        
        # Mostrar resultados
        print(f"Variable: {variable_name}")
        print(f"Valores válidos: {n}/{self.n}")
        print(f"Media: {y_mean:.4f}")
        print(f"Indice de Moran (I): {I:.4f}")
        print(f"Valor esperado E(I): {EI:.4f}")
        print(f"Z-score: {z_score:.4f}")
        print(f"P-valor: {p_value:.6f}")
        
        if p_value < 0.05:
            if I > EI:
                interpretacion = "Autocorrelación espacial POSITIVA significativa"
                print(f"RESULTADO: {interpretacion}")
            else:
                interpretacion = "Autocorrelación espacial NEGATIVA significativa"
                print(f"RESULTADO: {interpretacion}")
        else:
            interpretacion = "No hay autocorrelación espacial significativa"
            print(f"RESULTADO: {interpretacion}")
        
        return I, p_value, interpretacion
    
    def moran_local(self, variable_name):
        """Calcular Índices de Moran Local (LISA) para una variable"""
        
        if variable_name not in self.gdf.columns:
            print(f"Variable {variable_name} no encontrada")
            return None
        
        print(f"\n=== MORAN LOCAL (LISA): {variable_name.upper()} ===")
        
        n = self.n
        y = self.gdf[variable_name].values
        w = self.weights
        y_mean = statistics.mean(y[~np.isnan(y) & ~np.isinf(y)])
        
        # Calcular LISA para cada ubicación
        lisa_values = []
        lisa_clusters = []
        
        for i in range(n):
            if np.isnan(y[i]) or np.isinf(y[i]):
                lisa_values.append(np.nan)
                lisa_clusters.append("NA")
                continue
            
            # LISA local
            lisa_i = 0
            for j in range(n):
                if not (np.isnan(y[j]) or np.isinf(y[j])):
                    lisa_i += w[i][j] * (y[j] - y_mean)
            lisa_i = (y[i] - y_mean) * lisa_i
            
            lisa_values.append(lisa_i)
            
            # Clasificar cluster usando la corrección de la memoria
            # Calcular media ponderada de vecinos
            neighbors_mean = sum(w[i][j] * y[j] for j in range(n) 
                               if not (np.isnan(y[j]) or np.isinf(y[j])))
            
            # Clasificar basado en si el valor y sus vecinos están por encima o debajo de la media
            if y[i] > y_mean and neighbors_mean > y_mean:
                cluster = "HH"  # Alto-Alto
            elif y[i] < y_mean and neighbors_mean < y_mean:
                cluster = "LL"  # Bajo-Bajo
            elif y[i] > y_mean and neighbors_mean < y_mean:
                cluster = "HL"  # Alto-Bajo (outlier alto)
            else:  # y[i] < y_mean and neighbors_mean > y_mean
                cluster = "LH"  # Bajo-Alto (outlier bajo)
            
            lisa_clusters.append(cluster)
        
        # Contar tipos de clusters
        cluster_counts = defaultdict(int)
        for cluster in lisa_clusters:
            if cluster != "NA":
                cluster_counts[cluster] += 1
        
        print("Distribución de clusters LISA:")
        for cluster, count in cluster_counts.items():
            pct = (count / n) * 100
            print(f"  {cluster}: {count} propiedades ({pct:.1f}%)")
        
        return lisa_values, lisa_clusters
    
    def normal_cdf(self, x):
        """Función de distribución normal estándar"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def analizar_todos_los_indices(self):
        """Analizar autocorrelación espacial para todos los índices"""
        
        print("="*80)
        print("ANÁLISIS DE AUTOCORRELACIÓN ESPACIAL - ÍNDICES DE SATISFACCIÓN")
        print("="*80)
        
        # Crear matriz de pesos
        self.crear_matriz_pesos(k=8)
        
        resultados = {}
        
        for indice in self.indices_satisfaccion:
            if indice in self.gdf.columns:
                # Moran Global
                I, p_value, interpretacion = self.moran_global(indice)
                
                # Moran Local
                lisa_values, lisa_clusters = self.moran_local(indice)
                
                resultados[indice] = {
                    'moran_I': I,
                    'p_value': p_value,
                    'interpretacion': interpretacion,
                    'lisa_values': lisa_values,
                    'lisa_clusters': lisa_clusters
                }
            else:
                print(f"⚠️ Índice {indice} no encontrado en los datos")
        
        return resultados
    
    def crear_mapa_comparativo(self, resultados):
        """Crear mapas comparativos de clusters LISA"""
        
        print("\n=== GENERANDO MAPAS COMPARATIVOS ===")
        
        # Filtrar índices válidos
        indices_validos = [idx for idx in resultados.keys() if resultados[idx]['moran_I'] is not None]
        
        if not indices_validos:
            print("No hay índices válidos para mapear")
            return
        
        # Configurar subplots
        n_indices = len(indices_validos)
        cols = 3
        rows = (n_indices + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = [axes] if n_indices == 1 else axes
        else:
            axes = axes.flatten()
        
        # Colores para cada tipo de cluster
        color_map = {
            'HH': 'red',      # Alto-Alto: rojo
            'LL': 'blue',     # Bajo-Bajo: azul
            'HL': 'orange',   # Alto-Bajo: naranja
            'LH': 'green',    # Bajo-Alto: verde
            'NA': 'gray'      # No disponible: gris
        }
        
        for i, indice in enumerate(indices_validos):
            ax = axes[i]
            
            clusters = resultados[indice]['lisa_clusters']
            
            # Extraer coordenadas
            lons = [coord[0] for coord in self.coordinates]
            lats = [coord[1] for coord in self.coordinates]
            
            # Plotear por tipo de cluster
            for cluster_type in color_map.keys():
                cluster_lons = [lons[j] for j in range(len(lons)) if clusters[j] == cluster_type]
                cluster_lats = [lats[j] for j in range(len(lats)) if clusters[j] == cluster_type]
                
                if cluster_lons:
                    ax.scatter(cluster_lons, cluster_lats, 
                             c=color_map[cluster_type], 
                             label=f'{cluster_type}', 
                             alpha=0.7, s=20)
            
            ax.set_xlabel('Longitud')
            ax.set_ylabel('Latitud')
            ax.set_title(f'Clusters LISA - {indice}')
            ax.legend(title='Tipo de Cluster')
            ax.grid(True, alpha=0.3)
        
        # Ocultar subplots vacíos
        for i in range(len(indices_validos), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('mapas_lisa_satisfaccion.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Mapas guardados en: mapas_lisa_satisfaccion.png")
    
    def generar_reporte_final(self, resultados):
        """Generar reporte final del análisis"""
        
        print("\n" + "="*80)
        print("📊 REPORTE FINAL - AUTOCORRELACIÓN ESPACIAL DE SATISFACCIÓN")
        print("="*80)
        
        print(f"\n🏠 DATOS ANALIZADOS:")
        print(f"  • Total propiedades: {self.n}")
        print(f"  • Índices analizados: {len(resultados)}")
        
        print(f"\n📈 RESULTADOS DE AUTOCORRELACIÓN ESPACIAL:")
        
        for indice, resultado in resultados.items():
            if resultado['moran_I'] is not None:
                print(f"\n  🎯 {indice.upper()}:")
                print(f"    • Moran's I: {resultado['moran_I']:.4f}")
                print(f"    • P-valor: {resultado['p_value']:.6f}")
                print(f"    • Interpretación: {resultado['interpretacion']}")
                
                # Contar clusters LISA
                clusters = resultado['lisa_clusters']
                cluster_counts = defaultdict(int)
                for cluster in clusters:
                    if cluster != "NA":
                        cluster_counts[cluster] += 1
                
                print(f"    • Clusters LISA:")
                for cluster, count in cluster_counts.items():
                    pct = (count / self.n) * 100
                    print(f"      - {cluster}: {count} ({pct:.1f}%)")
        
        print(f"\n🔍 INTERPRETACIÓN PARA RECOMENDACIONES:")
        print(f"  • HH (Alto-Alto): Zonas de alta satisfacción consolidadas")
        print(f"  • LL (Bajo-Bajo): Zonas de baja satisfacción que requieren mejoras")
        print(f"  • HL (Alto-Bajo): Oportunidades premium en zonas en desarrollo")
        print(f"  • LH (Bajo-Alto): Propiedades subvaloradas en zonas buenas")

def main():
    """Función principal"""
    archivo_datos = "datos_integrados_satisfaccion.geojson"
    
    try:
        # Crear analizador
        analyzer = AnalisisMoranSatisfaccion(archivo_datos)
        
        # Ejecutar análisis completo
        resultados = analyzer.analizar_todos_los_indices()
        
        # Crear mapas comparativos
        analyzer.crear_mapa_comparativo(resultados)
        
        # Generar reporte final
        analyzer.generar_reporte_final(resultados)
        
        print("\n✅ Análisis de autocorrelación espacial completado!")
        
    except FileNotFoundError:
        print(f"❌ Error: No se encuentra el archivo {archivo_datos}")
        print("Ejecuta primero: python integrador_datos_satisfaccion.py")
    except Exception as e:
        print(f"❌ Error durante el análisis: {str(e)}")

if __name__ == "__main__":
    main()
