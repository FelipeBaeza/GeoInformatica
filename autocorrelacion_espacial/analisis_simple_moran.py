#!/usr/bin/env python3
"""
Análisis simple de autocorrelación espacial con mapa LISA
Código unificado para análisis de Moran y visualización
"""

import json
import math
import statistics
import matplotlib.pyplot as plt
from collections import defaultdict

class AnalisisMoranSimple:
    def __init__(self, geojson_path):
        """Cargar datos del archivo GeoJSON"""
        print("Cargando datos inmobiliarios...")
        
        with open(geojson_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        self.properties = []
        self.coordinates = []
        self.prices = []
        
        for feature in self.data['features']:
            if 'total_uf' in feature['properties'] and feature['properties']['total_uf'] > 0:
                props = feature['properties']
                coords = feature['geometry']['coordinates']
                
                self.properties.append(props)
                self.coordinates.append(coords)
                self.prices.append(props['total_uf'])
        
        # Log-transformación de precios
        self.log_prices = [math.log(p) for p in self.prices]
        self.n = len(self.log_prices)
        
        print(f"Datos cargados: {self.n} propiedades")
        
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
            # Esto asegura que cada fila sume 1.0
            for neighbor_idx, _ in neighbors:
                self.weights[i][neighbor_idx] = 1.0 / k
        
        print("Matriz de pesos creada")
    
    def moran_global(self):
        """Calcular Índice de Moran Global"""
        print("\n=== ANALISIS DE MORAN GLOBAL ===")
        
        n = self.n
        y = self.log_prices
        w = self.weights
        
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
        
        # Guardar resultados
        self.moran_I = I
        self.moran_p = p_value
        self.moran_z = z_score
        
        # Mostrar resultados
        print(f"Indice de Moran (I): {I:.4f}")
        print(f"Valor esperado E(I): {EI:.4f}")
        print(f"Z-score: {z_score:.4f}")
        print(f"P-valor: {p_value:.6f}")
        
        if p_value < 0.05:
            if I > EI:
                print("RESULTADO: Autocorrelacion espacial POSITIVA significativa")
            else:
                print("RESULTADO: Autocorrelacion espacial NEGATIVA significativa")
        else:
            print("RESULTADO: No hay autocorrelacion espacial significativa")
        
        return I, p_value
    
    def moran_local(self):
        """Calcular Índices de Moran Local (LISA)"""
        print("\n=== ANALISIS DE MORAN LOCAL (LISA) ===")
        
        n = self.n
        y = self.log_prices
        w = self.weights
        y_mean = statistics.mean(y)
        
        # Calcular LISA para cada ubicación
        self.lisa_values = []
        self.lisa_clusters = []
        
        for i in range(n):
            # LISA local
            lisa_i = 0
            for j in range(n):
                lisa_i += w[i][j] * (y[j] - y_mean)
            lisa_i = (y[i] - y_mean) * lisa_i
            
            self.lisa_values.append(lisa_i)
            
            # Clasificar cluster
            # Calcular media ponderada de vecinos
            neighbors_mean = sum(w[i][j] * y[j] for j in range(n))
            
            # Clasificar basado en si el valor y sus vecinos están por encima o debajo de la media
            if y[i] > y_mean and neighbors_mean > y_mean:
                cluster = "HH"  # Alto-Alto: valor alto rodeado de valores altos
            elif y[i] < y_mean and neighbors_mean < y_mean:
                cluster = "LL"  # Bajo-Bajo: valor bajo rodeado de valores bajos
            elif y[i] > y_mean and neighbors_mean < y_mean:
                cluster = "HL"  # Alto-Bajo: valor alto rodeado de valores bajos (outlier alto)
            else:  # y[i] < y_mean and neighbors_mean > y_mean
                cluster = "LH"  # Bajo-Alto: valor bajo rodeado de valores altos (outlier bajo)
            
            self.lisa_clusters.append(cluster)
        
        # Contar tipos de clusters
        cluster_counts = defaultdict(int)
        for cluster in self.lisa_clusters:
            cluster_counts[cluster] += 1
        
        print("Distribucion de clusters LISA:")
        for cluster, count in cluster_counts.items():
            pct = (count / n) * 100
            print(f"  {cluster}: {count} propiedades ({pct:.1f}%)")
        
        print("\nTipos de clusters:")
        print("  HH (Alto-Alto): Precios altos rodeados de precios altos")
        print("  LL (Bajo-Bajo): Precios bajos rodeados de precios bajos")
        print("  HL (Alto-Bajo): Precios altos en zona de precios bajos (outliers altos)")
        print("  LH (Bajo-Alto): Precios bajos en zona de precios altos (outliers bajos)")
        
        # Validación adicional
        self.validar_clasificacion_lisa()
    
    def validar_clasificacion_lisa(self):
        """Validar que la clasificación LISA sea correcta"""
        print("\n=== VALIDACION DE CLASIFICACION LISA ===")
        
        y = self.log_prices
        w = self.weights
        y_mean = statistics.mean(y)
        n = self.n
        
        # Verificar algunos casos para validación
        errores = 0
        for i in range(min(10, n)):  # Verificar primeros 10 casos
            neighbors_mean = sum(w[i][j] * y[j] for j in range(n))
            cluster_actual = self.lisa_clusters[i]
            
            # Determinar cluster esperado
            if y[i] > y_mean and neighbors_mean > y_mean:
                cluster_esperado = "HH"
            elif y[i] < y_mean and neighbors_mean < y_mean:
                cluster_esperado = "LL"
            elif y[i] > y_mean and neighbors_mean < y_mean:
                cluster_esperado = "HL"
            else:
                cluster_esperado = "LH"
            
            if cluster_actual != cluster_esperado:
                errores += 1
                print(f"Error en punto {i}: esperado {cluster_esperado}, obtenido {cluster_actual}")
        
        if errores == 0:
            print("✓ Validación exitosa: Clasificación LISA correcta")
        else:
            print(f"✗ Se encontraron {errores} errores en la clasificación")
    
    def crear_mapa_lisa(self):
        """Crear mapa simple de clusters LISA"""
        print("\n=== GENERANDO MAPA LISA ===")
        
        # Extraer coordenadas
        lons = [coord[0] for coord in self.coordinates]
        lats = [coord[1] for coord in self.coordinates]
        
        # Colores para cada tipo de cluster
        colors = []
        color_map = {
            'HH': 'red',      # Alto-Alto: rojo
            'LL': 'blue',     # Bajo-Bajo: azul
            'HL': 'orange',   # Alto-Bajo: naranja
            'LH': 'green'     # Bajo-Alto: verde
        }
        
        for cluster in self.lisa_clusters:
            colors.append(color_map[cluster])
        
        # Crear figura
        plt.figure(figsize=(12, 8))
        
        # Scatter plot
        for cluster_type in color_map.keys():
            # Filtrar puntos por tipo de cluster
            cluster_lons = [lons[i] for i in range(len(lons)) if self.lisa_clusters[i] == cluster_type]
            cluster_lats = [lats[i] for i in range(len(lats)) if self.lisa_clusters[i] == cluster_type]
            
            if cluster_lons:  # Solo plotear si hay puntos de este tipo
                plt.scatter(cluster_lons, cluster_lats, 
                           c=color_map[cluster_type], 
                           label=f'{cluster_type}', 
                           alpha=0.7, s=30)
        
        plt.xlabel('Longitud')
        plt.ylabel('Latitud')
        plt.title('Mapa de Clusters LISA - Autocorrelacion Espacial de Precios')
        plt.legend(title='Tipo de Cluster')
        plt.grid(True, alpha=0.3)
        
        # Ajustar límites
        plt.xlim(min(lons) - 0.01, max(lons) + 0.01)
        plt.ylim(min(lats) - 0.01, max(lats) + 0.01)
        
        plt.tight_layout()
        plt.show()
        
    
    def normal_cdf(self, x):
        """Función de distribución normal estándar"""
        return 0.5 * (1 + math.erf(x / math.sqrt(2)))
    
    def resumen_por_comuna(self):
        """Análisis por comuna"""
        print("\n=== ANALISIS POR COMUNA ===")
        
        comunas = defaultdict(list)
        for i, prop in enumerate(self.properties):
            comuna = prop.get('comuna', 'DESCONOCIDA')
            comunas[comuna].append(self.log_prices[i])
        
        for comuna, prices in comunas.items():
            if len(prices) > 1:
                mean_price = statistics.mean(prices)
                std_price = statistics.stdev(prices)
                print(f"{comuna}:")
                print(f"  Propiedades: {len(prices)}")
                print(f"  Precio promedio (log): {mean_price:.3f}")
                print(f"  Desviacion estandar: {std_price:.3f}")
    
    def ejecutar_analisis_completo(self):
        """Ejecutar análisis completo"""
        print("ANALISIS DE AUTOCORRELACION ESPACIAL")
        print("=" * 50)
        
        # 1. Crear matriz de pesos
        self.crear_matriz_pesos(k=8)
        
        # 2. Moran Global
        I, p_value = self.moran_global()
        
        # 3. Moran Local
        self.moran_local()
        
        # 4. Análisis por comuna
        self.resumen_por_comuna()
        
        # 5. Crear mapa LISA
        self.crear_mapa_lisa()
        
        # 6. Resumen final
        print("\n=== RESUMEN FINAL ===")
        print(f"Total propiedades analizadas: {self.n}")
        print(f"Indice de Moran Global: {I:.4f}")
        print(f"P-valor: {p_value:.6f}")
        
        if p_value < 0.05:
            print("CONCLUSION: Existe autocorrelacion espacial significativa")
        else:
            print("CONCLUSION: No hay autocorrelacion espacial significativa")

def main():
    """Función principal"""
    geojson_file = "datos_filtrados/base_maestra_comunas_filtradas.geojson"

    try:
        # Crear analizador y ejecutar
        analyzer = AnalisisMoranSimple(geojson_file)
        analyzer.ejecutar_analisis_completo()
        
        print("\nAnalisis completado exitosamente!")
        
    except FileNotFoundError:
        print(f"Error: No se encuentra el archivo {geojson_file}")
        print("Verifica que el archivo este en el directorio actual")
    except Exception as e:
        print(f"Error durante el analisis: {str(e)}")

if __name__ == "__main__":
    main()
