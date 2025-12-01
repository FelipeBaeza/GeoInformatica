#!/usr/bin/env python3
"""
API de Predicción para Propiedades en Venta

Este script proporciona funciones para:
1. Predecir satisfacción de una propiedad nueva
2. Comparar múltiples propiedades
3. Filtrar propiedades por perfil de usuario
4. Generar ranking de propiedades

Uso:
    from predecir_satisfaccion import PredictorSatisfaccion
    
    predictor = PredictorSatisfaccion()
    resultado = predictor.predecir(propiedad)
    ranking = predictor.ranking_propiedades(df, perfil='familia_con_ninos')
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import pickle
from pathlib import Path
import json
import re

# Paths
BASE_DIR = Path('/home/felipe/Documentos/GeoInformatica')
MODELOS_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana3_modelo_satisfaccion' / 'modelos'
OUTPUT_DIR = BASE_DIR / 'autocorrelacion_espacial' / 'semana3_modelo_satisfaccion' / 'resultados' / 'modelo_venta'

class PredictorSatisfaccion:
    """Clase para predecir satisfacción de propiedades en venta"""
    
    def __init__(self, modelo_path=None):
        """
        Inicializa el predictor cargando el modelo entrenado.
        
        Args:
            modelo_path: Path al archivo del modelo (opcional)
        """
        if modelo_path is None:
            modelo_path = MODELOS_DIR / 'modelo_satisfaccion_venta.pkl'
        
        if not Path(modelo_path).exists():
            raise FileNotFoundError(f"Modelo no encontrado: {modelo_path}\n"
                                  "Ejecute primero modelo_satisfaccion.py")
        
        with open(modelo_path, 'rb') as f:
            data = pickle.load(f)
        
        self.modelo = data['modelo']
        self.scaler = data['scaler']
        self.features = data['features']
        self.metricas = data.get('metricas', {})
        self.perfiles = data.get('perfiles', {})
        
        # Cargar dataset completo si existe
        self.df_propiedades = None
        csv_path = OUTPUT_DIR / 'propiedades_venta_con_satisfaccion.csv'
        if csv_path.exists():
            self.df_propiedades = pd.read_csv(csv_path)
        
        print(f"✅ Modelo cargado: {len(self.features)} features")
        print(f"   R² = {self.metricas.get('r2_test', 'N/A'):.4f}")
    
    def _preparar_features(self, propiedad):
        """Prepara las features para predicción"""
        X = pd.DataFrame([{f: propiedad.get(f, 0) for f in self.features}])
        
        # Rellenar NaN con medianas si es posible
        for col in X.columns:
            if X[col].isna().any() and self.df_propiedades is not None:
                if col in self.df_propiedades.columns:
                    X[col] = self.df_propiedades[col].median()
                else:
                    X[col] = 0
        
        return X
    
    def predecir(self, propiedad):
        """
        Predice la satisfacción para una propiedad.
        
        Args:
            propiedad: dict con las características de la propiedad
                       Mínimo: superficie_util, dormitorios, precio_m2_uf
        
        Returns:
            dict con satisfacción predicha y detalles
        """
        # Calcular features derivadas si no existen
        if 'precio_m2_uf' not in propiedad and 'precio_uf' in propiedad:
            propiedad['precio_m2_uf'] = propiedad['precio_uf'] / max(1, propiedad.get('superficie_util', 50))
        
        if 'm2_por_dormitorio' not in propiedad:
            dorms = propiedad.get('dormitorios', 1)
            if dorms == 0:
                dorms = 1
            propiedad['m2_por_dormitorio'] = propiedad.get('superficie_util', 50) / dorms
        
        if 'm2_por_habitante' not in propiedad:
            dorms = propiedad.get('dormitorios', 1)
            if dorms == 0:
                dorms = 1
            propiedad['m2_por_habitante'] = propiedad.get('superficie_util', 50) / (dorms * 2)
        
        if 'ratio_bano_dorm' not in propiedad:
            dorms = propiedad.get('dormitorios', 1)
            if dorms == 0:
                dorms = 1
            propiedad['ratio_bano_dorm'] = propiedad.get('banos', 1) / dorms
        
        if 'total_habitaciones' not in propiedad:
            propiedad['total_habitaciones'] = propiedad.get('dormitorios', 0) + propiedad.get('banos', 0)
        
        if 'es_departamento' not in propiedad:
            tipo = propiedad.get('tipo_propiedad', 'departamento').lower()
            propiedad['es_departamento'] = 1 if tipo == 'departamento' else 0
            propiedad['es_casa'] = 1 if tipo == 'casa' else 0
        
        # Preparar y escalar
        X = self._preparar_features(propiedad)
        X_scaled = self.scaler.transform(X)
        
        # Predecir
        satisfaccion = self.modelo.predict(X_scaled)[0]
        satisfaccion = min(10, max(0, satisfaccion))
        
        # Interpretar
        if satisfaccion >= 8:
            nivel = "Excelente"
            emoji = "🌟"
        elif satisfaccion >= 6:
            nivel = "Bueno"
            emoji = "✅"
        elif satisfaccion >= 4:
            nivel = "Regular"
            emoji = "⚠️"
        else:
            nivel = "Bajo"
            emoji = "❌"
        
        return {
            'satisfaccion': round(satisfaccion, 2),
            'nivel': nivel,
            'emoji': emoji,
            'escala': '0-10',
            'features_usadas': len(self.features),
            'r2_modelo': round(self.metricas.get('r2_test', 0), 4)
        }
    
    def comparar_propiedades(self, propiedades):
        """
        Compara múltiples propiedades y genera ranking.
        
        Args:
            propiedades: lista de dicts con características
        
        Returns:
            DataFrame con ranking
        """
        resultados = []
        for i, prop in enumerate(propiedades):
            pred = self.predecir(prop)
            resultados.append({
                'id': prop.get('id', i+1),
                'direccion': prop.get('direccion', f'Propiedad {i+1}'),
                'satisfaccion': pred['satisfaccion'],
                'nivel': pred['nivel'],
                'precio_uf': prop.get('precio_uf', 0),
                'superficie': prop.get('superficie_util', 0),
                'dormitorios': prop.get('dormitorios', 0),
                'tipo': prop.get('tipo_propiedad', 'N/A')
            })
        
        df = pd.DataFrame(resultados)
        df = df.sort_values('satisfaccion', ascending=False)
        df['ranking'] = range(1, len(df) + 1)
        
        return df
    
    def filtrar_por_satisfaccion(self, min_satisfaccion=6.0, tipo=None, comuna=None, 
                                  precio_max_uf=None, superficie_min=None):
        """
        Filtra propiedades del dataset por criterios.
        
        Args:
            min_satisfaccion: Satisfacción mínima (0-10)
            tipo: 'departamento' o 'casa'
            comuna: Nombre de la comuna
            precio_max_uf: Precio máximo en UF
            superficie_min: Superficie mínima en m²
        
        Returns:
            DataFrame filtrado
        """
        if self.df_propiedades is None:
            raise ValueError("Dataset no cargado")
        
        df = self.df_propiedades.copy()
        
        if 'satisfaccion_balanceado' in df.columns:
            df = df[df['satisfaccion_balanceado'] >= min_satisfaccion]
        
        if tipo:
            df = df[df['tipo_propiedad'] == tipo]
        
        if comuna:
            df = df[df['comuna'].str.lower().str.contains(comuna.lower())]
        
        if precio_max_uf:
            df = df[df['precio_uf'] <= precio_max_uf]
        
        if superficie_min:
            df = df[df['superficie_util'] >= superficie_min]
        
        return df.sort_values('satisfaccion_balanceado', ascending=False)
    
    def ranking_por_perfil(self, perfil='balanceado', top_n=20):
        """
        Genera ranking de propiedades para un perfil específico.
        
        Args:
            perfil: Uno de los perfiles disponibles
            top_n: Número de propiedades a mostrar
        
        Returns:
            DataFrame con ranking
        """
        if self.df_propiedades is None:
            raise ValueError("Dataset no cargado")
        
        col_sat = f'satisfaccion_{perfil}'
        if col_sat not in self.df_propiedades.columns:
            col_sat = 'satisfaccion_balanceado'
            print(f"⚠️ Perfil '{perfil}' no encontrado, usando 'balanceado'")
        
        df = self.df_propiedades.nlargest(top_n, col_sat)
        
        columnas = ['comuna', 'tipo_propiedad', 'precio_uf', 'superficie_util', 
                    'dormitorios', col_sat]
        columnas = [c for c in columnas if c in df.columns]
        
        return df[columnas].reset_index(drop=True)
    
    def explicar_prediccion(self, propiedad):
        """
        Explica qué factores influyen en la satisfacción de una propiedad.
        
        Args:
            propiedad: dict con características
        
        Returns:
            dict con explicación detallada
        """
        pred = self.predecir(propiedad)
        
        # Categorizar features
        explicacion = {
            'satisfaccion_total': pred['satisfaccion'],
            'nivel': pred['nivel'],
            'factores_positivos': [],
            'factores_negativos': [],
            'recomendaciones': []
        }
        
        # Analizar factores
        superficie = propiedad.get('superficie_util', 50)
        dormitorios = propiedad.get('dormitorios', 1) or 1
        precio_m2 = propiedad.get('precio_m2_uf', 50)
        
        m2_por_dorm = superficie / dormitorios
        
        # Espacio
        if m2_por_dorm >= 25:
            explicacion['factores_positivos'].append(f"✓ Buen espacio por dormitorio ({m2_por_dorm:.0f} m²/dorm)")
        elif m2_por_dorm < 15:
            explicacion['factores_negativos'].append(f"✗ Poco espacio por dormitorio ({m2_por_dorm:.0f} m²/dorm)")
            explicacion['recomendaciones'].append("Considerar propiedad con más m² o menos dormitorios")
        
        # Precio
        if self.df_propiedades is not None:
            mediana_precio = self.df_propiedades['precio_m2_uf'].median()
            if precio_m2 < mediana_precio * 0.8:
                explicacion['factores_positivos'].append(f"✓ Precio por m² bajo ({precio_m2:.1f} UF/m² vs mediana {mediana_precio:.1f})")
            elif precio_m2 > mediana_precio * 1.2:
                explicacion['factores_negativos'].append(f"✗ Precio por m² alto ({precio_m2:.1f} UF/m² vs mediana {mediana_precio:.1f})")
        
        # Tipo
        tipo = propiedad.get('tipo_propiedad', 'departamento')
        if tipo == 'casa' and superficie > 100:
            explicacion['factores_positivos'].append("✓ Casa con buena superficie")
        
        return explicacion
    
    def resumen_mercado(self):
        """Genera resumen del mercado de propiedades"""
        if self.df_propiedades is None:
            return "Dataset no disponible"
        
        df = self.df_propiedades
        
        resumen = {
            'total_propiedades': len(df),
            'departamentos': len(df[df['tipo_propiedad'] == 'departamento']),
            'casas': len(df[df['tipo_propiedad'] == 'casa']),
            'precio_promedio_uf': df['precio_uf'].mean(),
            'precio_mediano_uf': df['precio_uf'].median(),
            'superficie_promedio': df['superficie_util'].mean(),
            'satisfaccion_promedio': df.get('satisfaccion_balanceado', pd.Series([0])).mean(),
            'comunas': df['comuna'].value_counts().to_dict()
        }
        
        return resumen


def demo():
    """Demostración del uso del predictor"""
    print("\n" + "="*60)
    print("🏠 DEMO: Predictor de Satisfacción para Propiedades en Venta")
    print("="*60)
    
    try:
        predictor = PredictorSatisfaccion()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Ejemplo 1: Predecir para una propiedad
    print("\n📊 Ejemplo 1: Predicción individual")
    print("-"*40)
    propiedad = {
        'superficie_util': 65,
        'dormitorios': 2,
        'banos': 1,
        'precio_uf': 4500,
        'tipo_propiedad': 'departamento',
        'comuna': 'Santiago'
    }
    propiedad['precio_m2_uf'] = propiedad['precio_uf'] / propiedad['superficie_util']
    
    print(f"Propiedad: {propiedad['superficie_util']}m², {propiedad['dormitorios']} dorm, "
          f"{propiedad['precio_uf']:,} UF")
    
    resultado = predictor.predecir(propiedad)
    print(f"\n{resultado['emoji']} Satisfacción: {resultado['satisfaccion']}/10 ({resultado['nivel']})")
    
    # Ejemplo 2: Explicación
    print("\n📖 Ejemplo 2: Explicación de factores")
    print("-"*40)
    explicacion = predictor.explicar_prediccion(propiedad)
    
    if explicacion['factores_positivos']:
        print("Factores positivos:")
        for f in explicacion['factores_positivos']:
            print(f"   {f}")
    
    if explicacion['factores_negativos']:
        print("Factores negativos:")
        for f in explicacion['factores_negativos']:
            print(f"   {f}")
    
    # Ejemplo 3: Ranking por perfil
    print("\n🏆 Ejemplo 3: Top 5 propiedades para familias")
    print("-"*40)
    try:
        ranking = predictor.ranking_por_perfil('familia_con_ninos', top_n=5)
        print(ranking.to_string())
    except Exception as e:
        print(f"   No disponible: {e}")
    
    # Ejemplo 4: Resumen de mercado
    print("\n📈 Ejemplo 4: Resumen del mercado")
    print("-"*40)
    resumen = predictor.resumen_mercado()
    if isinstance(resumen, dict):
        print(f"   Total propiedades: {resumen['total_propiedades']:,}")
        print(f"   Departamentos: {resumen['departamentos']:,}")
        print(f"   Casas: {resumen['casas']:,}")
        print(f"   Precio promedio: {resumen['precio_promedio_uf']:,.0f} UF")
        print(f"   Satisfacción promedio: {resumen['satisfaccion_promedio']:.2f}/10")


if __name__ == '__main__':
    demo()
