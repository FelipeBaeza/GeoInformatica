#!/usr/bin/env python3
"""
Script 2: Modelo predictivo de satisfacción residencial

Implementa múltiples enfoques:
1. Modelo base (Random Forest): Satisfacción = f(internos, externos)
2. Modelo personalizado: Ajusta pesos según perfil usuario
3. Análisis de correlaciones: Necesidades ↔ Factores
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json

# Configuración
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "modelos"
REPORTS_DIR = BASE_DIR / "reportes"
FIGURES_DIR = BASE_DIR / "figuras"

for d in [OUTPUT_DIR, REPORTS_DIR, FIGURES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Configurar visualizaciones
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

class ModeloSatisfaccionResidencial:
    """
    Modelo predictivo de satisfacción residencial
    """
    
    def __init__(self):
        self.modelo = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        self.features_internas = []
        self.features_externas = []
        
    def cargar_datos(self, filepath):
        """Carga dataset integrado"""
        print("📂 Cargando datos integrados...")
        df = pd.read_csv(filepath)
        print(f"   ✓ {len(df)} propiedades cargadas")
        return df
    
    def definir_features(self, df):
        """Define qué columnas son factores internos vs externos"""
        
        # Factores INTERNOS de la propiedad
        self.features_internas = [
            'superficie_total',
            'superficie_util',
            'dormitorios',
            'banos',
            'estacionamientos',
            'ambientes',
            'numero_piso_unidad',
            'tiene_estacionamiento',
            'tiene_bodega',
            'm2_por_habitante',
            'razon_banos_dormitorios'
        ]
        
        # Factores EXTERNOS espaciales
        self.features_externas = [
            # Distancias clave
            'dist_educacion_min_m',
            'dist_salud_min_m',
            'dist_transporte_metro_m',
            'dist_seguridad_min_m',
            'dist_comercio_m',
            'dist_areas_verdes_m',
            'dist_ocio_m',
            
            # Densidades radio caminable (300m)
            'dens_educacion_300m_km2',
            'dens_salud_300m_km2',
            'dens_comercio_300m_km2',
            'dens_recreacion_300m_km2',
            'dens_total_300m_km2',
            'diversidad_servicios_300m',
            
            # Índices normalizados
            'dens_norm_total_300m_km2',
            'div_norm_servicios_300m'
        ]
        
        # Filtrar features que existen en el dataset
        self.features_internas = [f for f in self.features_internas if f in df.columns]
        self.features_externas = [f for f in self.features_externas if f in df.columns]
        
        print(f"\n📊 Features definidas:")
        print(f"   • Internas: {len(self.features_internas)}")
        print(f"   • Externas: {len(self.features_externas)}")
        
        return self.features_internas + self.features_externas
    
    def preparar_datos(self, df, target='precio_m2'):
        """
        Prepara datos para entrenamiento
        
        Target options:
        - 'precio_m2': Proxy de deseabilidad
        - 'precio_percentil_comuna': Valor relativo en zona
        """
        print(f"\n🎯 Variable objetivo: {target}")

        # Si el target es precio_m2 y no existe, intentar calcularlo a partir de precio y superficie
        if target == 'precio_m2' and 'precio_m2' not in df.columns:
            print("   ℹ️  'precio_m2' no existe en el dataset. Intentando calcularlo a partir de 'precio' y 'superficie_util'/'superficie_total'...")
            df = df.copy()
            if 'superficie_util' in df.columns and df['superficie_util'].notnull().any():
                df.loc[:, 'precio_m2'] = df['precio'] / df['superficie_util'].replace({0: np.nan})
                print("   ✓ Calculado 'precio_m2' = precio / superficie_util")
            elif 'superficie_total' in df.columns and df['superficie_total'].notnull().any():
                df.loc[:, 'precio_m2'] = df['precio'] / df['superficie_total'].replace({0: np.nan})
                print("   ✓ Calculado 'precio_m2' = precio / superficie_total")
            else:
                raise KeyError("No es posible calcular 'precio_m2': faltan columnas de superficie")

        # Definir features
        all_features = self.definir_features(df)

        # Filtrar filas con valores completos
        df_clean = df[all_features + [target]].dropna()
        print(f"   ✓ {len(df_clean)} muestras válidas (de {len(df)})")

        X = df_clean[all_features]
        y = df_clean[target]

        return X, y, df_clean
    
    def entrenar_modelo_base(self, X, y):
        """Entrena Random Forest básico"""
        print("\n🤖 Entrenando modelo base (Random Forest)...")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Escalar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Entrenar modelo
        self.modelo = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            random_state=42,
            n_jobs=-1
        )
        self.modelo.fit(X_train_scaled, y_train)
        
        # Evaluar
        y_pred_train = self.modelo.predict(X_train_scaled)
        y_pred_test = self.modelo.predict(X_test_scaled)
        
        metrics = {
            'train': {
                'r2': r2_score(y_train, y_pred_train),
                'rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'mae': mean_absolute_error(y_train, y_pred_train)
            },
            'test': {
                'r2': r2_score(y_test, y_pred_test),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'mae': mean_absolute_error(y_test, y_pred_test)
            }
        }
        
        print(f"\n📈 Métricas de desempeño:")
        print(f"   Train - R²: {metrics['train']['r2']:.3f}, RMSE: {metrics['train']['rmse']:.0f}, MAE: {metrics['train']['mae']:.0f}")
        print(f"   Test  - R²: {metrics['test']['r2']:.3f}, RMSE: {metrics['test']['rmse']:.0f}, MAE: {metrics['test']['mae']:.0f}")
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.modelo.feature_importances_
        }).sort_values('importance', ascending=False)
        
        return metrics, X_test, y_test, y_pred_test
    
    def analizar_importancia_features(self):
        """Analiza qué features son más importantes"""
        print("\n🔍 Top 15 features más importantes:")
        
        top15 = self.feature_importance.head(15)
        for idx, row in top15.iterrows():
            tipo = "🏠 INTERNA" if row['feature'] in self.features_internas else "🌆 EXTERNA"
            print(f"   {row['importance']:.3f} - {row['feature']:40s} {tipo}")
        
        # Visualizar
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.barplot(
            data=self.feature_importance.head(20),
            y='feature',
            x='importance',
            ax=ax
        )
        ax.set_title('Importancia de Features - Modelo de Satisfacción', fontsize=14, fontweight='bold')
        ax.set_xlabel('Importancia')
        ax.set_ylabel('')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'feature_importance.png', dpi=300)
        print(f"   💾 Gráfico guardado: {FIGURES_DIR / 'feature_importance.png'}")
        
    def calcular_satisfaccion_personalizada(self, propiedad, perfil_usuario):
        """
        Calcula satisfacción ajustada al perfil del usuario
        
        perfil_usuario = {
            'tiene_hijos': bool,
            'trabaja_presencial': bool,
            'prioridad_seguridad': scale(1-5),
            'prioridad_naturaleza': scale(1-5),
            'edad': int
        }
        """
        
        # Pesos base (todos igual importancia)
        pesos = {f: 1.0 for f in self.features_internas + self.features_externas}
        
        # Ajustar pesos según perfil
        if perfil_usuario.get('tiene_hijos', False):
            pesos['dist_educacion_min_m'] = 2.5  # Muy importante
            pesos['dens_educacion_300m_km2'] = 2.0
            pesos['dist_areas_verdes_m'] = 1.8
            pesos['dens_recreacion_300m_km2'] = 1.5
        
        if perfil_usuario.get('trabaja_presencial', False):
            pesos['dist_transporte_metro_m'] = 3.0  # Crítico
        
        if perfil_usuario.get('prioridad_seguridad', 3) >= 4:
            pesos['dist_seguridad_min_m'] = 2.0
        
        if perfil_usuario.get('prioridad_naturaleza', 3) >= 4:
            pesos['dist_areas_verdes_m'] = 2.5
            pesos['dens_recreacion_300m_km2'] = 2.0
        
        # Calcular score ponderado
        score_total = 0
        peso_total = 0
        
        for feature, peso in pesos.items():
            if feature in propiedad and not pd.isna(propiedad[feature]):
                # Normalizar features de distancia (menor es mejor)
                if 'dist_' in feature:
                    valor_norm = 1 - min(propiedad[feature] / 5000, 1)  # Normalizar a [0,1]
                else:
                    valor_norm = propiedad[feature] / propiedad[feature].max() if propiedad[feature].max() > 0 else 0
                
                score_total += valor_norm * peso
                peso_total += peso
        
        satisfaccion = (score_total / peso_total) * 100  # Escala 0-100
        
        return satisfaccion
    
    def analizar_correlaciones_necesidades(self, df):
        """
        Analiza correlación entre necesidades del usuario y factores externos
        """
        print("\n🔗 MATRIZ DE CORRELACIÓN: Necesidades ↔ Factores Externos")
        
        # Definir proxies de necesidades
        necesidades_proxies = {
            'familia_con_ninos': ['dist_educacion_min_m', 'dens_educacion_300m_km2', 'dist_areas_verdes_m'],
            'profesional_joven': ['dist_transporte_metro_m', 'dens_ocio_m', 'dens_comercio_300m_km2'],
            'adulto_mayor': ['dist_salud_min_m', 'numero_piso_unidad'],
            'trabajo_remoto': ['superficie_util', 'dist_areas_verdes_m'],
            'seguridad_prioritaria': ['dist_seguridad_min_m']
        }
        
        correlaciones = {}
        for necesidad, features in necesidades_proxies.items():
            features_disponibles = [f for f in features if f in df.columns]
            if features_disponibles:
                correlaciones[necesidad] = df[features_disponibles + ['precio_m2']].corr()['precio_m2'].drop('precio_m2')
        
        # Visualizar
        fig, ax = plt.subplots(figsize=(12, 6))
        corr_df = pd.DataFrame(correlaciones).T
        sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Correlación: Necesidades vs Precio/m² (Proxy Satisfacción)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'correlacion_necesidades.png', dpi=300)
        print(f"   💾 Gráfico guardado: {FIGURES_DIR / 'correlacion_necesidades.png'}")
        
        return correlaciones

def generar_recomendaciones(modelo, df):
    """Genera recomendaciones basadas en análisis"""
    print("\n" + "="*70)
    print("💡 RECOMENDACIONES DEL MODELO")
    print("="*70)
    
    top_features = modelo.feature_importance.head(10)
    
    print("\n1️⃣ Factores más determinantes de satisfacción:")
    for idx, row in top_features.iterrows():
        tipo = "INTERNO" if row['feature'] in modelo.features_internas else "EXTERNO"
        print(f"   • {row['feature']:40s} ({tipo})")
    
    print("\n2️⃣ Perfiles de usuario a considerar:")
    print("   • Familia con niños → Priorizar educación + áreas verdes")
    print("   • Profesional joven → Metro + ocio + comercio")
    print("   • Adulto mayor → Salud accesible + pisos bajos")
    print("   • Trabajo remoto → Espacios grandes + naturaleza")
    
    print("\n3️⃣ Umbrales recomendados:")
    print(f"   • Distancia a metro: < 800m (ideal < 500m)")
    print(f"   • Distancia a educación: < 500m (si hay niños)")
    print(f"   • Distancia a salud: < 1000m")
    print(f"   • Densidad servicios 300m: > 5 servicios/km²")

def main():
    print("🚀 MODELO PREDICTIVO DE SATISFACCIÓN RESIDENCIAL\n")
    
    # 1. Inicializar modelo
    modelo = ModeloSatisfaccionResidencial()
    
    # 2. Cargar datos integrados
    df = modelo.cargar_datos(DATA_DIR / "propiedades_con_factores_espaciales.csv")
    
    # 3. Preparar datos
    X, y, df_clean = modelo.preparar_datos(df, target='precio_m2')
    
    # 4. Entrenar modelo base
    metrics, X_test, y_test, y_pred = modelo.entrenar_modelo_base(X, y)
    
    # 5. Analizar importancia
    modelo.analizar_importancia_features()
    
    # 6. Analizar correlaciones
    correlaciones = modelo.analizar_correlaciones_necesidades(df_clean)
    
    # 7. Generar recomendaciones
    generar_recomendaciones(modelo, df_clean)
    
    # 8. Guardar modelo
    import joblib
    modelo_path = OUTPUT_DIR / "modelo_satisfaccion.pkl"
    joblib.dump(modelo, modelo_path)
    print(f"\n💾 Modelo guardado: {modelo_path}")
    
    print("\n✅ MODELADO COMPLETADO")

if __name__ == "__main__":
    main()
