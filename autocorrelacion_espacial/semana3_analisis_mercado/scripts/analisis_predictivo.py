#!/usr/bin/env python3
"""
Script para análisis de correlaciones habitabilidad-precio y 
desarrollo de modelos predictivos de valoración inmobiliaria

Autor: Proyecto GeoInformática  
Fecha: Octubre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

# Configuración de matplotlib
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def cargar_datos_mercado():
    """Cargar datos de mercado sintéticos"""
    ruta_datos = "../datos_mercado/propiedades_mercado_sintetico.geojson"
    
    if not os.path.exists(ruta_datos):
        print(f" Error: No se encuentra {ruta_datos}")
        return None
    
    datos = gpd.read_file(ruta_datos)
    print(f" Datos cargados: {len(datos)} propiedades con {len(datos.columns)} características")
    
    return datos

def analizar_correlaciones_precio_habitabilidad(datos):
    """Análisis detallado de correlaciones precio-habitabilidad"""
    print(" Analizando correlaciones precio-habitabilidad...")
    
    # Seleccionar variables para análisis
    variables_habitabilidad = [
        'idx_habitabilidad_global', 'idx_vida_urbana', 'idx_calidad_vida',
        'acc_educacion', 'acc_salud', 'acc_transporte', 
        'acc_entorno', 'acc_seguridad', 'acc_comercial'
    ]
    
    variables_distancia = [
        'dist_colegios_m', 'dist_universidades_m', 'dist_hospitales_m',
        'dist_metro_m', 'dist_areas_verdes_m', 'dist_centros_comerciales_m'
    ]
    
    variables_densidad = [
        'dens_educacion_1000m_km2', 'dens_salud_1000m_km2', 'dens_comercio_1000m_km2',
        'dens_transporte_1000m_km2', 'dens_areas_verdes_1000m_km2'
    ]
    
    variables_propiedad = [
        'metros_construidos', 'dormitorios', 'banos', 
        'estacionamientos', 'antiguedad_anos', 'piso'
    ]
    
    # Variable objetivo
    target = 'precio_uf_m2'
    
    # Calcular correlaciones
    correlaciones = {}
    
    for categoria, variables in [
        ('Habitabilidad', variables_habitabilidad),
        ('Distancias', variables_distancia), 
        ('Densidades', variables_densidad),
        ('Propiedad', variables_propiedad)
    ]:
        vars_existentes = [var for var in variables if var in datos.columns]
        corrs = datos[vars_existentes + [target]].corr()[target].drop(target)
        correlaciones[categoria] = corrs.to_dict()
    
    # Crear visualización
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, (categoria, corrs) in enumerate(correlaciones.items()):
        if i < len(axes):
            # Ordenar por valor absoluto
            corrs_ordenadas = dict(sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True))
            
            variables = list(corrs_ordenadas.keys())[:10]  # Top 10
            valores = [corrs_ordenadas[var] for var in variables]
            
            colors = ['red' if x < 0 else 'green' for x in valores]
            
            bars = axes[i].barh(range(len(variables)), valores, color=colors, alpha=0.7)
            axes[i].set_yticks(range(len(variables)))
            axes[i].set_yticklabels([var.replace('_', ' ').title()[:20] for var in variables], fontsize=10)
            axes[i].set_xlabel('Correlación con Precio UF/m²')
            axes[i].set_title(f'{categoria}', fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            # Añadir valores
            for bar, valor in zip(bars, valores):
                axes[i].text(valor + (0.01 if valor >= 0 else -0.01), bar.get_y() + bar.get_height()/2, 
                            f'{valor:.3f}', va='center', fontsize=9, 
                            ha='left' if valor >= 0 else 'right')
    
    plt.suptitle('Correlaciones con Precio UF/m² por Categoría', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, correlaciones

def preparar_datos_para_modelado(datos):
    """Preparar datos para entrenamiento de modelos"""
    print(" Preparando datos para modelado...")
    
    # Seleccionar características relevantes
    caracteristicas_numericas = [
        # Índices principales
        'idx_habitabilidad_global', 'idx_vida_urbana', 'idx_calidad_vida',
        
        # Accesibilidades
        'acc_educacion', 'acc_salud', 'acc_transporte', 
        'acc_entorno', 'acc_seguridad', 'acc_comercial',
        
        # Características de propiedad
        'metros_construidos', 'dormitorios', 'banos', 
        'estacionamientos', 'antiguedad_anos', 'piso',
        
        # Distancias clave (normalizar dividiendo por 1000)
        'dist_metro_m', 'dist_colegios_m', 'dist_hospitales_m',
        
        # Densidades clave  
        'dens_comercio_1000m_km2', 'dens_educacion_1000m_km2', 'dens_transporte_1000m_km2'
    ]
    
    # Filtrar características que existen
    caracteristicas_disponibles = [col for col in caracteristicas_numericas if col in datos.columns]
    
    # Preparar dataset
    X = datos[caracteristicas_disponibles].copy()
    
    # Normalizar distancias a kilómetros
    for col in X.columns:
        if 'dist_' in col and '_m' in col:
            X[col] = X[col] / 1000
    
    # Variable categórica: comuna y tipo
    le_comuna = LabelEncoder()
    le_tipo = LabelEncoder()
    
    X['comuna_encoded'] = le_comuna.fit_transform(datos['comuna'])
    X['tipo_propiedad_encoded'] = le_tipo.fit_transform(datos['tipo_propiedad'])
    
    # Variable objetivo
    y = datos['precio_uf_m2'].copy()
    
    # Eliminar outliers extremos (percentiles 1% y 99%)
    q_low = y.quantile(0.01)
    q_high = y.quantile(0.99)
    
    mask = (y >= q_low) & (y <= q_high)
    X = X[mask]
    y = y[mask]
    
    print(f" Dataset preparado: {len(X)} muestras, {len(X.columns)} características")
    print(f"   Rango precio: {y.min():.1f} - {y.max():.1f} UF/m²")
    
    return X, y, le_comuna, le_tipo

def entrenar_modelos(X, y):
    """Entrenar múltiples modelos de machine learning"""
    print(" Entrenando modelos predictivos...")
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Escalado para modelos que lo requieren
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Definir modelos
    modelos = {
        'linear_regression': {
            'modelo': LinearRegression(),
            'usar_escalado': True,
            'params': {}
        },
        'random_forest': {
            'modelo': RandomForestRegressor(random_state=42),
            'usar_escalado': False,
            'params': {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
        },
        'gradient_boosting': {
            'modelo': GradientBoostingRegressor(random_state=42),
            'usar_escalado': False,
            'params': {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1, 0.15],
                'max_depth': [3, 5, 7]
            }
        }
    }
    
    resultados = {}
    
    for nombre, config in modelos.items():
        print(f"  Entrenando {nombre}...")
        
        # Seleccionar datos
        X_train_modelo = X_train_scaled if config['usar_escalado'] else X_train
        X_test_modelo = X_test_scaled if config['usar_escalado'] else X_test
        
        # Optimización de hiperparámetros si hay parámetros
        if config['params']:
            grid_search = GridSearchCV(
                config['modelo'], 
                config['params'], 
                cv=5, 
                scoring='r2',
                n_jobs=-1
            )
            grid_search.fit(X_train_modelo, y_train)
            mejor_modelo = grid_search.best_estimator_
            mejores_params = grid_search.best_params_
        else:
            mejor_modelo = config['modelo']
            mejor_modelo.fit(X_train_modelo, y_train)
            mejores_params = {}
        
        # Predicciones
        y_pred_train = mejor_modelo.predict(X_train_modelo)
        y_pred_test = mejor_modelo.predict(X_test_modelo)
        
        # Métricas
        metricas = {
            'r2_train': r2_score(y_train, y_pred_train),
            'r2_test': r2_score(y_test, y_pred_test),
            'rmse_train': np.sqrt(mean_squared_error(y_train, y_pred_train)),
            'rmse_test': np.sqrt(mean_squared_error(y_test, y_pred_test)),
            'mae_train': mean_absolute_error(y_train, y_pred_train),
            'mae_test': mean_absolute_error(y_test, y_pred_test),
            'mejores_parametros': mejores_params
        }
        
        # Validación cruzada
        cv_scores = cross_val_score(mejor_modelo, X_train_modelo, y_train, cv=5, scoring='r2')
        metricas['cv_r2_mean'] = cv_scores.mean()
        metricas['cv_r2_std'] = cv_scores.std()
        
        resultados[nombre] = {
            'modelo': mejor_modelo,
            'scaler': scaler if config['usar_escalado'] else None,
            'metricas': metricas,
            'predicciones_test': y_pred_test
        }
        
        print(f"    R² test: {metricas['r2_test']:.3f}, RMSE test: {metricas['rmse_test']:.2f} UF/m²")
    
    return resultados, X_test, y_test

def analizar_importancia_caracteristicas(resultados, nombres_caracteristicas):
    """Analizar importancia de características en modelos"""
    print(" Analizando importancia de características...")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Random Forest - Importancia de características
    if 'random_forest' in resultados:
        rf_model = resultados['random_forest']['modelo']
        importancias = rf_model.feature_importances_
        
        # Crear DataFrame para ordenar
        imp_df = pd.DataFrame({
            'caracteristica': nombres_caracteristicas,
            'importancia': importancias
        }).sort_values('importancia', ascending=True)
        
        # Top 15 características más importantes
        top_imp = imp_df.tail(15)
        
        bars = axes[0].barh(range(len(top_imp)), top_imp['importancia'], color='green', alpha=0.7)
        axes[0].set_yticks(range(len(top_imp)))
        axes[0].set_yticklabels([c.replace('_', ' ').title() for c in top_imp['caracteristica']], fontsize=10)
        axes[0].set_xlabel('Importancia')
        axes[0].set_title('Random Forest - Importancia de Características', fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Añadir valores
        for bar, valor in zip(bars, top_imp['importancia']):
            axes[0].text(valor + 0.002, bar.get_y() + bar.get_height()/2, 
                        f'{valor:.3f}', va='center', fontsize=9)
    
    # Gradient Boosting - Importancia de características  
    if 'gradient_boosting' in resultados:
        gb_model = resultados['gradient_boosting']['modelo']
        importancias = gb_model.feature_importances_
        
        # Crear DataFrame para ordenar
        imp_df = pd.DataFrame({
            'caracteristica': nombres_caracteristicas,
            'importancia': importancias
        }).sort_values('importancia', ascending=True)
        
        # Top 15 características más importantes
        top_imp = imp_df.tail(15)
        
        bars = axes[1].barh(range(len(top_imp)), top_imp['importancia'], color='blue', alpha=0.7)
        axes[1].set_yticks(range(len(top_imp)))
        axes[1].set_yticklabels([c.replace('_', ' ').title() for c in top_imp['caracteristica']], fontsize=10)
        axes[1].set_xlabel('Importancia')
        axes[1].set_title('Gradient Boosting - Importancia de Características', fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Añadir valores
        for bar, valor in zip(bars, top_imp['importancia']):
            axes[1].text(valor + 0.002, bar.get_y() + bar.get_height()/2, 
                        f'{valor:.3f}', va='center', fontsize=9)
    
    plt.tight_layout()
    return fig

def generar_visualizaciones_predicciones(resultados, y_test):
    """Generar visualizaciones de predicciones vs valores reales"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colores = ['blue', 'green', 'red']
    
    for i, (nombre, resultado) in enumerate(resultados.items()):
        if i < len(axes):
            y_pred = resultado['predicciones_test']
            r2 = resultado['metricas']['r2_test']
            rmse = resultado['metricas']['rmse_test']
            
            # Scatter plot predicciones vs reales
            axes[i].scatter(y_test, y_pred, alpha=0.6, color=colores[i], s=20)
            
            # Línea perfecta
            min_val = min(y_test.min(), y_pred.min())
            max_val = max(y_test.max(), y_pred.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, linewidth=2)
            
            axes[i].set_xlabel('Precio Real (UF/m²)')
            axes[i].set_ylabel('Precio Predicho (UF/m²)')
            axes[i].set_title(f'{nombre.replace("_", " ").title()}\nR² = {r2:.3f}, RMSE = {rmse:.2f} UF/m²', 
                             fontweight='bold')
            axes[i].grid(True, alpha=0.3)
    
    # Gráfico de comparación de métricas
    if len(resultados) > 0:
        modelos = list(resultados.keys())
        r2_scores = [resultados[m]['metricas']['r2_test'] for m in modelos]
        rmse_scores = [resultados[m]['metricas']['rmse_test'] for m in modelos]
        
        x = np.arange(len(modelos))
        
        ax_metricas = axes[-1]
        
        # Gráfico de barras para R²
        bars1 = ax_metricas.bar(x - 0.2, r2_scores, 0.4, label='R² Score', color='green', alpha=0.7)
        
        # Segundo eje Y para RMSE
        ax2 = ax_metricas.twinx()
        bars2 = ax2.bar(x + 0.2, rmse_scores, 0.4, label='RMSE (UF/m²)', color='red', alpha=0.7)
        
        ax_metricas.set_xlabel('Modelos')
        ax_metricas.set_ylabel('R² Score', color='green')
        ax2.set_ylabel('RMSE (UF/m²)', color='red')
        ax_metricas.set_title('Comparación de Rendimiento de Modelos', fontweight='bold')
        ax_metricas.set_xticks(x)
        ax_metricas.set_xticklabels([m.replace('_', ' ').title() for m in modelos], rotation=45)
        
        # Añadir valores en las barras
        for bar, valor in zip(bars1, r2_scores):
            ax_metricas.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                            f'{valor:.3f}', ha='center', va='bottom', fontsize=10)
        
        for bar, valor in zip(bars2, rmse_scores):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                    f'{valor:.1f}', ha='center', va='bottom', fontsize=10)
        
        ax_metricas.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax_metricas.grid(True, alpha=0.3)
    
    plt.suptitle('Evaluación de Modelos Predictivos de Precios', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

def guardar_modelos_y_reportes(resultados, correlaciones, X, le_comuna, le_tipo):
    """Guardar modelos entrenados y generar reportes"""
    print(" Guardando modelos y generando reportes...")
    
    # Guardar mejor modelo
    mejor_modelo_nombre = max(resultados.keys(), 
                             key=lambda k: resultados[k]['metricas']['r2_test'])
    mejor_modelo_info = resultados[mejor_modelo_nombre]
    
    # Guardar modelo
    joblib.dump(mejor_modelo_info['modelo'], f'../modelos/mejor_modelo_{mejor_modelo_nombre}.pkl')
    
    if mejor_modelo_info['scaler']:
        joblib.dump(mejor_modelo_info['scaler'], f'../modelos/scaler_{mejor_modelo_nombre}.pkl')
    
    joblib.dump(le_comuna, '../modelos/encoder_comuna.pkl')
    joblib.dump(le_tipo, '../modelos/encoder_tipo.pkl')
    
    # Crear reporte completo
    reporte = {
        'fecha_analisis': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'dataset_info': {
            'total_muestras': len(X),
            'total_caracteristicas': len(X.columns),
            'caracteristicas_utilizadas': X.columns.tolist()
        },
        'correlaciones_por_categoria': correlaciones,
        'modelos_evaluados': {
            nombre: {
                'r2_test': float(resultado['metricas']['r2_test']),
                'rmse_test': float(resultado['metricas']['rmse_test']),
                'mae_test': float(resultado['metricas']['mae_test']),
                'cv_r2_mean': float(resultado['metricas']['cv_r2_mean']),
                'cv_r2_std': float(resultado['metricas']['cv_r2_std']),
                'mejores_parametros': resultado['metricas']['mejores_parametros']
            }
            for nombre, resultado in resultados.items()
        },
        'mejor_modelo': {
            'nombre': mejor_modelo_nombre,
            'r2_test': float(mejor_modelo_info['metricas']['r2_test']),
            'rmse_test': float(mejor_modelo_info['metricas']['rmse_test']),
            'interpretacion': f"El modelo explica {mejor_modelo_info['metricas']['r2_test']:.1%} de la variabilidad en precios con error promedio de {mejor_modelo_info['metricas']['rmse_test']:.1f} UF/m²"
        },
        'insights_principales': {
            'correlacion_habitabilidad_precio': float(max([
                corr for categoria_corrs in correlaciones.values() 
                for corr in categoria_corrs.values() 
                if 'habitabilidad' in str(categoria_corrs).lower()
            ], default=0)),
            'caracteristica_mas_correlacionada': max(
                [(var, corr) for categoria_corrs in correlaciones.values() 
                 for var, corr in categoria_corrs.items()], 
                key=lambda x: abs(x[1])
            )[0],
            'rendimiento_modelos': f"Mejor modelo: {mejor_modelo_nombre} con R² = {mejor_modelo_info['metricas']['r2_test']:.3f}"
        }
    }
    
    # Guardar reporte
    with open('../reportes/analisis_predictivo_completo.json', 'w', encoding='utf-8') as f:
        json.dump(reporte, f, indent=2, ensure_ascii=False)
    
    print(f" Mejor modelo guardado: {mejor_modelo_nombre} (R² = {mejor_modelo_info['metricas']['r2_test']:.3f})")
    print(f" Reporte completo guardado: analisis_predictivo_completo.json")
    
    return reporte

def main():
    """Función principal para análisis predictivo"""
    print(" ANÁLISIS PREDICTIVO DE VALORACIÓN INMOBILIARIA")
    print("="*70)
    
    try:
        # 1. Cargar datos
        datos = cargar_datos_mercado()
        if datos is None:
            return False
        
        # 2. Análisis de correlaciones
        fig_correlaciones, correlaciones = analizar_correlaciones_precio_habitabilidad(datos)
        fig_correlaciones.savefig('../visualizaciones/correlaciones_precio_habitabilidad.png', 
                                 dpi=300, bbox_inches='tight')
        plt.close(fig_correlaciones)
        print(" Análisis de correlaciones completado")
        
        # 3. Preparar datos
        X, y, le_comuna, le_tipo = preparar_datos_para_modelado(datos)
        
        # 4. Entrenar modelos
        resultados, X_test, y_test = entrenar_modelos(X, y)
        print(" Modelos entrenados")
        
        # 5. Análisis de importancia
        fig_importancia = analizar_importancia_caracteristicas(resultados, X.columns.tolist())
        fig_importancia.savefig('../visualizaciones/importancia_caracteristicas.png', 
                               dpi=300, bbox_inches='tight')
        plt.close(fig_importancia)
        print(" Análisis de importancia completado")
        
        # 6. Visualizaciones de predicciones
        fig_predicciones = generar_visualizaciones_predicciones(resultados, y_test)
        fig_predicciones.savefig('../visualizaciones/evaluacion_modelos_predictivos.png', 
                                dpi=300, bbox_inches='tight')
        plt.close(fig_predicciones)
        print(" Visualizaciones de predicciones generadas")
        
        # 7. Guardar modelos y reportes
        reporte = guardar_modelos_y_reportes(resultados, correlaciones, X, le_comuna, le_tipo)
        
        # 8. Mostrar resumen final
        print(f"\n RESULTADOS DEL ANÁLISIS PREDICTIVO:")
        print(f"   Muestras analizadas: {len(X):,}")
        print(f"   Características utilizadas: {len(X.columns)}")
        
        mejor_modelo = reporte['mejor_modelo']
        print(f"   Mejor modelo: {mejor_modelo['nombre']}")
        print(f"   R² del mejor modelo: {mejor_modelo['r2_test']:.3f}")
        print(f"   RMSE del mejor modelo: {mejor_modelo['rmse_test']:.1f} UF/m²")
        print(f"   Interpretación: {mejor_modelo['interpretacion']}")
        
        print(f"\n Análisis predictivo completado exitosamente!")
        
        return True
        
    except Exception as e:
        print(f" Error en análisis predictivo: {e}")
        return False

if __name__ == "__main__":
    import os
    exito = main()
    if not exito:
        exit(1)