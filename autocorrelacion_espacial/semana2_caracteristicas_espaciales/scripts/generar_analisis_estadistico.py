#!/usr/bin/env python3
"""
Script para generar análisis estadístico detallado de las características espaciales

Autor: Proyecto GeoInformática
Fecha: Octubre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

def cargar_datos():
    """Carga los datos finales con todas las características calculadas"""
    grilla_path = "../features/grilla_con_indices.geojson"
    
    if not os.path.exists(grilla_path):
        print(f" Error: No se encuentra {grilla_path}")
        return None
    
    grilla = gpd.read_file(grilla_path)
    return grilla

def analisis_correlaciones(grilla):
    """Análisis de correlaciones entre variables"""
    print(" Generando análisis de correlaciones...")
    
    # Seleccionar variables numéricas principales
    variables_principales = [
        'acc_educacion', 'acc_salud', 'acc_transporte', 'acc_entorno',
        'acc_seguridad', 'acc_comercial', 'idx_vida_urbana', 
        'idx_calidad_vida', 'idx_habitabilidad_global'
    ]
    
    # Filtrar variables que existen
    variables_existentes = [var for var in variables_principales if var in grilla.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Matriz de correlación principal
    corr_matrix = grilla[variables_existentes].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                square=True, ax=axes[0,0], cbar_kws={"shrink": .8})
    axes[0,0].set_title('Matriz de Correlación - Variables Principales', fontweight='bold')
    
    # 2. Correlaciones más fuertes
    corr_flat = corr_matrix.where(~mask).stack().reset_index()
    corr_flat.columns = ['Variable1', 'Variable2', 'Correlacion']
    corr_flat = corr_flat.dropna()
    corr_flat['Correlacion_Abs'] = np.abs(corr_flat['Correlacion'])
    
    top_correlaciones = corr_flat.nlargest(10, 'Correlacion_Abs')
    
    colors = ['red' if x < 0 else 'blue' for x in top_correlaciones['Correlacion']]
    bars = axes[0,1].barh(range(len(top_correlaciones)), top_correlaciones['Correlacion'], color=colors, alpha=0.7)
    axes[0,1].set_yticks(range(len(top_correlaciones)))
    axes[0,1].set_yticklabels([f"{row['Variable1']} - {row['Variable2']}" 
                               for _, row in top_correlaciones.iterrows()], fontsize=8)
    axes[0,1].set_xlabel('Coeficiente de Correlación')
    axes[0,1].set_title('Top 10 Correlaciones más Fuertes', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # Añadir valores en las barras
    for bar, valor in zip(bars, top_correlaciones['Correlacion']):
        axes[0,1].text(valor + (0.02 if valor >= 0 else -0.02), bar.get_y() + bar.get_height()/2, 
                      f'{valor:.3f}', va='center', fontsize=8, 
                      ha='left' if valor >= 0 else 'right')
    
    # 3. Distribución de correlaciones
    todas_correlaciones = corr_flat['Correlacion'].values
    axes[1,0].hist(todas_correlaciones, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1,0].axvline(np.mean(todas_correlaciones), color='red', linestyle='--', 
                     label=f'Media: {np.mean(todas_correlaciones):.3f}')
    axes[1,0].axvline(np.median(todas_correlaciones), color='orange', linestyle='--', 
                     label=f'Mediana: {np.median(todas_correlaciones):.3f}')
    axes[1,0].set_xlabel('Coeficiente de Correlación')
    axes[1,0].set_ylabel('Frecuencia')
    axes[1,0].set_title('Distribución de Correlaciones', fontweight='bold')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Correlación con habitabilidad global
    if 'idx_habitabilidad_global' in grilla.columns:
        corr_habitabilidad = grilla[variables_existentes].corrwith(grilla['idx_habitabilidad_global']).drop('idx_habitabilidad_global')
        corr_habitabilidad_sorted = corr_habitabilidad.abs().sort_values(ascending=True)
        
        colors_hab = ['red' if x < 0 else 'green' for x in corr_habitabilidad[corr_habitabilidad_sorted.index]]
        bars_hab = axes[1,1].barh(range(len(corr_habitabilidad_sorted)), 
                                  corr_habitabilidad[corr_habitabilidad_sorted.index], 
                                  color=colors_hab, alpha=0.7)
        axes[1,1].set_yticks(range(len(corr_habitabilidad_sorted)))
        axes[1,1].set_yticklabels(corr_habitabilidad_sorted.index, fontsize=10)
        axes[1,1].set_xlabel('Correlación con Habitabilidad Global')
        axes[1,1].set_title('Factores de Habitabilidad Global', fontweight='bold')
        axes[1,1].grid(True, alpha=0.3)
        
        # Añadir valores
        for bar, valor in zip(bars_hab, corr_habitabilidad[corr_habitabilidad_sorted.index]):
            axes[1,1].text(valor + (0.02 if valor >= 0 else -0.02), bar.get_y() + bar.get_height()/2, 
                          f'{valor:.3f}', va='center', fontsize=9, 
                          ha='left' if valor >= 0 else 'right')
    
    plt.suptitle('Análisis de Correlaciones - Características Espaciales', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, corr_matrix

def analisis_pca(grilla):
    """Análisis de Componentes Principales"""
    print(" Realizando Análisis de Componentes Principales...")
    
    # Seleccionar variables para PCA (solo numéricas)
    variables_pca = [col for col in grilla.columns 
                     if col.startswith(('acc_', 'idx_', 'dist_', 'dens_')) 
                     and grilla[col].dtype in ['float64', 'int64']]
    
    # Limpiar datos para PCA
    datos_pca = grilla[variables_pca].fillna(0)
    
    # Estandarizar datos
    scaler = StandardScaler()
    datos_escalados = scaler.fit_transform(datos_pca)
    
    # Aplicar PCA
    pca = PCA()
    componentes = pca.fit_transform(datos_escalados)
    
    # Crear visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Varianza explicada
    varianza_explicada = pca.explained_variance_ratio_
    varianza_acumulada = np.cumsum(varianza_explicada)
    
    axes[0,0].bar(range(1, len(varianza_explicada[:10])+1), varianza_explicada[:10], 
                  alpha=0.7, color='steelblue')
    axes[0,0].set_xlabel('Componente Principal')
    axes[0,0].set_ylabel('Varianza Explicada')
    axes[0,0].set_title('Varianza Explicada por Componente\n(Primeros 10)', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # Añadir porcentajes
    for i, v in enumerate(varianza_explicada[:10]):
        axes[0,0].text(i+1, v + 0.001, f'{v:.1%}', ha='center', va='bottom', fontsize=8)
    
    # 2. Varianza acumulada
    axes[0,1].plot(range(1, len(varianza_acumulada[:20])+1), varianza_acumulada[:20], 
                   'bo-', linewidth=2, markersize=6)
    axes[0,1].axhline(y=0.8, color='red', linestyle='--', alpha=0.7, label='80% Varianza')
    axes[0,1].axhline(y=0.95, color='orange', linestyle='--', alpha=0.7, label='95% Varianza')
    axes[0,1].set_xlabel('Número de Componentes')
    axes[0,1].set_ylabel('Varianza Acumulada')
    axes[0,1].set_title('Varianza Acumulada', fontweight='bold')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Biplot - Primeros dos componentes
    scatter = axes[1,0].scatter(componentes[:, 0], componentes[:, 1], 
                               c=grilla['idx_habitabilidad_global'] if 'idx_habitabilidad_global' in grilla.columns else 'blue',
                               cmap='viridis', alpha=0.6, s=20)
    
    axes[1,0].set_xlabel(f'PC1 ({varianza_explicada[0]:.1%} varianza)')
    axes[1,0].set_ylabel(f'PC2 ({varianza_explicada[1]:.1%} varianza)')
    axes[1,0].set_title('Biplot - Primeras dos Componentes', fontweight='bold')
    axes[1,0].grid(True, alpha=0.3)
    
    if 'idx_habitabilidad_global' in grilla.columns:
        cbar = plt.colorbar(scatter, ax=axes[1,0])
        cbar.set_label('Habitabilidad Global')
    
    # 4. Contribución de variables a PC1 y PC2
    componentes_principales = pca.components_[:2]  # Primeras 2 componentes
    contribuciones = pd.DataFrame(componentes_principales.T, 
                                 columns=['PC1', 'PC2'], 
                                 index=variables_pca)
    
    # Top variables para PC1
    top_pc1 = contribuciones['PC1'].abs().sort_values(ascending=True).tail(10)
    colors_pc1 = ['red' if x < 0 else 'blue' for x in contribuciones.loc[top_pc1.index, 'PC1']]
    
    axes[1,1].barh(range(len(top_pc1)), contribuciones.loc[top_pc1.index, 'PC1'], 
                   color=colors_pc1, alpha=0.7)
    axes[1,1].set_yticks(range(len(top_pc1)))
    axes[1,1].set_yticklabels(top_pc1.index, fontsize=8)
    axes[1,1].set_xlabel('Contribución a PC1')
    axes[1,1].set_title('Top Variables - Primera Componente', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('Análisis de Componentes Principales (PCA)', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Guardar información de PCA
    info_pca = {
        'varianza_explicada_pc1': float(varianza_explicada[0]),
        'varianza_explicada_pc2': float(varianza_explicada[1]),
        'varianza_acumulada_5pc': float(varianza_acumulada[4]),
        'componentes_para_80_varianza': int(np.where(varianza_acumulada >= 0.8)[0][0] + 1),
        'componentes_para_95_varianza': int(np.where(varianza_acumulada >= 0.95)[0][0] + 1),
        'top_variables_pc1': contribuciones['PC1'].abs().sort_values(ascending=False).head(5).to_dict()
    }
    
    return fig, info_pca

def estadisticas_descriptivas(grilla):
    """Generar estadísticas descriptivas completas"""
    print(" Generando estadísticas descriptivas...")
    
    # Variables de interés
    variables_interes = [
        'acc_educacion', 'acc_salud', 'acc_transporte', 'acc_entorno',
        'acc_seguridad', 'acc_comercial', 'idx_vida_urbana', 
        'idx_calidad_vida', 'idx_habitabilidad_global'
    ]
    
    variables_existentes = [var for var in variables_interes if var in grilla.columns]
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    axes = axes.flatten()
    
    estadisticas_resumen = {}
    
    for i, variable in enumerate(variables_existentes):
        if i < len(axes):
            datos = grilla[variable].dropna()
            
            # Calcular estadísticas
            stats_var = {
                'media': float(datos.mean()),
                'mediana': float(datos.median()),
                'desviacion_std': float(datos.std()),
                'min': float(datos.min()),
                'max': float(datos.max()),
                'q25': float(datos.quantile(0.25)),
                'q75': float(datos.quantile(0.75)),
                'asimetria': float(stats.skew(datos)),
                'curtosis': float(stats.kurtosis(datos))
            }
            estadisticas_resumen[variable] = stats_var
            
            # Histograma con curva normal superpuesta
            axes[i].hist(datos, bins=30, density=True, alpha=0.7, color='skyblue', edgecolor='black')
            
            # Curva normal teórica
            x = np.linspace(datos.min(), datos.max(), 100)
            normal_curve = stats.norm.pdf(x, datos.mean(), datos.std())
            axes[i].plot(x, normal_curve, 'r-', linewidth=2, label='Normal teórica')
            
            # Líneas de estadísticas
            axes[i].axvline(datos.mean(), color='red', linestyle='--', alpha=0.8, label=f'Media: {datos.mean():.2f}')
            axes[i].axvline(datos.median(), color='orange', linestyle='--', alpha=0.8, label=f'Mediana: {datos.median():.2f}')
            
            axes[i].set_title(f'{variable.replace("_", " ").title()}\nAsimetría: {stats_var["asimetria"]:.2f}', 
                             fontweight='bold', fontsize=10)
            axes[i].set_xlabel('Valor')
            axes[i].set_ylabel('Densidad')
            axes[i].legend(fontsize=8)
            axes[i].grid(True, alpha=0.3)
    
    # Ocultar ejes no utilizados
    for j in range(len(variables_existentes), len(axes)):
        axes[j].set_visible(False)
    
    plt.suptitle('Distribuciones y Estadísticas Descriptivas', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, estadisticas_resumen

def analisis_por_comuna(grilla):
    """Análisis comparativo por comuna"""
    print(" Generando análisis por comuna...")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Estadísticas por comuna
    variables_comuna = ['acc_educacion', 'acc_salud', 'acc_transporte', 'acc_entorno',
                       'acc_seguridad', 'acc_comercial', 'idx_habitabilidad_global']
    variables_existentes = [var for var in variables_comuna if var in grilla.columns]
    
    stats_por_comuna = grilla.groupby('comuna')[variables_existentes].agg(['mean', 'std', 'min', 'max'])
    
    # Crear heatmap de medias
    medias_comuna = stats_por_comuna.xs('mean', level=1, axis=1)
    sns.heatmap(medias_comuna.T, annot=True, cmap='RdYlGn', center=5, 
                ax=axes[0,0], cbar_kws={'label': 'Puntuación Media'})
    axes[0,0].set_title('Puntuaciones Medias por Comuna', fontweight='bold')
    axes[0,0].set_ylabel('Variables')
    
    # 2. Variabilidad por comuna (coeficiente de variación)
    cv_comuna = (stats_por_comuna.xs('std', level=1, axis=1) / 
                 stats_por_comuna.xs('mean', level=1, axis=1)) * 100
    
    sns.heatmap(cv_comuna.T, annot=True, cmap='YlOrRd', 
                ax=axes[0,1], cbar_kws={'label': 'Coef. Variación (%)'})
    axes[0,1].set_title('Variabilidad por Comuna\n(Coeficiente de Variación)', fontweight='bold')
    axes[0,1].set_ylabel('Variables')
    
    # 3. Ranking de comunas por variable
    ranking_data = []
    for variable in variables_existentes:
        ranking_var = medias_comuna[variable].rank(ascending=False).astype(int)
        ranking_data.extend([(comuna, variable, ranking) 
                           for comuna, ranking in ranking_var.items()])
    
    ranking_df = pd.DataFrame(ranking_data, columns=['Comuna', 'Variable', 'Ranking'])
    ranking_pivot = ranking_df.pivot(index='Variable', columns='Comuna', values='Ranking')
    
    sns.heatmap(ranking_pivot, annot=True, cmap='RdYlGn_r', center=2.5,
                ax=axes[1,0], cbar_kws={'label': 'Posición Ranking'})
    axes[1,0].set_title('Ranking por Variable y Comuna\n(1=Mejor, 4=Peor)', fontweight='bold')
    
    # 4. Distribución de habitabilidad por comuna (violin plot)
    if 'idx_habitabilidad_global' in grilla.columns:
        sns.violinplot(data=grilla, x='comuna', y='idx_habitabilidad_global', ax=axes[1,1])
        axes[1,1].set_title('Distribución de Habitabilidad\npor Comuna', fontweight='bold')
        axes[1,1].set_xlabel('Comuna')
        axes[1,1].set_ylabel('Índice de Habitabilidad Global')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
    
    plt.suptitle('Análisis Comparativo por Comuna', 
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig, medias_comuna

def generar_reporte_estadistico(estadisticas_resumen, info_pca, medias_comuna):
    """Generar reporte estadístico en formato JSON"""
    reporte = {
        'fecha_analisis': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'estadisticas_descriptivas': estadisticas_resumen,
        'analisis_pca': info_pca,
        'resumen_por_comuna': {
            comuna: {
                variable: float(valor) 
                for variable, valor in datos_comuna.items()
            }
            for comuna, datos_comuna in medias_comuna.iterrows()
        },
        'insights_principales': {
            'variable_mayor_variabilidad': max(estadisticas_resumen.items(), 
                                              key=lambda x: x[1]['desviacion_std'])[0],
            'variable_mas_simetrica': min(estadisticas_resumen.items(), 
                                         key=lambda x: abs(x[1]['asimetria']))[0],
            'componentes_principales_80_varianza': info_pca['componentes_para_80_varianza'],
            'mejor_comuna_habitabilidad': medias_comuna['idx_habitabilidad_global'].idxmax() if 'idx_habitabilidad_global' in medias_comuna.columns else None
        }
    }
    
    return reporte

def main():
    """Función principal para generar análisis estadístico"""
    print(" ANÁLISIS ESTADÍSTICO DETALLADO - SEMANA 2")
    print("="*60)
    
    # Crear directorios
    os.makedirs('../graficos', exist_ok=True)
    os.makedirs('../reportes', exist_ok=True)
    
    # Cargar datos
    grilla = cargar_datos()
    if grilla is None:
        return False
    
    print(f" Analizando {len(grilla)} puntos con {len(grilla.columns)} variables...")
    
    try:
        # 1. Análisis de correlaciones
        fig_corr, matriz_corr = analisis_correlaciones(grilla)
        fig_corr.savefig('../graficos/07_analisis_correlaciones.png', dpi=300, bbox_inches='tight')
        plt.close(fig_corr)
        print(" Análisis de correlaciones completado")
        
        # 2. Análisis PCA
        fig_pca, info_pca = analisis_pca(grilla)
        fig_pca.savefig('../graficos/08_analisis_pca.png', dpi=300, bbox_inches='tight')
        plt.close(fig_pca)
        print(" Análisis PCA completado")
        
        # 3. Estadísticas descriptivas
        fig_stats, estadisticas_resumen = estadisticas_descriptivas(grilla)
        fig_stats.savefig('../graficos/09_estadisticas_descriptivas.png', dpi=300, bbox_inches='tight')
        plt.close(fig_stats)
        print(" Estadísticas descriptivas completadas")
        
        # 4. Análisis por comuna
        fig_comuna, medias_comuna = analisis_por_comuna(grilla)
        fig_comuna.savefig('../graficos/10_analisis_por_comuna.png', dpi=300, bbox_inches='tight')
        plt.close(fig_comuna)
        print(" Análisis por comuna completado")
        
        # 5. Generar reporte estadístico
        reporte_estadistico = generar_reporte_estadistico(estadisticas_resumen, info_pca, medias_comuna)
        
        import json
        with open('../reportes/analisis_estadistico.json', 'w', encoding='utf-8') as f:
            json.dump(reporte_estadistico, f, indent=2, ensure_ascii=False)
        
        print(" Reporte estadístico guardado")
        
        # Guardar matriz de correlación como CSV
        matriz_corr.to_csv('../reportes/matriz_correlaciones.csv')
        print(" Matriz de correlaciones exportada")
        
        print(f"\n Análisis estadístico completado exitosamente!")
        print(f" Gráficos guardados en: semana2_caracteristicas_espaciales/graficos/")
        print(f" Reportes guardados en: semana2_caracteristicas_espaciales/reportes/")
        
        return True
        
    except Exception as e:
        print(f" Error durante el análisis estadístico: {e}")
        return False

if __name__ == "__main__":
    exito = main()
    if not exito:
        exit(1)