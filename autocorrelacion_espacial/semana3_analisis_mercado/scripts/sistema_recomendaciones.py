#!/usr/bin/env python3
"""
Sistema de Recomendaciones Inmobiliarias Personalizado
basado en análisis de habitabilidad y modelos predictivos

Autor: Proyecto GeoInformática  
Fecha: Octubre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')

class SistemaRecomendacionesInmobiliarias:
    """Sistema inteligente de recomendaciones inmobiliarias"""
    
    def __init__(self):
        self.datos = None
        self.modelo_precio = None
        self.scaler = None
        self.le_comuna = None
        self.le_tipo = None
        self.caracteristicas_modelo = None
        
    def cargar_datos_y_modelos(self):
        """Cargar datos de mercado y modelos entrenados"""
        print(" Cargando datos y modelos...")
        
        # Cargar datos de mercado
        self.datos = gpd.read_file("../datos_mercado/propiedades_mercado_sintetico.geojson")
        print(f" Datos cargados: {len(self.datos)} propiedades")
        
        # Cargar modelo predictivo
        self.modelo_precio = joblib.load("../modelos/mejor_modelo_gradient_boosting.pkl")
        
        # Cargar encoders
        self.le_comuna = joblib.load("../modelos/encoder_comuna.pkl")
        self.le_tipo = joblib.load("../modelos/encoder_tipo.pkl")
        
        # Cargar características del modelo
        with open('../reportes/analisis_predictivo_completo.json', 'r', encoding='utf-8') as f:
            reporte = json.load(f)
            self.caracteristicas_modelo = reporte['dataset_info']['caracteristicas_utilizadas']
        
        print(" Modelos y encoders cargados")
        
    def definir_perfil_usuario(self, presupuesto_uf, tipo_propiedad_pref=None, 
                              comuna_pref=None, prioridades=None):
        """Definir perfil y preferencias del usuario"""
        
        perfil = {
            'presupuesto_max_uf': presupuesto_uf,
            'tipo_propiedad_pref': tipo_propiedad_pref,  # 'casa', 'departamento', None
            'comuna_pref': comuna_pref,  # Lista de comunas preferidas o None
            'prioridades': prioridades or {
                'habitabilidad_global': 0.30,
                'transporte': 0.20,
                'educacion': 0.15,
                'salud': 0.15,
                'seguridad': 0.10,
                'comercio': 0.10
            }
        }
        
        return perfil
    
    def filtrar_propiedades_disponibles(self, perfil):
        """Filtrar propiedades según criterios básicos del usuario"""
        
        propiedades_filtradas = self.datos.copy()
        
        # Filtro por presupuesto
        propiedades_filtradas = propiedades_filtradas[
            propiedades_filtradas['precio_total_uf'] <= perfil['presupuesto_max_uf']
        ]
        
        # Filtro por tipo de propiedad
        if perfil['tipo_propiedad_pref']:
            propiedades_filtradas = propiedades_filtradas[
                propiedades_filtradas['tipo_propiedad'] == perfil['tipo_propiedad_pref']
            ]
        
        # Filtro por comuna
        if perfil['comuna_pref']:
            propiedades_filtradas = propiedades_filtradas[
                propiedades_filtradas['comuna'].isin(perfil['comuna_pref'])
            ]
        
        print(f" Propiedades que cumplen criterios básicos: {len(propiedades_filtradas)}")
        
        return propiedades_filtradas
    
    def calcular_score_personalizado(self, propiedades, prioridades):
        """Calcular score personalizado basado en prioridades del usuario"""
        
        # Mapeo de prioridades a características
        mapeo_caracteristicas = {
            'habitabilidad_global': 'idx_habitabilidad_global',
            'transporte': 'acc_transporte',
            'educacion': 'acc_educacion',
            'salud': 'acc_salud',
            'seguridad': 'acc_seguridad',
            'comercio': 'acc_comercial'
        }
        
        scores_personalizados = []
        
        for _, propiedad in propiedades.iterrows():
            score = 0
            
            for prioridad, peso in prioridades.items():
                if prioridad in mapeo_caracteristicas:
                    caracteristica = mapeo_caracteristicas[prioridad]
                    if caracteristica in propiedad:
                        # Normalizar a 0-1 y aplicar peso
                        valor_normalizado = propiedad[caracteristica] / 10.0
                        score += valor_normalizado * peso
            
            scores_personalizados.append(score)
        
        return np.array(scores_personalizados)
    
    def predecir_precios(self, propiedades):
        """Predecir precios usando el modelo entrenado"""
        
        # Preparar características numéricas
        caracteristicas_numericas = [col for col in self.caracteristicas_modelo 
                                   if col not in ['comuna_encoded', 'tipo_propiedad_encoded']]
        
        X_pred = propiedades[caracteristicas_numericas].copy()
        
        # Añadir variables codificadas
        X_pred['comuna_encoded'] = self.le_comuna.transform(propiedades['comuna'])
        X_pred['tipo_propiedad_encoded'] = self.le_tipo.transform(propiedades['tipo_propiedad'])
        
        # Normalizar distancias
        for col in X_pred.columns:
            if 'dist_' in col and '_m' in col:
                X_pred[col] = X_pred[col] / 1000
        
        # Reordenar columnas según el modelo
        X_pred = X_pred[self.caracteristicas_modelo]
        
        # Predicciones
        precios_predichos = self.modelo_precio.predict(X_pred)
        
        return precios_predichos
    
    def calcular_valor_relativo(self, propiedades):
        """Calcular relación calidad-precio (valor relativo)"""
        
        precios_predichos = self.predecir_precios(propiedades)
        precios_reales = propiedades['precio_uf_m2'].values
        
        # Ratio de valor: precio_predicho / precio_real
        # Valores > 1 indican buena oportunidad (precio bajo para la calidad)
        valor_relativo = precios_predichos / precios_reales
        
        return valor_relativo
    
    def generar_recomendaciones(self, perfil, top_n=10):
        """Generar recomendaciones principales"""
        
        # 1. Filtrar propiedades disponibles
        propiedades_candidatas = self.filtrar_propiedades_disponibles(perfil)
        
        if len(propiedades_candidatas) == 0:
            print(" No hay propiedades que cumplan los criterios básicos")
            return None
        
        # 2. Calcular scores personalizados
        scores_personalizados = self.calcular_score_personalizado(
            propiedades_candidatas, perfil['prioridades']
        )
        
        # 3. Calcular valor relativo
        valor_relativo = self.calcular_valor_relativo(propiedades_candidatas)
        
        # 4. Score combinado (70% preferencias, 30% valor)
        score_final = (0.7 * scores_personalizados) + (0.3 * (valor_relativo - 1))
        
        # 5. Añadir scores al dataframe
        propiedades_scored = propiedades_candidatas.copy()
        propiedades_scored['score_personalizado'] = scores_personalizados
        propiedades_scored['valor_relativo'] = valor_relativo
        propiedades_scored['score_final'] = score_final
        
        # 6. Ordenar por score final y tomar top N
        recomendaciones = propiedades_scored.nlargest(top_n, 'score_final')
        
        return recomendaciones
    
    def generar_explicacion_recomendacion(self, propiedad, perfil):
        """Generar explicación de por qué se recomienda una propiedad"""
        
        explicaciones = []
        
        # Habitabilidad general
        hab_global = propiedad['idx_habitabilidad_global']
        if hab_global >= 7:
            explicaciones.append(f" Excelente habitabilidad general ({hab_global:.1f}/10)")
        elif hab_global >= 5:
            explicaciones.append(f" Buena habitabilidad general ({hab_global:.1f}/10)")
        
        # Prioridades específicas del usuario
        mapeo_caracteristicas = {
            'transporte': ('acc_transporte', 'accesibilidad al transporte'),
            'educacion': ('acc_educacion', 'acceso a educación'),
            'salud': ('acc_salud', 'acceso a servicios de salud'),
            'seguridad': ('acc_seguridad', 'nivel de seguridad'),
            'comercio': ('acc_comercial', 'acceso a comercio')
        }
        
        # Analizar top 3 prioridades del usuario
        top_prioridades = sorted(perfil['prioridades'].items(), 
                               key=lambda x: x[1], reverse=True)[:3]
        
        for prioridad, peso in top_prioridades:
            if prioridad in mapeo_caracteristicas:
                caracteristica, descripcion = mapeo_caracteristicas[prioridad]
                if caracteristica in propiedad:
                    valor = propiedad[caracteristica]
                    if valor >= 7:
                        explicaciones.append(f" Excelente {descripcion} ({valor:.1f}/10)")
                    elif valor >= 5:
                        explicaciones.append(f" Buena {descripcion} ({valor:.1f}/10)")
        
        # Valor relativo
        valor_rel = propiedad['valor_relativo']
        if valor_rel > 1.1:
            explicaciones.append(f" Excelente relación calidad-precio (valor {valor_rel:.1%} sobre precio)")
        elif valor_rel > 1.05:
            explicaciones.append(f" Buena relación calidad-precio (valor {valor_rel:.1%} sobre precio)")
        
        # Características específicas de la propiedad
        if propiedad['metros_construidos'] >= 100:
            explicaciones.append(f" Espacioso: {propiedad['metros_construidos']:.0f}m² construidos")
        
        if propiedad['estacionamientos'] >= 2:
            explicaciones.append(f" {propiedad['estacionamientos']} estacionamientos")
        
        if propiedad['antiguedad_anos'] <= 5:
            explicaciones.append(f" Propiedad nueva ({propiedad['antiguedad_anos']:.1f} años)")
        
        return explicaciones[:5]  # Máximo 5 explicaciones
    
    def crear_reporte_recomendaciones(self, recomendaciones, perfil):
        """Crear reporte detallado de recomendaciones"""
        
        reporte = {
            'fecha_recomendacion': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'perfil_usuario': perfil,
            'total_propiedades_evaluadas': len(self.datos),
            'propiedades_que_cumplen_criterios': len(recomendaciones) if recomendaciones is not None else 0,
            'recomendaciones': []
        }
        
        if recomendaciones is not None and len(recomendaciones) > 0:
            for i, (_, propiedad) in enumerate(recomendaciones.iterrows(), 1):
                explicaciones = self.generar_explicacion_recomendacion(propiedad, perfil)
                
                rec = {
                    'posicion': i,
                    'id_propiedad': int(propiedad['id_propiedad']),
                    'comuna': propiedad['comuna'],
                    'tipo_propiedad': propiedad['tipo_propiedad'],
                    'precio_total_uf': float(propiedad['precio_total_uf']),
                    'precio_uf_m2': float(propiedad['precio_uf_m2']),
                    'metros_construidos': float(propiedad['metros_construidos']),
                    'dormitorios': int(propiedad['dormitorios']),
                    'banos': int(propiedad['banos']),
                    'estacionamientos': int(propiedad['estacionamientos']),
                    'antiguedad_anos': float(propiedad['antiguedad_anos']),
                    'piso': int(propiedad['piso']),
                    'indices_habitabilidad': {
                        'habitabilidad_global': float(propiedad['idx_habitabilidad_global']),
                        'vida_urbana': float(propiedad['idx_vida_urbana']),
                        'calidad_vida': float(propiedad['idx_calidad_vida'])
                    },
                    'scores': {
                        'score_personalizado': float(propiedad['score_personalizado']),
                        'valor_relativo': float(propiedad['valor_relativo']),
                        'score_final': float(propiedad['score_final'])
                    },
                    'explicaciones': explicaciones,
                    'coordenadas': {
                        'lat': float(propiedad.geometry.y),
                        'lon': float(propiedad.geometry.x)
                    }
                }
                
                reporte['recomendaciones'].append(rec)
        
        return reporte
    
    def crear_visualizacion_recomendaciones(self, recomendaciones, perfil):
        """Crear visualización de las recomendaciones"""
        
        if recomendaciones is None or len(recomendaciones) == 0:
            print(" No hay recomendaciones para visualizar")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Mapa de ubicaciones recomendadas
        ax1 = axes[0, 0]
        
        # Plotear todas las propiedades disponibles en gris
        propiedades_filtradas = self.filtrar_propiedades_disponibles(perfil)
        ax1.scatter(propiedades_filtradas.geometry.x, propiedades_filtradas.geometry.y, 
                   c='lightgray', alpha=0.3, s=10, label='Propiedades disponibles')
        
        # Plotear recomendaciones por score
        scatter = ax1.scatter(recomendaciones.geometry.x, recomendaciones.geometry.y,
                             c=recomendaciones['score_final'], cmap='RdYlGn', 
                             s=100, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Destacar top 3
        top_3 = recomendaciones.head(3)
        for i, (_, prop) in enumerate(top_3.iterrows()):
            ax1.annotate(f'{i+1}', (prop.geometry.x, prop.geometry.y), 
                        xytext=(5, 5), textcoords='offset points',
                        fontsize=12, fontweight='bold', color='white',
                        bbox=dict(boxstyle='circle', facecolor='red', alpha=0.8))
        
        ax1.set_title('Ubicación de Propiedades Recomendadas', fontweight='bold')
        ax1.set_xlabel('Coordenada X')
        ax1.set_ylabel('Coordenada Y')
        ax1.legend()
        plt.colorbar(scatter, ax=ax1, label='Score Final')
        
        # 2. Distribución de precios vs habitabilidad
        ax2 = axes[0, 1]
        
        scatter2 = ax2.scatter(recomendaciones['idx_habitabilidad_global'], 
                              recomendaciones['precio_uf_m2'],
                              c=recomendaciones['score_final'], cmap='RdYlGn',
                              s=60, alpha=0.7, edgecolors='black')
        
        ax2.set_xlabel('Habitabilidad Global (0-10)')
        ax2.set_ylabel('Precio (UF/m²)')
        ax2.set_title('Precio vs Habitabilidad', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Comparación de scores por comuna
        ax3 = axes[1, 0]
        
        score_por_comuna = recomendaciones.groupby('comuna')['score_final'].mean().sort_values(ascending=True)
        
        bars = ax3.barh(range(len(score_por_comuna)), score_por_comuna.values, 
                       color='green', alpha=0.7)
        ax3.set_yticks(range(len(score_por_comuna)))
        ax3.set_yticklabels(score_por_comuna.index)
        ax3.set_xlabel('Score Promedio')
        ax3.set_title('Score Promedio por Comuna', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # Añadir valores
        for bar, valor in zip(bars, score_por_comuna.values):
            ax3.text(valor + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{valor:.2f}', va='center', fontsize=10)
        
        # 4. Top 10 recomendaciones - tabla
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Crear tabla con información clave
        tabla_data = []
        for i, (_, prop) in enumerate(recomendaciones.head(10).iterrows()):
            tabla_data.append([
                f"{i+1}",
                prop['comuna'][:8],  # Truncar nombre comuna
                f"{prop['precio_total_uf']:.0f}",
                f"{prop['metros_construidos']:.0f}",
                f"{prop['idx_habitabilidad_global']:.1f}",
                f"{prop['score_final']:.2f}"
            ])
        
        tabla = ax4.table(cellText=tabla_data,
                         colLabels=['#', 'Comuna', 'Precio\n(UF)', 'M²', 'Hab.', 'Score'],
                         cellLoc='center',
                         loc='center')
        tabla.auto_set_font_size(False)
        tabla.set_fontsize(9)
        tabla.scale(1.2, 1.5)
        
        # Estilo de la tabla
        for i in range(len(tabla_data) + 1):
            for j in range(6):
                cell = tabla[(i, j)]
                if i == 0:  # Header
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(weight='bold', color='white')
                else:
                    cell.set_facecolor('#F2F2F2' if i % 2 == 0 else '#FFFFFF')
        
        ax4.set_title('Top 10 Recomendaciones', fontweight='bold', pad=20)
        
        plt.suptitle(f'Recomendaciones Inmobiliarias - Presupuesto: {perfil["presupuesto_max_uf"]:,.0f} UF', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig

def ejemplo_uso_sistema():
    """Ejemplo de uso del sistema de recomendaciones"""
    
    # Inicializar sistema
    sistema = SistemaRecomendacionesInmobiliarias()
    sistema.cargar_datos_y_modelos()
    
    # Casos de ejemplo
    casos_ejemplo = [
        {
            'nombre': 'Familia joven con niños',
            'perfil': sistema.definir_perfil_usuario(
                presupuesto_uf=8000,
                tipo_propiedad_pref='casa',
                comuna_pref=None,
                prioridades={
                    'habitabilidad_global': 0.25,
                    'educacion': 0.35,
                    'transporte': 0.15,
                    'salud': 0.15,
                    'seguridad': 0.10
                }
            )
        },
        {
            'nombre': 'Profesional soltero',
            'perfil': sistema.definir_perfil_usuario(
                presupuesto_uf=4000,
                tipo_propiedad_pref='departamento',
                comuna_pref=['Santiago'],
                prioridades={
                    'habitabilidad_global': 0.20,
                    'transporte': 0.40,
                    'comercio': 0.20,
                    'educacion': 0.05,
                    'salud': 0.10,
                    'seguridad': 0.05
                }
            )
        },
        {
            'nombre': 'Pareja adulta mayor',
            'perfil': sistema.definir_perfil_usuario(
                presupuesto_uf=6000,
                tipo_propiedad_pref=None,
                comuna_pref=['La Reina', 'Ñuñoa'],
                prioridades={
                    'habitabilidad_global': 0.30,
                    'salud': 0.35,
                    'transporte': 0.15,
                    'seguridad': 0.15,
                    'educacion': 0.05
                }
            )
        }
    ]
    
    reportes_generados = []
    
    for i, caso in enumerate(casos_ejemplo, 1):
        print(f"\n CASO {i}: {caso['nombre']}")
        print("="*50)
        
        # Generar recomendaciones
        recomendaciones = sistema.generar_recomendaciones(caso['perfil'], top_n=10)
        
        if recomendaciones is not None:
            print(f" {len(recomendaciones)} recomendaciones generadas")
            
            # Crear reporte
            reporte = sistema.crear_reporte_recomendaciones(recomendaciones, caso['perfil'])
            
            # Guardar reporte
            nombre_archivo = f"../reportes/recomendaciones_{caso['nombre'].lower().replace(' ', '_')}.json"
            with open(nombre_archivo, 'w', encoding='utf-8') as f:
                json.dump(reporte, f, indent=2, ensure_ascii=False)
            
            reportes_generados.append(nombre_archivo)
            
            # Mostrar top 3
            print(f"\n TOP 3 RECOMENDACIONES:")
            for j, (_, prop) in enumerate(recomendaciones.head(3).iterrows(), 1):
                print(f"\n{j}. ID {prop['id_propiedad']} - {prop['tipo_propiedad'].title()} en {prop['comuna']}")
                print(f"   Precio: {prop['precio_total_uf']:.0f} UF ({prop['precio_uf_m2']:.1f} UF/m²)")
                print(f"   {prop['metros_construidos']:.0f}m², {prop['dormitorios']}D/{prop['banos']}B, {prop['estacionamientos']} est.")
                print(f"   Habitabilidad: {prop['idx_habitabilidad_global']:.1f}/10")
                print(f"   Score: {prop['score_final']:.3f}")
                
                explicaciones = sistema.generar_explicacion_recomendacion(prop, caso['perfil'])
                for exp in explicaciones[:3]:
                    print(f"   {exp}")
            
            # Crear visualización
            fig = sistema.crear_visualizacion_recomendaciones(recomendaciones, caso['perfil'])
            if fig:
                nombre_grafico = f"../visualizaciones/recomendaciones_{caso['nombre'].lower().replace(' ', '_')}.png"
                fig.savefig(nombre_grafico, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f" Visualización guardada: {nombre_grafico}")
        
        else:
            print(" No se encontraron propiedades que cumplan los criterios")
    
    return reportes_generados

def main():
    """Función principal del sistema de recomendaciones"""
    print(" SISTEMA DE RECOMENDACIONES INMOBILIARIAS")
    print("="*60)
    
    try:
        reportes = ejemplo_uso_sistema()
        
        print(f"\n Sistema de recomendaciones ejecutado exitosamente!")
        print(f" Reportes generados: {len(reportes)}")
        for reporte in reportes:
            print(f"   - {reporte}")
        
        return True
        
    except Exception as e:
        print(f" Error en sistema de recomendaciones: {e}")
        return False

if __name__ == "__main__":
    exito = main()
    if not exito:
        exit(1)