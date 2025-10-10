#!/usr/bin/env python3
"""
Script para generar datos sintéticos de mercado inmobiliario
basados en las características espaciales calculadas en Semana 2

Autor: Proyecto GeoInformática  
Fecha: Octubre 2025
"""

import geopandas as gpd
import pandas as pd
import numpy as np
from scipy import stats
import os
import json
import warnings
warnings.filterwarnings('ignore')

def cargar_grilla_con_caracteristicas():
    """Cargar la grilla con todas las características espaciales de Semana 2"""
    ruta_grilla = "../../semana2_caracteristicas_espaciales/features/grilla_con_indices.geojson"
    
    if not os.path.exists(ruta_grilla):
        print(f" Error: No se encuentra {ruta_grilla}")
        return None
    
    grilla = gpd.read_file(ruta_grilla)
    print(f" Grilla cargada: {len(grilla)} puntos con {len(grilla.columns)} características")
    
    return grilla

def definir_modelos_precio():
    """Definir modelos de precio base por comuna y tipo de propiedad"""
    
    # Precios base por comuna (UF/m² aproximados reales de Santiago 2025)
    precios_base_comuna = {
        'La Reina': {'casa': 75, 'departamento': 85},
        'Santiago': {'casa': 45, 'departamento': 50}, 
        'Ñuñoa': {'casa': 55, 'departamento': 60},
        'Estación Central': {'casa': 40, 'departamento': 45}
    }
    
    # Factores de ajuste por características espaciales
    factores_caracteristicas = {
        # Accesibilidad (peso: 40%)
        'acc_educacion': 0.15,      # +15% por punto de accesibilidad educativa
        'acc_salud': 0.12,          # +12% por punto de accesibilidad salud
        'acc_transporte': 0.20,     # +20% por punto de accesibilidad transporte
        'acc_entorno': 0.10,        # +10% por punto de entorno
        'acc_seguridad': 0.08,      # +8% por punto de seguridad
        'acc_comercial': 0.05,      # +5% por punto comercial
        
        # Índices superiores (peso: 30%)
        'idx_vida_urbana': 0.12,    # +12% por punto de vida urbana
        'idx_calidad_vida': 0.18,   # +18% por punto de calidad de vida
        
        # Distancias importantes (peso: 20% - impacto negativo)
        'dist_metro_m': -0.00008,   # -8% por cada 1000m al metro
        'dist_colegios_m': -0.00005, # -5% por cada 1000m a colegios
        'dist_hospitales_m': -0.00003, # -3% por cada 1000m a hospitales
        
        # Densidades (peso: 10%)
        'dens_comercio_1000m_km2': 0.002,   # +0.2% por densidad comercial
        'dens_areas_verdes_1000m_km2': 0.003  # +0.3% por densidad áreas verdes
    }
    
    return precios_base_comuna, factores_caracteristicas

def generar_tipos_propiedades(grilla):
    """Generar tipos de propiedades realistas por comuna"""
    
    # Probabilidades de tipo por comuna (basado en realidad de Santiago)
    prob_tipo_comuna = {
        'La Reina': {'casa': 0.5, 'departamento': 0.5},
        'Santiago': {'casa': 0.3, 'departamento': 0.7},
        'Ñuñoa': {'casa': 0.6, 'departamento': 0.4}, 
        'Estación Central': {'casa': 0.4, 'departamento': 0.6}
    }
    
    tipos_propiedad = []
    
    for _, punto in grilla.iterrows():
        comuna = punto['comuna']
        probs = prob_tipo_comuna[comuna]
        
        # Generar tipo basado en probabilidad
        tipo = np.random.choice(['casa', 'departamento'], 
                               p=[probs['casa'], probs['departamento']])
        tipos_propiedad.append(tipo)
    
    return tipos_propiedad

def generar_caracteristicas_propiedades(grilla, tipos_propiedad):
    """Generar características realistas de propiedades"""
    
    caracteristicas = []
    
    for i, (_, punto) in enumerate(grilla.iterrows()):
        tipo = tipos_propiedad[i]
        comuna = punto['comuna']
        
        # Generar características basadas en tipo y comuna
        if tipo == 'casa':
            # Casas: más metros, más dormitorios
            metros = np.random.normal(180, 50)  # Media 180m², std 50m²
            metros = max(80, min(400, metros))  # Rango 80-400m²
            
            dormitorios = np.random.choice([2, 3, 4, 5], p=[0.1, 0.4, 0.4, 0.1])
            baños = np.random.choice([1, 2, 3, 4], p=[0.1, 0.5, 0.3, 0.1])
            estacionamientos = np.random.choice([1, 2, 3], p=[0.3, 0.6, 0.1])
            
        else:  # departamento
            # Departamentos: menos metros, menos dormitorios
            if comuna in ['La Reina', 'Ñuñoa']:
                metros = np.random.normal(90, 30)  # Deptos más grandes en comunas premium
            else:
                metros = np.random.normal(70, 25)  # Deptos más pequeños
            
            metros = max(35, min(200, metros))  # Rango 35-200m²
            
            dormitorios = np.random.choice([1, 2, 3, 4], p=[0.2, 0.5, 0.25, 0.05])
            baños = np.random.choice([1, 2, 3], p=[0.3, 0.6, 0.1])
            estacionamientos = np.random.choice([0, 1, 2], p=[0.2, 0.7, 0.1])
        
        # Antigüedad (años)
        antiguedad = np.random.exponential(15)  # Media 15 años, distribución exponencial
        antiguedad = min(50, max(0, antiguedad))
        
        # Piso (solo para departamentos)
        if tipo == 'departamento':
            # Probabilidades que suman exactamente 1
            probs_piso = [0.15, 0.12, 0.10, 0.08, 0.08, 0.07, 0.06, 0.06, 0.05, 0.05,
                         0.05, 0.04, 0.03, 0.02, 0.02, 0.01, 0.00, 0.00, 0.00, 0.01]
            piso = np.random.choice(range(1, 21), p=probs_piso)
        else:
            piso = 1
        
        caracteristicas.append({
            'tipo_propiedad': tipo,
            'metros_construidos': round(metros, 1),
            'metros_totales': round(metros * np.random.uniform(1.0, 1.3), 1),  # Terreno más grande
            'dormitorios': dormitorios,
            'banos': baños,
            'estacionamientos': estacionamientos,
            'antiguedad_anos': round(antiguedad, 1),
            'piso': piso
        })
    
    return caracteristicas

def calcular_precios_sinteticos(grilla, tipos_propiedad, caracteristicas, 
                               precios_base_comuna, factores_caracteristicas):
    """Calcular precios sintéticos realistas basados en características espaciales"""
    
    precios_data = []
    
    for i, (_, punto) in enumerate(grilla.iterrows()):
        comuna = punto['comuna']
        tipo = tipos_propiedad[i]
        caract = caracteristicas[i]
        
        # Precio base por comuna y tipo
        precio_base_uf_m2 = precios_base_comuna[comuna][tipo]
        
        # Factor de habitabilidad global (peso principal: 50%)
        factor_habitabilidad = 1 + (punto['idx_habitabilidad_global'] - 5.0) * 0.1
        
        # Factores por características específicas
        factor_total = factor_habitabilidad
        
        # Aplicar factores de características espaciales
        for caracteristica, peso in factores_caracteristicas.items():
            if caracteristica in punto:
                valor = punto[caracteristica]
                if 'dist_' in caracteristica:
                    # Para distancias, aplicar factor negativo
                    factor_total += valor * peso
                else:
                    # Para índices y densidades, factor positivo
                    factor_total += (valor / 10.0) * peso  # Normalizar a escala 0-1
        
        # Factores por características de la propiedad
        # Antigüedad (depreciation)
        factor_antiguedad = max(0.7, 1 - caract['antiguedad_anos'] * 0.008)  # -0.8% por año
        
        # Metros construidos (economía de escala)
        if caract['metros_construidos'] > 120:
            factor_metros = 1.1  # Premium por propiedades grandes
        elif caract['metros_construidos'] < 60:
            factor_metros = 0.9  # Descuento por propiedades pequeñas
        else:
            factor_metros = 1.0
        
        # Piso (para departamentos)
        if tipo == 'departamento':
            if caract['piso'] >= 10:
                factor_piso = 1.05  # Premium pisos altos
            elif caract['piso'] <= 2:
                factor_piso = 0.95  # Descuento pisos bajos
            else:
                factor_piso = 1.0
        else:
            factor_piso = 1.0
        
        # Calcular precio final
        precio_uf_m2 = precio_base_uf_m2 * factor_total * factor_antiguedad * factor_metros * factor_piso
        
        # Añadir ruido aleatorio (±10% variabilidad del mercado)
        ruido = np.random.normal(1.0, 0.1)
        precio_uf_m2 *= ruido
        
        # Asegurar precios realistas
        precio_uf_m2 = max(20, min(150, precio_uf_m2))  # Rango 20-150 UF/m²
        
        precio_total_uf = precio_uf_m2 * caract['metros_construidos']
        
        # Conversión a pesos (UF ≈ $37,000 CLP en octubre 2025)
        uf_a_clp = 37000
        precio_total_clp = precio_total_uf * uf_a_clp
        
        precios_data.append({
            'precio_uf_m2': round(precio_uf_m2, 2),
            'precio_total_uf': round(precio_total_uf, 1),
            'precio_total_clp': int(precio_total_clp),
            'precio_clp_m2': int(precio_uf_m2 * uf_a_clp),
            'factor_habitabilidad': round(factor_habitabilidad, 3),
            'factor_total_aplicado': round(factor_total, 3)
        })
    
    return precios_data

def crear_dataset_mercado_completo(grilla, tipos_propiedad, caracteristicas, precios_data):
    """Crear dataset completo combinando características espaciales y de mercado"""
    
    # Convertir listas a DataFrames
    df_caracteristicas = pd.DataFrame(caracteristicas)
    df_precios = pd.DataFrame(precios_data)
    
    # Combinar con grilla original
    dataset_completo = pd.concat([
        grilla.reset_index(drop=True),
        df_caracteristicas.reset_index(drop=True),
        df_precios.reset_index(drop=True)
    ], axis=1)
    
    # Añadir ID único
    dataset_completo['id_propiedad'] = range(1, len(dataset_completo) + 1)
    
    # Reordenar columnas
    columnas_principales = [
        'id_propiedad', 'comuna', 'tipo_propiedad', 
        'precio_uf_m2', 'precio_total_uf', 'precio_total_clp',
        'metros_construidos', 'metros_totales', 
        'dormitorios', 'banos', 'estacionamientos',
        'antiguedad_anos', 'piso',
        'idx_habitabilidad_global', 'idx_vida_urbana', 'idx_calidad_vida'
    ]
    
    # Mantener columnas principales al inicio
    otras_columnas = [col for col in dataset_completo.columns 
                     if col not in columnas_principales and col != 'geometry']
    
    columnas_finales = columnas_principales + otras_columnas + ['geometry']
    dataset_completo = dataset_completo[columnas_finales]
    
    return dataset_completo

def generar_estadisticas_mercado(dataset):
    """Generar estadísticas del mercado sintético"""
    
    estadisticas = {
        'fecha_generacion': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'total_propiedades': len(dataset),
        'comunas_incluidas': dataset['comuna'].unique().tolist(),
        
        'distribucion_tipos': {
            'casas': int(dataset[dataset['tipo_propiedad'] == 'casa'].shape[0]),
            'departamentos': int(dataset[dataset['tipo_propiedad'] == 'departamento'].shape[0]),
            'porcentaje_casas': float(dataset[dataset['tipo_propiedad'] == 'casa'].shape[0] / len(dataset) * 100),
            'porcentaje_departamentos': float(dataset[dataset['tipo_propiedad'] == 'departamento'].shape[0] / len(dataset) * 100)
        },
        
        'precios_uf_m2': {
            'promedio_general': float(dataset['precio_uf_m2'].mean()),
            'mediana_general': float(dataset['precio_uf_m2'].median()),
            'desviacion_std': float(dataset['precio_uf_m2'].std()),
            'minimo': float(dataset['precio_uf_m2'].min()),
            'maximo': float(dataset['precio_uf_m2'].max()),
            'percentil_25': float(dataset['precio_uf_m2'].quantile(0.25)),
            'percentil_75': float(dataset['precio_uf_m2'].quantile(0.75))
        },
        
        'precios_por_comuna': {},
        'precios_por_tipo': {},
        
        'correlaciones_habitabilidad': {
            'precio_vs_habitabilidad': float(dataset['precio_uf_m2'].corr(dataset['idx_habitabilidad_global'])),
            'precio_vs_vida_urbana': float(dataset['precio_uf_m2'].corr(dataset['idx_vida_urbana'])),
            'precio_vs_calidad_vida': float(dataset['precio_uf_m2'].corr(dataset['idx_calidad_vida']))
        },
        
        'metricas_propiedades': {
            'metros_promedio': float(dataset['metros_construidos'].mean()),
            'dormitorios_promedio': float(dataset['dormitorios'].mean()),
            'antiguedad_promedio': float(dataset['antiguedad_anos'].mean()),
            'estacionamientos_promedio': float(dataset['estacionamientos'].mean())
        }
    }
    
    # Estadísticas por comuna
    for comuna in dataset['comuna'].unique():
        subset = dataset[dataset['comuna'] == comuna]
        estadisticas['precios_por_comuna'][comuna] = {
            'precio_promedio_uf_m2': float(subset['precio_uf_m2'].mean()),
            'precio_mediana_uf_m2': float(subset['precio_uf_m2'].median()),
            'total_propiedades': len(subset),
            'habitabilidad_promedio': float(subset['idx_habitabilidad_global'].mean())
        }
    
    # Estadísticas por tipo
    for tipo in dataset['tipo_propiedad'].unique():
        subset = dataset[dataset['tipo_propiedad'] == tipo]
        estadisticas['precios_por_tipo'][tipo] = {
            'precio_promedio_uf_m2': float(subset['precio_uf_m2'].mean()),
            'precio_mediana_uf_m2': float(subset['precio_uf_m2'].median()),
            'metros_promedio': float(subset['metros_construidos'].mean()),
            'total_propiedades': len(subset)
        }
    
    return estadisticas

def main():
    """Función principal para generar datos de mercado sintéticos"""
    print(" GENERADOR DE DATOS DE MERCADO SINTÉTICOS")
    print("="*60)
    
    # Establecer seed para reproducibilidad
    np.random.seed(42)
    
    try:
        # 1. Cargar grilla con características espaciales
        grilla = cargar_grilla_con_caracteristicas()
        if grilla is None:
            return False
        
        print(f" Procesando {len(grilla)} ubicaciones en {grilla['comuna'].nunique()} comunas...")
        
        # 2. Definir modelos de precio
        precios_base_comuna, factores_caracteristicas = definir_modelos_precio()
        print(" Modelos de precio definidos")
        
        # 3. Generar tipos de propiedades
        tipos_propiedad = generar_tipos_propiedades(grilla)
        print(" Tipos de propiedades generados")
        
        # 4. Generar características de propiedades
        caracteristicas = generar_caracteristicas_propiedades(grilla, tipos_propiedad)
        print(" Características de propiedades generadas")
        
        # 5. Calcular precios sintéticos
        precios_data = calcular_precios_sinteticos(grilla, tipos_propiedad, caracteristicas,
                                                 precios_base_comuna, factores_caracteristicas)
        print(" Precios sintéticos calculados")
        
        # 6. Crear dataset completo
        dataset_completo = crear_dataset_mercado_completo(grilla, tipos_propiedad, 
                                                        caracteristicas, precios_data)
        print(" Dataset completo creado")
        
        # 7. Guardar dataset
        ruta_salida = '../datos_mercado/propiedades_mercado_sintetico.geojson'
        dataset_completo.to_file(ruta_salida, driver='GeoJSON')
        print(f" Dataset guardado: {ruta_salida}")
        
        # 8. Generar estadísticas
        estadisticas = generar_estadisticas_mercado(dataset_completo)
        
        with open('../reportes/estadisticas_mercado_sintetico.json', 'w', encoding='utf-8') as f:
            json.dump(estadisticas, f, indent=2, ensure_ascii=False)
        print(" Estadísticas guardadas")
        
        # 9. Mostrar resumen
        print(f"\n RESUMEN DEL MERCADO SINTÉTICO:")
        print(f"   Total propiedades: {estadisticas['total_propiedades']:,}")
        print(f"   Casas: {estadisticas['distribucion_tipos']['casas']} ({estadisticas['distribucion_tipos']['porcentaje_casas']:.1f}%)")
        print(f"   Departamentos: {estadisticas['distribucion_tipos']['departamentos']} ({estadisticas['distribucion_tipos']['porcentaje_departamentos']:.1f}%)")
        print(f"   Precio promedio: {estadisticas['precios_uf_m2']['promedio_general']:.1f} UF/m²")
        print(f"   Rango precios: {estadisticas['precios_uf_m2']['minimo']:.1f} - {estadisticas['precios_uf_m2']['maximo']:.1f} UF/m²")
        print(f"   Correlación precio-habitabilidad: {estadisticas['correlaciones_habitabilidad']['precio_vs_habitabilidad']:.3f}")
        
        print(f"\n Datos de mercado sintéticos generados exitosamente!")
        
        return True
        
    except Exception as e:
        print(f" Error generando datos de mercado: {e}")
        return False

if __name__ == "__main__":
    exito = main()
    if not exito:
        exit(1)