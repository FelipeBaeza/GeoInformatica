#!/usr/bin/env python3
"""
Integrador de Datos para Análisis de Satisfacción Inmobiliaria
Semana 3: Integración de todos los datasets geoespaciales
Basado en metodología de Claudio Álvarez pero adaptado para satisfacción del usuario
"""

import geopandas as gpd
import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

class IntegradorDatosSatisfaccion:
    """
    Integrador de datos geoespaciales para análisis de satisfacción
    Inspirado en la metodología de Claudio Álvarez pero adaptado para satisfacción
    """
    
    def __init__(self, directorio_datos="datos_filtrados/"):
        self.directorio = directorio_datos
        self.datasets = {}
        self.propiedades_base = None
        self.df_integrado = None
        self.reporte_calidad = {}
        
        print("🏠 Inicializando Integrador de Datos para Satisfacción")
        print("=" * 60)
        
    def cargar_todos_los_datasets(self):
        """Cargar todos los datasets geoespaciales"""
        
        # Mapeo de archivos a categorías de satisfacción
        mapeo_satisfaccion = {
            # TRANSPORTE (Impacto alto en satisfacción)
            'metro': 'Lineas_de_metro_de_Santiago.geojson',
            'estaciones_carga': 'estaciones_carga_filtradas.geojson',
            
            # EDUCACIÓN (Crítico para familias)
            'colegios': 'establecimientos_educacion_escolar.geojson',
            'universidades': 'establecimientos_educacion_superior.geojson',
            'jardines': 'establecimientos_parvularia_filtrados.geojson',
            
            # SALUD (Esencial para satisfacción)
            'salud': 'puntos_medicos_farmacias_hospitales_filtrados.geojson',
            'clinicas': 'redes_de_clinicas_filtradas.geojson',
            
            # SEGURIDAD (Factor crítico de satisfacción)
            'cuarteles': 'cuarteles_filtrados.geojson',
            'bomberos': 'cuerpos_de_bomberos_filtrados.geojson',
            'pdi': 'unidades_operativas_pdi_filtradas.geojson',
            'delincuencia': 'delincuencia_comunas_anual.geojson',
            
            # COMERCIO Y SERVICIOS (Conveniencia diaria)
            'tiendas': 'tiendas_filtradas.geojson',
            'servicios': 'servicios_filtrados.geojson',
            
            # OCIO Y CALIDAD DE VIDA
            'ocio': 'ocio_filtrado.geojson',
            'turismo': 'atracciones_turisticas_filtradas.geojson',
            'puntos_interes': 'puntos_de_interes_filtrados.geojson',
            
            # PROBLEMAS URBANOS (Impacto negativo)
            'vertederos': 'vertederos_ilegales_filtrados.geojson',
            'campamentos': 'campamentos.geojson',
            
            # ADMINISTRACIÓN
            'municipios': 'municipios_filtrados.geojson',
            'sernam': 'centros_sernam_filtrados.geojson'
        }
        
        print("🔄 Cargando datasets para análisis de satisfacción...")
        
        for categoria, archivo in mapeo_satisfaccion.items():
            try:
                ruta = Path(self.directorio) / archivo
                if ruta.exists():
                    gdf = gpd.read_file(ruta)
                    
                    # Validar que tenga geometrías válidas
                    if not gdf.empty and 'geometry' in gdf.columns:
                        # Filtrar solo geometrías válidas
                        gdf = gdf[gdf.geometry.is_valid]
                        
                        self.datasets[categoria] = {
                            'data': gdf,
                            'count': len(gdf),
                            'bbox': gdf.total_bounds if not gdf.empty else None,
                            'columns': list(gdf.columns),
                            'geom_types': gdf.geometry.type.value_counts().to_dict()
                        }
                        print(f"  ✅ {categoria}: {len(gdf)} registros")
                    else:
                        print(f"  ⚠️ {categoria}: Sin geometrías válidas")
                else:
                    print(f"  ❌ {categoria}: Archivo no encontrado - {archivo}")
            except Exception as e:
                print(f"  ⚠️ {categoria}: Error - {str(e)}")
        
        print(f"\n📊 Total datasets cargados: {len(self.datasets)}")
        return self.datasets
    
    def validar_calidad_espacial(self):
        """Validar calidad de datos espaciales para satisfacción"""
        
        print("\n🔍 Validando calidad de datos espaciales...")
        
        for categoria, info in self.datasets.items():
            gdf = info['data']
            
            validacion = {
                'total_registros': len(gdf),
                'geometrias_validas': gdf.geometry.is_valid.sum(),
                'geometrias_invalidas': (~gdf.geometry.is_valid).sum(),
                'valores_nulos': gdf.isnull().sum().sum(),
                'duplicados_espaciales': 0,
                'tipos_geometria': info['geom_types'],
                'bbox': info['bbox']
            }
            
            # Detectar duplicados espaciales para puntos
            if not gdf.empty:
                puntos = gdf[gdf.geometry.type == 'Point']
                if not puntos.empty:
                    coords = [(geom.x, geom.y) for geom in puntos.geometry]
                    validacion['duplicados_espaciales'] = len(coords) - len(set(coords))
            
            self.reporte_calidad[categoria] = validacion
            
            # Imprimir resumen
            print(f"\n📊 {categoria.upper()}:")
            print(f"  • Total registros: {validacion['total_registros']}")
            print(f"  • Geometrías válidas: {validacion['geometrias_validas']}")
            print(f"  • Valores nulos: {validacion['valores_nulos']}")
            print(f"  • Duplicados espaciales: {validacion['duplicados_espaciales']}")
            print(f"  • Tipos geometría: {validacion['tipos_geometria']}")
        
        return self.reporte_calidad
    
    def cargar_propiedades_base(self):
        """Cargar propiedades base"""
        
        print("\n🏠 Cargando propiedades base...")
        
        try:
            self.propiedades_base = gpd.read_file(
                f"{self.directorio}/base_maestra_comunas_filtradas.geojson"
            )
            
            # Validar datos
            print(f"  ✅ {len(self.propiedades_base)} propiedades cargadas")
            print(f"  • Columnas: {list(self.propiedades_base.columns)}")
            print(f"  • CRS: {self.propiedades_base.crs}")
            
            # Verificar variable de precio
            if 'total_uf' in self.propiedades_base.columns:
                precios = self.propiedades_base['total_uf']
                print(f"  • Precios - Min: {precios.min():.0f} UF, Max: {precios.max():.0f} UF")
                print(f"  • Precios - Media: {precios.mean():.0f} UF, Mediana: {precios.median():.0f} UF")
            
            return True
            
        except Exception as e:
            print(f"  ❌ Error cargando propiedades: {str(e)}")
            return False
    
    def _calcular_distancia_minima(self, propiedades, amenidades):
        """Calcular distancia mínima a amenidades (en metros)"""
        
        if amenidades.empty:
            return [float('inf')] * len(propiedades)
        
        distancias = []
        
        for idx, prop in propiedades.iterrows():
            try:
                # Calcular distancia a todas las amenidades
                dists = amenidades.geometry.distance(prop.geometry)
                dist_min_grados = dists.min()
                
                # Convertir a metros (aproximación para Santiago)
                dist_min_metros = dist_min_grados * 111000  # 1 grado ≈ 111km
                distancias.append(dist_min_metros)
                
            except Exception as e:
                distancias.append(float('inf'))
        
        return distancias
    
    def _contar_en_radio(self, propiedades, amenidades, radio_metros):
        """Contar amenidades en un radio dado"""
        
        if amenidades.empty:
            return [0] * len(propiedades)
        
        radio_grados = radio_metros / 111000
        conteos = []
        
        for idx, prop in propiedades.iterrows():
            try:
                # Crear buffer y contar amenidades dentro
                buffer = prop.geometry.buffer(radio_grados)
                dentro = amenidades[amenidades.geometry.within(buffer)]
                conteos.append(len(dentro))
                
            except Exception as e:
                conteos.append(0)
        
        return conteos
    
    def _calcular_accesibilidad(self, propiedades, amenidades):
        """Calcular índice de accesibilidad ponderado por distancia"""
        
        if amenidades.empty:
            return [0] * len(propiedades)
        
        accesibilidad = []
        
        for idx, prop in propiedades.iterrows():
            try:
                # Calcular distancias a todas las amenidades
                dists = amenidades.geometry.distance(prop.geometry)
                dists_metros = dists * 111000
                
                # Función de decaimiento: 1/(1 + d/1000)
                # Más peso a amenidades cercanas
                pesos = 1 / (1 + dists_metros / 1000)
                accesibilidad_total = pesos.sum()
                
                accesibilidad.append(accesibilidad_total)
                
            except Exception as e:
                accesibilidad.append(0)
        
        return accesibilidad
    
    def integrar_con_propiedades(self):
        """Integrar todos los datasets con las propiedades base"""
        
        if self.propiedades_base is None:
            print("❌ Error: Primero debe cargar las propiedades base")
            return None
        
        print("\n🔄 Integrando datasets con propiedades...")
        
        # Crear DataFrame para métricas
        metricas_satisfaccion = pd.DataFrame(index=self.propiedades_base.index)
        
        # Procesar cada categoría
        for categoria, info in self.datasets.items():
            if categoria == 'delincuencia':  # Tratar diferente (datos por comuna)
                continue
                
            amenidades = info['data']
            
            if amenidades.empty:
                print(f"  ⚠️ {categoria}: Dataset vacío, saltando...")
                continue
            
            print(f"  🔄 Procesando {categoria}...")
            
            try:
                # Métricas básicas de satisfacción
                metricas_satisfaccion[f'dist_min_{categoria}'] = self._calcular_distancia_minima(
                    self.propiedades_base, amenidades
                )
                
                metricas_satisfaccion[f'count_500m_{categoria}'] = self._contar_en_radio(
                    self.propiedades_base, amenidades, 500
                )
                
                metricas_satisfaccion[f'count_1km_{categoria}'] = self._contar_en_radio(
                    self.propiedades_base, amenidades, 1000
                )
                
                # Métricas específicas para servicios críticos
                if categoria in ['metro', 'colegios', 'salud', 'tiendas']:
                    metricas_satisfaccion[f'accesibilidad_{categoria}'] = self._calcular_accesibilidad(
                        self.propiedades_base, amenidades
                    )
                
                print(f"    ✅ {categoria}: Métricas calculadas")
                
            except Exception as e:
                print(f"    ❌ {categoria}: Error - {str(e)}")
        
        # Integrar con propiedades base
        self.df_integrado = pd.concat([
            self.propiedades_base, 
            metricas_satisfaccion
        ], axis=1)
        
        print(f"\n✅ Integración completada: {len(self.df_integrado)} propiedades con {len(self.df_integrado.columns)} variables")
        
        return self.df_integrado
    
    def _normalizar_y_combinar(self, variables_pesos):
        """Normalizar variables y combinar con pesos"""
        
        resultado = np.zeros(len(self.df_integrado))
        
        for variable, direccion, peso in variables_pesos:
            if variable.name not in self.df_integrado.columns:
                continue
                
            # Manejar valores infinitos
            var_clean = variable.replace([np.inf, -np.inf], np.nan)
            
            if var_clean.isna().all():
                continue
            
            # Normalizar 0-1
            scaler = MinMaxScaler()
            var_values = var_clean.fillna(var_clean.median()).values.reshape(-1, 1)
            var_norm = scaler.fit_transform(var_values).flatten()
            
            # Invertir si la relación es inversa (distancias)
            if direccion == -1:
                var_norm = 1 - var_norm
            
            # Aplicar peso
            resultado += peso * var_norm
        
        return resultado
    
    def crear_indices_satisfaccion(self):
        """Crear índices compuestos específicos para satisfacción"""
        
        if self.df_integrado is None:
            print("❌ Error: Primero debe integrar los datos")
            return None
        
        print("\n🎯 Creando índices de satisfacción...")
        
        df = self.df_integrado.copy()
        
        # 1. ÍNDICE DE ACCESIBILIDAD TRANSPORTE
        try:
            variables_transporte = []
            if 'dist_min_metro' in df.columns:
                variables_transporte.append((df['dist_min_metro'], -1, 0.8))
            if 'dist_min_estaciones_carga' in df.columns:
                variables_transporte.append((df['dist_min_estaciones_carga'], -1, 0.2))
            
            if variables_transporte:
                df['indice_transporte'] = self._normalizar_y_combinar(variables_transporte)
                print("  ✅ Índice de transporte creado")
        except Exception as e:
            print(f"  ❌ Error en índice transporte: {e}")
        
        # 2. ÍNDICE DE SERVICIOS ESENCIALES
        try:
            variables_servicios = []
            if 'dist_min_salud' in df.columns:
                variables_servicios.append((df['dist_min_salud'], -1, 0.4))
            if 'dist_min_colegios' in df.columns:
                variables_servicios.append((df['dist_min_colegios'], -1, 0.3))
            if 'count_1km_salud' in df.columns:
                variables_servicios.append((df['count_1km_salud'], 1, 0.2))
            if 'count_1km_colegios' in df.columns:
                variables_servicios.append((df['count_1km_colegios'], 1, 0.1))
            
            if variables_servicios:
                df['indice_servicios'] = self._normalizar_y_combinar(variables_servicios)
                print("  ✅ Índice de servicios creado")
        except Exception as e:
            print(f"  ❌ Error en índice servicios: {e}")
        
        # 3. ÍNDICE DE SEGURIDAD
        try:
            variables_seguridad = []
            if 'dist_min_cuarteles' in df.columns:
                variables_seguridad.append((df['dist_min_cuarteles'], -1, 0.3))
            if 'dist_min_bomberos' in df.columns:
                variables_seguridad.append((df['dist_min_bomberos'], -1, 0.2))
            if 'dist_min_pdi' in df.columns:
                variables_seguridad.append((df['dist_min_pdi'], -1, 0.2))
            if 'count_1km_cuarteles' in df.columns:
                variables_seguridad.append((df['count_1km_cuarteles'], 1, 0.3))
            
            if variables_seguridad:
                df['indice_seguridad'] = self._normalizar_y_combinar(variables_seguridad)
                print("  ✅ Índice de seguridad creado")
        except Exception as e:
            print(f"  ❌ Error en índice seguridad: {e}")
        
        # 4. ÍNDICE DE CONVENIENCIA DIARIA
        try:
            variables_conveniencia = []
            if 'count_500m_tiendas' in df.columns:
                variables_conveniencia.append((df['count_500m_tiendas'], 1, 0.4))
            if 'count_500m_servicios' in df.columns:
                variables_conveniencia.append((df['count_500m_servicios'], 1, 0.3))
            if 'dist_min_tiendas' in df.columns:
                variables_conveniencia.append((df['dist_min_tiendas'], -1, 0.3))
            
            if variables_conveniencia:
                df['indice_conveniencia'] = self._normalizar_y_combinar(variables_conveniencia)
                print("  ✅ Índice de conveniencia creado")
        except Exception as e:
            print(f"  ❌ Error en índice conveniencia: {e}")
        
        # 5. ÍNDICE DE CALIDAD DE VIDA
        try:
            variables_calidad = []
            if 'count_1km_ocio' in df.columns:
                variables_calidad.append((df['count_1km_ocio'], 1, 0.4))
            if 'count_1km_turismo' in df.columns:
                variables_calidad.append((df['count_1km_turismo'], 1, 0.2))
            if 'count_1km_puntos_interes' in df.columns:
                variables_calidad.append((df['count_1km_puntos_interes'], 1, 0.4))
            
            if variables_calidad:
                df['indice_calidad_vida'] = self._normalizar_y_combinar(variables_calidad)
                print("  ✅ Índice de calidad de vida creado")
        except Exception as e:
            print(f"  ❌ Error en índice calidad de vida: {e}")
        
        # 6. ÍNDICE DE PROBLEMAS URBANOS (impacto negativo)
        try:
            variables_problemas = []
            if 'dist_min_vertederos' in df.columns:
                variables_problemas.append((df['dist_min_vertederos'], 1, 0.6))  # Más lejos es mejor
            if 'dist_min_campamentos' in df.columns:
                variables_problemas.append((df['dist_min_campamentos'], 1, 0.4))
            
            if variables_problemas:
                df['indice_problemas'] = self._normalizar_y_combinar(variables_problemas)
                print("  ✅ Índice de problemas urbanos creado")
        except Exception as e:
            print(f"  ❌ Error en índice problemas: {e}")
        
        # 7. ÍNDICE COMPUESTO DE SATISFACCIÓN GENERAL
        try:
            indices_principales = []
            pesos_principales = []
            
            if 'indice_transporte' in df.columns:
                indices_principales.append((df['indice_transporte'], 1, 0.25))
            if 'indice_servicios' in df.columns:
                indices_principales.append((df['indice_servicios'], 1, 0.25))
            if 'indice_seguridad' in df.columns:
                indices_principales.append((df['indice_seguridad'], 1, 0.20))
            if 'indice_conveniencia' in df.columns:
                indices_principales.append((df['indice_conveniencia'], 1, 0.15))
            if 'indice_calidad_vida' in df.columns:
                indices_principales.append((df['indice_calidad_vida'], 1, 0.10))
            if 'indice_problemas' in df.columns:
                indices_principales.append((df['indice_problemas'], 1, 0.05))
            
            if indices_principales:
                df['satisfaccion_estimada_base'] = self._normalizar_y_combinar(indices_principales)
                print("  ✅ Índice compuesto de satisfacción creado")
        except Exception as e:
            print(f"  ❌ Error en índice compuesto: {e}")
        
        self.df_integrado = df
        
        # Mostrar resumen de índices creados
        indices_creados = [col for col in df.columns if col.startswith('indice_') or col == 'satisfaccion_estimada_base']
        print(f"\n📊 Índices creados: {len(indices_creados)}")
        for indice in indices_creados:
            valores = df[indice]
            print(f"  • {indice}: Min={valores.min():.3f}, Max={valores.max():.3f}, Media={valores.mean():.3f}")
        
        return df
    
    def generar_reporte_integracion(self):
        """Generar reporte completo de la integración"""
        
        print("\n" + "="*80)
        print("📊 REPORTE DE INTEGRACIÓN DE DATOS - SEMANA 3")
        print("="*80)
        
        if self.df_integrado is None:
            print("❌ No hay datos integrados para reportar")
            return
        
        # Resumen general
        print(f"\n🏠 PROPIEDADES BASE:")
        print(f"  • Total propiedades: {len(self.df_integrado)}")
        print(f"  • Total variables: {len(self.df_integrado.columns)}")
        
        # Datasets procesados
        print(f"\n📁 DATASETS PROCESADOS:")
        for categoria, info in self.datasets.items():
            print(f"  • {categoria}: {info['count']} registros")
        
        # Variables de distancia creadas
        vars_distancia = [col for col in self.df_integrado.columns if col.startswith('dist_min_')]
        print(f"\n📏 VARIABLES DE DISTANCIA: {len(vars_distancia)}")
        for var in vars_distancia[:10]:  # Mostrar solo las primeras 10
            valores = self.df_integrado[var].replace([np.inf, -np.inf], np.nan)
            if not valores.isna().all():
                print(f"  • {var}: {valores.min():.0f}m - {valores.max():.0f}m (media: {valores.mean():.0f}m)")
        
        # Variables de conteo creadas
        vars_conteo = [col for col in self.df_integrado.columns if col.startswith('count_')]
        print(f"\n🔢 VARIABLES DE CONTEO: {len(vars_conteo)}")
        for var in vars_conteo[:10]:  # Mostrar solo las primeras 10
            valores = self.df_integrado[var]
            print(f"  • {var}: {valores.min():.0f} - {valores.max():.0f} (media: {valores.mean():.1f})")
        
        # Índices de satisfacción
        indices = [col for col in self.df_integrado.columns if col.startswith('indice_') or col == 'satisfaccion_estimada_base']
        print(f"\n🎯 ÍNDICES DE SATISFACCIÓN: {len(indices)}")
        for indice in indices:
            valores = self.df_integrado[indice]
            print(f"  • {indice}: {valores.min():.3f} - {valores.max():.3f} (media: {valores.mean():.3f})")
        
        # Calidad de datos
        print(f"\n✅ CALIDAD DE DATOS:")
        total_vars = len(self.df_integrado.columns)
        vars_con_nulos = self.df_integrado.isnull().any().sum()
        print(f"  • Variables con valores nulos: {vars_con_nulos}/{total_vars}")
        print(f"  • Completitud promedio: {((total_vars - vars_con_nulos) / total_vars * 100):.1f}%")
        
        return self.df_integrado
    
    def guardar_datos_integrados(self, archivo_salida="datos_integrados_satisfaccion.geojson"):
        """Guardar datos integrados"""
        
        if self.df_integrado is None:
            print("❌ No hay datos para guardar")
            return False
        
        try:
            # Convertir a GeoDataFrame si no lo es
            if not isinstance(self.df_integrado, gpd.GeoDataFrame):
                gdf_salida = gpd.GeoDataFrame(self.df_integrado, geometry=self.propiedades_base.geometry)
            else:
                gdf_salida = self.df_integrado
            
            # Guardar
            gdf_salida.to_file(archivo_salida, driver='GeoJSON')
            print(f"✅ Datos guardados en: {archivo_salida}")
            print(f"   • {len(gdf_salida)} propiedades")
            print(f"   • {len(gdf_salida.columns)} variables")
            
            return True
            
        except Exception as e:
            print(f"❌ Error guardando datos: {str(e)}")
            return False

def main():
    """Función principal para ejecutar la integración"""
    
    print("🚀 INICIANDO INTEGRACIÓN DE DATOS - SEMANA 3")
    print("="*60)
    
    # Crear integrador
    integrador = IntegradorDatosSatisfaccion()
    
    # Paso 1: Cargar todos los datasets
    datasets = integrador.cargar_todos_los_datasets()
    
    # Paso 2: Validar calidad
    reporte_calidad = integrador.validar_calidad_espacial()
    
    # Paso 3: Cargar propiedades base
    if not integrador.cargar_propiedades_base():
        print("❌ Error cargando propiedades base. Terminando.")
        return
    
    # Paso 4: Integrar datos
    df_integrado = integrador.integrar_con_propiedades()
    
    if df_integrado is None:
        print("❌ Error en la integración. Terminando.")
        return
    
    # Paso 5: Crear índices de satisfacción
    df_final = integrador.crear_indices_satisfaccion()
    
    # Paso 6: Generar reporte
    integrador.generar_reporte_integracion()
    
    # Paso 7: Guardar datos
    integrador.guardar_datos_integrados()
    
    print("\n🎉 INTEGRACIÓN COMPLETADA EXITOSAMENTE")
    print("📁 Archivo generado: datos_integrados_satisfaccion.geojson")
    print("🔜 Listo para Semana 4: Diseño del cuestionario de preferencias")

if __name__ == "__main__":
    main()
