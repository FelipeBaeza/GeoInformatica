#!/usr/bin/env python3
"""
Creación del diccionario de datos y metadatos para el proyecto
Documenta todas las fuentes, campos y transformaciones aplicadas
"""

import json
import geopandas as gpd
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

def extraer_esquema_archivo(archivo_path):
    """Extrae esquema detallado de un archivo GeoJSON"""
    
    try:
        gdf = gpd.read_file(archivo_path)
        
        if gdf.empty:
            return {
                'archivo': archivo_path.name,
                'estado': 'vacio',
                'columnas': 0,
                'filas': 0
            }
        
        # Información básica
        esquema = {
            'archivo': archivo_path.name,
            'filas': len(gdf),
            'columnas': len(gdf.columns) - 1,  # Excluir geometry
            'crs': str(gdf.crs),
            'tipo_geometria': gdf.geom_type.mode()[0] if not gdf.geom_type.empty else 'Desconocido',
            'bounds': gdf.total_bounds.tolist(),
            'campos': {}
        }
        
        # Análisis de cada campo
        for col in gdf.columns:
            if col == 'geometry':
                continue
                
            serie = gdf[col]
            
            campo_info = {
                'tipo_datos': str(serie.dtype),
                'valores_unicos': serie.nunique(),
                'valores_nulos': serie.isna().sum(),
                'completitud_pct': ((len(serie) - serie.isna().sum()) / len(serie)) * 100,
            }
            
            # Ejemplos de valores (no nulos)
            valores_no_nulos = serie.dropna()
            if not valores_no_nulos.empty:
                ejemplos = valores_no_nulos.head(3).tolist()
                campo_info['ejemplos'] = ejemplos
            
            # Estadísticas para campos numéricos
            if pd.api.types.is_numeric_dtype(serie):
                campo_info['estadisticas'] = {
                    'min': serie.min() if not serie.isna().all() else None,
                    'max': serie.max() if not serie.isna().all() else None,
                    'media': serie.mean() if not serie.isna().all() else None,
                    'mediana': serie.median() if not serie.isna().all() else None
                }
            
            # Valores más frecuentes para campos categóricos
            if not pd.api.types.is_numeric_dtype(serie) or serie.nunique() < 20:
                freq = serie.value_counts().head(5)
                campo_info['valores_frecuentes'] = freq.to_dict()
            
            esquema['campos'][col] = campo_info
        
        return esquema
        
    except Exception as e:
        return {
            'archivo': archivo_path.name,
            'estado': 'error',
            'error': str(e)
        }

def clasificar_archivos_por_categoria():
    """Clasifica archivos por categoría temática"""
    
    clasificacion = {
        'transporte_movilidad': [
            'Lineas_de_metro_de_Santiago.geojson',
            'estaciones_carga_filtradas.geojson',
            'circuito_turistico_filtrado.geojson'
        ],
        'servicios_amenidades': [
            'servicios_filtrados.geojson',
            'tiendas_filtradas.geojson',
            'puntos_de_interes_filtrados.geojson',
            'puntos_medicos_farmacias_hospitales_filtrados.geojson',
            'redes_de_clinicas_filtradas.geojson',
            'atracciones_turisticas_filtradas.geojson'
        ],
        'educacion': [
            'establecimientos_educacion_escolar.geojson',
            'establecimientos_educacion_superior.geojson',
            'establecimientos_parvularia_filtrados.geojson'
        ],
        'seguridad': [
            'cuarteles_filtrados.geojson',
            'cuerpos_de_bomberos_filtrados.geojson',
            'unidades_operativas_pdi_filtradas.geojson'
        ],
        'ambiente_espacios': [
            'areas_verdes_filtradas.geojson',
            'ocio_filtrado.geojson',
            'estaciones_calidad_aire_filtrada.geojson',
            'estaciones_metereológicas_filtradas.geojson',
            'vertedores_ilegales_filtrados.geojson'
        ],
        'limites_administrativos': [
            'comunas_buffer.geojson',
            'unidades_vecinales_filtradas.geojson',
            'poblacion_filtrada.geojson',
            'poblacion2_filtrada.geojson',
            'municipios_filtrados.geojson'
        ],
        'socioeconomico': [
            'delincuencia_comunas_anual.geojson',
            'campamentos.geojson',
            'centros_sernam_filtrados.geojson',
            'base_maestra_comunas_filtradas.geojson'
        ]
    }
    
    return clasificacion

def definir_fuentes_originales():
    """Define las fuentes originales de cada dataset"""
    
    fuentes = {
        # Transporte
        'Lineas_de_metro_de_Santiago.geojson': {
            'fuente_original': 'Metro de Santiago / DTPM',
            'tipo_fuente': 'Oficial',
            'url': 'https://www.metro.cl',
            'licencia': 'Datos abiertos',
            'fecha_descarga': '2024-2025',
            'cobertura_temporal': 'Actual',
            'descripcion': 'Red de líneas del Metro de Santiago'
        },
        
        # Servicios OSM
        'servicios_filtrados.geojson': {
            'fuente_original': 'OpenStreetMap',
            'tipo_fuente': 'Colaborativa',
            'url': 'https://www.openstreetmap.org',
            'licencia': 'ODbL',
            'fecha_descarga': '2024-2025',
            'cobertura_temporal': 'Actual',
            'descripcion': 'Servicios urbanos diversos extraídos de OSM'
        },
        
        'tiendas_filtradas.geojson': {
            'fuente_original': 'OpenStreetMap',
            'tipo_fuente': 'Colaborativa', 
            'url': 'https://www.openstreetmap.org',
            'licencia': 'ODbL',
            'fecha_descarga': '2024-2025',
            'cobertura_temporal': 'Actual',
            'descripcion': 'Comercio y tiendas de OSM'
        },
        
        # Educación oficial
        'establecimientos_educacion_escolar.geojson': {
            'fuente_original': 'MINEDUC - Ministerio de Educación',
            'tipo_fuente': 'Oficial',
            'url': 'https://www.mineduc.cl',
            'licencia': 'Datos abiertos gubernamentales',
            'fecha_descarga': '2024-2025',
            'cobertura_temporal': '2024',
            'descripcion': 'Establecimientos de educación escolar registrados'
        },
        
        # Censo y límites
        'poblacion_filtrada.geojson': {
            'fuente_original': 'INE - Instituto Nacional de Estadísticas',
            'tipo_fuente': 'Oficial',
            'url': 'https://www.ine.cl',
            'licencia': 'Datos abiertos gubernamentales',
            'fecha_descarga': '2024-2025',
            'cobertura_temporal': 'Censo 2017',
            'descripcion': 'Manzanas censales con datos de población y vivienda'
        },
        
        'unidades_vecinales_filtradas.geojson': {
            'fuente_original': 'INE - Instituto Nacional de Estadísticas',
            'tipo_fuente': 'Oficial',
            'url': 'https://www.ine.cl',
            'licencia': 'Datos abiertos gubernamentales',
            'fecha_descarga': '2024-2025',
            'cobertura_temporal': '2024',
            'descripcion': 'Unidades vecinales con indicadores socioeconómicos'
        }
    }
    
    return fuentes

def definir_variables_para_features():
    """Define qué variables usar para cada tipo de feature"""
    
    variables_features = {
        'accesibilidad_transporte': {
            'descripcion': 'Variables para calcular accesibilidad a transporte público',
            'archivos_fuente': ['Lineas_de_metro_de_Santiago.geojson'],
            'variables_usar': ['geometry'],
            'features_derivar': [
                'distancia_metro_m',
                'tiempo_metro_min',
                'estaciones_1km'
            ],
            'notas': 'Falta agregar estaciones puntuales desde OSM'
        },
        
        'densidad_servicios': {
            'descripcion': 'Densidad de servicios urbanos en diferentes radios',
            'archivos_fuente': [
                'servicios_filtrados.geojson',
                'tiendas_filtradas.geojson', 
                'puntos_medicos_farmacias_hospitales_filtrados.geojson'
            ],
            'variables_usar': ['geometry', 'amenity', 'shop', 'healthcare'],
            'features_derivar': [
                'servicios_300m',
                'servicios_600m', 
                'servicios_1km',
                'tiendas_300m',
                'salud_1km'
            ]
        },
        
        'educacion_acceso': {
            'descripcion': 'Acceso a educación por niveles',
            'archivos_fuente': [
                'establecimientos_educacion_escolar.geojson',
                'establecimientos_parvularia_filtrados.geojson',
                'establecimientos_educacion_superior.geojson'
            ],
            'variables_usar': ['geometry', 'tipo', 'nivel'],
            'features_derivar': [
                'colegios_1km',
                'jardines_500m',
                'universidades_2km',
                'distancia_colegio_m'
            ]
        },
        
        'seguridad_proximidad': {
            'descripcion': 'Proximidad a servicios de seguridad',
            'archivos_fuente': [
                'cuarteles_filtrados.geojson',
                'unidades_operativas_pdi_filtradas.geojson',
                'cuerpos_de_bomberos_filtrados.geojson'
            ],
            'variables_usar': ['geometry', 'tipo_unidad', 'categoria'],
            'features_derivar': [
                'seguridad_1km_dummy',
                'distancia_pdi_m',
                'distancia_bomberos_m'
            ]
        },
        
        'ambiente_calidad': {
            'descripcion': 'Calidad ambiental y espacios verdes',
            'archivos_fuente': [
                'areas_verdes_filtradas.geojson',
                'ocio_filtrado.geojson',
                'vertedores_ilegales_filtrados.geojson'
            ],
            'variables_usar': ['geometry', 'area_m2', 'tipo'],
            'features_derivar': [
                'pct_area_verde_600m',
                'distancia_parque_m',
                'area_parque_cercano_m2',
                'vertederos_1km_dummy'
            ]
        },
        
        'contexto_socioeconomico': {
            'descripcion': 'Variables de contexto socioeconómico por zona',
            'archivos_fuente': [
                'unidades_vecinales_filtradas.geojson',
                'poblacion_filtrada.geojson'
            ],
            'variables_usar': ['geometry', 'total_poblacion', 'densidad_vivienda', 'indicadores_ndpet*'],
            'features_derivar': [
                'densidad_poblacional',
                'indicadores_socioeconomicos_uv',
                'tipologia_barrio'
            ],
            'metodo_join': 'point_in_polygon'
        }
    }
    
    return variables_features

def crear_diccionario_completo(esquemas_archivos, carpeta_salida):
    """Crea el diccionario de datos completo del proyecto"""
    
    diccionario = {
        'proyecto': {
            'nombre': 'Sistema de Valoración Inmobiliaria Geoespacial con Satisfacción Personalizada',
            'fecha_creacion': datetime.now().isoformat(),
            'version': '1.0.0',
            'descripcion': 'Predicción dual de precio y satisfacción personalizada para propiedades inmobiliarias',
            'crs_proyecto': 'EPSG:32719 (UTM 19S)',
            'area_estudio': 'Gran Santiago - Comunas seleccionadas'
        },
        
        'metadatos_generales': {
            'total_archivos': len(esquemas_archivos),
            'total_geometrias': sum(e.get('filas', 0) for e in esquemas_archivos),
            'clasificacion_tematica': clasificar_archivos_por_categoria(),
            'fuentes_originales': definir_fuentes_originales(),
            'variables_para_features': definir_variables_para_features()
        },
        
        'esquemas_archivos': esquemas_archivos,
        
        'proceso_normalizacion': {
            'descripcion': 'Todos los archivos fueron normalizados a EPSG:32719',
            'transformaciones_aplicadas': [
                'Detección y asignación de CRS faltantes',
                'Reproyección a UTM 19S (EPSG:32719)',
                'Reparación de geometrías inválidas',
                'Eliminación de geometrías nulas',
                'Cálculo de áreas y perímetros en metros',
                'Agregación de coordenadas de centroides',
                'Eliminación de campos Shape__Area/Length en grados'
            ],
            'campos_agregados': [
                'area_m2: Área en metros cuadrados (solo polígonos)',
                'perimeter_m: Perímetro en metros (solo polígonos)', 
                'centroide_x: Coordenada X del centroide (UTM)',
                'centroide_y: Coordenada Y del centroide (UTM)',
                'crs_original: CRS original antes de normalización',
                'fecha_normalizacion: Timestamp del procesamiento'
            ]
        },
        
        'pipeline_features': {
            'descripcion': 'Variables espaciales a derivar para modelos de precio y satisfacción',
            'escalas_analisis': ['300m', '600m', '1000m', '2000m'],
            'tipos_features': [
                'Distancias euclidianas y tiempos de viaje',
                'Densidades por buffer circular',
                'Porcentajes de cobertura (áreas verdes)',
                'Dummies de proximidad (seguridad, des-amenidades)',
                'Variables socioeconómicas por join espacial',
                'Coordenadas UTM para tendencias espaciales'
            ]
        }
    }
    
    return diccionario

def generar_documentacion_uso(diccionario, carpeta_salida):
    """Genera documentación de uso para el equipo"""
    
    readme_content = f"""# Diccionario de Datos - Proyecto Valoración Inmobiliaria

##  Resumen del Proyecto
**Nombre:** {diccionario['proyecto']['nombre']}  
**Versión:** {diccionario['proyecto']['version']}  
**Fecha:** {diccionario['proyecto']['fecha_creacion'][:10]}  
**CRS del Proyecto:** {diccionario['proyecto']['crs_proyecto']}

##  Estadísticas Generales
- **Total archivos:** {diccionario['metadatos_generales']['total_archivos']}
- **Total geometrías:** {diccionario['metadatos_generales']['total_geometrias']:,}
- **Área de estudio:** {diccionario['proyecto']['area_estudio']}

##  Estructura de Carpetas
```
autocorrelacion_espacial/
 datos_filtrados/          # Datos originales (NO USAR para análisis)
 datos_normalizados/       # Datos listos para análisis (USAR ESTOS)
 features/                 # Features derivadas (se generarán)
 scripts/                  # Scripts de procesamiento
 reportes/                # Reportes de calidad y validación
```

##  Categorías de Datos

### Transporte y Movilidad
{chr(10).join(f"- {archivo}" for archivo in diccionario['metadatos_generales']['clasificacion_tematica']['transporte_movilidad'])}

### Servicios y Amenidades  
{chr(10).join(f"- {archivo}" for archivo in diccionario['metadatos_generales']['clasificacion_tematica']['servicios_amenidades'])}

### Educación
{chr(10).join(f"- {archivo}" for archivo in diccionario['metadatos_generales']['clasificacion_tematica']['educacion'])}

### Seguridad
{chr(10).join(f"- {archivo}" for archivo in diccionario['metadatos_generales']['clasificacion_tematica']['seguridad'])}

### Ambiente y Espacios
{chr(10).join(f"- {archivo}" for archivo in diccionario['metadatos_generales']['clasificacion_tematica']['ambiente_espacios'])}

### Límites Administrativos
{chr(10).join(f"- {archivo}" for archivo in diccionario['metadatos_generales']['clasificacion_tematica']['limites_administrativos'])}

### Socioeconómico
{chr(10).join(f"- {archivo}" for archivo in diccionario['metadatos_generales']['clasificacion_tematica']['socioeconomico'])}

##  Campos Agregados Durante Normalización
{chr(10).join(f"- **{campo.split(':')[0]}:** {campo.split(': ')[1]}" for campo in diccionario['proceso_normalizacion']['campos_agregados'])}

##  Notas Importantes

### USAR SIEMPRE datos_normalizados/
-  **SÍ:** `gdf = gpd.read_file('datos_normalizados/servicios_filtrados.geojson')`
-  **NO:** `gdf = gpd.read_file('datos_filtrados/servicios_filtrados.geojson')`

### CRS y Métricas
- Todos los archivos están en **EPSG:32719** (UTM 19S)
- Distancias y áreas ya están en **metros** y **metros cuadrados**
- NO calcular `.distance()` o `.buffer()` en WGS84 (grados)

### Campos Clave por Archivo
- **OSM:** Usar `osm_id2` como ID único
- **Censo:** Usar `GEOCODIGO` para joins
- **Áreas:** Campo `area_m2` calculado correctamente
- **Coordenadas:** `centroide_x`, `centroide_y` para análisis

##  Próximos Pasos
1. **Completar estaciones metro:** Descargar desde OSM
2. **Generar features espaciales:** Scripts en desarrollo
3. **Validar con datos de propiedades:** Cuando estén disponibles

##  Contacto
Ver reportes detallados en `reportes/` para más información técnica.
"""

    # Guardar README
    with open(carpeta_salida / 'README_datos.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    return readme_content

if __name__ == "__main__":
    # Configuración
    base_path = Path(__file__).parent.parent
    carpeta_datos = base_path / "datos_normalizados"
    carpeta_reportes = base_path / "reportes"
    
    carpeta_reportes.mkdir(exist_ok=True)
    
    # Extraer esquemas de todos los archivos
    archivos_geojson = list(carpeta_datos.glob('*.geojson'))
    
    print(f" Creando diccionario de datos para {len(archivos_geojson)} archivos...")
    
    esquemas = []
    for archivo in sorted(archivos_geojson):
        print(f"   Procesando: {archivo.name}")
        esquema = extraer_esquema_archivo(archivo)
        esquemas.append(esquema)
    
    # Crear diccionario completo
    diccionario = crear_diccionario_completo(esquemas, carpeta_reportes)
    
    # Guardar diccionario completo (con manejo seguro de tipos)
    def safe_json_default(obj):
        """Convertidor seguro para tipos no serializables"""
        if hasattr(obj, 'isoformat'):
            return obj.isoformat()
        elif hasattr(obj, '__str__'):
            return str(obj)
        return repr(obj)
    
    with open(carpeta_reportes / 'diccionario_datos.json', 'w', encoding='utf-8') as f:
        json.dump(diccionario, f, ensure_ascii=False, indent=2, default=safe_json_default)
    
    # Generar documentación de uso
    readme = generar_documentacion_uso(diccionario, carpeta_reportes)
    
    print(f"\n DICCIONARIO DE DATOS CREADO")
    print(f" Diccionario completo: {carpeta_reportes / 'diccionario_datos.json'}")
    print(f" Documentación de uso: {carpeta_reportes / 'README_datos.md'}")
    print(f"\n Total archivos documentados: {len(esquemas)}")
    
    # Mostrar resumen de categorías
    clasificacion = clasificar_archivos_por_categoria()
    print(f"\n RESUMEN POR CATEGORÍAS:")
    for categoria, archivos in clasificacion.items():
        print(f"   {categoria}: {len(archivos)} archivos")
    
    print(f"\n Diccionario de datos completado!")