# Semana 1: Preparación y Normalización de Datos

## Descripción General

Esta fase inicial del proyecto se enfoca en la **preparación, limpieza y normalización** de todos los datasets geoespaciales necesarios para el análisis de habitabilidad urbana en Santiago.

### Objetivos
- Normalizar todos los datasets a un sistema de coordenadas común
- Filtrar datos por área de interés (4 comunas principales)
- Estandarizar formatos y estructuras de datos
- Crear una base de datos geoespacial limpia y consistente

## Metodología

### 1. Identificación de Datasets Relevantes
- **29 datasets geoespaciales** identificados
- **17 categorías de servicios** urbanos
- **Cobertura completa** de servicios esenciales
- **Formatos múltiples**: Shapefile, GeoJSON, CSV con coordenadas

### 2. Normalización y Filtrado

#### Sistema de Coordenadas
- **CRS objetivo**: EPSG:32719 (UTM Zone 19S)
- **Precisión métrica** para cálculos espaciales
- **Reproyección automática** desde múltiples CRS origen

#### Filtrado Geográfico
- **Área de interés**: 4 comunas principales
- Las Condes
- Providencia 
- Santiago
- Ñuñoa
- **Buffer de inclusión**: Márgenes para capturar servicios limítrofes
- **Validación geométrica** de todos los elementos

#### 🧹 Limpieza de Datos
- **Eliminación de duplicados** espaciales
- **Validación de geometrías** (detección de elementos inválidos)
- **Estandarización de campos** de atributos
- **Control de calidad** automatizado

## Scripts y Códigos Explicados

### `normalizar_datasets.py`
**QUÉ HACE**: Script principal que coordina todo el proceso de normalización de datasets geoespaciales heterogéneos hacia un formato estándar común.

**FUNCIONALIDADES PRINCIPALES**:
- `definir_area_interes()`: Establece límites geográficos de las 4 comunas objetivo (La Reina, Santiago, Ñuñoa, Estación Central)
- `listar_datasets_disponibles()`: Escanea directorios y cataloga todos los archivos geoespaciales encontrados
- `detectar_formato_origen()`: Identifica automáticamente tipo de archivo (Shapefile, GeoJSON, CSV) y CRS origen
- `reproyectar_a_utm()`: Convierte coordenadas desde cualquier CRS origen hacia EPSG:32719 (UTM 19S)
- `filtrar_por_area()`: Aplica máscara geográfica para mantener solo elementos dentro/cerca de las comunas
- `validar_geometrias()`: Detecta y corrige geometrías inválidas (auto-intersecciones, topología corrupta)
- `eliminar_duplicados_espaciales()`: Identifica y remueve elementos duplicados usando distancia mínima
- `estandarizar_campos()`: Normaliza nombres de columnas y tipos de datos para consistencia
- `aplicar_control_calidad()`: Ejecuta batería de validaciones automáticas sobre datos procesados
- `exportar_dataset_normalizado()`: Guarda resultado en formato GeoJSON con metadatos completos

**FLUJO DE PROCESAMIENTO**:
1. Carga dataset original en cualquier formato/CRS
2. Reproyecta a UTM 19S para cálculos métricos precisos 
3. Filtra geográficamente por área de interés con buffer
4. Valida y corrige geometrías problemáticas
5. Elimina duplicados espaciales usando tolerancia geométrica
6. Estandariza estructura de atributos
7. Aplica control de calidad automatizado
8. Exporta como GeoJSON normalizado

**ENTRADA**: Datasets originales en formatos variados
**SALIDA**: GeoJSON normalizados en ../datos_normalizados/

### `validar_normalizacion.py`
**QUÉ HACE**: Sistema de control de calidad que valida exhaustivamente todos los datasets normalizados para garantizar integridad y consistencia.

**FUNCIONALIDADES PRINCIPALES**:
- `validar_crs_consistente()`: Verifica que todos los datasets usen EPSG:32719
- `validar_geometrias_validas()`: Confirma que todas las geometrías sean topológicamente correctas
- `validar_cobertura_geografica()`: Verifica que elementos estén dentro del área esperada
- `validar_campos_requeridos()`: Confirma presencia de atributos mínimos necesarios
- `detectar_duplicados_residuales()`: Identifica duplicados que pudieron escapar normalización
- `validar_tipos_datos()`: Verifica consistencia de tipos de datos entre datasets
- `calcular_estadisticas_calidad()`: Genera métricas de completitud y precisión
- `generar_reporte_validacion()`: Crea reporte detallado con todos los resultados
- `identificar_problemas_criticos()`: Marca issues que requieren corrección manual
- `sugerir_acciones_correctivas()`: Propone soluciones específicas para problemas encontrados

**VALIDACIONES EJECUTADAS**:
1. **Espaciales**: CRS, geometrías válidas, cobertura geográfica
2. **Estructurales**: Campos requeridos, tipos de datos, valores nulos 
3. **Semánticas**: Clasificaciones coherentes, rangos válidos
4. **Integridad**: Duplicados, consistencia entre datasets
5. **Metadatos**: Documentación completa, trazabilidad

**ENTRADA**: Datasets normalizados de ../datos_normalizados/
**SALIDA**: Reportes de validación JSON + recomendaciones de corrección

### `organizar_estructura.py` 
**QUÉ HACE**: Organiza automáticamente la estructura de directorios del proyecto y clasifica datasets por categorías temáticas.

**FUNCIONALIDADES PRINCIPALES**:
- `crear_estructura_directorios()`: Crea jerarquía estándar de carpetas del proyecto
- `clasificar_datasets_por_categoria()`: Agrupa datasets por temática (educación, salud, transporte, etc.)
- `generar_inventario_completo()`: Cataloga todos los archivos con metadatos descriptivos
- `establecer_nomenclatura_estandar()`: Aplica convención de nombres consistente
- `crear_enlaces_simbolicos()`: Facilita acceso a datasets desde múltiples ubicaciones
- `documentar_linaje_datos()`: Registra origen y transformaciones aplicadas a cada dataset

**CATEGORÍAS ORGANIZACIONALES**:
- **Educación**: colegios, universidades, bibliotecas
- **Salud**: hospitales, consultorios, farmacias
- **Transporte**: metro, buses, ciclovías
- **Comercio**: centros comerciales, mercados, tiendas
- **Seguridad**: comisarías, bomberos, emergencias
- **Recreación**: parques, plazas, áreas verdes
- **Cultura**: museos, teatros, centros culturales
- **Servicios**: municipalidades, servicios públicos

## Estructura de Archivos

```
semana1_preparacion_datos/
 scripts/
 normalizar_datasets.py # Script principal de normalización
 validar_normalizacion.py # Sistema de control de calidad 
 organizar_estructura.py # Organización de directorios
 datos_originales/
 areas_ocio_filtrado.geojson
 atracciones_turisticas_filtradas.geojson
 estaciones_carga_filtradas.geojson
 poblacion_filtrada.geojson
 servicios_filtrados.geojson
 tiendas_filtradas.geojson
 unidades_vecinales_filtradas.geojson
 ... (29 datasets originales)
 datos_normalizados/
 areas_ocio_filtrado.geojson
 atracciones_turisticas_filtradas.geojson
 bibliotecas_filtradas.geojson
 campamentos.geojson
 centros_comerciales_filtrados.geojson
 ... (29 datasets normalizados)
 reportes/
 normalizacion_completa.json # Reporte de normalización
 validacion_calidad.json # Control de calidad
```

## Datasets Procesados

### Educación y Cultura
- **bibliotecas_filtradas.geojson**: Bibliotecas públicas y universitarias
- **colegios_filtrados.geojson**: Instituciones educacionales básica/media
- **universidades_filtradas.geojson**: Educación superior
- **museos_filtrados.geojson**: Espacios culturales y patrimoniales

### Salud y Bienestar
- **puntos_medicos_farmacias_hospitales_filtrados.geojson**: Red de salud integral
- **centros_sernam_filtrados.geojson**: Centros de apoyo social
- **gimnasios_filtrados.geojson**: Espacios de actividad física

### Transporte y Movilidad
- **estaciones_metro_filtradas.geojson**: Red de metro de Santiago
- **paraderos_buses_filtrados.geojson**: Sistema de transporte público
- **ciclovias_filtradas.geojson**: Infraestructura ciclista
- **estaciones_carga_filtradas.geojson**: Puntos de carga eléctrica

### Seguridad y Emergencias 
- **comisarias_filtradas.geojson**: Unidades policiales
- **cuarteles_filtrados.geojson**: Destacamentos militares
- **cuerpos_de_bomberos_filtrados.geojson**: Estaciones de bomberos

### Comercio y Servicios
- **centros_comerciales_filtrados.geojson**: Grandes superficies comerciales
- **mercados_filtrados.geojson**: Mercados y ferias
- **tiendas_filtradas.geojson**: Comercio minorista
- **servicios_filtrados.geojson**: Servicios diversos

### Recreación y Espacio Público
- **areas_verdes_filtradas.geojson**: Parques y áreas recreativas
- **areas_ocio_filtrado.geojson**: Espacios de entretenimiento
- **atracciones_turisticas_filtradas.geojson**: Puntos de interés turístico

### Datos Territoriales y Demográficos
- **unidades_vecinales_filtradas.geojson**: División administrativa local
- **poblacion_filtrada.geojson**: Datos demográficos
- **poblacion2_filtrada.geojson**: Datos poblacionales complementarios
- **campamentos.geojson**: Asentamientos informales
- **delincuencia_comunas_anual.geojson**: Estadísticas de seguridad

### Datos de Referencia
- **puntos_de_interes_filtrados.geojson**: POIs generales
- **base_maestra_comunas_filtradas.geojson**: Límites comunales
- **comunas_buffer.geojson**: Áreas de análisis con buffer

## Métricas de Procesamiento

### Cobertura Geográfica
- **Área total procesada**: ~250 km²
- **4 comunas principales** completamente cubiertas
- **Buffer de inclusión**: 500m para servicios limítrofes
- **Validación geométrica**: 100% de elementos verificados

### Control de Calidad
- **Elementos con geometría válida**: >99.8%
- **Duplicados eliminados**: Variable por dataset
- **CRS unificado**: 100% EPSG:32719
- **Campos estandarizados**: Todos los datasets

### Estadísticas por Categoría
```
Educación: 847 elementos procesados
Salud: 312 elementos procesados 
Transporte: 1,156 elementos procesados
Comercio: 2,341 elementos procesados
Seguridad: 89 elementos procesados
Recreación: 423 elementos procesados
```

## Tecnologías Utilizadas

### Bibliotecas Principales
```python
geopandas==1.1.1 # Manipulación de datos geoespaciales
pandas==2.3.3 # Análisis de datos tabulares
shapely>=2.0.0 # Operaciones geométricas
fiona>=1.8.0 # I/O de archivos geoespaciales
pyproj>=3.4.0 # Transformaciones de coordenadas
```

### Herramientas de Validación
- **Validación geométrica**: Shapely
- **Control de CRS**: PyProj
- **Detección de duplicados**: Spatial indexing
- **Estadísticas de calidad**: Pandas

## Metodología de Normalización

### 1. Análisis de Entrada
```python
# Detectar CRS original
crs_original = dataset.crs

# Analizar estructura de campos
campos_disponibles = dataset.columns.tolist()

# Validar geometrías
geometrias_validas = dataset.geometry.is_valid.sum()
```

### 2. Transformación Espacial
```python
# Reproyección a UTM 19S
dataset_utm = dataset.to_crs('EPSG:32719')

# Filtrado por área de interés
dataset_filtrado = gpd.clip(dataset_utm, area_comunas)

# Validación post-transformación
assert dataset_filtrado.crs.to_epsg() == 32719
```

### 3. Control de Calidad
```python
# Eliminación de duplicados espaciales
dataset_sin_duplicados = eliminar_duplicados_espaciales(dataset_filtrado)

# Validación de geometrías
dataset_valido = dataset_sin_duplicados[dataset_sin_duplicados.geometry.is_valid]

# Estadísticas de calidad
estadisticas = generar_reporte_calidad(dataset_valido)
```

## Validaciones Implementadas

### Consistencia Espacial
- **CRS unificado** en todos los datasets
- **Geometrías válidas** (sin auto-intersecciones)
- **Cobertura geográfica** verificada
- **Precisión de coordenadas** validada

### Integridad de Datos
- **Campos requeridos** presentes
- **Tipos de datos** consistentes
- **Valores nulos** controlados
- **Duplicados** eliminados

### Calidad Temática
- **Clasificaciones** coherentes
- **Atributos descriptivos** preservados
- **Metadatos** documentados
- **Trazabilidad** del procesamiento

## Instrucciones de Ejecución

### Normalización Completa
```bash
cd semana1_preparacion_datos/scripts/
python normalizar_datasets.py
```

### Validación de Calidad
```bash
python validar_normalizacion.py
```

### Verificación Manual
```python
import geopandas as gpd

# Cargar cualquier dataset normalizado
dataset = gpd.read_file('../datos_normalizados/colegios_filtrados.geojson')

# Verificar CRS
print(f"CRS: {dataset.crs}") # Debe ser EPSG:32719

# Verificar cobertura
print(f"Bounds: {dataset.total_bounds}")

# Verificar calidad
print(f"Geometrías válidas: {dataset.geometry.is_valid.sum()}/{len(dataset)}")
```

## Checklist de Completitud

### Preparación de Datos
- [x] 29 datasets identificados y catalogados
- [x] Sistema de coordenadas unificado (EPSG:32719)
- [x] Filtrado geográfico por 4 comunas principales
- [x] Eliminación de duplicados espaciales
- [x] Validación de geometrías
- [x] Control de calidad automatizado
- [x] Documentación completa de metadatos

### Estructura Organizacional
- [x] Separación datos originales/normalizados
- [x] Scripts de procesamiento documentados
- [x] Reportes de calidad generados
- [x] Sistema de archivos organizado

### Próximas Etapas (Semana 2)
- [ ] Generación de grilla regular de evaluación
- [ ] Cálculo de distancias euclidianas
- [ ] Análisis de densidades por buffer
- [ ] Creación de índices de accesibilidad
- [ ] Visualización de características espaciales

## Información Técnica

**Proyecto**: Sistema de Recomendación Inmobiliaria Basado en Análisis Geoespacial 
**Fase**: Semana 1 - Preparación y Normalización de Datos 
**Estado**: **COMPLETADO** 
**Calidad**: **VALIDADO** 
**Próximo paso**: Semana 2 - Ingeniería de Características Espaciales

---

> **Nota**: Todos los datasets normalizados están listos para ser utilizados en análisis espaciales posteriores. La estructura organizacional permite fácil extensión y mantenimiento del proyecto.