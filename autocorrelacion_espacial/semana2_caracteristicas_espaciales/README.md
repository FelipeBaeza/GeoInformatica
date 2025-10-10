# Semana 2: Ingeniería de Características Espaciales

## Descripción General

Esta fase del proyecto se enfoca en la **generación de características espaciales cuantitativas** para evaluar la habitabilidad urbana en 4 comunas de Santiago: La Reina, Santiago, Ñuñoa y Estación Central.

### Objetivos
- Crear una grilla sistemática de evaluación con 3,149 puntos
- Calcular 72 características espaciales por ubicación
- Generar índices de accesibilidad y habitabilidad
- Realizar análisis estadístico y visual de los resultados
- Crear visualizaciones interactivas simples para análisis

## Scripts y Códigos Explicados

### `generar_grilla.py`
**QUÉ HACE**: Crea una grilla regular de puntos de evaluación sobre las 4 comunas objetivo.

**FUNCIONALIDADES**:
- `crear_grilla_regular()`: Genera puntos espaciados cada 250m en sistema UTM
- `filtrar_por_comunas()`: Mantiene solo puntos dentro de límites comunales
- `validar_cobertura()`: Verifica que la grilla cubra toda el área de estudio
- `exportar_grilla()`: Guarda la grilla en formato GeoJSON para análisis posterior

**ENTRADA**: Shapefile de comunas filtradas
**SALIDA**: grilla_evaluacion.geojson (3,149 puntos)

### `calcular_distancias.py`
**QUÉ HACE**: Calcula distancias euclidianas desde cada punto de la grilla hacia servicios urbanos.

**FUNCIONALIDADES**:
- `calcular_distancia_euclidiana()`: Mide distancia directa entre puntos y servicios
- `encontrar_mas_cercano()`: Identifica el servicio más próximo a cada punto de grilla
- `procesar_categoria_servicios()`: Procesa una categoría completa (ej: educación, salud)
- `normalizar_distancias()`: Convierte distancias a escala 0-10 (10=muy cerca, 0=muy lejos)
- `generar_reporte_distancias()`: Crea estadísticas descriptivas de distancias calculadas

**ENTRADA**: grilla_evaluacion.geojson + datasets de servicios
**SALIDA**: grilla_con_distancias.geojson (21 nuevas columnas de distancias)

### `calcular_densidades.py`
**QUÉ HACE**: Calcula densidades de servicios en buffers circulares alrededor de cada punto.

**FUNCIONALIDADES**:
- `crear_buffer_circular()`: Crea áreas circulares de 300m, 600m y 1000m alrededor de cada punto
- `contar_servicios_en_buffer()`: Cuenta cuántos servicios hay dentro de cada buffer
- `calcular_densidad_normalizada()`: Convierte conteos a densidad por km² y normaliza 0-10
- `procesar_multiples_buffers()`: Calcula densidades para los 3 radios simultáneamente
- `validar_geometrias()`: Verifica que buffers y servicios tengan geometrías válidas

**ENTRADA**: grilla_con_distancias.geojson + datasets de servicios
**SALIDA**: grilla_con_densidades.geojson (+42 columnas de densidades)

### `crear_indices_accesibilidad.py`
**QUÉ HACE**: Combina distancias y densidades para crear índices compuestos de accesibilidad.

**FUNCIONALIDADES**:
- `calcular_indice_educacion()`: Pondera proximidad a colegios, universidades y bibliotecas
- `calcular_indice_salud()`: Integra acceso a hospitales, consultorios y farmacias
- `calcular_indice_transporte()`: Evalúa conectividad metro, buses y ciclovías
- `calcular_indice_comercial()`: Mide acceso a centros comerciales y mercados
- `calcular_indice_seguridad()`: Combina proximidad a comisarías y bomberos
- `calcular_indice_entorno()`: Evalúa calidad del entorno urbano (parques, cultura)
- `aplicar_pesos_ponderados()`: Asigna pesos diferentes a distancias vs densidades

**ENTRADA**: grilla_con_densidades.geojson
**SALIDA**: grilla_con_indices_accesibilidad.geojson (+6 índices principales)

### `generar_analisis_estadistico.py`
**QUÉ HACE**: Calcula estadísticas descriptivas y análisis de correlaciones entre variables.

**FUNCIONALIDADES**:
- `calcular_estadisticas_descriptivas()`: Media, mediana, std, min, max para cada variable
- `generar_matriz_correlaciones()`: Calcula correlaciones de Pearson entre todas las variables
- `identificar_outliers()`: Detecta valores atípicos usando método IQR
- `analisis_por_comuna()`: Compara estadísticas entre comunas
- `generar_ranking_habitabilidad()`: Ordena puntos por índice de habitabilidad global
- `crear_reporte_estadistico()`: Genera reporte JSON con todos los análisis

**ENTRADA**: grilla_con_indices.geojson
**SALIDA**: analisis_estadistico.json + estadisticas_por_comuna.json

### `generar_graficos.py`
**QUÉ HACE**: Crea 10 visualizaciones comprehensivas de los resultados del análisis espacial.

**FUNCIONALIDADES**:
- `configurar_matplotlib()`: Establece parámetros estéticos profesionales para gráficos
- `cargar_datos()`: Carga y valida datos geoespaciales procesados
- `crear_grafico_distribucion_comunas()`: Muestra distribución de puntos por comuna
- `crear_grafico_indices_principales()`: Boxplots de los 6 índices de accesibilidad por comuna
- `crear_grafico_indices_superiores()`: Análisis de índices compuestos (vida urbana, calidad vida)
- `crear_grafico_distancias_densidades()`: Scatter plots correlacionando distancias vs densidades
- `crear_grafico_mapa_habitabilidad()`: Mapa espacial con colores según habitabilidad
- `crear_dashboard_resumen()`: Panel ejecutivo con métricas principales y comparaciones
- `mostrar_graficos_interactivos()`: **NUEVO** - Muestra gráficos en pantalla de forma simple

**ENTRADA**: grilla_con_indices.geojson
**SALIDA**: 10 archivos PNG + visualización interactiva en pantalla

### `mostrar_graficos.py` *(NUEVO)*
**QUÉ HACE**: Script simplificado para visualizar gráficos de forma interactiva sin generar archivos.

**FUNCIONALIDADES**:
- Carga datos ya procesados
- Muestra gráficos uno por uno en ventanas simples
- No requiere navegador web ni URLs externas
- El usuario cierra cada ventana para continuar al siguiente
- Perfecto para revisión rápida de resultados

**USO**: `python mostrar_graficos.py`

### `ejecutar_semana2.py`
**QUÉ HACE**: Script maestro que ejecuta todo el pipeline de la Semana 2 en secuencia.

**FUNCIONALIDADES**:
- `validar_prerequisitos()`: Verifica que existan todos los datos de entrada
- `ejecutar_pipeline_completo()`: Corre todos los scripts en orden correcto
- `monitorear_progreso()`: Muestra progreso de cada etapa
- `validar_salidas()`: Confirma que cada etapa genere sus archivos esperados
- `generar_reporte_final()`: Resume todo el proceso ejecutado

**ENTRADA**: Datos normalizados de Semana 1
**SALIDA**: Todos los archivos de análisis espacial + reporte de ejecución

### `generar_resumen_ejecutivo.py`
**QUÉ HACE**: Crea reportes ejecutivos con métricas principales y hallazgos clave.

**FUNCIONALIDADES**:
- `extraer_metricas_clave()`: Identifica indicadores más relevantes
- `generar_ranking_comunas()`: Ordena comunas por habitabilidad promedio 
- `identificar_mejores_ubicaciones()`: Encuentra top 10 puntos con mayor habitabilidad
- `calcular_correlaciones_principales()`: Identifica variables más correlacionadas
- `crear_reporte_markdown()`: Genera documento ejecutivo en formato Markdown
- `exportar_datos_dashboard()`: Prepara datos para dashboard interactivo

## Metodología

### 1. Generación de Grilla Regular
- **Resolución**: 250m x 250m
- **Cobertura**: 213.3 km² en 4 comunas
- **Puntos totales**: 3,149 ubicaciones de evaluación
- **Sistema de coordenadas**: EPSG:32719 (UTM 19S)

### 2. Cálculo de Características Espaciales

#### Distancias Euclidianas (21 métricas)
- Educación: colegios, universidades, bibliotecas
- Salud: hospitales, consultorios, farmacias
- Transporte: metro, buses, ciclovías
- Comercio: centros comerciales, mercados
- Seguridad: comisarías, bomberos
- Recreación y cultura: parques, museos, gimnasios

#### Densidades por Buffer (42 métricas)
- **Radios de análisis**: 300m, 600m, 1000m
- **Normalización**: Escala 0-10 para comparabilidad
- **14 categorías** de servicios evaluadas

#### Índices de Accesibilidad (9 métricas)
- **Accesibilidad Educativa**: proximidad + diversidad educativa
- **Accesibilidad en Salud**: cobertura médica integral
- **Accesibilidad al Transporte**: conectividad multimodal
- **Accesibilidad al Entorno**: calidad del entorno urbano
- **Accesibilidad a Seguridad**: cobertura de seguridad
- **Accesibilidad Comercial**: diversidad comercial

#### Índices Superiores (3 métricas)
- **Vida Urbana**: servicios + cultura + recreación
- **Calidad de Vida**: salud + educación + transporte
- **Habitabilidad Global**: índice maestro integral

## Estructura de Archivos

```
semana2_caracteristicas_espaciales/
 scripts/
 generar_grilla.py # Generación de grilla regular
 calcular_distancias.py # Cálculo de distancias euclidianas 
 calcular_densidades.py # Análisis de densidades por buffer
 crear_indices_accesibilidad.py # Índices de accesibilidad
 ejecutar_semana2.py # Pipeline automatizado
 generar_graficos.py # Visualizaciones principales
 generar_analisis_estadistico.py # Análisis estadístico detallado
 features/
 grilla_regular.geojson # Grilla base generada
 grilla_con_distancias.geojson # + distancias calculadas
 grilla_con_densidades.geojson # + densidades por buffer
 grilla_con_indices.geojson # Dataset final completo
 graficos/
 01_distribucion_comunas.png # Distribución de puntos
 02_indices_principales.png # Índices de accesibilidad
 03_indices_superiores.png # Índices de habitabilidad
 04_distancias_densidades.png # Análisis distancias vs densidades
 05_mapa_habitabilidad.png # Mapas espaciales
 06_dashboard_resumen.png # Dashboard ejecutivo
 07_analisis_correlaciones.png # Matriz de correlaciones
 08_analisis_pca.png # Componentes principales
 09_estadisticas_descriptivas.png # Distribuciones estadísticas
 10_analisis_por_comuna.png # Comparativa por comuna
 reportes/
 pipeline_semana2.json # Reporte de ejecución
 graficos_resumen.json # Índice de visualizaciones
 analisis_estadistico.json # Estadísticas detalladas
 matriz_correlaciones.csv # Matriz de correlaciones
 resultados_analisis/ # CARPETA PRINCIPAL DE RESULTADOS
 README.md # Análisis detallado en párrafos
 INDICE_IMAGENES.md # Catálogo de visualizaciones
 imagenes/ # Todas las imágenes del análisis
 01_distribucion_comunas.png
 02_indices_principales.png
 03_indices_superiores.png
 04_distancias_densidades.png
 05_mapa_habitabilidad.png
 06_dashboard_resumen.png
 07_analisis_correlaciones.png
 08_analisis_pca.png
 09_estadisticas_descriptivas.png
 10_analisis_por_comuna.png
```

## Guía de Ejecución

### Método 1: Ejecución Completa (Recomendado)
```bash
# Ejecutar todo el pipeline de la Semana 2
cd scripts/
python ejecutar_semana2.py
```

### Método 2: Ejecución Manual Paso a Paso
```bash
cd scripts/

# 1. Generar grilla de evaluación
python generar_grilla.py

# 2. Calcular distancias a servicios
python calcular_distancias.py

# 3. Calcular densidades por buffers
python calcular_densidades.py

# 4. Crear índices de accesibilidad
python crear_indices_accesibilidad.py

# 5. Generar análisis estadístico
python generar_analisis_estadistico.py

# 6. Crear visualizaciones (con opción interactiva)
python generar_graficos.py
```

### Método 3: Solo Visualización Interactiva **NUEVO**
```bash
# Para ver solo los gráficos de forma simple
cd scripts/
python mostrar_graficos.py
```

## Visualización de Gráficos

### **Opción 1: Ver Resultados Completos (Recomendado)**
**Ubicación**: `resultados_analisis/`

**Contenido**:
- **README.md**: Análisis completo en párrafos explicativos
- **imagenes/**: Todas las visualizaciones organizadas
- **INDICE_IMAGENES.md**: Catálogo detallado de cada gráfico
- **Análisis profesional** listo para presentaciones

### **Opción 2: Gráficos Interactivos Simples**
**Comando**: `python mostrar_graficos.py`

**Características**:
- Muestra gráficos directamente en pantalla
- No abre navegador web ni URLs externas
- Interfaz simple: cierra cada ventana para continuar
- Perfecto para revisión rápida de resultados

### **Opción 3: Generación Completa + Interactivos**
**Comando**: `python generar_graficos.py`

**Características**:
- Genera 10 archivos PNG de alta resolución
- Al final pregunta si deseas ver gráficos interactivos
- Combina guardado permanente + visualización inmediata

### **Opción 4: Archivos PNG Originales**
**Ubicación**: `../graficos/*.png`

Los 10 gráficos originales (también copiados en resultados_analisis/imagenes/)

## Metodología

## Ejecución del Proyecto

### Opción 1: Pipeline Automatizado (Recomendado)
```bash
cd scripts/
python ejecutar_semana2.py
```

### Opción 2: Ejecución Manual por Etapas
```bash
# 1. Generar grilla de evaluación
python generar_grilla.py

# 2. Calcular distancias a servicios
python calcular_distancias.py

# 3. Calcular densidades por buffer
python calcular_densidades.py

# 4. Crear índices de accesibilidad
python crear_indices_accesibilidad.py
```

### Opción 3: Generar Visualizaciones
```bash
# Gráficos principales
python generar_graficos.py

# Análisis estadístico detallado
python generar_analisis_estadistico.py
```

## Resultados Principales

### Cobertura del Análisis
- **3,149 puntos** evaluados sistemáticamente
- **72 características** espaciales por ubicación
- **226,728 cálculos** espaciales realizados
- **4 comunas** analizadas integralmente

### Hallazgos Clave
- **Variabilidad espacial significativa** en accesibilidad
- **Patrones diferenciados** entre comunas
- **Correlaciones fuertes** entre servicios complementarios
- **Identificación de zonas** de alta/baja habitabilidad

### Métricas de Calidad
- **Sin valores faltantes** en características principales
- **Normalización 0-10** para comparabilidad
- **Validación espacial** de todos los cálculos
- **Consistencia geográfica** verificada

## Visualizaciones Generadas

### Gráficos Principales
1. **Distribución por Comunas**: Cobertura y estadísticas
2. **Índices Principales**: Boxplots de accesibilidad por comuna
3. **Índices Superiores**: Análisis de habitabilidad integral
4. **Distancias vs Densidades**: Patrones espaciales comparativos
5. **Mapa de Habitabilidad**: Distribución espacial geográfica
6. **Dashboard Resumen**: Vista ejecutiva integral

### Análisis Estadístico Avanzado
7. **Análisis de Correlaciones**: Relaciones entre variables
8. **Componentes Principales (PCA)**: Reducción dimensional
9. **Estadísticas Descriptivas**: Distribuciones y normalidad
10. **Análisis por Comuna**: Comparativas detalladas

## Dependencias Técnicas

```python
# Geoespaciales
geopandas==1.1.1
pandas==2.3.3
numpy==2.3.3

# Análisis espacial
scipy==1.16.2

# Visualización
matplotlib==3.10.7
seaborn==0.13.2

# Análisis estadístico
scikit-learn>=1.0.0
```

## Notas Técnicas

### Optimizaciones Implementadas
- **KDTree** para cálculo eficiente de distancias
- **Procesamiento vectorizado** con NumPy/Pandas
- **Gestión de memoria** para datasets grandes
- **Validación automática** de resultados

### Consideraciones Espaciales
- **CRS consistente** (EPSG:32719) en todos los cálculos
- **Precisión métrica** para distancias y áreas
- **Filtrado geográfico** por límites comunales
- **Buffer circular** para análisis de densidad

### Escalabilidad
- **Diseño modular** para fácil extensión
- **Configuración paramétrica** de resolución de grilla
- **Validación robusta** de archivos de entrada
- **Reporting automático** de progreso y errores

## Próximos Pasos (Semana 3)

La siguiente fase incorporará:
- **Análisis de mercado inmobiliario**
- **Integración de precios de propiedades**
- **Modelos predictivos de valor**
- **Correlación habitabilidad-precio**

## Información de Contacto

**Proyecto**: Sistema de Recomendación Inmobiliaria Basado en Análisis Geoespacial 
**Fase**: Semana 2 - Ingeniería de Características Espaciales 
**Autor**: Proyecto GeoInformática 
**Fecha**: Octubre 2025