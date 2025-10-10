# Sistema de Recomendación Inmobiliaria Basado en Análisis Geoespacial

## Descripción del Proyecto

Este proyecto desarrolla un **sistema integral de análisis geoespacial** para evaluar la habitabilidad urbana y generar recomendaciones inmobiliarias inteligentes en Santiago de Chile. Utiliza técnicas avanzadas de análisis espacial, ingeniería de características y modelado predictivo para identificar las mejores ubicaciones residenciales.

### Objetivos Principales

1. **Análizar la habitabilidad urbana** mediante características espaciales cuantitativas
2. **Evaluar accesibilidad** a servicios esenciales (educación, salud, transporte, etc.)
3. **Generar índices de calidad de vida** basados en datos geoespaciales objetivos
4. **Desarrollar modelos predictivos** para valoración inmobiliaria
5. **Crear herramientas de recomendación** personalizadas para compradores

### Área de Estudio

**Región Metropolitana de Santiago - 4 Comunas Principales:**
- **Las Condes**: Sector oriente de alto desarrollo
- **Providencia**: Zona central consolidada 
- **Santiago**: Centro histórico y comercial
- **Ñuñoa**: Sector oriente residencial

**Cobertura Total:** 213.3 km² con 3,149 puntos de evaluación sistemática

## Metodología General

### Enfoque Multi-Fase
```
Semana 1: Preparación de Datos → Semana 2: Características Espaciales → Semana 3: Análisis de Mercado
```

### Tecnologías Aplicadas
- **Análisis Geoespacial**: Cálculo de distancias, densidades y buffers espaciales
- **Ingeniería de Características**: 72 variables cuantitativas por ubicación 
- **Análisis Estadístico**: PCA, correlaciones, análisis descriptivo
- **Visualización Avanzada**: Mapas, dashboards y gráficos estadísticos
- **Machine Learning**: Modelos predictivos de valoración (Semana 3)

## Estructura del Proyecto

```
GeoInformatica/
 semana1_preparacion_datos/ # Fase 1: Datos limpios y normalizados
 scripts/ # Scripts de normalización
 datos_originales/ # Datasets sin procesar (29 archivos)
 datos_normalizados/ # Datasets limpios y filtrados
 reportes/ # Reportes de calidad

 semana2_caracteristicas_espaciales/ # Fase 2: Ingeniería de características
 scripts/ # Análisis espacial y visualización
 features/ # Características calculadas
 graficos/ # Visualizaciones (10 gráficos)
 reportes/ # Análisis estadístico

 semana3_analisis_mercado/ # Fase 3: Modelos predictivos
 scripts/ # Modelos y predicciones
 datos_mercado/ # Precios inmobiliarios
 modelos/ # Algoritmos entrenados
 reportes/ # Resultados y recomendaciones
```

## Estado del Desarrollo

### **Semana 1: COMPLETADO** 
**Preparación y Normalización de Datos**
- [x] 29 datasets geoespaciales normalizados
- [x] Sistema de coordenadas unificado (EPSG:32719)
- [x] Filtrado por área de interés (4 comunas)
- [x] Control de calidad automatizado
- [x] Eliminación de duplicados y geometrías inválidas

**Logros:**
- **6,168 elementos** procesados exitosamente
- **100% CRS consistente** en todos los datasets
- **99.8% geometrías válidas** tras limpieza
- **Base de datos espacial** lista para análisis

### **Semana 2: COMPLETADO**
**Ingeniería de Características Espaciales**
- [x] Grilla regular de 3,149 puntos de evaluación
- [x] 21 métricas de distancia euclidiana
- [x] 42 métricas de densidad por buffer (300m, 600m, 1km)
- [x] 9 índices de accesibilidad especializados
- [x] 3 índices superiores de habitabilidad
- [x] 10 visualizaciones comprensivas
- [x] Análisis estadístico avanzado (PCA, correlaciones)

**Logros:**
- **72 características espaciales** por ubicación
- **226,728 cálculos espaciales** realizados
- **Cobertura sistemática** de 213.3 km²
- **Dashboard ejecutivo** con métricas clave
- **Análisis de componentes principales** completado

### **Semana 3: PLANIFICADO**
**Análisis de Mercado y Modelos Predictivos**
- [ ] Integración de datos de precios inmobiliarios
- [ ] Análisis de correlación habitabilidad-precio
- [ ] Desarrollo de modelos predictivos (Random Forest, XGBoost)
- [ ] Sistema de recomendaciones personalizadas
- [ ] Validación y métricas de rendimiento
- [ ] Dashboard final interactivo

## Resultados Principales (Semana 1-2)

### Cobertura de Análisis
```
 Comunas analizadas: 4
 Puntos de evaluación: 3,149
 Características por punto: 72
 Servicios categorizados: 17
 Total cálculos espaciales: 226,728
```

### Hallazgos Espaciales Clave
- **Variabilidad significativa** en accesibilidad entre comunas
- **Las Condes**: Mayor accesibilidad a servicios premium
- **Santiago**: Mejor conectividad de transporte público
- **Providencia**: Balance óptimo en múltiples categorías
- **Ñuñoa**: Buen acceso a educación y recreación

### Métricas de Habitabilidad
```
Habitabilidad promedio general: 6.2/10
Rango de variación: 2.1 - 9.4
Comuna con mejor habitabilidad: Las Condes (7.1/10)
Desviación estándar: 1.3
```

## Stack Tecnológico

### Procesamiento Geoespacial
```python
geopandas==1.1.1 # Manipulación de datos geoespaciales
shapely>=2.0.0 # Operaciones geométricas
fiona>=1.8.0 # I/O archivos geoespaciales
pyproj>=3.4.0 # Transformaciones coordenadas
```

### Análisis y Modelado
```python
pandas==2.3.3 # Análisis de datos
numpy==2.3.3 # Computación numérica
scipy==1.16.2 # Análisis espacial avanzado
scikit-learn==1.7.2 # Machine learning
```

### Visualización
```python
matplotlib==3.10.7 # Gráficos base
seaborn==0.13.2 # Visualización estadística
```

### Optimizaciones
- **KDTree** para búsquedas espaciales eficientes
- **Procesamiento vectorizado** con NumPy
- **Operaciones por lotes** para grandes datasets
- **Validación automática** de calidad

## Visualizaciones Generadas

### Gráficos Principales (Semana 2)
1. ** Distribución por Comunas**: Cobertura y estadísticas generales
2. ** Índices de Accesibilidad**: Boxplots por categoría y comuna
3. ** Índices de Habitabilidad**: Correlaciones y rankings
4. ** Distancias vs Densidades**: Patrones espaciales comparativos
5. ** Mapas de Habitabilidad**: Distribución geográfica de índices
6. ** Dashboard Ejecutivo**: Resumen integral de métricas

### Análisis Estadístico Avanzado
7. ** Matriz de Correlaciones**: Relaciones entre 72 variables
8. ** Componentes Principales**: Reducción dimensional (PCA)
9. ** Estadísticas Descriptivas**: Distribuciones y normalidad
10. ** Análisis por Comuna**: Comparativas detalladas

## Casos de Uso

### Para Compradores de Vivienda
- **Evaluación objetiva** de ubicaciones potenciales
- **Comparación cuantitativa** entre alternativas
- **Identificación de zonas** con mejor relación calidad-precio
- **Filtrado personalizado** según preferencias

### Para Desarrolladores Inmobiliarios
- **Identificación de oportunidades** de desarrollo
- **Análisis de competencia** espacial
- **Optimización de ubicaciones** de proyectos
- **Evaluación de potencial** de valorización

### Para Planificadores Urbanos
- **Identificación de déficits** de servicios
- **Análisis de equidad** territorial
- **Planificación de infraestructura** futura
- **Evaluación de políticas** urbanas

## Próximos Pasos (Semana 3)

### Integración de Datos de Mercado
- Scraping de portales inmobiliarios (Portal Inmobiliario, Yapo, etc.)
- Normalización de precios por m² y tipo de propiedad
- Geolocalización precisa de propiedades

### Modelado Predictivo
- Correlación habitabilidad-precio
- Modelos de regresión avanzados
- Validación cruzada y métricas de rendimiento
- Interpretación de importancia de características

### Sistema de Recomendaciones
- Algoritmo de matching personalizado
- Interface de consulta interactiva
- Ranking automatizado de opciones
- Explicabilidad de recomendaciones

## Información del Proyecto

**Título**: Sistema de Recomendación Inmobiliaria Basado en Análisis Geoespacial 
**Institución**: Proyecto GeoInformática 
**Período**: Octubre 2025 
**Estado Actual**: Semana 2 Completada 
**Próxima Fase**: Semana 3 - Análisis de Mercado 

### Repositorio y Documentación
- **Código fuente**: Scripts organizados por semana
- **Datos procesados**: Datasets normalizados y características
- **Visualizaciones**: 10 gráficos comprensivos generados
- **Reportes**: Análisis estadístico y métricas de calidad

---

> **Nota**: Este proyecto representa un enfoque integral y sistemático para el análisis inmobiliario basado en datos geoespaciales objetivos, proporcionando herramientas cuantitativas para la toma de decisiones en el mercado inmobiliario urbano.