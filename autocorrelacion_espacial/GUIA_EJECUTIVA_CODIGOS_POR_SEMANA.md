# GUÍA EJECUTIVA: CÓDIGOS POR SEMANA

## Información General del Proyecto

**Proyecto**: Análisis Espacial de Habitabilidad Urbana en Santiago
**Estructura**: 3 Semanas de desarrollo progresivo
**Tecnología Principal**: Python + GeoPandas + Análisis Espacial
**Objetivo**: Desarrollar sistema integral de evaluación de habitabilidad urbana

---

# SEMANA 1: PREPARACIÓN Y NORMALIZACIÓN DE DATOS

## **Objetivo de la Semana 1**
Preparar, limpiar y normalizar todos los datos geoespaciales necesarios para el análisis. Esta semana establece la base de datos sólida y confiable para todas las semanas posteriores.

## **Ubicación de Archivos**
```
/autocorrelacion_espacial/semana1_preparacion_datos/
 scripts/ # Códigos principales
 datos_originales/ # Datos de entrada sin procesar
 datos_normalizados/ # Datos procesados y listos
 reportes/ # Reportes de calidad y análisis
```

## **Códigos de la Semana 1**

### **`analizar_crs_geometrias.py`**
**¿Qué hace?**
- Analiza los sistemas de coordenadas (CRS) de todos los archivos GeoJSON
- Identifica inconsistencias de proyección
- Genera reporte de calidad geométrica

**¿Cuándo usarlo?**
- PRIMER script a ejecutar
- Antes de cualquier procesamiento
- Para diagnóstico inicial de datos

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana1_preparacion_datos/scripts
python analizar_crs_geometrias.py
```

**¿Qué produce?**
- `reportes/analisis_crs_geometrias.json`: Reporte técnico detallado
- Diagnóstico en consola sobre sistemas de coordenadas

---

### **`normalizar_crs.py`**
**¿Qué hace?**
- Convierte todos los archivos GeoJSON a un CRS uniforme (EPSG:32719 - UTM Zone 19S)
- Corrige problemas de proyección
- Optimiza geometrías para análisis espacial

**¿Cuándo usarlo?**
- DESPUÉS de `analizar_crs_geometrias.py`
- Obligatorio antes de cualquier análisis espacial
- Solo una vez por dataset

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana1_preparacion_datos/scripts
python normalizar_crs.py
```

**¿Qué produce?**
- Archivos GeoJSON normalizados en `datos_normalizados/`
- Reporte de transformaciones aplicadas

---

### **`crear_diccionario_datos.py`**
**¿Qué hace?**
- Analiza estructura y contenido de cada dataset
- Crea diccionario de datos con metadatos completos
- Identifica campos relevantes para análisis

**¿Cuándo usarlo?**
- DESPUÉS de normalización CRS
- Para documentación y comprensión de datos
- Antes de diseñar análisis específicos

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana1_preparacion_datos/scripts
python crear_diccionario_datos.py
```

**¿Qué produce?**
- `reportes/diccionario_datos.json`: Metadatos estructurados
- Documentación técnica de campos y tipos

---

### **`validar_calidad.py`**
**¿Qué hace?**
- Ejecuta 15+ validaciones de calidad de datos
- Detecta geometrías inválidas, duplicados, valores atípicos
- Genera reporte comprehensivo de calidad

**¿Cuándo usarlo?**
- DESPUÉS de normalización y diccionario
- Para validar integridad antes de análisis
- Cuando se sospechen problemas de datos

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana1_preparacion_datos/scripts
python validar_calidad.py
```

**¿Qué produce?**
- `reportes/reporte_calidad.json`: Validaciones detalladas
- `reportes/reporte_calidad_resumen.txt`: Resumen ejecutivo

---

### **`ejecutar_semana1.py`** **[SCRIPT PRINCIPAL]**
**¿Qué hace?**
- Ejecuta TODA la secuencia de Semana 1 automáticamente
- Coordina ejecución de todos los scripts anteriores
- Genera reporte final consolidado

**¿Cuándo usarlo?**
- Para ejecutar toda la Semana 1 de una sola vez
- Cuando se quiere proceso automatizado completo
- Recomendado para usuarios principiantes

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana1_preparacion_datos/scripts
python ejecutar_semana1.py
```

**¿Qué produce?**
- Todos los outputs de scripts individuales
- Reporte consolidado de la semana completa

## **Secuencia Recomendada Semana 1**

### **Opción A: Ejecución Automática (Recomendada)**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana1_preparacion_datos/scripts
python ejecutar_semana1.py
```

### **Opción B: Ejecución Manual Paso a Paso**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana1_preparacion_datos/scripts

# Paso 1: Analizar sistemas de coordenadas
python analizar_crs_geometrias.py

# Paso 2: Normalizar proyecciones
python normalizar_crs.py

# Paso 3: Crear documentación de datos
python crear_diccionario_datos.py

# Paso 4: Validar calidad final
python validar_calidad.py
```

## **Requisitos Previos Semana 1**
- Datos GeoJSON originales en `datos_originales/`
- Python 3.8+ con librerías: geopandas, pandas, shapely, json

## **Indicadores de Éxito Semana 1**
- Todos los archivos tienen CRS uniforme (EPSG:32719)
- Reporte de calidad sin errores críticos
- Diccionario de datos generado completamente
- Datos listos para análisis espacial

---

# SEMANA 2: CARACTERÍSTICAS ESPACIALES Y ANÁLISIS

## **Objetivo de la Semana 2**
Calcular índices de habitabilidad urbana basados en accesibilidad a servicios, generar grillas de análisis espacial, y producir visualizaciones comprehensivas del territorio.

## **Ubicación de Archivos**
```
/autocorrelacion_espacial/semana2_caracteristicas_espaciales/
 scripts/ # Códigos de análisis
 features/ # Características calculadas
 graficos/ # Visualizaciones generadas
 reportes/ # Reportes técnicos
 resultados_analisis/ # Análisis detallado de resultados
```

## **Códigos de la Semana 2**

### **`generar_grilla.py`**
**¿Qué hace?**
- Crea grilla regular de 200x200m sobre área metropolitana
- Genera puntos de análisis espacialmente distribuidos
- Establece base geométrica para todos los cálculos

**¿Cuándo usarlo?**
- PRIMER script de Semana 2
- Después de completar Semana 1
- Una sola vez por análisis

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts
python generar_grilla.py
```

**¿Qué produce?**
- `features/grilla_puntos.geojson`: Grilla de análisis
- Base geométrica para cálculos posteriores

---

### **`calcular_distancias.py`**
**¿Qué hace?**
- Calcula distancias desde cada punto de grilla a servicios más cercanos
- Procesa 8 categorías: educación, salud, transporte, seguridad, comercio, etc.
- Optimiza cálculos usando algoritmos espaciales eficientes

**¿Cuándo usarlo?**
- DESPUÉS de `generar_grilla.py`
- Proceso más lento (puede tomar 15-30 minutos)
- Base para índices de accesibilidad

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts
python calcular_distancias.py
```

**¿Qué produce?**
- `features/distancias_servicios.geojson`: Distancias calculadas
- Progreso detallado en consola

---

### **`calcular_densidades.py`**
**¿Qué hace?**
- Calcula densidad de servicios en buffers de 500m, 1km, 2km
- Complementa análisis de distancias con análisis de concentración
- Procesa mismas 8 categorías de servicios

**¿Cuándo usarlo?**
- DESPUÉS de `calcular_distancias.py` 
- En paralelo o secuencial con distancias
- Para enriquecer análisis de accesibilidad

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts
python calcular_densidades.py
```

**¿Qué produce?**
- `features/densidades_servicios.geojson`: Densidades por radios
- Complemento a análisis de distancias

---

### **`crear_indices_accesibilidad.py`**
**¿Qué hace?**
- Integra distancias y densidades en índices normalizados (0-10)
- Calcula índices compuestos: Vida Urbana, Calidad de Vida, Habitabilidad Global
- Aplica metodología de ponderación y normalización

**¿Cuándo usarlo?**
- DESPUÉS de distancias y densidades
- Paso crítico para generar métricas finales
- Prerequisito para visualizaciones

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts
python crear_indices_accesibilidad.py
```

**¿Qué produce?**
- `features/indices_accesibilidad.geojson`: Índices normalizados
- Base de datos final para análisis

---

### **`generar_analisis_estadistico.py`**
**¿Qué hace?**
- Realiza análisis estadístico completo de todos los índices
- Calcula correlaciones, PCA, estadísticas descriptivas
- Genera matrices de correlación y componentes principales

**¿Cuándo usarlo?**
- DESPUÉS de `crear_indices_accesibilidad.py`
- Para análisis avanzado de patrones
- Base para interpretación de resultados

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts
python generar_analisis_estadistico.py
```

**¿Qué produce?**
- `reportes/analisis_estadistico.json`: Análisis completo
- Datos para visualizaciones avanzadas

---

### **`generar_graficos.py`**
**¿Qué hace?**
- Genera 10 visualizaciones comprehensivas del análisis
- Incluye mapas temáticos, correlaciones, PCA, comparaciones por comuna
- Produce gráficos de alta calidad para presentación

**¿Cuándo usarlo?**
- DESPUÉS de análisis estadístico
- Para visualizar resultados
- Para presentaciones y reportes

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts
python generar_graficos.py
```

**¿Qué produce?**
- 10 archivos PNG en `graficos/`
- Visualizaciones listas para presentación

---

### **`mostrar_graficos.py`** **[VISUALIZACIÓN SIMPLE]**
**¿Qué hace?**
- Muestra gráficos directamente en pantalla sin navegador
- Versión simplificada para visualización rápida
- Evita problemas de apertura de navegador web

**¿Cuándo usarlo?**
- DESPUÉS de `generar_graficos.py`
- Para ver resultados de forma simple
- Alternativa a abrir archivos PNG manualmente

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts
python mostrar_graficos.py
```

**¿Qué produce?**
- Visualización interactiva directa
- Gráficos mostrados uno por uno

---

### **`generar_resumen_ejecutivo.py`**
**¿Qué hace?**
- Crea resumen ejecutivo automático de todos los resultados
- Integra estadísticas, interpretaciones y recomendaciones
- Genera documento final para tomadores de decisión

**¿Cuándo usarlo?**
- AL FINAL de toda la Semana 2
- Para generar reporte final
- Para comunicar resultados a no técnicos

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts
python generar_resumen_ejecutivo.py
```

**¿Qué produce?**
- `reportes/resumen_ejecutivo.md`: Reporte final
- Documento listo para presentación

---

### **`ejecutar_semana2.py`** **[SCRIPT PRINCIPAL]**
**¿Qué hace?**
- Ejecuta TODA la secuencia de Semana 2 automáticamente
- Coordina todos los scripts en orden correcto
- Maneja dependencias y validaciones

**¿Cuándo usarlo?**
- Para ejecutar toda la Semana 2 automáticamente
- Después de completar Semana 1
- Recomendado para proceso completo

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts
python ejecutar_semana2.py
```

**¿Qué produce?**
- Todos los outputs de la Semana 2
- Proceso completo automatizado

## **Secuencia Recomendada Semana 2**

### **Opción A: Ejecución Automática (Recomendada)**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts
python ejecutar_semana2.py

# Para ver resultados
python mostrar_graficos.py
```

### **Opción B: Ejecución Manual Paso a Paso**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts

# Paso 1: Crear grilla de análisis
python generar_grilla.py

# Paso 2: Calcular distancias (proceso lento)
python calcular_distancias.py

# Paso 3: Calcular densidades
python calcular_densidades.py

# Paso 4: Crear índices finales
python crear_indices_accesibilidad.py

# Paso 5: Análisis estadístico
python generar_analisis_estadistico.py

# Paso 6: Generar visualizaciones
python generar_graficos.py

# Paso 7: Ver resultados
python mostrar_graficos.py

# Paso 8: Generar reporte final
python generar_resumen_ejecutivo.py
```

### **Opción C: Solo Visualización (si ya se ejecutó todo)**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts

# Solo mostrar gráficos existentes
python mostrar_graficos.py
```

## **Requisitos Previos Semana 2**
- Semana 1 completada exitosamente
- Datos normalizados disponibles
- Librerías adicionales: matplotlib, seaborn, sklearn

## **Indicadores de Éxito Semana 2**
- Grilla de 3,149 puntos generada
- Distancias y densidades calculadas para 8 categorías
- 9 índices de accesibilidad creados
- 10 visualizaciones generadas
- Análisis estadístico completado

---

# SEMANA 3: ANÁLISIS DE MERCADO Y PREDICCIONES

## **Objetivo de la Semana 3**
Desarrollar modelos predictivos de precios inmobiliarios basados en los índices de habitabilidad, crear sistema de recomendaciones y generar análisis de mercado territorial.

## **Ubicación de Archivos**
```
/autocorrelacion_espacial/semana3_analisis_mercado/
 scripts/ # Códigos de modelado
 datos_mercado/ # Datos inmobiliarios
 modelos/ # Modelos entrenados
 visualizaciones/ # Gráficos de mercado
 reportes/ # Análisis de mercado
```

## **Códigos de la Semana 3**

### **`generar_datos_mercado_sinteticos.py`**
**¿Qué hace?**
- Genera dataset sintético de precios inmobiliarios
- Relaciona precios con índices de habitabilidad de Semana 2
- Simula mercado realista con variabilidad territorial

**¿Cuándo usarlo?**
- PRIMER script de Semana 3
- DESPUÉS de completar Semana 2
- Base para modelos predictivos

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana3_analisis_mercado/scripts
python generar_datos_mercado_sinteticos.py
```

**¿Qué produce?**
- `datos_mercado/propiedades_sinteticas.geojson`: Dataset de mercado
- Base para modelado predictivo

---

### **`analisis_predictivo.py`**
**¿Qué hace?**
- Entrena múltiples modelos de machine learning (Random Forest, XGBoost, etc.)
- Evalúa capacidad predictiva de índices de habitabilidad
- Genera análisis de importancia de variables

**¿Cuándo usarlo?**
- DESPUÉS de generar datos sintéticos
- Para crear modelos predictivos
- Base para sistema de recomendaciones

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana3_analisis_mercado/scripts
python analisis_predictivo.py
```

**¿Qué produce?**
- Modelos entrenados en `modelos/`
- `reportes/evaluacion_modelos.json`: Métricas de rendimiento
- Análisis de importancia de variables

---

### **`sistema_recomendaciones.py`**
**¿Qué hace?**
- Implementa sistema de recomendación territorial
- Encuentra ubicaciones óptimas según preferencias de usuario
- Genera ranking personalizado de habitabilidad

**¿Cuándo usarlo?**
- DESPUÉS de entrenar modelos
- Para aplicaciones prácticas
- Sistema final de usuario

**¿Cómo ejecutarlo?**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana3_analisis_mercado/scripts
python sistema_recomendaciones.py
```

**¿Qué produce?**
- Sistema interactivo de recomendaciones
- Ranking personalizado de ubicaciones
- Mapas de recomendación territorial

## **Secuencia Recomendada Semana 3**

### **Secuencia Completa**
```bash
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana3_analisis_mercado/scripts

# Paso 1: Generar datos de mercado
python generar_datos_mercado_sinteticos.py

# Paso 2: Entrenar modelos predictivos
python analisis_predictivo.py

# Paso 3: Sistema de recomendaciones
python sistema_recomendaciones.py
```

## **Requisitos Previos Semana 3**
- Semana 2 completada exitosamente
- Índices de accesibilidad disponibles
- Librerías adicionales: scikit-learn, xgboost

## **Indicadores de Éxito Semana 3**
- Dataset sintético de mercado generado
- Modelos predictivos entrenados con R² > 0.85
- Sistema de recomendaciones funcional
- Análisis de mercado completado

---

# EJECUCIÓN COMPLETA DEL PROYECTO

## **Ejecución Rápida Completa (Todas las Semanas)**

Si quieres ejecutar todo el proyecto de principio a fin:

```bash
# Paso 1: Semana 1 - Preparación de datos
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana1_preparacion_datos/scripts
python ejecutar_semana1.py

# Paso 2: Semana 2 - Análisis espacial
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana2_caracteristicas_espaciales/scripts
python ejecutar_semana2.py

# Paso 3: Ver resultados Semana 2
python mostrar_graficos.py

# Paso 4: Semana 3 - Análisis de mercado
Ubicacion: GeoInformatica/autocorrelacion_espacial/semana3_analisis_mercado/scripts
python generar_datos_mercado_sinteticos.py
python analisis_predictivo.py
python sistema_recomendaciones.py
```

## **Checklist de Ejecución Completa**

### Semana 1: Datos 
- [ ] `ejecutar_semana1.py` completado sin errores
- [ ] Reporte de calidad sin issues críticos
- [ ] Datos normalizados disponibles

### Semana 2: Análisis 
- [ ] `ejecutar_semana2.py` completado
- [ ] 10 gráficos generados correctamente
- [ ] `mostrar_graficos.py` funciona correctamente
- [ ] Índices de habitabilidad calculados

### Semana 3: Mercado 
- [ ] Datos sintéticos generados
- [ ] Modelos entrenados exitosamente 
- [ ] Sistema de recomendaciones funcional

## **Solución de Problemas Comunes**

### Error: "Archivo no encontrado"
```bash
# Verificar ubicación actual
pwd
# Verificar estructura de carpetas
ls -la
```

### Error: "Librería no instalada"
```bash
# Instalar dependencias
pip install geopandas pandas matplotlib seaborn scikit-learn xgboost
```

### Error: "Datos no disponibles"
- Verificar que Semana 1 se ejecutó completamente
- Revisar carpeta `datos_normalizados/`

### Visualización no funciona
```bash
# Usar script alternativo
cd semana2_caracteristicas_espaciales/scripts
python mostrar_graficos.py
```

## **Contacto y Documentación Adicional**

- **Documentación detallada**: Ver `README.md` en cada carpeta de semana
- **Análisis de resultados**: Ver `resultados_analisis/README.md` en Semana 2 
- **Código fuente**: Todos los scripts están comentados línea por línea
- **Reportes técnicos**: Disponibles en carpetas `reportes/` de cada semana

---

*Guía creada: 10 de octubre de 2025* 
*Proyecto: Análisis Espacial de Habitabilidad Urbana en Santiago* 
*Autor: Felipe Baeza*