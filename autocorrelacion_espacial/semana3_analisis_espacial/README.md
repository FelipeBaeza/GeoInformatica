# Semana 3: Análisis de Patrones Espaciales# Semana 3: Análisis Espacial Avanzado



## ¿Qué hicimos esta semana?## 🎯 Objetivos



En la tercera semana, utilizamos las métricas calculadas en la Semana 2 para descubrir **patrones espaciales**: identificamos si existen zonas similares que se agrupan geográficamente (clusters) y segmentamos Santiago en "submercados" con características homogéneas.Esta fase del proyecto se enfoca en **responder la pregunta central**: ¿El entorno geoespacial afecta las preferencias y el tipo de vivienda que el usuario desea?



---Para responder, implementamos tres análisis fundamentales:



## La Pregunta Central1. **Autocorrelación Espacial Global** (Índice de Moran) - ¿Existe agrupamiento espacial?

2. **Autocorrelación Espacial Local** (LISA) - ¿Dónde están los clusters?

**¿Los barrios similares tienden a estar cerca unos de otros?**3. **Identificación de Submercados** (K-Means Clustering) - ¿Cómo se segmenta el territorio?



Por ejemplo:---

- ¿Los barrios con buenos servicios están agrupados geográficamente?

- ¿Las zonas con acceso limitado al metro forman clusters?## 📁 Estructura de Archivos

- ¿Podemos identificar "tipos" de barrios según sus características?

```

---semana3_analisis_espacial/

├── README.md                          # Este archivo

## Lo que Hicimos├── ejecutar_semana3.py               # Script principal de ejecución

├── scripts/

### 1. **Autocorrelación Espacial Global (Índice de Moran)**│   ├── calcular_autocorrelacion_global.py    # Índice de Moran global

│   ├── calcular_autocorrelacion_local.py     # LISA clusters

**¿Qué es?** Es una medida estadística que nos dice si los valores similares tienden a estar cerca geográficamente.│   └── identificar_submercados.py            # K-Means clustering

├── reportes/

**¿Cómo funciona?** │   ├── autocorrelacion_global_reporte.md     # Reporte Moran global

- **Valor positivo:** Barrios similares están juntos (ej: zonas ricas juntas, zonas pobres juntas)│   ├── autocorrelacion_global_resultados.csv # Datos CSV

- **Valor cercano a 0:** Distribución aleatoria, sin patrones claros│   ├── autocorrelacion_global_reporte.json   # Datos JSON

- **Valor negativo:** Barrios diferentes están juntos (caso raro en ciudades)│   ├── autocorrelacion_local_reporte.md      # Reporte LISA

│   ├── autocorrelacion_local_resultados.json # Resultados LISA

**Analogía:** Imagina un mapa de temperaturas. Si las zonas frías están juntas y las cálidas también, hay autocorrelación positiva. Si están mezcladas aleatoriamente, no hay autocorrelación.│   ├── submercados_reporte.md                # Reporte submercados

│   └── submercados_perfiles.json             # Perfiles submercados

**Resultado:** Calculamos el Índice de Moran para **40+ variables** (densidades y distancias). Encontramos que la mayoría muestran **autocorrelación positiva significativa**, lo que significa que sí existen patrones geográficos claros.├── mapas/

│   ├── autocorrelacion_global_resumen.png    # Resumen visual Moran

### 2. **Autocorrelación Espacial Local (LISA)**│   ├── moran_scatterplot_*.png               # Scatterplots (múltiples)

│   ├── lisa_clusters_*.png                   # Mapas LISA (múltiples)

**¿Qué es?** Mientras que el Índice de Moran nos da un valor global, LISA nos muestra **dónde** están los clusters específicamente.│   └── lisa_significancia_*.png              # Mapas significancia

└── submercados/

**Tipos de clusters identificados:**    ├── mapa_submercados.png                  # Mapa de submercados

    ├── determinacion_k_optimo.png            # Análisis k óptimo

- **🔴 High-High (HH):** Zonas con valores altos rodeadas de zonas con valores altos    ├── grilla_con_submercados.geojson        # Datos con labels

  - *Ejemplo:* Providencia tiene alta densidad de metro, y sus vecinos también    └── lisa_*.geojson                        # Resultados LISA por variable

  ```

- **🔵 Low-Low (LL):** Zonas con valores bajos rodeadas de zonas con valores bajos

  - *Ejemplo:* Zonas periféricas sin metro, rodeadas de otras sin metro---

  

- **🟠 High-Low (HL):** Zona con valor alto rodeada de zonas con valores bajos (outlier)## 🔧 Requisitos Previos

  - *Ejemplo:* Un colegio excelente en un barrio con pocos servicios

  ### Dependencias de Python

- **🟢 Low-High (LH):** Zona con valor bajo rodeada de zonas con valores altos (outlier)

  - *Ejemplo:* Un terreno vacío en medio de una zona comercialAntes de ejecutar los scripts, asegúrate de tener instaladas todas las dependencias:



**Resultado:** Creamos mapas que muestran estos clusters para cada variable, identificando claramente las zonas homogéneas.```bash

# Con pip

### 3. **Identificación de Submercados (K-Means Clustering)**pip install geopandas pandas numpy pysal matplotlib seaborn scikit-learn scipy



**¿Qué es?** Dividimos todo Santiago en "submercados" o grupos de celdas que comparten características similares.# O con conda (RECOMENDADO)

conda install -c conda-forge geopandas pysal matplotlib seaborn scikit-learn scipy

**¿Cómo funciona?** Usamos un algoritmo llamado K-Means que agrupa las celdas según todas sus características (densidades, distancias, etc.). El resultado es una segmentación de la ciudad en 5-8 tipos diferentes de barrios.```



**Ejemplo de submercados identificados:**### Librerías específicas:

- **geopandas**: Manipulación de datos geoespaciales

1. **Submercado 1 - "Centro Urbano Premium":**- **pysal (libpysal + esda)**: Análisis espacial y autocorrelación

   - Alta densidad de todos los servicios- **scikit-learn**: Algoritmos de clustering

   - Muy cerca del metro (< 300m)- **matplotlib + seaborn**: Visualizaciones

   - Muchas áreas verdes- **scipy**: Estadísticas y clustering jerárquico

   - *Zonas típicas:* Providencia, partes de Las Condes

### Datos de entrada requeridos

2. **Submercado 2 - "Residencial con Servicios":**

   - Buena densidad de colegios y supermercadosLos scripts esperan encontrar el siguiente archivo:

   - Acceso moderado al metro (500-800m)```

   - *Zonas típicas:* Ñuñoa residencial../semana2_caracteristicas_espaciales/features/grilla_con_densidades.geojson

```

3. **Submercado 3 - "Periferia con Carencias":**

   - Baja densidad de serviciosSi completaste la Semana 2 correctamente, este archivo ya debe existir.

   - Lejos del metro (> 1,500m)

   - Pocas áreas verdes---

   - *Zonas típicas:* Bordes de las comunas

## 🚀 Ejecución

---

### Opción 1: Ejecutar todo automáticamente (RECOMENDADO)

## Resultados Obtenidos

```bash

### Archivos Generadoscd semana3_analisis_espacial

python3 ejecutar_semana3.py

1. **`reportes/`**: Contiene 3 reportes principales```

   - `autocorrelacion_global_reporte.md`: Resultados del Índice de Moran

   - `autocorrelacion_local_reporte.md`: Descripción de clusters LISAEste script:

   - `submercados_reporte.md`: Perfiles de cada submercado- ✅ Verifica que todas las dependencias estén instaladas

- ✅ Ejecuta los 3 análisis secuencialmente

2. **`mapas/`**: Visualizaciones de los análisis- ✅ Genera todos los reportes y visualizaciones

   - Mapas de calor mostrando autocorrelación- ✅ Muestra un resumen final

   - Mapas de clusters LISA coloreados

   - Scatterplots de Moran**Tiempo estimado:** 5-15 minutos (depende de tu hardware)



3. **`submercados/`**: Datos y mapas de segmentación### Opción 2: Ejecutar scripts individuales

   - `mapa_submercados.png`: Visualización de los submercados

   - `grilla_con_submercados.geojson`: Datos con etiquetas de submercadoSi prefieres ejecutar cada análisis por separado:



### Hallazgos Clave```bash

cd semana3_analisis_espacial/scripts

- ✅ **Autocorrelación positiva** en la mayoría de variables (p < 0.001)

- ✅ **5-7 submercados** claramente diferenciados identificados# 1. Autocorrelación global

- ✅ **Desigualdad espacial** evidente: servicios concentrados en el centropython3 calcular_autocorrelacion_global.py

- ✅ **Clusters significativos** de alta y baja habitabilidad

# 2. Autocorrelación local (requiere resultados del paso 1)

---python3 calcular_autocorrelacion_local.py



## Scripts Utilizados# 3. Identificación de submercados

python3 identificar_submercados.py

Los scripts en `scripts/` realizan los análisis:```



1. **`calcular_autocorrelacion_global.py`**: Calcula el Índice de Moran para todas las variables---



2. **`calcular_autocorrelacion_local.py`**: Identifica clusters LISA y genera mapas## 📊 Descripción de los Análisis



3. **`identificar_submercados.py`**: Aplica K-Means clustering para segmentar el territorio### 1. Autocorrelación Espacial Global (Índice de Moran)



4. **`ejecutar_semana3.py`**: Script maestro que ejecuta todo en secuencia**¿Qué hace?**

- Calcula el Índice de Moran (I) para todas las características espaciales

---- Determina si existe clustering espacial significativo

- Identifica qué variables muestran autocorrelación espacial

## ¿Por Qué es Importante?

**¿Qué significa?**

Esta etapa es crucial porque:- **I > 0 (p < 0.05):** Clustering positivo → valores similares se agrupan espacialmente

- **I ≈ 0:** Distribución aleatoria → no hay patrón espacial

1. **Valida hipótesis:** Confirma que la ubicación geográfica SÍ importa (hay patrones claros)- **I < 0 (p < 0.05):** Dispersión → valores diferentes se agrupan



2. **Identifica desigualdades:** Muestra objetivamente dónde están las carencias de servicios**¿Por qué es importante?**

Si hay autocorrelación espacial significativa:

3. **Segmenta el mercado:** Los submercados nos permiten entender que no todo Santiago es igual- ✅ Confirma que la ubicación SÍ importa

- ✅ Justifica el uso de modelos espaciales (GWR/MGWR)

4. **Prepara para predicción:** En la Semana 4 usaremos estos patrones para modelos predictivos- ✅ Los modelos OLS simples tendrán sesgo



---**Salidas:**

- CSV con índices de Moran para todas las variables

## Ejemplo Práctico: Acceso al Metro- JSON con reporte completo

- Markdown con interpretación

**Índice de Moran: 0.78** (p < 0.001) → Autocorrelación MUY fuerte- Gráficos: distribución de I, I vs p-valor, top variables

- Moran scatterplots para top 5 características

**Interpretación:** Las zonas con buena acceso al metro están fuertemente agrupadas geográficamente. No es aleatorio: hay un "corredor de metro" claramente definido.

---

**Clusters LISA identificados:**

- 🔴 **HH (High-High):** Providencia, Las Condes centro → Muy cerca del metro, vecinos también### 2. Autocorrelación Espacial Local (LISA)

- 🔵 **LL (Low-Low):** Bordes de Ñuñoa → Lejos del metro, vecinos también

- 🟠 **HL (Outliers):** Algunas estaciones terminales rodeadas de zonas sin metro**¿Qué hace?**

- Calcula Indicadores Locales de Asociación Espacial (LISA)

---- Identifica clusters espaciales específicos:

  - **HH (High-High):** Hot spots - zonas de alta habitabilidad

## Estadísticas Clave  - **LL (Low-Low):** Cold spots - zonas de baja habitabilidad

  - **HL (High-Low):** Outliers altos - puntos buenos en zonas malas

- ✅ **40+ variables** analizadas con Índice de Moran  - **LH (Low-High):** Outliers bajos - puntos malos en zonas buenas

- ✅ **~12,000 celdas** clasificadas en clusters LISA

- ✅ **5-7 submercados** identificados**¿Por qué es importante?**

- ✅ **Significancia estadística** (p < 0.001) en mayoría de variables- ✅ Identifica dónde están los mejores/peores lugares

- ✅ Detecta anomalías espaciales que requieren investigación

---- ✅ Permite segmentación geográfica del territorio



## Próximos Pasos**Salidas:**

- Mapas de clusters LISA por variable

Con estos patrones espaciales identificados, en la **Semana 4** recolectaremos datos reales de propiedades y construiremos modelos de Machine Learning que predigan precios considerando tanto características de la propiedad como del entorno geográfico.- Mapas de significancia estadística

- GeoJSON con clasificación de cada punto

---- Reporte con estadísticas de clusters



**Nota técnica:** Los análisis usan matrices de vecindad espacial (Queen contiguity) y test de permutaciones (999 iteraciones) para validación estadística robusta.---


### 3. Identificación de Submercados

**¿Qué hace?**
- Aplica K-Means clustering a todas las características
- Determina k óptimo mediante:
  - Método del codo (elbow)
  - Silhouette score
  - Davies-Bouldin index
- Caracteriza cada submercado identificando características distintivas

**¿Por qué es importante?**
- ✅ Segmenta el territorio en zonas homogéneas
- ✅ Cada submercado tiene un perfil único
- ✅ Permite personalizar recomendaciones por zona
- ✅ Fundamenta modelos locales (GWR/MGWR)

**Salidas:**
- Mapa de submercados identificados
- Gráficos de determinación de k óptimo
- JSON con perfiles detallados de cada submercado
- GeoJSON con clasificación de puntos
- Reporte con características distintivas

---

## 📈 Interpretación de Resultados

### ¿Cómo saber si el análisis fue exitoso?

1. **Autocorrelación Global:**
   - ✅ Debería haber **al menos 50% de variables con I significativo (p < 0.05)**
   - ✅ Las características importantes (accesibilidad, densidades) deben mostrar clustering
   - ⚠️ Si ninguna variable es significativa, podría indicar distribución aleatoria (poco probable)

2. **LISA:**
   - ✅ Debería identificar **Hot Spots (HH) y Cold Spots (LL) claros**
   - ✅ Los clusters deberían tener sentido geográfico (zonas contiguas)
   - ⚠️ Demasiados outliers pueden indicar ruido en datos

3. **Submercados:**
   - ✅ k óptimo típicamente entre **4-8 clusters**
   - ✅ Silhouette score > 0.3 (aceptable), > 0.5 (bueno)
   - ✅ Cada cluster debe tener al menos 5-10% de puntos
   - ✅ Características distintivas deben ser interpretables

---

## 🎯 Respuesta a la Pregunta Central

### ¿El entorno afecta el tipo de vivienda que el usuario desea?

**Después de completar estos análisis, podrás responder:**

✅ **SÍ**, si encuentras:
- Autocorrelación espacial significativa (I > 0, p < 0.05)
- Clusters espaciales claros (Hot Spots y Cold Spots)
- Submercados diferenciados con perfiles únicos
- Heterogeneidad espacial en características

❌ **NO** (poco probable), si encuentras:
- Ninguna variable muestra autocorrelación significativa
- No se forman clusters espaciales coherentes
- Todos los submercados son homogéneos

**Evidencia esperada:**
Basándonos en el TFM analizado, esperamos:
- 70-90% de características con autocorrelación significativa
- 4-8 submercados claramente diferenciados
- Hot Spots en zonas como Providencia, Las Condes
- Cold Spots en periferia o zonas industriales

---

## 🔍 Análisis Detallado de Salidas

### Archivos de Reporte

#### `autocorrelacion_global_reporte.md`
Contiene:
- Resumen ejecutivo de autocorrelación
- Top 15 variables con mayor clustering
- Tabla completa de todas las variables
- Interpretación y conclusiones

**Busca:**
- Cuántas variables son significativas (p < 0.05)
- Qué índices tienen mayor autocorrelación
- Si índices clave (calidad de vida, accesibilidad) son significativos

#### `autocorrelacion_local_reporte.md`
Contiene:
- Distribución de tipos de clusters (HH, LL, HL, LH)
- Estadísticas por variable analizada
- Interpretación de Hot Spots y Cold Spots

**Busca:**
- Porcentaje de puntos con clusters significativos
- Distribución geográfica de Hot Spots vs Cold Spots
- Consistencia entre variables (zonas siempre buenas/malas)

#### `submercados_reporte.md`
Contiene:
- Métricas de calidad del clustering
- Perfil de cada submercado
- Características distintivas (z-scores)
- Interpretación para recomendaciones

**Busca:**
- Qué caracteriza a cada submercado
- Cuáles son las diferencias clave entre submercados
- Dónde se ubica cada submercado

---

## 🗺️ Análisis de Mapas

### Mapas LISA

Los mapas de clusters LISA muestran:
- 🔴 **Rojo:** Hot Spots (HH) - zonas de alta habitabilidad
- 🔵 **Azul:** Cold Spots (LL) - zonas de baja habitabilidad
- 🟠 **Naranja:** Outliers altos (HL) - anomalías positivas
- 🔷 **Celeste:** Outliers bajos (LH) - anomalías negativas
- ⚪ **Gris:** No significativo

**Interpretación:**
- Las zonas rojas son **premium** → candidatas para usuarios exigentes
- Las zonas azules requieren **mayor desarrollo** → oportunidades
- Los outliers requieren **investigación caso por caso**

### Mapa de Submercados

Muestra la segmentación territorial con diferentes colores por cluster.

**Interpretación:**
- Cada color representa un submercado con perfil único
- Los límites deberían ser relativamente continuos (no muy fragmentados)
- Compara con comunas reales para validar coherencia

---

## 🚨 Problemas Comunes y Soluciones

### Problema 1: "Import Error: No module named 'esda'"
**Solución:**
```bash
pip install pysal
# o
conda install -c conda-forge pysal
```

### Problema 2: "FileNotFoundError: grilla_con_densidades.geojson"
**Solución:**
- Asegúrate de haber completado Semana 2
- Verifica que el archivo exista en `../semana2_caracteristicas_espaciales/features/`
- Ejecuta los scripts desde el directorio correcto

### Problema 3: "MemoryError" o proceso muy lento
**Solución:**
- Reduce el número de permutaciones en Moran (999 → 499)
- Procesa menos variables en LISA (solo top 5-10)
- Cierra otros programas para liberar RAM

### Problema 4: "Ninguna variable muestra autocorrelación significativa"
**Posibles causas:**
- Grilla muy dispersa (aumenta espaciamiento → menos vecinos)
- Matriz de pesos incorrecta (prueba KNN en lugar de Queen)
- Datos con mucho ruido (revisar calidad Semana 1)

### Problema 5: "Silhouette score muy bajo (<0.2)"
**Solución:**
- Prueba diferentes valores de k
- Normalización incorrecta → verificar StandardScaler
- Demasiadas variables correlacionadas → aplicar PCA primero

---

## 📚 Conceptos Clave

### Índice de Moran (I)

Mide autocorrelación espacial global:

$$I = \frac{n}{\sum_{ij} w_{ij}} \frac{\sum_{ij} w_{ij}(x_i - \bar{x})(x_j - \bar{x})}{\sum_i (x_i - \bar{x})^2}$$

Donde:
- $n$: número de observaciones
- $w_{ij}$: peso espacial entre i y j
- $x_i$: valor en ubicación i
- $\bar{x}$: media

### LISA (Local Moran)

Descompone I global en contribuciones locales:

$$I_i = \frac{x_i - \bar{x}}{s^2} \sum_j w_{ij}(x_j - \bar{x})$$

Clasifica cada punto en cuadrantes del Moran Scatterplot.

### K-Means Clustering

Minimiza varianza intra-cluster:

$$\text{argmin}_C \sum_{i=1}^k \sum_{x \in C_i} ||x - \mu_i||^2$$

Donde $\mu_i$ es el centroide del cluster $i$.

---

## 🎓 Referencias Bibliográficas

1. **Anselin, L. (1995)**. "Local Indicators of Spatial Association—LISA". *Geographical Analysis*, 27(2), 93-115.

2. **Tobler, W. (1970)**. "A Computer Movie Simulating Urban Growth in the Detroit Region". *Economic Geography*, 46, 234-240.

3. **Fotheringham, A. S., Brunsdon, C., & Charlton, M. (2002)**. *Geographically Weighted Regression: The Analysis of Spatially Varying Relationships*. Wiley.

4. **Getis, A., & Ord, J. K. (1992)**. "The Analysis of Spatial Association by Use of Distance Statistics". *Geographical Analysis*, 24(3), 189-206.

5. **Rey, S. J., & Anselin, L. (2010)**. "PySAL: A Python Library of Spatial Analytical Methods". In *Handbook of Applied Spatial Analysis* (pp. 175-193). Springer.

---

## 🚀 Próximos Pasos

Una vez completada la Semana 3:

### Fase 4: Adquisición de Datos de Mercado (1-2 semanas)
- Web scraping de portales inmobiliarios
- Geocodificación de propiedades
- Enriquecimiento con características espaciales

### Fase 5: Modelado Hedónico y Predictivo (2-3 semanas)
- Modelo OLS base
- Modelo MGWR espacial
- Random Forest con features espaciales

### Fase 6: Sistema de Recomendación (3-4 semanas)
- Motor de puntuación personalizado
- API REST (FastAPI)
- Dashboard interactivo

---

## 💡 Consejos y Mejores Prácticas

1. **Documenta todo:** Toma capturas de pantalla de los mapas más importantes

2. **Interpreta, no solo ejecutes:** Lee los reportes Markdown generados y extrae conclusiones

3. **Valida coherencia:** Los patrones espaciales deben tener sentido geográfico

4. **Compara con la realidad:** ¿Los Hot Spots corresponden a zonas que conoces como buenas?

5. **Itera si es necesario:** Si los resultados no son claros, ajusta parámetros

6. **Respalda tus datos:** Los archivos GeoJSON generados son la base para fases siguientes

---

## ✅ Checklist de Completitud

- [ ] Todas las dependencias instaladas
- [ ] Script `ejecutar_semana3.py` ejecutado sin errores
- [ ] Reportes Markdown generados y revisados
- [ ] Mapas visualizados e interpretados
- [ ] Al menos 50% de variables con autocorrelación significativa
- [ ] Clusters LISA identificados (HH, LL claros)
- [ ] 4-8 submercados identificados con perfiles únicos
- [ ] GeoJSON `grilla_con_submercados.geojson` generado
- [ ] Respuesta documentada a pregunta central

---

## 📞 Soporte

Si encuentras problemas:

1. Revisa la sección "Problemas Comunes y Soluciones"
2. Verifica que completaste correctamente Semana 1 y 2
3. Revisa los mensajes de error en detalle
4. Consulta documentación de PySAL: https://pysal.org/

---

**¡Éxito con el análisis espacial! 🚀**
