# 📊 Guía de Visualizaciones del Modelo de Satisfacción Residencial

## Introducción: ¿De qué trata este proyecto?

Imagina que quieres comprar una casa o departamento, pero no sabes cuál elegir entre miles de opciones. Este proyecto creó una **herramienta inteligente** que analiza propiedades inmobiliarias y te dice qué tan "satisfecho" podrías estar viviendo en cada una de ellas, considerando no solo las características de la propiedad (tamaño, precio, dormitorios), sino también **todo lo que hay alrededor**: parques, colegios, hospitales, transporte público, comercio, y más.

En términos simples, el sistema funciona así:
1. **Recopila datos** de miles de propiedades en venta (7,702 en total)
2. **Analiza la ubicación** de cada propiedad y mide qué tan lejos está de servicios importantes
3. **Entrena un modelo** que aprende a predecir la satisfacción
4. **Genera recomendaciones personalizadas** según el tipo de usuario (familias, profesionales jóvenes, inversionistas, etc.)

---

## El Script: `generar_visualizaciones.py`

Este archivo es el encargado de crear todos los gráficos y mapas que nos permiten **entender visualmente** los resultados del modelo. A continuación, explicamos cada parte del código de forma sencilla.

---

## Paso 1: Preparación del Entorno

```python
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import folium
```

**¿Qué hace?** El script comienza importando las "herramientas" necesarias para trabajar con datos geográficos y crear gráficos. Es como un chef que primero saca todos los ingredientes y utensilios antes de cocinar.

**Librerías principales:**
- `pandas`: Para manejar tablas de datos (como Excel, pero más potente)
- `geopandas`: Para trabajar con datos que tienen ubicación geográfica (latitud/longitud)
- `matplotlib` y `seaborn`: Para crear gráficos estáticos
- `folium`: Para crear mapas interactivos que puedes explorar con el mouse

---

## Paso 2: Carga de Datos

```python
df = pd.read_csv(csv_path)
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.longitude, df.latitude))
```

**¿Qué hace?** Carga el archivo con todas las propiedades (7,702 en total) y las convierte en un formato geográfico. Cada propiedad ahora es un "punto" en el mapa con sus coordenadas.

**Datos cargados:**
- 4 comunas de Santiago: La Reina, Ñuñoa, Santiago Centro, Estación Central
- Información de cada propiedad: precio, tamaño, dormitorios, baños, tipo (casa o departamento)
- Índice de satisfacción predicho por el modelo (del 1 al 10)

---

## Los Mapas: Entendiendo la Ubicación Espacial

### 🗺️ Mapa 1: Ubicación del Área de Estudio

**¿Qué muestra?** Un mapa general con la distribución de todas las propiedades analizadas. Los puntos azules son departamentos y los rojos son casas.

**¿Cómo interpretarlo?**
- Las zonas con muchos puntos agrupados indican mayor oferta inmobiliaria
- Se puede ver que Santiago Centro y Estación Central tienen más departamentos
- La Reina tiene una mezcla más equilibrada de casas y departamentos

**¿Por qué es útil?** Permite ver de un vistazo dónde están concentradas las propiedades y entender la cobertura geográfica del estudio.

---

### 🗺️ Mapa 2: Distribución de Precio por m²

**¿Qué muestra?** El mismo mapa, pero ahora cada punto está coloreado según su precio por metro cuadrado (en UF). Los colores van del verde (más barato) al rojo (más caro).

**¿Cómo interpretarlo?**
- **Colores verdes/amarillos**: Propiedades con buen precio por m² (más económicas)
- **Colores naranjas/rojos**: Propiedades más caras por m²
- Se puede observar que ciertas zonas tienen precios consistentemente más altos

**Estadísticas mostradas:**
- Precio mínimo por m²: El valor más bajo encontrado
- Precio mediano: El valor del "medio" (50% son más caras, 50% más baratas)
- Precio máximo: El valor más alto encontrado

**¿Por qué es útil?** Ayuda a identificar zonas más económicas vs. zonas premium sin tener que revisar propiedad por propiedad.

---

### 🗺️ Mapa 3: Resultado del Análisis (Satisfacción Predicha)

**¿Qué muestra?** El mapa más importante: muestra el **índice de satisfacción** que el modelo predijo para cada propiedad. La escala va del 1 (baja satisfacción) al 10 (alta satisfacción).

**¿Cómo interpretarlo?**
- **Puntos verdes (8-10)**: Propiedades con alta satisfacción predicha - excelente combinación de precio, características y ubicación
- **Puntos amarillos (5-7)**: Satisfacción media - propiedades "normales"
- **Puntos rojos (1-4)**: Baja satisfacción - probablemente muy caras para lo que ofrecen o mal ubicadas

**¿Qué significa el R² = 0.8635?** Este número indica que el modelo explica el **86.35% de la variabilidad** en la satisfacción. Es un resultado excelente que significa que las predicciones son muy confiables.

---

## Los Gráficos Estadísticos: Entendiendo los Números

### 📊 Gráfico 1: Histogramas de Variables Clave

**¿Qué muestra?** Seis histogramas que muestran cómo se distribuyen las principales características de las propiedades.

**¿Qué es un histograma?** Es un gráfico de barras que muestra qué tan frecuentes son ciertos valores. Las barras más altas indican valores más comunes.

**Variables mostradas:**

1. **Precio (UF)**: Distribución de precios. La mayoría de propiedades están en un rango medio, con pocas muy baratas o muy caras.

2. **Superficie (m²)**: Tamaño de las propiedades. Los departamentos suelen ser más pequeños que las casas.

3. **Precio/m² (UF)**: Cuánto cuesta cada metro cuadrado. Útil para comparar propiedades de diferentes tamaños.

4. **Dormitorios**: Número de habitaciones. El pico más alto muestra el número de dormitorios más común.

5. **Baños**: Cantidad de baños por propiedad.

6. **Satisfacción**: La variable que predice el modelo. Muestra que la mayoría de propiedades tienen satisfacción media-alta.

**Las líneas en cada histograma:**
- **Línea roja punteada**: La **media** (promedio de todos los valores)
- **Línea naranja punteada**: La **mediana** (el valor del medio)

**¿Por qué es útil?** Permite entender qué es "normal" en el mercado inmobiliario de estas comunas.

---

### 📊 Gráfico 2: Análisis por Comuna

**¿Qué muestra?** Cuatro gráficos que comparan las comunas entre sí.

**Subgráficos:**

1. **Boxplot de Precio/m² por Comuna**: Muestra el rango de precios en cada comuna.
   - La "caja" contiene el 50% central de los datos
   - La línea dentro de la caja es la mediana
   - Los puntos fuera de los "bigotes" son valores atípicos (outliers)

2. **Boxplot de Satisfacción por Comuna**: Misma idea, pero para la satisfacción predicha.

3. **Cantidad de Propiedades por Comuna y Tipo**: Gráfico de barras apiladas que muestra cuántas casas y departamentos hay en cada comuna.

4. **Superficie Promedio por Comuna y Tipo**: Compara el tamaño típico de las propiedades en cada zona.

**¿Cómo interpretarlo?**
- Si una comuna tiene boxplots de precio más bajos pero satisfacción similar, ofrece mejor valor
- Comunas con más departamentos pequeños tendrán superficies promedio más bajas
- Las diferencias entre comunas reflejan sus características socioeconómicas

---

### 📊 Gráfico 3: Matriz de Correlaciones

**¿Qué muestra?** Un mapa de calor que indica qué variables están relacionadas entre sí.

**¿Qué es una correlación?**
- **Correlación positiva (+1)**: Cuando una variable sube, la otra también sube (color rojo/azul intenso)
- **Correlación negativa (-1)**: Cuando una variable sube, la otra baja (color opuesto)
- **Sin correlación (0)**: No hay relación entre las variables (color blanco/neutro)

**Ejemplos de interpretación:**
- `precio_m2_uf` vs `satisfaccion`: Si es negativo, propiedades más caras por m² tienden a tener menor satisfacción (menos valor por tu dinero)
- `superficie_util` vs `dormitorios`: Correlación positiva esperada (casas más grandes tienen más dormitorios)
- `dist_areas_verdes_m` vs `satisfaccion`: Si es negativo, estar cerca de parques aumenta la satisfacción

**¿Por qué es útil?** Ayuda a entender qué factores influyen más en la satisfacción y cómo se relacionan entre sí.

---

### 📊 Gráfico 4: Diagramas de Dispersión

**¿Qué muestra?** Cuatro gráficos que exploran relaciones entre pares de variables.

**¿Qué es un diagrama de dispersión?** Cada punto representa una propiedad, y su posición en el gráfico depende de dos valores. Patrones visuales revelan relaciones.

**Subgráficos:**

1. **Precio vs Superficie (color = satisfacción)**
   - Cada punto es una propiedad
   - Posición horizontal: superficie en m²
   - Posición vertical: precio en UF
   - Color: satisfacción (verde = alta, rojo = baja)
   - **Interpretación**: Las propiedades verdes (alta satisfacción) suelen estar en la parte inferior izquierda (buen precio para su tamaño) o tener colores mixtos indicando otros factores importantes

2. **Precio/m² vs Satisfacción**
   - Muestra la relación directa entre cuánto pagas por m² y qué tan satisfecho estarías
   - La **línea roja de tendencia** indica la dirección general de la relación
   - Si la línea baja de izquierda a derecha: pagar más por m² reduce la satisfacción

3. **Dormitorios vs Satisfacción (por tipo)**
   - Puntos azules: departamentos
   - Puntos rojos: casas
   - Muestra si tener más dormitorios aumenta la satisfacción diferenciadamente

4. **Predicción vs Real del Modelo**
   - El gráfico más importante para validar el modelo
   - Eje X: satisfacción real (calculada)
   - Eje Y: satisfacción predicha por el modelo
   - **La línea diagonal roja**: Si todos los puntos estuvieran sobre esta línea, el modelo sería perfecto
   - **R² = 0.8635**: Los puntos están muy cerca de la línea, indicando predicciones muy precisas

---

### 📊 Gráfico 5: Importancia de Variables y Comparación de Modelos

**¿Qué muestra?** Dos visualizaciones clave sobre el funcionamiento del modelo.

**Subgráfico izquierdo: Top 15 Variables Más Importantes**

Este gráfico de barras horizontales muestra qué características influyen más en la predicción de satisfacción.

**Las variables más importantes (según el modelo):**

1. **precio_m2_uf** (1,351 puntos): El precio por metro cuadrado es el factor MÁS importante. Tiene sentido: todos queremos pagar menos por más.

2. **superficie_util** (1,017 puntos): El tamaño de la propiedad es el segundo factor. Espacios más grandes = mayor comodidad.

3. **precio_uf** (616 puntos): El precio total también importa, aunque menos que el precio por m².

4. **dormitorios** (445 puntos): La cantidad de habitaciones influye significativamente.

5. **dist_areas_verdes_m** (367 puntos): Primera variable **espacial** - la distancia a parques y áreas verdes importa mucho.

6. **dist_seguridad_min_m** (293 puntos): Qué tan cerca está la comisaría o carabineros más cercano.

7. **longitude/latitude** (284/283 puntos): La ubicación geográfica exacta captura patrones que las otras variables no.

8. **dist_transporte_min_m** (279 puntos): Cercanía al metro o buses.

9. **dist_ocio_m** (233 puntos): Acceso a lugares de entretenimiento y ocio.

**Colores del gráfico:**
- **Barras azules**: Variables espaciales (distancias, densidades) - factores del entorno
- **Barras verdes**: Variables de la propiedad misma (precio, tamaño, dormitorios)

**Subgráfico derecho: Comparación de Modelos**

Compara tres modelos de inteligencia artificial que se probaron:

| Modelo | R² (precisión) | RMSE (error) |
|--------|----------------|--------------|
| **LightGBM (actual)** | 0.8635 | 0.3357 |
| Random Forest | 0.8453 | 0.3573 |
| GWRF (anterior) | 0.7922 | 0.3953 |

**¿Qué significan estas métricas?**

- **R² (coeficiente de determinación)**: Qué porcentaje de la variabilidad explica el modelo. **Más alto = mejor.** El 0.8635 significa que el modelo explica el 86.35% de por qué unas propiedades satisfacen más que otras.

- **RMSE (error cuadrático medio)**: Qué tan equivocadas están las predicciones en promedio. **Más bajo = mejor.** Un RMSE de 0.3357 significa que las predicciones se desvían en promedio 0.34 puntos (en una escala de 1-10).

**Conclusión**: Se eligió **LightGBM** porque tiene el mejor R² y el menor error.

---

## 🌐 Visualización Interactiva: Mapa Folium

**Archivo generado:** `mapa_interactivo.html`

**¿Qué es?** Un mapa web que puedes abrir en el navegador y explorar interactivamente.

**Características:**
- **Zoom con la rueda del mouse**: Acercarse y alejarse del mapa
- **Arrastrar para moverse**: Navegar por diferentes zonas
- **Clic en marcadores**: Ver información detallada de cada propiedad
- **Clusters**: Los marcadores se agrupan automáticamente cuando hay muchos juntos
- **Heatmap (mapa de calor)**: Visualiza la densidad de propiedades con alta satisfacción

**Popup de cada propiedad incluye:**
- Tipo (departamento o casa)
- Comuna
- Precio en UF
- Superficie en m²
- Precio por m²
- Número de dormitorios
- **Índice de satisfacción** con código de color

**¿Cómo usarlo?**
1. Abre el archivo `mapa_interactivo.html` en cualquier navegador
2. Navega por el mapa buscando zonas de interés
3. Haz clic en los marcadores para ver detalles
4. Usa el control de capas para activar/desactivar el heatmap

---

## Los Perfiles de Usuario

El modelo no solo predice satisfacción general, sino que puede personalizarla según diferentes tipos de compradores:

### 👨‍👩‍👧‍👦 Familia con Niños
**Prioridades:**
- Espacio amplio (más dormitorios) - peso 2.5x
- Cercanía a colegios - peso 2.5x
- Áreas verdes cercanas - peso 2.0x
- Seguridad del barrio - peso 2.0x

### 👔 Profesional Joven
**Prioridades:**
- Transporte público cercano (metro, buses) - peso 2.5x
- Comercio y vida nocturna - peso 2.0x
- Buen precio - peso 1.8x
- Espacio no es tan importante (depto pequeño está OK) - peso 0.8x

### 💰 Inversionista
**Prioridades:**
- Valor del dinero (ROI) - peso 3.0x
- Transporte (facilita arriendo) - peso 2.0x
- Seguridad (arrendatarios lo valoran) - peso 1.5x

### 👴 Adulto Mayor
**Prioridades:**
- Cercanía a hospitales y farmacias - peso 3.0x
- Seguridad - peso 2.0x
- Áreas verdes para caminar - peso 1.5x

### ⚖️ Balanceado
Considera todos los factores con igual importancia.

---

## Resumen: ¿Qué Logramos?

Este proyecto desarrolló un sistema completo que:

1. **Recolectó datos** de 7,702 propiedades en venta de 4 comunas de Santiago

2. **Integró factores espaciales** calculando la distancia de cada propiedad a:
   - Áreas verdes y parques
   - Establecimientos educacionales
   - Centros de salud
   - Estaciones de metro y paraderos
   - Comercio y entretenimiento
   - Comisarías y bomberos

3. **Creó un índice de satisfacción** que combina:
   - Valor relativo (¿es buen precio para lo que ofrece?)
   - Características físicas (tamaño, dormitorios, baños)
   - Accesibilidad a servicios (todo lo que hay cerca)

4. **Entrenó un modelo de inteligencia artificial** (LightGBM) que puede:
   - Predecir la satisfacción con 86.35% de precisión
   - Identificar qué factores importan más
   - Personalizar recomendaciones según el tipo de usuario

5. **Generó visualizaciones** que permiten:
   - Explorar los datos de forma intuitiva
   - Entender patrones geográficos de precio y satisfacción
   - Validar que el modelo funciona correctamente

---

## Archivos Generados

| Archivo | Descripción |
|---------|-------------|
| `mapa_01_ubicacion_area_estudio.png` | Mapa con todas las propiedades |
| `mapa_02_precio_m2.png` | Mapa de precios por metro cuadrado |
| `mapa_03_satisfaccion_predicha.png` | Mapa de satisfacción predicha |
| `grafico_01_histogramas.png` | Distribución de variables principales |
| `grafico_02_analisis_comunas.png` | Comparación entre comunas |
| `grafico_03_correlaciones.png` | Matriz de correlaciones |
| `grafico_04_dispersion.png` | Relaciones entre variables |
| `grafico_05_importancia_metricas.png` | Variables importantes y métricas |
| `mapa_interactivo.html` | Mapa web explorable |
| `INDICE_VISUALIZACIONES.json` | Índice de todos los archivos |

---

## 📍 Estadística Descriptiva Espacial: Comercios y Seguridad

Esta sección presenta el análisis exploratorio de datos espaciales (EDA espacial) que responde a dos preguntas clave de investigación sobre la distribución de infraestructura urbana en las 4 comunas de estudio.

---

### Pregunta 1: ¿Cuál es la distribución de comercios por zonas de una ciudad?

El análisis reveló patrones significativos en la distribución espacial de los 532 comercios identificados en el área de estudio.

#### Resultados por Comuna

| Comuna | Total Comercios | Área (km²) | Densidad (comercios/km²) | % del Total |
|--------|-----------------|------------|--------------------------|-------------|
| **Santiago** | 383 | 42.03 | **9.11** | 72.0% |
| La Reina | 74 | 44.70 | 1.66 | 13.9% |
| Ñuñoa | 52 | 31.77 | 1.64 | 9.8% |
| Estación Central | 23 | 31.39 | 0.73 | 4.3% |

#### Patrones Identificados

1. **Concentración centro-periferia**: Santiago Centro concentra el **72%** de todos los comercios del área de estudio, con una densidad de 9.11 comercios/km² - más de 5 veces el promedio de las otras comunas.

2. **Tipos de comercio predominantes**:
   - Autopartes (14.7%)
   - Tiendas de conveniencia (11.9%)
   - Botillerías/alcohol (6.7%)
   - Peluquerías (4.7%)
   - Panaderías (4.2%)

3. **Heterogeneidad espacial**: La desviación estándar de 168 comercios indica alta variabilidad entre comunas, sugiriendo especialización funcional del territorio.

#### Visualización: Mapa de Distribución de Comercios

📍 **Archivo**: `autocorrelacion_espacial/semana2_caracteristicas_espaciales/graficos/mapa_distribucion_comercios.png`

**Interpretación del mapa:**
- **Panel izquierdo**: Puntos rojos indican la ubicación exacta de cada comercio. Se observa clara aglomeración en Santiago Centro.
- **Panel derecho**: Mapa coroplético que muestra la densidad por comuna. Los tonos más oscuros indican mayor concentración comercial.

---

### Pregunta 2: ¿En qué lugares existe una alta concentración de servicios de seguridad?

Se analizaron **57 servicios de seguridad**: Carabineros/Policía (28 unidades), PDI (25 unidades) y Bomberos (4 compañías).

> **Nota**: Los datos incluyen registros de `cuarteles_filtrados.geojson`, `cuerpos_de_bomberos_filtrados.geojson` y datos adicionales de `servicios_filtrados.geojson` (amenity = police/fire_station).

#### Resultados por Comuna

| Comuna | Carabineros | Bomberos | PDI | Total | Densidad (serv/km²) |
|--------|-------------|----------|-----|-------|---------------------|
| **Santiago** | 17 | 3 | 11 | 31 | **0.738** |
| Estación Central | 7 | 0 | 4 | 11 | 0.350 |
| Ñuñoa | 2 | 1 | 6 | 9 | 0.283 |
| La Reina | 2 | 0 | 4 | 6 | 0.134 |

#### Zonas de Alta Concentración

Las comunas con densidad superior al promedio (0.376 servicios/km²) son:
- **Santiago**: 0.738 servicios/km² - Concentra más de la mitad de todos los servicios

#### Patrones Identificados

1. **Distribución estratégica**: Los servicios de seguridad se concentran en zonas de alta actividad comercial y poblacional.

2. **Complementariedad institucional**: Carabineros, Bomberos y PDI muestran patrones de distribución complementarios, maximizando la cobertura territorial.

3. **Relación comercio-seguridad**: Las comunas con mayor densidad comercial (Santiago) también tienen mayor presencia de servicios de seguridad.

#### Visualización: Mapa de Servicios de Seguridad

📍 **Archivo**: `autocorrelacion_espacial/semana2_caracteristicas_espaciales/graficos/mapa_servicios_seguridad.png`

**Interpretación del mapa:**
- **Panel izquierdo**: Ubicación de cada servicio con simbología diferenciada:
  - 🔺 Triángulos azules: Carabineros
  - 🟥 Cuadrados rojos: Bomberos  
  - 🟢 Círculos verdes: PDI
- **Panel derecho**: Mapa coroplético de concentración. Los tonos azules más intensos indican mayor presencia de seguridad.

---

### Mapa Integrado: Comercios y Servicios de Seguridad

📍 **Archivo**: `autocorrelacion_espacial/semana2_caracteristicas_espaciales/graficos/mapa_integrado_comercios_seguridad.png`

Este mapa combina ambas capas de información, permitiendo visualizar la **correlación espacial** entre actividad comercial y presencia de servicios de seguridad. Se observa que:

- Las zonas con alta densidad comercial (puntos naranjas) coinciden con mayor presencia de servicios de seguridad
- Santiago Centro emerge como el principal nodo urbano del área de estudio
- Las comunas periféricas (La Reina, Ñuñoa) muestran distribuciones más dispersas

---

### Gráficos Estadísticos Comparativos

📍 **Archivo**: `autocorrelacion_espacial/semana2_caracteristicas_espaciales/graficos/graficos_estadisticos_espaciales.png`

Los gráficos de barras permiten comparar visualmente:
1. **Total de comercios** por comuna (arriba izquierda)
2. **Densidad de comercios** con línea de promedio (arriba derecha)
3. **Servicios de seguridad apilados** por tipo (abajo izquierda)
4. **Densidad de servicios** de seguridad (abajo derecha)

---

### Implicaciones para la Valoración Inmobiliaria

Los hallazgos de este análisis espacial tienen relevancia directa para el modelo de satisfacción residencial:

1. **Proximidad a comercios**: Las propiedades cercanas a zonas comerciales densas pueden tener mayor valoración por accesibilidad a servicios.

2. **Cobertura de seguridad**: La cercanía a servicios de seguridad puede ser un factor positivo en la percepción de satisfacción, especialmente para familias con niños y adultos mayores.

3. **Diferenciación comunal**: La clara diferencia en densidades sugiere que el factor "comuna" captura características urbanas distintivas que afectan la satisfacción residencial.

---

### Archivos Generados - Estadística Descriptiva Espacial

| Archivo | Descripción |
|---------|-------------|
| `mapa_distribucion_comercios.png/pdf` | Mapa de distribución y densidad de comercios |
| `mapa_servicios_seguridad.png/pdf` | Mapa de ubicación y concentración de seguridad |
| `mapa_integrado_comercios_seguridad.png/pdf` | Vista combinada de comercios y seguridad |
| `graficos_estadisticos_espaciales.png/pdf` | Gráficos de barras comparativos |
| `estadistica_descriptiva_espacial.json` | Reporte completo en formato JSON |
| `ESTADISTICA_DESCRIPTIVA_ESPACIAL.md` | Resumen ejecutivo en Markdown |

---

## Conclusiones Principales

1. **El precio por metro cuadrado es el factor más determinante** en la satisfacción residencial. No es sorprendente: todos queremos maximizar el espacio por nuestro dinero.

2. **La ubicación importa muchísimo**: Las distancias a áreas verdes, seguridad y transporte están entre las 10 variables más importantes.

3. **El modelo es muy preciso** (R² = 0.8635), lo que significa que podemos confiar en sus recomendaciones.

4. **Los perfiles de usuario personalizan las recomendaciones**: Una familia con niños y un profesional joven recibirán sugerencias diferentes aunque miren las mismas propiedades.

5. **La visualización de datos geográficos es poderosa**: Los mapas permiten identificar patrones que los números solos no revelan.

6. **La distribución de comercios y seguridad es heterogénea**: Santiago concentra el 72% de comercios y el 54% de servicios de seguridad (31 de 57), evidenciando una clara centralización urbana.

---

*Documento generado para el Proyecto GeoInformática - Diciembre 2025*
