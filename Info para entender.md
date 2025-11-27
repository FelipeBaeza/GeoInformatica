# 🏠 Proyecto GeoInformática: Predicción de Satisfacción Residencial

> **Guía para entender el proyecto** - Este documento explica de forma simple qué hace el proyecto, cómo funciona y cómo usarlo.

---

## Resumen Ejecutivo - ¿De qué trata este proyecto?

Este proyecto busca responder una pregunta simple pero importante: **¿qué hace que una persona esté satisfecha con su vivienda?** Para responderla, combinamos información de propiedades en arriendo (superficie, dormitorios, precio) con datos del entorno urbano (cercanía a metros, colegios, hospitales, áreas verdes, etc.) en la Región Metropolitana de Santiago.

El resultado principal es un modelo que puede predecir qué tan satisfactoria será una vivienda para un potencial arrendatario, basándose tanto en las características internas del departamento como en la calidad del barrio donde se ubica. El modelo mejorado alcanza un R² de 0.55, lo que significa que explica más de la mitad de la variación en satisfacción residencial, una mejora del 150% respecto al modelo inicial que solo usaba el precio.

## ¿Qué datos usamos?

Trabajamos con dos tipos de información que se complementan entre sí. Por un lado tenemos los **datos internos de las propiedades**: 1,944 departamentos en arriendo con información como superficie útil, número de dormitorios, baños, estacionamientos y precio mensual. Por otro lado tenemos los **datos del entorno urbano**: información geográfica de Santiago que incluye ubicación de estaciones de metro, colegios, hospitales, comisarías, áreas verdes, comercios y más.

Todos los datos geográficos fueron normalizados al sistema de coordenadas UTM Zona 19S (EPSG:32719), que es el estándar para Chile central. Esto permite calcular distancias reales en metros y hacer análisis espaciales precisos. Después de limpiar datos duplicados y valores atípicos extremos, quedamos con 1,672 propiedades de alta calidad para el análisis.

## ¿Cómo organizamos el trabajo? (Semana a semana)

El proyecto se dividió en etapas claras que van construyendo una sobre otra, como bloques de lego.

**Semana 1 - Preparación de datos:** Nos dedicamos a recolectar y limpiar toda la información. Esto incluyó descargar datos de portales inmobiliarios, obtener capas geográficas de servicios urbanos (transporte, educación, salud, etc.) y asegurarnos de que todo estuviera en el mismo sistema de coordenadas. Es como preparar todos los ingredientes antes de cocinar.

**Semana 2 - Características espaciales:** Creamos una grilla de análisis sobre Santiago y calculamos dos tipos de métricas para cada zona: distancias (¿a cuántos metros está el metro más cercano?) y densidades (¿cuántos colegios hay en un radio de 600 metros?). También generamos índices compuestos de habitabilidad que resumen la calidad del entorno urbano en una sola cifra.

**Semana 3 - Modelado y análisis:** Aquí está el corazón del proyecto. Integramos las propiedades con los datos espaciales, entrenamos varios modelos de Machine Learning y evaluamos cuál predice mejor la satisfacción residencial. También analizamos si los errores del modelo tienen patrones espaciales (autocorrelación) y generamos visualizaciones para comunicar los resultados.

**Semana 4 - Ampliación (opcional):** Scripts adicionales para recolectar más datos si se quiere expandir el análisis a otras zonas o períodos.

## ¿Qué modelos usamos y por qué?

Usamos **Random Forest** como algoritmo principal. Imagina que tienes 200 expertos inmobiliarios y cada uno da su opinión sobre el precio de una propiedad basándose en diferentes criterios; Random Forest es exactamente eso: un "bosque" de árboles de decisión que votan para dar una predicción final. Lo elegimos porque funciona bien con datos mixtos, no necesita que las relaciones sean lineales, y nos dice qué variables son más importantes.

Probamos tres enfoques diferentes:

1. **Modelo Global (Baseline):** Un solo modelo para todas las propiedades. Es simple pero no considera que diferentes zonas pueden tener dinámicas distintas. Obtuvo R² = 0.22.

2. **GWRF (Geographically Weighted Random Forest):** La idea es entrenar modelos separados para diferentes zonas de la ciudad, porque lo que importa en Providencia puede ser diferente a lo que importa en Maipú. Probamos dividir por clusters espaciales y por niveles de densidad urbana.

3. **Modelo Mejorado con Satisfacción Compuesta:** En lugar de predecir solo el precio, creamos una variable de "satisfacción" que combina el índice de habitabilidad del barrio con las características de la propiedad. Este modelo alcanzó R² = 0.55, el mejor resultado.

## Explicación simple del flujo de predicción

Imagina que quieres saber si un departamento en particular será satisfactorio para vivir. El proceso funciona así:

**Paso 1 - Recolectar información de la propiedad:** Necesitas saber la superficie, dormitorios, baños, si tiene estacionamiento, y su ubicación exacta (latitud/longitud).

**Paso 2 - Calcular métricas del entorno:** Con la ubicación, el sistema calcula automáticamente qué tan cerca está del metro, de colegios, hospitales, áreas verdes, etc. También calcula cuántos servicios hay en un radio de 300, 600 y 1000 metros.

**Paso 3 - Generar índices de habitabilidad:** Todas esas métricas se combinan en índices fáciles de interpretar: índice de vida urbana, índice de calidad de vida, índice de habitabilidad global. Un departamento en una zona con metro cerca, buenos colegios y parques tendrá un índice alto.

**Paso 4 - Predecir satisfacción:** El modelo toma todas estas variables (internas + espaciales + índices) y predice un puntaje de satisfacción del 1 al 10. Un puntaje de 7+ indica alta satisfacción esperada.

La ventaja de este enfoque es que no solo mira el precio o el tamaño del departamento, sino que considera todo el contexto urbano. Un departamento pequeño pero cerca del metro y de servicios puede ser más satisfactorio que uno grande pero aislado.

## Resultados: ¿Qué tan bien funcionan los modelos?

Aquí están los números clave de cada modelo que probamos:

| Modelo | R² (precisión) | ¿Qué predice? |
|--------|----------------|---------------|
| Modelo Baseline (solo precio) | 0.22 | Precio por m² |
| GWRF por Cluster | 0.08 | Precio por m² |
| GWRF por Densidad | 0.09 | Precio por m² |
| **Modelo Mejorado** | **0.55** | Satisfacción compuesta |

**¿Cómo interpretar el R²?** Es un número entre 0 y 1 que indica qué porcentaje de la variación el modelo puede explicar. Un R² de 0.55 significa que el modelo explica el 55% de por qué unas viviendas son más satisfactorias que otras. El 45% restante depende de factores que no capturamos (gustos personales, estado del edificio, vecinos, etc.).

**¿Por qué el modelo mejorado es mejor?** La clave fue cambiar lo que intentamos predecir. Los modelos iniciales intentaban predecir el precio por m², pero el precio no es lo mismo que satisfacción: un departamento caro no necesariamente hace feliz a quien lo arrienda. El modelo mejorado predice un índice de satisfacción que combina la calidad del entorno (índice de habitabilidad) con las características de la propiedad. Este cambio de enfoque produjo una mejora del 150% en precisión.

## ¿Por qué algunos modelos funcionaron mejor que otros?

**El modelo baseline tuvo bajo rendimiento** porque solo miraba el precio, y el precio en Santiago depende de muchos factores difíciles de capturar (especulación, prestigio del barrio, etc.). Además, no incluía información del entorno urbano.

**Los modelos GWRF no mejoraron como esperábamos** porque nuestro dataset, aunque tiene casi 2,000 propiedades, queda pequeño cuando lo dividimos en zonas. Si un cluster tiene solo 50 propiedades, el modelo local no tiene suficientes ejemplos para aprender bien. Es como querer predecir el clima de todo Chile con datos de solo 3 ciudades.

**El modelo mejorado funcionó mejor** por dos razones: primero, incorpora los índices de habitabilidad calculados en la Semana 2 (que resumen la calidad del entorno); segundo, predice una variable más coherente con lo que realmente queremos medir (satisfacción, no precio). Un dato interesante: el análisis de autocorrelación mostró que los residuos (errores) del modelo NO tienen patrón espacial (Moran I = -0.007, p-value = 0.99), lo que significa que el modelo no está sesgado hacia ninguna zona en particular.

## ¿Qué variables importan más para predecir satisfacción?

El modelo nos dice qué factores tienen mayor peso en la predicción. Los resultados son muy reveladores:

**Top 5 variables más importantes:**
1. **Superficie útil (26%)** - El tamaño sigue siendo rey. Más metros = más satisfacción.
2. **Índice de vida urbana (17%)** - Qué tan bien conectado está el barrio con servicios y transporte.
3. **Índice de habitabilidad global (14%)** - La calidad general del entorno.
4. **Accesibilidad a transporte (10%)** - Cercanía a metro y transporte público.
5. **Distancia al metro (5%)** - Específicamente, qué tan cerca está la estación más cercana.

**La lección principal:** Las características del entorno urbano (índices de habitabilidad, acceso a transporte) son casi tan importantes como las características físicas de la propiedad. Esto confirma lo que intuimos: la satisfacción residencial no depende solo del departamento, sino del barrio donde está ubicado.

## ¿Cómo usar este proyecto?

**Si quieres ejecutar el análisis completo:**
```bash
cd autocorrelacion_espacial/semana3_modelo_satisfaccion
./ejecutar_pipeline.sh
```

**Si quieres ejecutar scripts individuales:**
```bash
source venv_viz/bin/activate
python scripts/01_integrar_datos.py      # Integra propiedades con datos espaciales
python scripts/02_modelo_satisfaccion.py # Entrena modelo baseline
python scripts/06_modelo_satisfaccion_mejorado.py  # Modelo mejorado
python scripts/08_visualizaciones_cartograficas.py # Genera mapas y gráficos
```

**Para predecir satisfacción de una nueva propiedad:**
1. Asegúrate de tener coordenadas (latitud/longitud)
2. El sistema calculará automáticamente las métricas espaciales
3. El modelo devolverá un puntaje de satisfacción del 1 al 10

## Estructura del proyecto

```
autocorrelacion_espacial/
├── semana1_preparacion_datos/     # Limpieza y normalización de datos
├── semana2_caracteristicas_espaciales/  # Cálculo de distancias, densidades e índices
├── semana3_modelo_satisfaccion/   # Modelado y visualizaciones
│   ├── scripts/                   # 7 scripts del pipeline
│   ├── data/                      # Datos procesados
│   ├── resultados/                # Métricas y predicciones
│   ├── graficos/                  # Mapas y gráficos generados
│   └── modelos/                   # Modelos entrenados (.pkl)
└── semana4_recoleccion_datos/     # Scripts para ampliar datos
```

**Archivos clave generados:**
- `graficos/mapa_interactivo.html` - Mapa web interactivo de propiedades
- `graficos/mapa_01-03_*.png` - 3 mapas temáticos con elementos cartográficos
- `graficos/grafico_01-05_*.png` - 5 gráficos estadísticos
- `resultados/modelo_mejorado/metricas_modelo_mejorado.json` - Métricas del mejor modelo
- `modelos/modelo_satisfaccion_mejorado.pkl` - Modelo entrenado listo para usar

## Visualizaciones generadas

El proyecto genera automáticamente todas las visualizaciones requeridas:

**3 Mapas temáticos (con elementos cartográficos completos):**
- Mapa de ubicación del área de estudio
- Mapa de distribución de precios por m²
- Mapa de resultados del análisis (satisfacción predicha)

**5 Gráficos estadísticos:**
- Histogramas de variables clave
- Análisis por comuna
- Matriz de correlaciones
- Diagramas de dispersión
- Boxplots comparativos

**1 Visualización interactiva:**
- Mapa HTML con Folium donde puedes hacer clic en cada propiedad para ver sus detalles

## Limitaciones y próximos pasos

**Limitaciones actuales:**
- El modelo funciona solo para la Región Metropolitana de Santiago
- Necesita coordenadas precisas para funcionar bien
- No considera factores temporales (el mercado cambia con el tiempo)

**Próximos pasos sugeridos:**
1. Expandir a otras regiones de Chile
2. Incorporar datos temporales para detectar tendencias
3. Agregar información de calidad de construcción y antigüedad
4. Crear una interfaz web para que usuarios puedan consultar predicciones

## Conclusión

Este proyecto demuestra que la satisfacción residencial es predecible cuando combinamos información de la propiedad con datos del entorno urbano. El modelo mejorado alcanza un R² de 0.55, lo que significa que más de la mitad de la variación en satisfacción puede explicarse con las variables que medimos. Las características del barrio (acceso a transporte, servicios, calidad de vida urbana) son casi tan importantes como el tamaño del departamento.

---
*Proyecto GeoInformática - Felipe Baeza - Noviembre 2025*
