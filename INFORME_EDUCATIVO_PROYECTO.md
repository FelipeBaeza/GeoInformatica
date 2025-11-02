# Proyecto de Evaluación de Propiedades: Una Guía Completa para Principiantes

## Introducción: ¿Qué estamos construyendo?

Imagine que usted quiere comprar o arrendar una casa o departamento en Santiago, pero se siente abrumado por la cantidad de opciones disponibles y no sabe cuáles realmente se ajustan a su estilo de vida y necesidades específicas. Nuestro proyecto está construyendo un sistema inteligente de recomendaciones inmobiliarias que realiza un estudio exhaustivo del mercado integrando factores geoespaciales de las propiedades, con el fin de generar recomendaciones personalizadas según las características y necesidades de cada usuario objetivo.

El propósito fundamental de nuestro sistema es presentar alternativas de propiedades, cuya estimación de satisfacción dada por valoraciones de arrendatarios, compradoros o cotizantes, maximice la precisión en la estimación del nivel de satisfacción del usuario, considerando elementos del entorno como infraestructura, servicios, transporte, aspectos ambientales, entre otros factores que influyen directamente en la decisión de compra o arriendo de un inmueble. En otras palabras, nuestro sistema proporciona recomendaciones más acertadas basadas en los diferentes factores que cada usuario considera importantes para encontrar su casa ideal.

Este sistema combina dos enfoques complementarios: primero, un análisis exhaustivo del mercado inmobiliario que identifica patrones de precios basados en características objetivas como el tamaño, la ubicación y los servicios cercanos (modelo hedónico de mercado). Segundo, un sistema personalizado de satisfacción que evalúa qué tan bien se adapta cada propiedad a los gustos y necesidades específicas de cada usuario, como preferencias por estar cerca de parques, centros comerciales, colegios, transporte público, o evitar áreas con alta contaminación acústica.

La innovación principal de nuestro proyecto radica en integrar ambos enfoques para generar recomendaciones precisas y personalizadas. Mientras que otros sistemas solo muestran listados genéricos de propiedades, nosotros le diremos "esta propiedad tiene una puntuación de satisfacción de 9.2/10 para usted específicamente, considerando que prefiere estar cerca del metro y parques, valora la seguridad del barrio, y desea evitar zonas con alta contaminación del aire". Esta combinación permite tomar decisiones más informadas y encontrar propiedades que realmente se ajusten al estilo de vida y prioridades de cada persona.

## Metodología General: Cómo Funciona Nuestro Sistema

### La Base: Datos Geoespaciales

Todo nuestro sistema se construye sobre datos geoespaciales, que son información digital que describe lugares específicos en el mapa de Santiago. Piense en estos datos como capas transparentes que se superponen sobre un mapa de la ciudad. Una capa muestra dónde están todos los colegios, otra capa muestra las estaciones de metro, otra las comisarías, y así sucesivamente. Cuando combinamos todas estas capas, obtenemos una imagen completa de qué servicios y características están disponibles en cada ubicación de la ciudad.

Para nuestro proyecto, hemos recopilado 29 capas diferentes de información que incluyen educación (colegios, universidades, jardines infantiles), salud (hospitales, consultorios, farmacias), seguridad (comisarías, cuarteles de bomberos), transporte (metro, estaciones de carga eléctrica), recreación (parques, áreas verdes, circuitos turísticos), servicios (municipalidades, centros del SERNAM), y datos socioeconómicos (población, campamentos, delincuencia). Esta diversidad de datos nos permite construir un perfil completo de cada ubicación en Santiago.

### El Desafío de los Sistemas de Coordenadas

Un aspecto técnico fundamental que debemos manejar correctamente son los sistemas de coordenadas geográficas. Imagine que usted tiene varios mapas de Santiago, pero cada uno fue dibujado con una regla diferente: uno en centímetros, otro en pulgadas, otro en pasos. Para poder combinar información de todos estos mapas, primero debe convertir todas las medidas a la misma unidad.

En el mundo digital, este problema se llama sistemas de referencia de coordenadas (CRS, por sus siglas en inglés). Algunos de nuestros datos originales estaban en el sistema WGS84 (que usa grados como unidades), otros en Web Mercator (optimizado para mapas web), y algunos sin sistema definido. Para hacer cálculos precisos de distancias y áreas en metros - que es lo que necesitamos para nuestro modelo - convertimos todos los datos al sistema UTM 19S (EPSG:32719), que es el estándar técnico para Chile y usa metros como unidad base.

### Procesamiento y Limpieza de Datos

Los datos geoespaciales, como cualquier información del mundo real, vienen con imperfecciones. Algunas geometrías pueden estar "rotas" (como un polígono que no cierra correctamente), puede haber información duplicada, o coordenadas que están fuera del área de estudio. Nuestro proceso de limpieza identifica y corrige automáticamente estos problemas.

Por ejemplo, si encontramos un hospital que según las coordenadas está ubicado en el océano Pacífico, sabemos que hay un error y debemos investigar. Si encontramos dos registros del mismo colegio con nombres ligeramente diferentes, los identificamos como duplicados potenciales. Este proceso de control de calidad es crucial porque errores en los datos base se propagan y amplifican en los modelos finales.

## SEMANA 1: Preparación y Normalización de Datos

### Objetivos de la Primera Semana

La primera semana del proyecto se enfocó en establecer una base sólida y confiable para todo el trabajo posterior. Los objetivos específicos fueron: asegurar que todos nuestros 29 conjuntos de datos geoespaciales estuvieran en el mismo sistema de coordenadas, validar que las geometrías fueran correctas y utilizables, documentar exhaustivamente qué contiene cada archivo, y crear una estructura organizacional clara para las fases siguientes del proyecto.

### Proceso de Análisis Inicial

Comenzamos realizando un diagnóstico completo de nuestros datos originales. Creamos un script automatizado que examina cada archivo geoespacial y extrae información clave: qué sistema de coordenadas usa, qué tipos de geometrías contiene (puntos, líneas, polígonos), cuántas características tiene, cuál es su extensión geográfica, y si hay problemas evidentes como geometrías inválidas o coordenadas fuera del rango esperado.

Este análisis reveló varios desafíos importantes. Encontramos que nuestros datos provenían de tres sistemas de coordenadas diferentes: 27 archivos en WGS84 (el estándar internacional que usan GPS y Google Maps), 2 archivos en Web Mercator (el sistema que usan los mapas web como OpenStreetMap), y algunos archivos sin sistema de coordenadas definido. Además, identificamos que algunos archivos tenían campos calculados incorrectamente, como áreas expresadas en grados cuadrados en lugar de metros cuadrados.

### Proceso de Normalización

La normalización fue el proceso central de la primera semana. Desarrollamos un algoritmo inteligente que puede detectar automáticamente el sistema de coordenadas de un archivo basándose en el rango de sus coordenadas. Si las coordenadas están entre -180 y 180 en ambas dimensiones, probablemente es WGS84. Si están en rangos de millones, probablemente es un sistema proyectado como UTM o Web Mercator.

Una vez identificado el sistema original, aplicamos una transformación matemática precisa para convertir todas las coordenadas al sistema UTM 19S. Este proceso requiere algoritmos complejos que consideran la curvatura de la Tierra y las distorsiones inherentes a proyectar una superficie esférica en un plano. Afortunadamente, las librerías especializadas como PROJ manejan estos cálculos por nosotros, pero es importante entender que no es simplemente cambiar números, sino aplicar transformaciones geométricas precisas.

Durante la normalización, también agregamos campos calculados útiles: área en metros cuadrados, perímetro en metros, y coordenadas del centroide (centro geométrico) de cada característica. Estos campos serán fundamentales para calcular indicadores espaciales en las fases siguientes.

### Validación de Calidad

Después de la normalización, implementamos un proceso exhaustivo de validación de calidad. Este proceso verifica que todas las transformaciones se aplicaron correctamente y que los datos resultantes son consistentes y utilizables. Verificamos que todos los archivos efectivamente estén en el sistema UTM 19S, que las geometrías sean válidas (por ejemplo, que los polígonos estén cerrados correctamente), que las coordenadas estén dentro del rango esperado para Santiago, y que no hayamos perdido información durante el proceso.

También implementamos verificaciones de consistencia entre archivos. Por ejemplo, verificamos que las comunas definidas en diferentes archivos tengan límites coherentes, que los puntos de servicios estén ubicados dentro de las áreas urbanas, y que no haya duplicaciones evidentes entre diferentes fuentes de datos.

El resultado de este proceso fue tranquilizador: procesamos exitosamente 10,151 geometrías individuales sin errores críticos. Identificamos 19 problemas menores (principalmente duplicados potenciales y algunos valores atípicos espaciales) que documentamos para revisión manual posterior, pero ninguno que impida continuar con el proyecto.

### Documentación y Organización

Finalmente, creamos una documentación completa que registra todo lo que hicimos, por qué lo hicimos, y qué resultados obtuvimos. Esta documentación incluye reportes técnicos detallados en formato JSON y CSV para análisis automatizado, resúmenes ejecutivos en formato Markdown para revisión humana, y guías de uso que explican cómo utilizar los datos procesados.

Organizamos todos los archivos en una estructura de carpetas clara: `datos_filtrados` contiene los archivos originales (que ya no debemos usar para análisis), `datos_normalizados` contiene los archivos procesados listos para usar, `scripts` contiene todos los programas que desarrollamos, `reportes` contiene la documentación generada, y `features` está preparado para recibir los indicadores espaciales que calcularemos en la Semana 2.

### Resultados y Preparación para la Siguiente Fase

Al finalizar la Semana 1, tenemos una base de datos geoespacial completamente normalizada y validada. Los 29 archivos están en el sistema de coordenadas correcto, las geometrías están reparadas, la calidad está documentada, y toda la información está organizada para facilitar el trabajo en equipo.

Esta preparación nos permite ahora avanzar con confianza hacia la Semana 2, donde calcularemos indicadores espaciales específicos que servirán como base para nuestro sistema de recomendaciones personalizadas. Podremos medir distancias precisas entre propiedades y servicios que cada usuario considera importantes, calcular densidades de amenidades en diferentes radios según las preferencias individuales, y crear índices de accesibilidad personalizados, todo basado en medidas precisas en metros gracias al trabajo de normalización realizado.

La inversión de tiempo en esta preparación inicial se justifica completamente: cada recomendación posterior será más precisa, más confiable y más relevante para cada usuario porque trabajamos sobre una base sólida y bien documentada que nos permite generar evaluaciones espaciales exactas.

## SEMANA 2: Ingeniería de Características Espaciales

### Objetivos de la Segunda Semana

La segunda semana del proyecto se enfocó en transformar nuestros datos geoespaciales normalizados en características cuantitativas específicas que alimentarán directamente el sistema de recomendaciones personalizadas. Los objetivos fueron generar una grilla regular de puntos de evaluación que cubra toda el área de estudio, calcular distancias precisas desde cada punto hacia diferentes tipos de servicios, medir densidades de amenidades en múltiples radios, y crear índices compuestos de accesibilidad que combinen múltiples factores para obtener puntuaciones más significativas.

Esta fase representa la transición de datos descriptivos (qué hay en cada lugar) a datos analíticos (qué tan accesible y conveniente es cada lugar para diferentes necesidades). Es como pasar de un inventario de lo que existe en la ciudad a una evaluación sistemática de la calidad de vida que cada ubicación puede ofrecer.

### Creación de la Grilla de Evaluación

Para poder evaluar sistemáticamente toda el área metropolitana, creamos una grilla regular de puntos de evaluación espaciados cada 200 metros. Imagine que colocamos una malla invisible sobre Santiago, con puntos de referencia cada dos cuadras aproximadamente. Esta grilla nos permite tener una cobertura uniforme y comparable de todo el territorio.

La grilla resultante contiene 3,149 puntos distribuidos proporcionalmente entre las cuatro comunas de estudio: La Reina (1,121 puntos), Estación Central (786 puntos), Santiago (627 puntos) y Ñuñoa (615 puntos). Cada punto representa una ubicación específica donde calculamos todas las características espaciales, permitiendo después interpolar valores para cualquier dirección exacta que consulte un usuario.

Este enfoque sistemático es crucial porque nos permite comparar objetivamente diferentes ubicaciones usando exactamente los mismos criterios y métodos de cálculo. Sin esta grilla uniforme, sería imposible generar recomendaciones consistentes y confiables.

### Cálculo de Características de Distancia

Para cada uno de los 3,149 puntos de la grilla, calculamos la distancia euclidiana (línea recta) al servicio más cercano en 17 categorías diferentes. Estas categorías incluyen educación básica (458 establecimientos), educación superior (363 instituciones), servicios de salud (96 centros), transporte público (52 estaciones), áreas verdes (2,162 espacios), comercio (497 establecimientos), y servicios de seguridad (40 instalaciones), entre otros.

Los resultados muestran patrones interesantes de accesibilidad urbana. Por ejemplo, la distancia promedio a un establecimiento de educación básica es de 468 metros, con algunas ubicaciones teniendo un colegio a solo 6 metros y las más alejadas a 4.2 kilómetros. Para servicios de salud, la distancia promedio es mayor (1,637 metros), reflejando la menor densidad de estos servicios especializados.

Además de las distancias individuales por categoría, creamos distancias agrupadas que representan la proximidad al servicio más cercano dentro de grandes grupos funcionales: educación (cualquier tipo), salud (todos los servicios médicos), seguridad (cualquier servicio policial o de emergencia), y transporte (metro o estaciones de carga eléctrica).

### Cálculo de Densidades por Buffers Circulares

Las distancias nos dicen qué tan lejos está el servicio más cercano, pero no nos informan sobre la cantidad de opciones disponibles en el área. Para capturar esta dimensión de diversidad y abundancia, calculamos densidades de servicios dentro de buffers circulares de 300, 600 y 1,000 metros alrededor de cada punto.

Estos tres radios representan diferentes escalas de movilidad urbana: 300 metros es aproximadamente una caminata de 3-4 minutos (el barrio inmediato), 600 metros representa unos 7-8 minutos caminando (el área local), y 1,000 metros equivale a 12-15 minutos de caminata o un trayecto corto en transporte (el distrito).

Los resultados revelan la estructura policéntrica de Santiago. En educación, encontramos densidades que van desde cero hasta 191 establecimientos por kilómetro cuadrado en el radio de 300 metros, con un promedio de 15.7. Para recreación, las densidades son sistemáticamente más altas (promedio de 26.1 servicios por km²), reflejando la abundancia de plazas, parques y espacios de esparcimiento.

Todas las densidades se normalizaron a una escala de 0 a 10 para facilitar su interpretación y comparación. Una puntuación de 10 representa la máxima densidad encontrada en el área de estudio para esa categoría y radio específico.

### Creación de Índices de Accesibilidad Compuesta

Las distancias y densidades por separado proporcionan información valiosa, pero el valor real para el sistema de recomendaciones viene de combinarlas inteligentemente. Desarrollamos seis índices individuales de accesibilidad que integran tanto la proximidad como la abundancia de diferentes tipos de servicios.

El índice de accesibilidad educativa combina la distancia al establecimiento más cercano (60% del peso) con la densidad educativa en un radio de 600 metros (40% del peso). Esta ponderación refleja que para educación, la proximidad es más importante que tener muchas opciones, pero cierta diversidad sigue siendo valiosa.

Para salud, aplicamos una ponderación diferente (70% distancia, 30% densidad) usando un radio mayor (1,000 metros) porque los servicios de salud especializados naturalmente requieren mayor área de cobertura y la proximidad es crítica para emergencias.

El índice de conectividad de transporte enfatiza fuertemente la proximidad (80% distancia, 20% densidad) porque para transporte público, lo crucial es tener una estación cerca, no necesariamente muchas opciones.

### Índices Compuestos de Nivel Superior

Para facilitar la toma de decisiones, creamos tres índices de nivel superior que combinan múltiples dimensiones de accesibilidad:

El Índice de Vida Urbana integra accesibilidad educativa, de salud y de transporte, capturando los servicios esenciales para la vida urbana funcional. Sus valores van de 0.30 a 8.44 puntos (promedio 4.56), mostrando una significativa variabilidad en la calidad de los servicios básicos urbanos.

El Índice de Calidad de Vida combina calidad del entorno, seguridad percibida y accesibilidad comercial, enfocándose en factores que afectan el bienestar y la satisfacción residencial. Este índice muestra valores de 0 a 7.04 (promedio 3.59), indicando que la calidad de vida varía considerablemente entre ubicaciones.

El Índice de Habitabilidad Global es la síntesis final que combina vida urbana (60%) y calidad de vida (40%), proporcionando una puntuación integral de qué tan deseable es cada ubicación. Los valores van de 0.18 a 7.44 (promedio 4.17), ofreciendo una métrica unificada para comparar ubicaciones.

### Validación y Control de Calidad

Todos los índices fueron sometidos a rigurosas validaciones para asegurar coherencia matemática y significado práctico. Verificamos que todos los valores estén en el rango esperado de 0-10, que las correlaciones entre variables relacionadas sean lógicas, que no existan valores extremos sin justificación geográfica, y que los patrones espaciales reflejen la realidad urbana conocida.

Las validaciones confirmaron la robustez de nuestros cálculos: procesamos exitosamente 3,149 puntos sin errores críticos, generamos 21 columnas de distancias y 42 de densidades, creamos 9 índices de accesibilidad sin valores fuera de rango, y documentamos completamente todos los procesos para reproducibilidad.

### Resultados y Preparación para la Siguiente Fase

Al finalizar la Semana 2, hemos transformado 29 capas de datos geoespaciales crudos en un conjunto comprehensivo de 72 características cuantitativas por ubicación. Estas características capturan sistemáticamente la accesibilidad, diversidad y calidad del entorno urbano en cada punto de nuestro área de estudio.

La grilla final contiene información detallada sobre distancias a 17 categorías de servicios, densidades en tres radios diferentes para seis grandes grupos funcionales, y nueve índices de accesibilidad que sintetizan múltiples factores. Esta riqueza de información espacial constituye la base fundamental sobre la cual construiremos el sistema de recomendaciones personalizadas.

Los datos están completamente procesados, validados y listos para alimentar tanto el modelo hedónico de análisis de mercado como el sistema de satisfacción personalizada que desarrollaremos en las siguientes fases del proyecto.

---

## Próximas Etapas del Proyecto

### Semana 3-4: Análisis del Mercado Inmobiliario (Modelo Hedónico)
Desarrollaremos el modelo que analiza patrones de precios del mercado basado en características objetivas de las propiedades y su entorno espacial, permitiendo entender cómo diferentes factores influyen en el valor de mercado.

### Semana 5-6: Sistema de Recomendaciones Personalizadas
Construiremos el sistema central que evalúa qué tan bien se adapta cada propiedad a las preferencias específicas de cada usuario, generando puntuaciones de satisfacción personalizadas basadas en sus necesidades y prioridades individuales.

### Semana 7-8: Integración y Sistema de Recomendaciones Final
Combinaremos el análisis de mercado con las preferencias personalizadas para crear un sistema de recomendaciones integral que presente las mejores opciones para cada usuario, validando la precisión de nuestras recomendaciones con casos reales.

### Semanas 3-4: Análisis del Mercado Inmobiliario (Modelo Hedónico)
Desarrollaremos el modelo que analiza patrones de precios del mercado basado en características objetivas de las propiedades y su entorno espacial, permitiendo entender cómo diferentes factores influyen en el valor de mercado.

### Semanas 5-6: Sistema de Recomendaciones Personalizadas
Construiremos el sistema central que evalúa qué tan bien se adapta cada propiedad a las preferencias específicas de cada usuario, generando puntuaciones de satisfacción personalizadas basadas en sus necesidades y prioridades individuales.

### Semanas 7-8: Integración y Sistema de Recomendaciones Final
Combinaremos el análisis de mercado con las preferencias personalizadas para crear un sistema de recomendaciones integral que presente las mejores opciones para cada usuario, validando la precisión de nuestras recomendaciones con casos reales.

---

*Este documento se actualiza progresivamente con cada fase completada del proyecto. La próxima actualización incluirá los detalles de la Semana 2: Ingeniería de Características Espaciales.*