# 🏠 Proyecto GeoInformática: Predicción de Satisfacción Residencial

> **Guía para entender el proyecto** - Este documento explica de forma simple qué hace el proyecto, cómo funciona y cómo usarlo.

---

## Resumen Ejecutivo

Este proyecto responde una pregunta importante: **¿qué hace que una vivienda sea satisfactoria para vivir?** 

Para responderla, combinamos información de **7,702 propiedades en venta** (departamentos y casas) con datos del entorno urbano de Santiago: cercanía a metros, colegios, hospitales, áreas verdes, comercio, etc.

**Resultado principal:** Un modelo de Machine Learning (LightGBM) que predice la satisfacción residencial con un **R² de 0.8635**, lo que significa que el modelo explica el **86.35%** de la variación en satisfacción.

---

## 📊 Datos Utilizados

### Propiedades
- **Fuente:** Portal Inmobiliario (datos_nuevos/DATOS_FILTRADOS)
- **Total:** 7,702 propiedades en venta
  - 5,135 departamentos
  - 2,567 casas
- **Comunas:** La Reina, Ñuñoa, Santiago, Estación Central
- **Precio:** 800 - 73,000 UF
- **Superficie:** 16 - 900 m²

### Datos Espaciales
- **75 características espaciales** del entorno urbano
- Distancias a: metro, colegios, hospitales, áreas verdes, seguridad, comercio
- Densidades de servicios en diferentes radios (300m, 600m, 1000m)
- Índices compuestos de habitabilidad

---

## 🔄 Flujo del Proyecto (3 Semanas)

\`\`\`
Semana 1                    Semana 2                    Semana 3
┌──────────────┐           ┌──────────────┐           ┌──────────────┐
│ PREPARACIÓN  │           │ CARACTERÍSTICAS│          │   MODELO     │
│    DATOS     │    ──►    │   ESPACIALES  │    ──►   │   LIGHTGBM   │
└──────────────┘           └──────────────┘           └──────────────┘
     │                          │                          │
     ▼                          ▼                          ▼
- Limpiar datos           - Crear grilla            - Entrenar modelo
- Normalizar CRS          - Calcular distancias     - Evaluar métricas  
- Validar calidad         - Calcular densidades     - Predecir satisfacción
                          - Generar índices
\`\`\`

### Semana 1: Preparación de Datos
- Descarga y limpieza de datos de propiedades
- Normalización al sistema de coordenadas UTM 19S (EPSG:32719)
- Validación de calidad de datos geográficos

### Semana 2: Características Espaciales
- Creación de grilla de análisis sobre Santiago
- Cálculo de distancias a servicios (metro, colegios, hospitales, etc.)
- Cálculo de densidades de servicios en radios de 300m, 600m, 1000m
- Generación de índices compuestos de habitabilidad

### Semana 3: Modelo de Satisfacción
- Integración de propiedades con datos espaciales
- Entrenamiento de modelo LightGBM
- Evaluación y validación cruzada
- API de predicción para nuevas propiedades

---

## 🤖 El Modelo: LightGBM

### ¿Por qué LightGBM?

Después de comparar **21 modelos diferentes** (Random Forest, XGBoost, CatBoost, Neural Networks, etc.), **LightGBM** demostró ser el mejor:

| Métrica | LightGBM | Random Forest (baseline) | Mejora |
|---------|----------|--------------------------|--------|
| R² Test | **0.8635** | 0.8453 | +2.2% |
| RMSE | **0.3357** | 0.3573 | -6.0% |
| CV R² | **0.8650** | 0.8449 | +2.4% |
| Tiempo | **3s** | 4.5s | 33% más rápido |

### Ventajas de LightGBM
1. ✅ Mayor precisión (+2.2% sobre Random Forest)
2. ✅ Menor error de predicción (-6% RMSE)
3. ✅ Entrenamiento más rápido
4. ✅ Baja autocorrelación espacial de residuos (Moran's I = 0.07)

---

## 🧭 ¿Qué es Random Forest y por qué lo usamos inicialmente?

Random Forest (Bosque Aleatorio) es un algoritmo de ensamble que combina muchos árboles de decisión independientes. Cada árbol se entrena con una muestra aleatoria de los datos y un subconjunto aleatorio de variables; la predicción final se obtiene promediando las salidas de todos los árboles.

Por qué se eligió al inicio:
- Es robusto frente a datos ruidosos y outliers.
- Maneja bien variables mixtas (numéricas y categóricas) sin mucha transformación.
- Proporciona medidas de importancia de variables útiles para interpretación inicial.
- Requiere poco ajuste por defecto y funciona como buen baseline para problemas tabulares.

En este proyecto se usó Random Forest como punto de partida porque permitió validar rápidamente que las features internas y espaciales contenían señal para predecir satisfacción, y ofreció una referencia estable para comparar modelos más sofisticados.

## ⚙️ ¿Qué es LightGBM y por qué lo usamos ahora?

LightGBM es un algoritmo de boosting basado en árboles (gradient boosting) diseñado para ser rápido y eficiente con grandes cantidades de datos y muchas features. A diferencia de Random Forest (que entrena árboles en paralelo), LightGBM construye los árboles de forma secuencial corrigiendo errores previos, lo que suele producir modelos más precisos cuando se hace un ajuste razonable de hiperparámetros.

Por qué lo elegimos tras la comparación exhaustiva:
- Mejor rendimiento en R² y RMSE en validación cruzada para nuestros datos.
- Mayor eficiencia en tiempo de entrenamiento y uso de memoria.
- Capacidad para explotar interacciones complejas entre variables (especialmente útiles con 75 features espaciales).

## 🔍 ¿Cómo cambia esto los resultados y la interpretación?

- Precisión: LightGBM mejora la capacidad predictiva global (R² sube de ~0.79 en GWRF a ~0.86), lo que reduce el error promedio de predicción.
- Estabilidad: la validación cruzada muestra menor varianza (CV R² más consistente), por lo que las predicciones son más confiables fuera de la muestra de entrenamiento.
- Interpretabilidad: aunque Random Forest y LightGBM ofrecen importancias de variables, LightGBM permite además técnicas como SHAP para interpretar efectos locales; sin embargo, los modelos de boosting pueden ser algo más complejos de interpretar que un RF simple.
- Comportamiento espacial: dado que nuestras features ya codifican información espacial (grilla, distancias, densidades), un modelo global como LightGBM aprovecha esa información y reduce la necesidad de un modelo explícitamente local como GWRF. Aún así, si se requiere explicar variaciones a muy pequeña escala, puede ser útil complementar con análisis local (p. ej. GWRF o maps de residuos).

En resumen: mantenemos Random Forest como referencia e instrumento de interpretación, pero usamos LightGBM en producción por su mayor precisión y eficiencia con las features espaciales que ya calculamos.


## 📈 Variables Más Importantes

El modelo identificó estas variables como las más predictivas de satisfacción:

| # | Variable | Importancia | Categoría |
|---|----------|-------------|-----------|
| 1 | **Precio/m² (UF)** | 1,351 | Económica |
| 2 | **Superficie útil** | 1,017 | Propiedad |
| 3 | **Precio total (UF)** | 616 | Económica |
| 4 | **Dormitorios** | 445 | Propiedad |
| 5 | **Dist. áreas verdes** | 367 | Espacial |
| 6 | **Dist. seguridad** | 293 | Espacial |
| 7 | **Ubicación (lon)** | 284 | Geográfica |
| 8 | **Ubicación (lat)** | 283 | Geográfica |
| 9 | **Dist. transporte** | 279 | Espacial |
| 10 | **Dist. ocio** | 233 | Espacial |

**Hallazgo clave:** Las características espaciales (distancia a áreas verdes, transporte, seguridad) son casi tan importantes como las características de la propiedad.

---

## 🎭 Perfiles de Usuario

El modelo calcula satisfacción para 5 perfiles diferentes:

| Perfil | Descripción | Prioridades |
|--------|-------------|-------------|
| **familia_con_ninos** | Familias con hijos escolares | Espacio, educación, áreas verdes |
| **profesional_joven** | Profesionales 25-35 años | Transporte, comercio, precio |
| **inversionista** | Compra para arriendo | Valor, transporte, seguridad |
| **adulto_mayor** | Personas 65+ años | Salud, seguridad, comercio |
| **balanceado** | Perfil equilibrado | Todos los factores igual |

---

## 🔧 Cómo Usar el Proyecto

### 1. Ejecutar el modelo completo
\`\`\`bash
cd autocorrelacion_espacial/semana3_modelo_satisfaccion
source /home/felipe/Documentos/GeoInformatica/.venv/bin/activate
python scripts/modelo_satisfaccion.py
\`\`\`

### 2. Predecir satisfacción de una propiedad
\`\`\`python
from scripts.predecir_satisfaccion import PredictorSatisfaccion

# Cargar predictor
predictor = PredictorSatisfaccion()

# Definir propiedad
propiedad = {
    'superficie_util': 60,
    'dormitorios': 2,
    'banos': 1,
    'precio_uf': 3000,
    'tipo_propiedad': 'departamento',
    'latitude': -33.45,
    'longitude': -70.65,
}

# Predecir
resultado = predictor.predecir(propiedad)
print(f"Satisfacción: {resultado['satisfaccion']}/10 {resultado['emoji']}")
# Output: Satisfacción: 5.99/10 ⚠️
\`\`\`

### 3. Comparar múltiples propiedades
\`\`\`python
propiedades = [
    {'superficie_util': 60, 'dormitorios': 2, 'precio_uf': 3000, ...},
    {'superficie_util': 80, 'dormitorios': 3, 'precio_uf': 4500, ...},
    {'superficie_util': 45, 'dormitorios': 1, 'precio_uf': 2000, ...},
]

ranking = predictor.comparar_propiedades(propiedades)
print(ranking[['ranking', 'satisfaccion', 'precio_uf']])
\`\`\`

---

## 📁 Estructura del Proyecto

\`\`\`
autocorrelacion_espacial/
├── semana1_preparacion_datos/
│   └── scripts/           # Limpieza y normalización
├── semana2_caracteristicas_espaciales/
│   ├── scripts/           # Cálculo de distancias y densidades
│   └── features/          # Grilla con índices espaciales
└── semana3_modelo_satisfaccion/
    ├── scripts/
    │   ├── modelo_satisfaccion.py      # Entrenamiento LightGBM
    │   ├── predecir_satisfaccion.py    # API de predicción
    │   └── comparar_modelos.py         # Comparación de 21 modelos
    ├── modelos/
    │   └── modelo_satisfaccion_venta.pkl  # Modelo entrenado
    ├── resultados/
    │   ├── modelo_venta/               # Métricas actuales
    │   └── comparacion_modelos/        # Resultados comparación
    └── graficos/                        # Visualizaciones
\`\`\`

---

## 📊 Visualizaciones del Proyecto

El proyecto genera visualizaciones que cumplen con los requisitos mínimos establecidos:
- **3 mapas temáticos** con elementos cartográficos (flecha norte, escala, leyenda)
- **5 gráficos estadísticos** que exploran los datos y resultados
- **1 visualización interactiva** funcional

### 🗺️ Mapas Temáticos (3)

| Mapa | Archivo | Descripción |
|------|---------|-------------|
| **Mapa 1** | `mapa_01_ubicacion_area_estudio.png` | Muestra la distribución de las 7,702 propiedades en las 4 comunas de estudio (La Reina, Ñuñoa, Santiago, Estación Central). Diferencia departamentos (azul) de casas (rojo). |
| **Mapa 2** | `mapa_02_precio_m2.png` | Representa el precio por metro cuadrado en UF mediante escala de colores (verde=barato, rojo=caro). Permite identificar zonas de alto y bajo costo. |
| **Mapa 3** | `mapa_03_satisfaccion_predicha.png` | Resultado principal del análisis: el índice de satisfacción predicho por LightGBM (1-10). Colores verdes indican alta satisfacción, rojos baja. |

**Elementos cartográficos incluidos:** Flecha de norte, barra de escala, leyenda, etiquetas de ejes, título descriptivo, estadísticas en recuadro.

### 📊 Gráficos Estadísticos (5)

| Gráfico | Archivo | Qué muestra | Interpretación |
|---------|---------|-------------|----------------|
| **Gráfico 1** | `grafico_01_histogramas.png` | Histogramas de 6 variables clave: precio, superficie, precio/m², dormitorios, baños y satisfacción | Permite ver la distribución de cada variable. La línea roja indica la media, la naranja la mediana. |
| **Gráfico 2** | `grafico_02_analisis_comunas.png` | Análisis comparativo por comuna: boxplots de precio y satisfacción, conteo de propiedades por tipo | Revela diferencias entre comunas: La Reina tiene precios más altos, Santiago más oferta de departamentos. |
| **Gráfico 3** | `grafico_03_correlaciones.png` | Matriz de correlación entre variables principales y espaciales | Valores cercanos a 1 (azul) o -1 (rojo) indican relaciones fuertes. El precio/m² tiene correlación negativa con satisfacción. |
| **Gráfico 4** | `grafico_04_dispersion.png` | 4 diagramas de dispersión: precio vs superficie, precio/m² vs satisfacción, dormitorios vs satisfacción, predicción vs real | Visualiza relaciones bivariadas y valida que el modelo predice bien (puntos cercanos a la diagonal). |
| **Gráfico 5** | `grafico_05_importancia_metricas.png` | Top 15 variables más importantes + comparación de métricas entre modelos (LightGBM vs RF vs GWRF) | Confirma que precio/m², superficie y variables espaciales son las más predictivas. LightGBM supera a los otros modelos. |

### 🌐 Visualización Interactiva (1)

**Archivo:** `mapa_interactivo.html`

Mapa web interactivo creado con Folium que permite:
- **Explorar propiedades:** Clic en marcadores para ver detalles (tipo, comuna, precio, superficie, satisfacción)
- **Ver patrones espaciales:** Heatmap de satisfacción residencial
- **Controlar capas:** Activar/desactivar marcadores y heatmap
- **Navegar:** Zoom, pan, vista de calle

**Cómo abrirlo:** Doble clic en el archivo HTML o abrirlo en cualquier navegador web.

---

## 📁 Archivos Generados

### Modelo
- `modelos/modelo_satisfaccion_venta.pkl` - Modelo LightGBM entrenado

### Resultados
- `resultados/modelo_venta/metricas_modelo_venta.json` - Métricas del modelo
- `resultados/modelo_venta/propiedades_venta_con_satisfaccion.csv` - Dataset con predicciones
- `resultados/comparacion_modelos/comparacion_modelos.csv` - Comparación de 21 modelos

### Visualizaciones
- `graficos/mapa_01_ubicacion_area_estudio.png` - Mapa de ubicación
- `graficos/mapa_02_precio_m2.png` - Mapa de precios
- `graficos/mapa_03_satisfaccion_predicha.png` - Mapa de resultados
- `graficos/grafico_01_histogramas.png` - Histogramas
- `graficos/grafico_02_analisis_comunas.png` - Análisis por comuna
- `graficos/grafico_03_correlaciones.png` - Correlaciones
- `graficos/grafico_04_dispersion.png` - Dispersión
- `graficos/grafico_05_importancia_metricas.png` - Importancia y métricas
- `graficos/mapa_interactivo.html` - Mapa interactivo
- `graficos/INDICE_VISUALIZACIONES.json` - Índice de visualizaciones

---

## 📉 Interpretación de Resultados

### Escala de Satisfacción (0-10)

| Rango | Nivel | Interpretación |
|-------|-------|----------------|
| 8-10 | 🌟 Excelente | Propiedad altamente satisfactoria |
| 6-8 | ✅ Bueno | Buena relación calidad-precio-ubicación |
| 4-6 | ⚠️ Regular | Cumple pero con aspectos a mejorar |
| 0-4 | ❌ Bajo | Baja satisfacción esperada |

### R² = 0.8635 significa...
- El modelo explica el **86.35%** de la variación en satisfacción
- Es un modelo muy preciso para datos geoespaciales
- El 13.65% restante depende de factores no medidos (estado del edificio, vista, vecinos, etc.)

---

## 🔮 Limitaciones y Próximos Pasos

### Limitaciones Actuales
- Solo funciona para la Región Metropolitana de Santiago
- Requiere coordenadas precisas (lat/lon)
- No considera factores temporales (el mercado cambia)
- No incluye calidad de construcción o antigüedad

### Posibles Mejoras
1. Expandir a otras regiones de Chile
2. Incorporar datos temporales (tendencias de precio)
3. Agregar información de calidad de construcción
4. Crear interfaz web para consultas

---

## 📋 Requisitos Técnicos

\`\`\`bash
# Instalar dependencias
pip install pandas numpy geopandas scikit-learn lightgbm matplotlib seaborn

# O usar requirements.txt
pip install -r autocorrelacion_espacial/semana3_modelo_satisfaccion/requirements.txt
\`\`\`

### Dependencias Principales
- Python 3.10+
- pandas, numpy, geopandas
- scikit-learn
- **lightgbm** (modelo principal)
- matplotlib, seaborn

---

## 📝 Resumen

| Aspecto | Valor |
|---------|-------|
| **Propiedades analizadas** | 7,702 |
| **Modelo** | LightGBM |
| **Precisión (R²)** | 0.8635 |
| **Error (RMSE)** | 0.3357 |
| **Features** | 42 |
| **Variables espaciales** | 30 |
| **Perfiles de usuario** | 5 |
| **Tiempo entrenamiento** | ~3 segundos |

---

*Proyecto GeoInformática - Felipe Baeza - Diciembre 2025*
*Modelo actualizado a LightGBM tras comparación exhaustiva de 21 modelos de ML*
