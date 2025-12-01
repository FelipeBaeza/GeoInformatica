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

## 📊 Archivos Generados

### Modelo
- \`modelos/modelo_satisfaccion_venta.pkl\` - Modelo LightGBM entrenado

### Resultados
- \`resultados/modelo_venta/metricas_modelo_venta.json\` - Métricas del modelo
- \`resultados/modelo_venta/propiedades_venta_con_satisfaccion.csv\` - Dataset con predicciones
- \`resultados/comparacion_modelos/comparacion_modelos.csv\` - Comparación de 21 modelos

### Gráficos
- \`graficos/feature_importance_venta.png\` - Importancia de variables
- \`graficos/prediccion_vs_real_venta.png\` - Scatter plot predicción vs real
- \`graficos/distribucion_satisfaccion_venta.png\` - Distribución por tipo
- \`graficos/comparacion_r2_modelos.png\` - Comparación de modelos

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
