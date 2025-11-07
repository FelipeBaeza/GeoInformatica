# Semana 3: Modelo de Satisfacción con Autocorrelación Espacial

## 📋 Descripción

Esta semana implementamos un modelo predictivo de satisfacción (precio/m²) utilizando Random Forest, incorporando factores espaciales generados en la Semana 2, y analizamos la autocorrelación espacial de los residuos para validar si el modelo captura adecuadamente la estructura espacial de los datos.

## 🎯 Objetivos

1. Integrar datos de propiedades con factores espaciales (distancias y densidades)
2. Construir un modelo de Random Forest para predecir precio/m²
3. Evaluar el rendimiento del modelo (R², RMSE, MAE)
4. Analizar autocorrelación espacial en los residuos (Índice de Moran)
5. Implementar modelos espacialmente explícitos (GWRF - Geographically Weighted Random Forest)

## 📂 Estructura

```
semana3_modelo_satisfaccion/
├── scripts/
│   ├── 00_imputar_valores_faltantes.py    # Limpieza de datos faltantes
│   ├── 01_integrar_datos.py                # Join espacial con features de semana2
│   ├── 02_modelo_satisfaccion.py           # Modelo Random Forest global
│   ├── 03_autocorrelacion_residuos.py      # Cálculo de Moran's I
│   ├── 04_validar_autocorrelacion.py       # Test de permutación
│   └── 05_implementar_gwrf.py              # Modelos GWRF y Stacking
├── data/
│   ├── propiedades_con_factores_espaciales.csv    # Dataset integrado
│   └── propiedades_con_residuos.geojson           # Con residuos del modelo
├── resultados/
│   ├── random_forest_model.pkl              # Modelo entrenado
│   ├── metricas_modelo.csv                  # Métricas de evaluación
│   ├── feature_importances.csv              # Importancia de variables
│   ├── autocorrelacion_residuos.csv         # Índice de Moran
│   └── gwrf/                                # Resultados GWRF
│       ├── gwrf_por_cluster.pkl
│       ├── meta_model_stack.pkl
│       └── comparacion_modelos_gwrf.csv
├── graficos/
│   ├── predicciones_vs_real.png
│   ├── feature_importance.png
│   ├── mapa_residuos.png
│   └── comparacion_gwrf_vs_baseline.png
└── README.md
```

## 🚀 Flujo de Ejecución

### Paso 1: Integrar Datos
```bash
python3 scripts/01_integrar_datos.py
```
**Qué hace:**
- Carga el dataset de propiedades (`clean_alquiler_02_11_2023cc.csv`)
- Realiza join espacial con la grilla de factores espaciales (semana2)
- Agrega 42 columnas de densidades y 21 de distancias
- Genera: `propiedades_con_factores_espaciales.csv`

**Output esperado:**
```
✓ Propiedades cargadas: 1554 registros
✓ Features espaciales agregadas:
  • Densidades: 42 columnas
  • Distancias: 21 columnas
✓ CSV guardado: data/propiedades_con_factores_espaciales.csv
```

### Paso 2: Modelo Random Forest Global
```bash
python3 scripts/02_modelo_satisfaccion.py
```
**Qué hace:**
- Entrena un Random Forest con todas las features espaciales
- Divide datos en train/test (80/20)
- Evalúa con R², RMSE, MAE
- Genera gráficos de predicciones vs real e importancia de features

**Métricas esperadas:**
```
Test Set:
  • R²:   0.85 - 0.90
  • RMSE: $2,500 - $3,500
  • MAE:  $1,800 - $2,500
```

### Paso 3: Análisis de Autocorrelación
```bash
python3 scripts/03_autocorrelacion_residuos.py
```
**Qué hace:**
- Calcula residuos del modelo (real - predicho)
- Calcula Índice de Moran I para detectar autocorrelación espacial
- Genera mapa de residuos y histograma

**Interpretación:**
- **Moran I > 0.3**: Autocorrelación positiva fuerte → modelo no captura estructura espacial
- **Moran I 0.1 - 0.3**: Autocorrelación moderada
- **Moran I < 0.1**: Autocorrelación baja o nula → modelo OK

### Paso 4: Validación Estadística
```bash
python3 scripts/04_validar_autocorrelacion.py
```
**Qué hace:**
- Test de permutación (999 iteraciones) para validar significancia de Moran's I
- Genera distribución bajo hipótesis nula
- Calcula p-value

**Criterio:**
- Si **p < 0.05**: Autocorrelación estadísticamente significativa → necesitamos GWRF

### Paso 5: GWRF y Stacking (SI hay autocorrelación)
```bash
python3 scripts/05_implementar_gwrf.py
```
**Qué hace:**
- Implementa 3 estrategias GWRF:
  1. **GWRF por Comuna**: Entrena un RF por cada comuna
  2. **GWRF por Cluster (KMeans)**: Particiona espacialmente con clustering en densidades
  3. **GWRF por Densidad**: Divide en terciles de densidad poblacional
  
- **Stacking**: Combina predicciones de todos los modelos con meta-learner Ridge

**Output:**
```
✓ RESULTADOS COMPARATIVOS:
  Modelo                    R²      RMSE      MAE
  ────────────────────────────────────────────────
  RF Global (Baseline)     0.85    $3,200    $2,100
  GWRF por Cluster         0.92    $2,100    $1,400
  GWRF por Densidad        0.89    $2,400    $1,600
  Stacking (Meta-Model)    0.94    $1,800    $1,200
```

## 📊 Métricas de Evaluación

### R² (Coeficiente de Determinación)
- **Qué mide**: % de varianza explicada por el modelo
- **Rango**: 0 a 1 (1 = perfecto)
- **Meta**: R² > 0.80

### RMSE (Root Mean Squared Error)
- **Qué mide**: Error promedio en unidades del target ($)
- **Penaliza**: Errores grandes más fuertemente
- **Meta**: Minimizar

### MAE (Mean Absolute Error)
- **Qué mide**: Error absoluto promedio
- **Interpretación**: Más robusta a outliers que RMSE
- **Meta**: Minimizar

## 🧮 ¿Por qué Random Forest?

### Ventajas para este proyecto:

1. **Maneja features espaciales complejas** (42 densidades + 21 distancias)
2. **Robusto a outliers** (importante en precios de propiedades)
3. **No requiere escalado** de variables
4. **Captura relaciones no lineales** (ej: precio vs distancia a metro no es lineal)
5. **Provee importancia de features** (interpretabilidad)
6. **Reduce overfitting** (ensemble de árboles)

### Limitación:
❌ **NO captura autocorrelación espacial directamente** → Por eso implementamos GWRF

## 🗺️ GWRF (Geographically Weighted Random Forest)

### Concepto
En lugar de un modelo global único, GWRF entrena **modelos locales** en diferentes regiones del espacio, capturando heterogeneidad espacial.

### Estrategias Implementadas:

#### 1. GWRF por Comuna
- **Partición**: Por comuna administrativa
- **Ventaja**: Captura políticas locales, identidad de barrio
- **Desventaja**: Algunas comunas tienen pocos datos

#### 2. GWRF por Cluster (KMeans)
- **Partición**: Clustering en primeras 10 densidades
- **Ventaja**: Agrupa zonas espacialmente similares, balanceo automático
- **Mejor para**: Cuando queremos particiones "naturales"

#### 3. GWRF por Densidad
- **Partición**: Terciles de densidad poblacional
- **Ventaja**: Captura gradiente urbano (centro ↔ periferia)
- **Mejor para**: Ciudades con clara estructura centro-periferia

### Stacking (Meta-Ensemble)
Combina las predicciones de todos los modelos GWRF + RF global usando un Ridge Regression como meta-learner:

```
prediccion_final = Ridge([pred_rf_global, pred_gwrf_cluster, pred_gwrf_densidad, ...])
```

**Ventaja**: Aprovecha fortalezas de cada estrategia

## 📈 Resultados Esperados

### Escenario 1: Sin Autocorrelación (Moran I < 0.1, p > 0.05)
✅ RF Global es suficiente (R² ≈ 0.85-0.90)  
✅ No necesitas GWRF

### Escenario 2: Autocorrelación Moderada (Moran I 0.1-0.3, p < 0.05)
⚠️ GWRF mejora 5-10% en R²  
⚠️ Implementar GWRF por Cluster

### Escenario 3: Autocorrelación Fuerte (Moran I > 0.3, p < 0.05)
🚨 RF Global es insuficiente (R² < 0.70)  
🚨 GWRF + Stacking mejoran 15-25% en R²  
🚨 Considera también SAR/CAR models

## 🔧 Configuración de Hiperparámetros

### Random Forest Global
```python
RandomForestRegressor(
    n_estimators=100,      # Árboles en el bosque
    max_depth=15,          # Profundidad máxima
    min_samples_split=5,   # Min muestras para split
    min_samples_leaf=2,    # Min muestras en hoja
    max_features='sqrt',   # Features por split
    random_state=42        # Reproducibilidad
)
```

### GWRF (por cluster)
```python
RandomForestRegressor(
    n_estimators=100,
    max_depth=12,          # Menor depth (menos datos por cluster)
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)
```

### Meta-Model (Stacking)
```python
Ridge(alpha=1.0)  # Regularización L2
```

## 🐛 Troubleshooting

### Error: "No se encontró propiedades_con_factores_espaciales.csv"
**Solución**: Ejecuta primero `01_integrar_datos.py`

### Error: "Cluster con datos insuficientes"
**Causa**: KMeans creó clusters muy pequeños  
**Solución**: El script ahora ajusta automáticamente el número de clusters según tamaño del dataset:
```python
max_clusters = max(2, min(5, len(df) // 50))  # Min 50 muestras por cluster
```

### Métricas bajas (R² < 0.5)
**Posibles causas**:
1. Dataset con muchos outliers → revisar limpieza
2. Features espaciales con valores constantes (dens_* = 0) → regenerar semana2
3. Pocos datos → considerar aumentar muestra

### GWRF no mejora sobre RF Global
**Interpretación**: No hay heterogeneidad espacial significativa  
**Acción**: Usar RF Global (más simple es mejor)

## 📚 Referencias

- Breiman, L. (2001). Random Forests. *Machine Learning*, 45(1), 5-32.
- Georganos, S., et al. (2019). Geographical random forests. *International Journal of Geographical Information Science*, 33(4), 804-831.
- Anselin, L. (1988). Spatial Econometrics: Methods and Models.

## 👥 Autores

Equipo GeoInformática  
Universidad de Chile  
Noviembre 2025

---

## 🚦 Inicio Rápido

```bash
# 1. Integrar datos
python3 scripts/01_integrar_datos.py

# 2. Entrenar modelo baseline
python3 scripts/02_modelo_satisfaccion.py

# 3. Analizar autocorrelación
python3 scripts/03_autocorrelacion_residuos.py

# 4. Validar significancia
python3 scripts/04_validar_autocorrelacion.py

# 5. GWRF + Stacking (si Moran I significativo)
python3 scripts/05_implementar_gwrf.py
```

¡Listo! 🎉
