# Modelo de Satisfacción Residencial - Propiedades en Venta

## 📋 Descripción
Modelo predictivo de satisfacción residencial para propiedades en venta en Santiago de Chile, utilizando Machine Learning (Random Forest + Gradient Boosting) con factores espaciales externos.

## 🚀 Ejecución Rápida

```bash
# Activar entorno virtual
source /home/felipe/Documentos/GeoInformatica/.venv/bin/activate

# Entrenar modelo (procesa datos de DATOS_FILTRADOS)
python scripts/modelo_satisfaccion.py

# Usar predictor
python scripts/predecir_satisfaccion.py
```

## 📁 Estructura del Proyecto

```
semana3_modelo_satisfaccion/
├── scripts/                    # Scripts activos
│   ├── modelo_satisfaccion.py  # Entrenamiento del modelo
│   └── predecir_satisfaccion.py # API de predicción
│
├── scripts_obsoletos/          # Scripts del pipeline antiguo (respaldo)
│
├── resultados/modelo_venta/    # Resultados del modelo
│   ├── metricas_modelo_venta.json
│   ├── feature_importance_venta.csv
│   └── propiedades_venta_con_satisfaccion.csv
│
├── graficos/                   # Visualizaciones
│   ├── feature_importance_venta.png
│   ├── prediccion_vs_real_venta.png
│   └── distribucion_satisfaccion_venta.png
│
├── modelos/                    # Modelo entrenado
│   └── modelo_satisfaccion_venta.pkl
│
└── README.md
```

## 📈 Resultados del Modelo

| Métrica | Valor |
|---------|-------|
| R² Test | **0.852** |
| RMSE | 0.349 |
| MAE | 0.276 |
| CV R² (5-fold) | 0.850 ± 0.016 |
| Features | 42 |

### Top Features más importantes
1. **dist_salud_m** (50%) - Distancia a centros de salud
2. **precio_m2_uf** (16%) - Precio por metro cuadrado
3. **dens_comercio_600m_km2** (8%) - Densidad comercial
4. **dist_transporte_min_m** (7%) - Acceso a transporte

## 🎭 Perfiles de Usuario

| Perfil | Descripción |
|--------|-------------|
| familia_con_ninos | Prioriza espacio, educación, seguridad |
| profesional_joven | Prioriza transporte, comercio, precio |
| inversionista | Prioriza ROI, transporte, seguridad |
| adulto_mayor | Prioriza salud, seguridad, áreas verdes |
| balanceado | Equilibrado en todas las dimensiones |

## 🔮 Uso del Predictor

```python
from predecir_satisfaccion import PredictorSatisfaccion

predictor = PredictorSatisfaccion()

resultado = predictor.predecir({
    'superficie_util': 65,
    'dormitorios': 2,
    'precio_uf': 4500,
    'tipo_propiedad': 'departamento'
})

print(f"Satisfacción: {resultado['satisfaccion']}/10")
```
