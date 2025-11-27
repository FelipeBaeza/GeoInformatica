# Semana 3: Modelo de Satisfacción Residencial

## 📋 Descripción
Pipeline completo para análisis de autocorrelación espacial y predicción de satisfacción residencial utilizando Random Forest y GWRF (Geographically Weighted Random Forest).

## 🚀 Ejecución Rápida

```bash
# Opción 1: Pipeline completo (automatizado)
./ejecutar_pipeline.sh

# Opción 2: Scripts individuales (con venv)
source venv_viz/bin/activate
python scripts/01_integrar_datos.py
python scripts/02_modelo_satisfaccion.py
python scripts/03_autocorrelacion_residuos.py
python scripts/04_validar_autocorrelacion.py
python scripts/05_implementar_gwrf.py
python scripts/06_modelo_satisfaccion_mejorado.py
python scripts/08_visualizaciones_cartograficas.py
```

## 📁 Estructura del Proyecto

```
semana3_modelo_satisfaccion/
├── scripts/                          # Scripts activos del pipeline
│   ├── 01_integrar_datos.py         # Integración de propiedades + grilla espacial
│   ├── 02_modelo_satisfaccion.py    # Modelo baseline Random Forest
│   ├── 03_autocorrelacion_residuos.py  # Análisis Moran's I de residuos
│   ├── 04_validar_autocorrelacion.py   # Test de permutación estadístico
│   ├── 05_implementar_gwrf.py       # GWRF + Stacking (meta-modelo)
│   ├── 06_modelo_satisfaccion_mejorado.py  # Modelo con idx_habitabilidad
│   └── 08_visualizaciones_cartograficas.py # Mapas y gráficos finales
│
├── scripts_obsoletos/               # Scripts deprecados (respaldo)
│   ├── 00_imputar_valores_faltantes.py
│   ├── 04_integrar_datos_y_reentrenar.py
│   └── 07_visualizaciones_completas.py
│
├── data/                            # Datos procesados
│   ├── propiedades_con_factores_espaciales.csv
│   ├── propiedades_con_factores_espaciales.geojson
│   └── propiedades_con_residuos.geojson
│
├── resultados/                      # Resultados del análisis
│   ├── autocorrelacion_residuos.csv
│   ├── test_autocorrelacion.csv
│   ├── distribucion_permutaciones.csv
│   ├── gwrf/                        # Resultados GWRF + Stacking
│   └── modelo_mejorado/             # Modelo con satisfacción compuesta
│
├── graficos/                        # Visualizaciones generadas
│   ├── mapa_01_ubicacion_area_estudio.png
│   ├── mapa_02_precio_m2.png
│   ├── mapa_03_resultado_analisis.png
│   ├── grafico_01-05_*.png
│   ├── mapa_interactivo.html
│   └── reporte_eda.json
│
├── modelos/                         # Modelos entrenados (.pkl)
├── venv_viz/                        # Entorno virtual Python
├── ejecutar_pipeline.sh             # Script de ejecución automatizada
└── README.md                        # Este archivo
```

## 📊 Flujo del Pipeline

```
01_integrar_datos → 02_modelo → 03_moran → 04_test → 05_gwrf → 06_mejorado → 08_viz
```

## 📈 Resultados Principales

| Modelo | R² Test | Target |
|--------|---------|--------|
| Original | 0.22 | precio_m2 |
| **Mejorado** | **0.55** | satisfaccion_compuesta |

**Mejora: +150%**

### Features más importantes
1. `superficie_util` (26%)
2. `idx_vida_urbana` (17%)
3. `idx_habitabilidad_global` (14%)
4. `acc_transporte` (10%)

## ✅ Requisitos Cumplidos

- ✅ 3 mapas temáticos con elementos cartográficos
- ✅ 5 gráficos estadísticos
- ✅ 1 visualización interactiva (Folium)
- ✅ Datos organizados
- ✅ Código documentado
- ✅ Análisis EDA completo

## 🔧 Dependencias

```bash
source venv_viz/bin/activate
# Ya incluye: pandas, numpy, geopandas, sklearn, matplotlib, seaborn, folium, plotly
```

## 👤 Autor
Felipe Baeza - Proyecto GeoInformática - Noviembre 2025
