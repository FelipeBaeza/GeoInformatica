# TerraMatch - Pipeline de Análisis Geoespacial

Sistema de Recomendación Inmobiliaria basado en Machine Learning y Análisis Espacial.

## Ejecución Rápida

```bash
# Desde la carpeta autocorrelacion_espacial/
cd /home/felipe/Documentos/GeoInformatica/autocorrelacion_espacial

# Ejecutar pipeline completo
python ejecutar_pipeline_completo.py
```

## ¿Qué hace cada semana?

### Semana 1: Preparación de Datos
- Recolecta 29 datasets de servicios urbanos
- Filtra datos de 4 comunas (Santiago, Ñuñoa, La Reina, Estación Central)
- Normaliza coordenadas a UTM 19S (EPSG:32719)
- Valida calidad geométrica y elimina duplicados

### Semana 2: Características Espaciales
- Genera grilla de ~3,149 puntos (200m espaciado)
- Calcula distancias a 21 categorías de servicios
- Calcula densidades en buffers de 300m, 600m y 1000m
- Crea índices de accesibilidad compuestos

### Semana 3: Modelo y Visualizaciones
- Integra propiedades con características espaciales (matching)
- Entrena modelo LightGBM (R² = 0.86)
- Genera 3 mapas temáticos y 5 gráficos estadísticos
- Crea mapa interactivo HTML

## Opciones de Ejecución

Al ejecutar el pipeline, selecciona:

| Opción | Descripción |
|--------|-------------|
| 1 | Solo Semana 1 (Preparación) |
| 2 | Solo Semana 2 (Características) |
| 3 | Solo Semana 3 (Modelo) |
| 4 | Pipeline completo (1→2→3) |
| 5 | Semanas 2 y 3 (datos ya preparados) |

## Dependencias

```bash
pip install geopandas pandas numpy scipy shapely scikit-learn matplotlib seaborn folium lightgbm
```

## Resultados

Después de ejecutar el pipeline:

```
autocorrelacion_espacial/
├── semana1_preparacion_datos/
│   └── datos_normalizados/     # Datos limpios
├── semana2_caracteristicas_espaciales/
│   └── features/               # Grilla con métricas espaciales
└── semana3_modelo_satisfaccion/
    ├── modelos/                # Modelo LightGBM entrenado
    ├── graficos/               # Mapas y gráficos
    └── resultados/             # Predicciones de satisfacción
```

## Ver Resultados

```bash
# Abrir mapa interactivo
xdg-open semana3_modelo_satisfaccion/graficos/mapa_interactivo.html

# Ver gráficos
ls semana3_modelo_satisfaccion/graficos/
```

## Métricas del Modelo

- **R²**: 0.8635 (86% precisión)
- **Moran's I**: 0.0695 (baja autocorrelación)
- **Propiedades**: 7,702
- **Features espaciales**: ~40 variables
