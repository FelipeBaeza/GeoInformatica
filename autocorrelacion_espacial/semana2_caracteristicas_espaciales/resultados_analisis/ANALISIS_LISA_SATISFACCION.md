# Análisis LISA - Factores de Satisfacción Residencial

## Pregunta de Investigación

**¿Los factores que harían que una propiedad sea más satisfactoria están ligados a una zona/comuna en particular?**

---

## Resumen Ejecutivo

| Métrica | Valor |
|---------|-------|
| Moran's I Global | 0.9749 |
| Z-score | 83.6803 |
| P-value | 0.0000 |
| Autocorrelación | **SIGNIFICATIVA** |
| Tipo | Positiva |

---

## Respuesta a la Pregunta de Investigación

### ✅ SÍ, existe una correlación espacial significativa

Los factores de satisfacción **están fuertemente ligados a zonas geográficas específicas**. Esto significa que:

1. **Las propiedades con alta satisfacción se agrupan** en ciertas zonas (Hot Spots)
2. **Las propiedades con baja satisfacción también se agrupan** en otras zonas (Cold Spots)
3. **La ubicación es un determinante clave** de la satisfacción potencial

---

## Distribución de Clústeres LISA

| Clúster | Propiedades | Porcentaje | Interpretación |
|---------|-------------|------------|----------------|
| No Significativo | 801 | 45.98% | Sin patrón definido |
| Low-Low | 510 | 29.28% | Cold Spots - Zonas con potencial |
| High-High | 429 | 24.63% | Hot Spots - Zonas premium |
| Low-High | 2 | 0.11% | Outliers negativos |

---

## Análisis por Comuna

### Santiago

| Métrica | Valor |
|---------|-------|
| Total propiedades | 486 |
| Satisfacción promedio | 0.6842 |
| % Hot Spots (High-High) | 72.2% |
| % Cold Spots (Low-Low) | 0.6% |
| Índice Accesibilidad | 0.8623 |
| Índice Densidad | 3.0683 |
| Índice Diversidad | 7.7663 |

### Estación Central

| Métrica | Valor |
|---------|-------|
| Total propiedades | 272 |
| Satisfacción promedio | 0.4942 |
| % Hot Spots (High-High) | 11.4% |
| % Cold Spots (Low-Low) | 1.1% |
| Índice Accesibilidad | 0.8078 |
| Índice Densidad | 1.4516 |
| Índice Diversidad | 6.4359 |

### Ñuñoa

| Métrica | Valor |
|---------|-------|
| Total propiedades | 195 |
| Satisfacción promedio | 0.4444 |
| % Hot Spots (High-High) | 24.1% |
| % Cold Spots (Low-Low) | 6.2% |
| Índice Accesibilidad | 0.7652 |
| Índice Densidad | 1.4213 |
| Índice Diversidad | 5.5812 |

### La Reina

| Métrica | Valor |
|---------|-------|
| Total propiedades | 33 |
| Satisfacción promedio | 0.4033 |
| % Hot Spots (High-High) | 0.0% |
| % Cold Spots (Low-Low) | 15.2% |
| Índice Accesibilidad | 0.7286 |
| Índice Densidad | 0.9858 |
| Índice Diversidad | 5.4545 |

---

## Implicaciones para la Recomendación de Propiedades

1. **Usuarios que priorizan servicios cercanos**: Recomendar propiedades en zonas High-High (Hot Spots)
2. **Usuarios con presupuesto limitado**: Considerar zonas Low-Low con potencial de desarrollo
3. **La comuna es un proxy de satisfacción**: Santiago y Ñuñoa tienden a tener mejores indicadores

---

*Análisis generado automáticamente - Proyecto GeoInformática*
*Fecha: 2025-12-20 20:46*