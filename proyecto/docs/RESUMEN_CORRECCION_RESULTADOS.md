# Resumen de Correcciones: Sección "Resultados Preliminares"

## Objetivo
Corregir la sección de Resultados Preliminares del informe académico (`informe_v1.tex`) para asegurar consistencia con la implementación real del proyecto TerraMatch.

## Problemas Identificados y Corregidos

### 1. Referencias a Modelos No Implementados
**Problema:** El texto original mencionaba modelos exploratorios (GWRF, XGBoost, CatBoost, Neural Networks) como si fueran parte del sistema final.

**Solución:** Se clarificó que solo Random Forest fue usado como **línea base de comparación** y LightGBM es el **modelo final implementado**.

### 2. Métricas Incorrectas o Incompletas
**Problema:** 
- LightGBM aparecía con métricas vacías
- Confusión entre unidades (UF/m² vs escala 1-10 de satisfacción)
- Referencias a métricas de modelos exploratorios

**Solución:**
Se actualizaron las métricas correctas del modelo final:
- R² = 0.8635
- RMSE = 0.3357 (en escala de satisfacción 1-10)
- MAE = 0.2661
- CV R² = 0.8650 ± 0.0078

### 3. Falta de Estadísticas Descriptivas Reales
**Problema:** No se presentaban estadísticas concretas del dataset analizado.

**Solución:** Se agregaron estadísticas descriptivas basadas en los datos reales:

#### Composición del Dataset
- **Total:** 7,702 propiedades
- **Departamentos:** 5,135 (66.7%)
- **Casas:** 2,567 (33.3%)
- **Comunas:** La Reina, Estación Central, Ñuñoa, Santiago

#### Índices de Accesibilidad (escala 1-10)
| Dimensión    | Media | Desv. Est. | Mín  | Máx  |
|--------------|-------|------------|------|------|
| Educación    | 5.79  | 0.95       | 0.95 | 8.33 |
| Salud        | 4.36  | 2.17       | 0.00 | 10.00|
| Transporte   | 3.60  | 2.66       | 0.00 | 10.00|
| Entorno      | 4.68  | 1.90       | 0.43 | 10.00|
| Seguridad    | 3.74  | 2.02       | 0.00 | 10.00|
| Comercial    | 2.00  | 1.65       | 0.00 | 10.00|

### 4. Estructura Narrativa Mejorada
**Cambios realizados:**
- Organización en 4 subsecciones claras:
  1. Análisis Exploratorio de Datos (EDA)
  2. Estadísticas Descriptivas Espaciales
  3. Primeras Visualizaciones
  4. Patrones Identificados y Selección del Modelo

- Flujo narrativo coherente: EDA → Características espaciales → Visualizaciones → Decisión del modelo

## Archivos Modificados

### 1. informe_v1.tex
**Ubicación:** `/home/felipe/Documentos/GeoInformatica/proyecto/docs/informe_v1.tex`

**Secciones actualizadas:**
- Líneas 947-1050: Sección completa de "Resultados Preliminares"
- Subsección 5.1: Análisis Exploratorio de Datos
- Subsección 5.2: Estadísticas Descriptivas Espaciales
- Subsección 5.3: Primeras Visualizaciones
- Subsección 5.4: Patrones Identificados y Selección del Modelo

### 2. Script de Demostración Creado
**Ubicación:** `/home/felipe/Documentos/GeoInformatica/proyecto/scripts/generar_estadisticas_preliminares.py`

**Funcionalidades:**
- Carga datos de `propiedades_venta_con_satisfaccion.csv`
- Carga métricas de `metricas_modelo_venta.json`
- Genera todas las estadísticas reportadas en el informe
- Calcula composición del dataset por tipo y comuna
- Analiza índices de accesibilidad espacial
- Muestra métricas del modelo final
- Compara LightGBM con línea base Random Forest

**Ejecución:**
```bash
cd /home/felipe/Documentos/GeoInformatica/proyecto
python3 scripts/generar_estadisticas_preliminares.py
```

## Resultados del Script de Demostración

El script genera un reporte completo con:

### Estadísticas del Dataset
- 7,702 propiedades totales
- Distribución: 66.7% departamentos, 33.3% casas
- 4 comunas analizadas
- Precio promedio: 6,619.81 UF
- Precio/m² promedio: 67.93 UF/m²
- Superficie promedio: 97.5 m²

### Métricas del Modelo LightGBM
```
R² (Test Set): 0.8635
RMSE: 0.3357
MAE: 0.2661
R² CV (5-fold): 0.8650 ± 0.0078
Características: 42
```

### Comparación con Baseline
```
Random Forest (baseline):
  R² = 0.8431, RMSE = 0.3598, MAE = 0.2813

LightGBM (final):
  R² = 0.8635, RMSE = 0.3357, MAE = 0.2661

Mejoras: +2.42% R², -6.70% RMSE, -5.40% MAE
```

## Validación de Consistencia

### ✅ Datos Alineados
- Todas las estadísticas provienen de archivos reales del proyecto
- Métricas extraídas de `metricas_modelo_venta.json`
- Estadísticas descriptivas calculadas de `propiedades_venta_con_satisfaccion.csv`

### ✅ Modelo Final Correcto
- Solo se menciona LightGBM como modelo final
- Random Forest solo como línea base de comparación
- Eliminadas referencias a GWRF, XGBoost, CatBoost, Neural Networks

### ✅ Unidades Correctas
- Satisfacción: escala 1-10
- Precios: UF y UF/m²
- Distancias: metros
- Índices de accesibilidad: escala 1-10

### ✅ Narrativa Coherente
- Proceso claro: EDA → Características → Visualizaciones → Selección
- Justificación de decisiones basada en datos
- Métricas interpretadas en contexto

## Impacto en el Informe

La sección de Resultados Preliminares ahora:
1. **Es factualmente correcta:** Todas las cifras provienen de datos reales
2. **Es consistente:** Solo menciona el modelo realmente implementado
3. **Es completa:** Incluye EDA, estadísticas espaciales, visualizaciones y justificación
4. **Es reproducible:** El script permite verificar todos los números
5. **Es académicamente rigurosa:** Presenta datos, análisis, interpretación y decisiones

## Próximos Pasos (Opcional)

Si se desea mayor profundidad, se podría:
1. Agregar análisis de correlación específicos entre variables
2. Incluir tests estadísticos (Shapiro-Wilk, Kolmogorov-Smirnov)
3. Documentar el proceso de selección de hiperparámetros
4. Agregar análisis de residuos del modelo
5. Incluir gráficos de distribución por comuna

## Notas Importantes

- **Script funcional:** El script de demostración ejecuta correctamente y genera las estadísticas
- **Datos verificados:** Todas las métricas han sido verificadas contra los archivos JSON
- **LaTeX compilable:** El documento informe_v1.tex mantiene su estructura compilable
- **Sin referencias ficticias:** Se eliminaron todas las métricas inventadas o exploratorias

---

**Fecha de actualización:** 2025
**Proyecto:** TerraMatch - Sistema de Recomendación Inmobiliaria
**Archivo de referencia:** `informe_v1.tex` (Sección 5: Resultados Preliminares)
