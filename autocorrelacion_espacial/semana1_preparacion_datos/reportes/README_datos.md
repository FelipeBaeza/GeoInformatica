# Diccionario de Datos - Proyecto Valoración Inmobiliaria

##  Resumen del Proyecto
**Nombre:** Sistema de Valoración Inmobiliaria Geoespacial con Satisfacción Personalizada  
**Versión:** 1.0.0  
**Fecha:** 2025-12-23  
**CRS del Proyecto:** EPSG:32719 (UTM 19S)

##  Estadísticas Generales
- **Total archivos:** 0
- **Total geometrías:** 0
- **Área de estudio:** Gran Santiago - Comunas seleccionadas

##  Estructura de Carpetas
```
autocorrelacion_espacial/
 datos_filtrados/          # Datos originales (NO USAR para análisis)
 datos_normalizados/       # Datos listos para análisis (USAR ESTOS)
 features/                 # Features derivadas (se generarán)
 scripts/                  # Scripts de procesamiento
 reportes/                # Reportes de calidad y validación
```

##  Categorías de Datos

### Transporte y Movilidad
- Lineas_de_metro_de_Santiago.geojson
- estaciones_carga_filtradas.geojson
- circuito_turistico_filtrado.geojson

### Servicios y Amenidades  
- servicios_filtrados.geojson
- tiendas_filtradas.geojson
- puntos_de_interes_filtrados.geojson
- puntos_medicos_farmacias_hospitales_filtrados.geojson
- redes_de_clinicas_filtradas.geojson
- atracciones_turisticas_filtradas.geojson

### Educación
- establecimientos_educacion_escolar.geojson
- establecimientos_educacion_superior.geojson
- establecimientos_parvularia_filtrados.geojson

### Seguridad
- cuarteles_filtrados.geojson
- cuerpos_de_bomberos_filtrados.geojson
- unidades_operativas_pdi_filtradas.geojson

### Ambiente y Espacios
- areas_verdes_filtradas.geojson
- ocio_filtrado.geojson
- estaciones_calidad_aire_filtrada.geojson
- estaciones_metereológicas_filtradas.geojson
- vertedores_ilegales_filtrados.geojson

### Límites Administrativos
- comunas_buffer.geojson
- unidades_vecinales_filtradas.geojson
- poblacion_filtrada.geojson
- poblacion2_filtrada.geojson
- municipios_filtrados.geojson

### Socioeconómico
- delincuencia_comunas_anual.geojson
- campamentos.geojson
- centros_sernam_filtrados.geojson
- base_maestra_comunas_filtradas.geojson

##  Campos Agregados Durante Normalización
- **area_m2:** Área en metros cuadrados (solo polígonos)
- **perimeter_m:** Perímetro en metros (solo polígonos)
- **centroide_x:** Coordenada X del centroide (UTM)
- **centroide_y:** Coordenada Y del centroide (UTM)
- **crs_original:** CRS original antes de normalización
- **fecha_normalizacion:** Timestamp del procesamiento

##  Notas Importantes

### USAR SIEMPRE datos_normalizados/
-  **SÍ:** `gdf = gpd.read_file('datos_normalizados/servicios_filtrados.geojson')`
-  **NO:** `gdf = gpd.read_file('datos_filtrados/servicios_filtrados.geojson')`

### CRS y Métricas
- Todos los archivos están en **EPSG:32719** (UTM 19S)
- Distancias y áreas ya están en **metros** y **metros cuadrados**
- NO calcular `.distance()` o `.buffer()` en WGS84 (grados)

### Campos Clave por Archivo
- **OSM:** Usar `osm_id2` como ID único
- **Censo:** Usar `GEOCODIGO` para joins
- **Áreas:** Campo `area_m2` calculado correctamente
- **Coordenadas:** `centroide_x`, `centroide_y` para análisis

##  Próximos Pasos
1. **Completar estaciones metro:** Descargar desde OSM
2. **Generar features espaciales:** Scripts en desarrollo
3. **Validar con datos de propiedades:** Cuando estén disponibles

##  Contacto
Ver reportes detallados en `reportes/` para más información técnica.
