# 🚀 Guía de Deployment - Backend GeoInformática

> **Instrucciones verificadas y funcionales para levantar el proyecto completo**  
> Última actualización: 23 de diciembre de 2025

---

## 📋 Prerequisitos

- Docker y Docker Compose instalados
- Puertos disponibles: `5432` (PostgreSQL), `8000` (Backend), `3000` (Frontend)
- ~2 GB de espacio en disco para datos

---

## 🏗️ Paso 1: Levantar el Stack con Docker Compose

```bash
# Desde el directorio raíz del proyecto
cd nombreDirectorioRaiz

# Construir y levantar todos los servicios
sudo docker compose up -d --build

# Verificar que los 3 servicios estén "healthy"
sudo docker compose ps
```

**Salida esperada:**
```
NAME                      STATUS
geoinformatica-db         Up 5 minutes (healthy)
geoinformatica-backend    Up 5 minutes (healthy)
geoinformatica-frontend   Up 5 minutes (healthy)
```

⏱️ **Tiempo estimado:** 2-3 minutos

---

## 🗄️ Paso 2: Verificar y Crear Base de Datos

```bash
# Crear base de datos si no existe
sudo docker exec geoinformatica-db psql -U postgres -c "CREATE DATABASE inmobiliaria_db;"

# Habilitar PostGIS
sudo docker exec geoinformatica-db psql -U postgres -d inmobiliaria_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"
sudo docker exec geoinformatica-db psql -U postgres -d inmobiliaria_db -c "CREATE EXTENSION IF NOT EXISTS postgis_topology;"

# Cambiar contraseña de postgres (si es necesario)
sudo docker exec geoinformatica-db psql -U postgres -c "ALTER USER postgres WITH PASSWORD 'geo_pass';"

# Reiniciar backend para que cree las tablas
sudo docker restart geoinformatica-backend
```

**Verificar conexión:**
```bash
curl http://localhost:8000/api/v1/health
```

Debe mostrar: `"database":"✅ Conectada"`

---

## 📍 Paso 3: Insertar Comunas Base

Las comunas son **requeridas** antes de cargar propiedades (relación `comuna_id`).

```bash
docker exec -i geoinformatica-db psql -U postgres -d inmobiliaria_db <<EOF
INSERT INTO comunas (nombre) VALUES 
    ('Santiago'), 
    ('Ñuñoa'), 
    ('La Reina'), 
    ('Estación Central')
ON CONFLICT (nombre) DO NOTHING;
EOF
```

**Salida esperada:**
```
INSERT 0 4
```

**Verificar:**
```bash
docker exec -it geoinformatica-db psql -U postgres -d inmobiliaria_db -c \
  "SELECT id, nombre FROM comunas ORDER BY id;"
```

---

## 🏠 Paso 4: Cargar Propiedades (8,051 registros)

Carga las propiedades desde los 8 archivos GeoJSON en `datos_nuevos/DATOS_FILTRADOS/`.

```bash
docker exec -it geoinformatica-backend python scripts/cargar_propiedades_geojson.py
```

**Salida esperada:**
```
======================================================================
🏠 CARGA DE PROPIEDADES DESDE GEOJSON
======================================================================
✅ Conectado a geoinformatica-db:5432/inmobiliaria_db

🗑️  Limpiando propiedades existentes...
   Propiedades después de limpiar: 0

📍 Configurando comunas...
   Comunas disponibles: 9

📂 Directorio de datos: datos_nuevos/DATOS_FILTRADOS

📁 Archivos GeoJSON encontrados: 8
   ✅ departamentos_la_reina.geojson: 245/245 insertados
   ✅ departamentos_Santiago.geojson: 1337/1337 insertados
   ✅ departamentos_estacion_central.geojson: 1879/1879 insertados
   ✅ casas_nunoa.geojson: 802/802 insertados
   ✅ casas_estacon_central.geojson: 220/220 insertados
   ✅ departamentos_nunoa.geojson: 1853/1853 insertados
   ✅ casas_Santiago.geojson: 463/463 insertados
   ✅ casas_la_reina.geojson: 1252/1252 insertados

======================================================================
📊 RESUMEN DE CARGA
======================================================================
   Total features en archivos: 8051
   ✅ Propiedades insertadas: 8051
   ❌ Errores: 0

🏠 Total en base de datos: 8051

📍 Distribución por comuna:
   Ñuñoa: 2655
   Estación Central: 2099
   Santiago: 1800
   La Reina: 1497

======================================================================
✅ CARGA COMPLETADA
======================================================================
```

⏱️ **Tiempo estimado:** 15-30 segundos

---

## 🗺️ Paso 5: Cargar Puntos de Interés (2,801 servicios)

Carga servicios (metro, colegios, hospitales, supermercados, etc.) desde `datos_normalizados/`.

```bash
docker exec -it geoinformatica-backend python scripts/cargar_servicios.py
```

**Salida esperada:**
```
================================================================================
CARGA DE PUNTOS DE INTERÉS (SERVICIOS)
================================================================================

Directorio de datos: /app/datos_normalizados/datos_normalizados

Conectando a la base de datos...
✓ Conexión establecida

Limpiando tabla puntos_interes...
✓ Tabla limpiada

================================================================================
PROCESANDO ARCHIVOS GEOJSON
================================================================================

Procesando: establecimientos_educacion_escolar.geojson
  - Features encontrados: 458
  ✓ Insertados: 458 registros

Procesando: establecimientos_educacion_superior.geojson
  - Features encontrados: 363
  ✓ Insertados: 363 registros

Procesando: Estaciones_metro_Santiago.geojson
  - Features encontrados: 120
  ✓ Insertados: 120 registros

[... más archivos ...]

Procesando: tiendas_filtradas.geojson
  - Features encontrados: 497
  ✓ Insertados: 497 registros

Procesando: servicios_filtrados.geojson
  - Features encontrados: 773
  ✓ Insertados: 773 registros

================================================================================
RESUMEN
================================================================================
Archivos procesados:     14
Total POIs insertados:   2801
================================================================================

✓ Carga completada exitosamente
```

⚠️ **Notas sobre errores esperados:**
- `Lineas_de_metro_de_Santiago.geojson`: Falla porque tiene LineStrings, no Points (no afecta, tenemos las estaciones).
- `areas_verdes_filtradas.geojson` y `ocio_filtrado.geojson`: Algunos tienen geometrías Polygon en vez de Point (omitidos, no crítico).

⏱️ **Tiempo estimado:** 30-60 segundos

---

## ✅ Paso 6: Verificar Datos Cargados

### Resumen General
```bash
docker exec -it geoinformatica-db psql -U postgres -d inmobiliaria_db -c "
SELECT 
    (SELECT COUNT(*) FROM comunas) as comunas,
    (SELECT COUNT(*) FROM propiedades) as propiedades,
    (SELECT COUNT(*) FROM puntos_interes) as pois;
"
```

**Salida esperada:**
```
 comunas | propiedades | pois 
---------+-------------+------
       4 |        8051 | 2801
```

### Distribución por Comuna
```bash
docker exec -it geoinformatica-db psql -U postgres -d inmobiliaria_db -c "
SELECT 
    c.nombre, 
    COUNT(p.id) as total_propiedades,
    ROUND(AVG(p.precio)::numeric, 0) as precio_promedio_uf
FROM comunas c
LEFT JOIN propiedades p ON c.id = p.comuna_id
GROUP BY c.nombre
ORDER BY total_propiedades DESC;
"
```

### Tipos de Puntos de Interés
```bash
docker exec -it geoinformatica-db psql -U postgres -d inmobiliaria_db -c "
SELECT tipo, COUNT(*) as total
FROM puntos_interes
GROUP BY tipo
ORDER BY total DESC
LIMIT 15;
"
```

**Salida esperada:**
```
       tipo        | total
-------------------+-------
 servicio          |   773
 comercio          |   497
 colegio           |   914
 universidad       |   363
 metro             |   120
 centro_medico     |    94
 comisaria         |    38
 bombero           |     2
```

### Verificar Geometrías PostGIS
```bash
docker exec -it geoinformatica-db psql -U postgres -d inmobiliaria_db -c "
SELECT 
    COUNT(*) as total_propiedades,
    COUNT(geometria) as con_geometria,
    COUNT(geometria) * 100.0 / COUNT(*) as porcentaje
FROM propiedades;
"
```

**Todas las propiedades deben tener geometría (100%).**

---

## 🌐 Paso 7: Probar la API

### Backend (FastAPI)
```bash
# Abrir en navegador o curl
curl http://localhost:8000/api/v1/health

# Documentación interactiva
xdg-open http://localhost:8000/docs  # Linux
```

**Endpoints disponibles:**
- `GET /api/v1/health` - Health check
- `GET /api/v1/propiedades` - Listar propiedades
- `GET /api/v1/comunas/stats` - Estadísticas por comuna
- `POST /api/v1/ml/recomendaciones` - Recomendaciones ML
- `POST /api/v1/prediccion/precio` - Predicción de precio
- `POST /api/v1/satisfaccion/predecir` - Satisfacción de venta

### Frontend (Nuxt 3)
```bash
# Abrir en navegador
xdg-open http://localhost:3000
```

---

## 🛠️ Comandos Útiles

### Ver logs en tiempo real
```bash
# Backend
docker compose logs -f backend

# Frontend
docker compose logs -f frontend

# Base de datos
docker compose logs -f db
```

### Reiniciar un servicio
```bash
docker compose restart backend
docker compose restart frontend
```

### Acceso directo a PostgreSQL
```bash
# Conectar a la BD
docker exec -it geoinformatica-db psql -U postgres -d inmobiliaria_db

# Dentro de psql:
\dt              # Listar tablas
\d propiedades   # Describir tabla
\q               # Salir
```

### Limpiar y reiniciar desde cero
```bash
# Detener y eliminar todo (incluyendo volúmenes)
docker compose down -v

# Levantar nuevamente
docker compose up -d --build

# Volver a cargar datos (Pasos 3-5)
```

---

## 🔧 Troubleshooting

### Error: "Cannot connect to database"
```bash
# Verificar que la DB esté corriendo
docker compose ps db

# Ver logs de la DB
docker compose logs db

# Reiniciar la DB
docker compose restart db
```

### Error: "Port 5432 already in use"
Tienes PostgreSQL corriendo en tu host. Opciones:
1. Detener PostgreSQL local: `sudo systemctl stop postgresql`
2. Cambiar puerto en `docker-compose.yml`: `"5433:5432"`

### Error: "No such file or directory: datos_nuevos/DATOS_FILTRADOS"
```bash
# Verificar que exista el directorio
ls datos_nuevos/DATOS_FILTRADOS/

# Si está en otro lado, ajustar volúmenes en docker-compose.yml
```

### Error: "Module 'psycopg2' not found" (al ejecutar desde host)
```bash
# Instalar dependencias en venv local
cd geo-proyect-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Ver salud de contenedores
```bash
docker compose ps
docker inspect geoinformatica-backend | grep -i health
```

---

## 📊 Resumen de Datos Cargados

| Entidad           | Total  | Fuente                                  |
|-------------------|--------|-----------------------------------------|
| **Comunas**       | 4      | Insercción manual (SQL)                 |
| **Propiedades**   | 8,051  | `datos_nuevos/DATOS_FILTRADOS/*.geojson`|
| **Puntos Interés**| 2,801  | `datos_normalizados/*.geojson`          |

### Desglose de Propiedades
- **Ñuñoa:** 2,655 (33%)
- **Estación Central:** 2,099 (26%)
- **Santiago:** 1,800 (22%)
- **La Reina:** 1,497 (19%)

### Desglose de Servicios
- Educación (colegios, universidades, párvulos): 1,277
- Comercio y servicios: 1,270
- Transporte (metro): 120
- Salud (centros médicos, clínicas): 94
- Seguridad (comisarías, bomberos): 40

---

## 🎯 Próximos Pasos

1. **Modelo ML de Satisfacción:** Verificar que existe `modelos/modelo_satisfaccion_venta.pkl`
2. **Calcular Distancias:** Ejecutar análisis espacial para poblar campos `dist_*_m`
3. **Análisis Semana 3:** Integrar datos de autocorrelación espacial (LISA, submercados)
4. **Frontend:** Probar mapa interactivo y filtros

---

## 📞 Soporte

Si encuentras problemas:
1. Revisar logs: `docker compose logs -f`
2. Verificar que todos los archivos GeoJSON existan
3. Confirmar que PostgreSQL tenga extensión PostGIS: 
   ```sql
   SELECT PostGIS_version();
   ```

---

**✅ Deployment completado exitosamente**

Backend: http://localhost:8000  
Frontend: http://localhost:3000  
Docs API: http://localhost:8000/docs
