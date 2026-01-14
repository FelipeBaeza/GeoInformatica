# GeoInformatica - Sistema de Analisis Inmobiliario

Sistema de analisis geoespacial del mercado inmobiliario en Santiago de Chile con prediccion de precios y satisfaccion residencial usando Machine Learning.

## Arquitectura del Proyecto

```
GeoInformatica/
├── geo-proyect-backend/     # API REST con FastAPI + PostgreSQL/PostGIS
├── geo-proyect-frontend/    # Aplicacion web con Nuxt 3 + Vue 3 + Leaflet
├── datos_nuevos/            # Datos GeoJSON de propiedades
├── autocorrelacion_espacial/ # Scripts de analisis y datos normalizados
└── docker-compose.yml       # Orquestacion de servicios
```

## Tecnologias

| Componente | Tecnologia |
|------------|------------|
| Backend | FastAPI, SQLAlchemy, Pydantic |
| Frontend | Nuxt 3, Vue 3, Leaflet, Chart.js, TailwindCSS |
| Base de Datos | PostgreSQL 15 + PostGIS 3.3 |
| ML | Random Forest (precios), LightGBM (satisfaccion) |
| Contenedores | Docker, Docker Compose |

## Inicio Rapido con Docker

### Prerequisitos

- Docker Desktop instalado y ejecutandose
- Puertos disponibles: 3000, 5432, 8000
- ~2 GB de espacio en disco

### Paso 1: Levantar los servicios

```bash
# Clonar el repositorio
git clone <url-del-repositorio>
cd GeoInformatica

# Construir y levantar todos los servicios
docker compose up -d --build

# Verificar que los 3 servicios esten corriendo
docker compose ps
```

Esperar hasta que todos los contenedores muestren estado `healthy`:

```
NAME                      STATUS
geoinformatica-db         Up (healthy)
geoinformatica-backend    Up (healthy)
geoinformatica-frontend   Up (healthy)
```

### Paso 2: Habilitar PostGIS

```bash
docker exec geoinformatica-db psql -U postgres -d inmobiliaria_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"
docker exec geoinformatica-db psql -U postgres -d inmobiliaria_db -c "CREATE EXTENSION IF NOT EXISTS postgis_topology;"
```

### Paso 3: Insertar comunas base

```bash
docker exec geoinformatica-db psql -U postgres -d inmobiliaria_db -c "INSERT INTO comunas (nombre) VALUES ('Santiago'), ('Ñuñoa'), ('La Reina'), ('Estación Central') ON CONFLICT (nombre) DO NOTHING;"
```

### Paso 4: Cargar propiedades (8,051 registros)

```bash
docker exec -it geoinformatica-backend python scripts/cargar_propiedades_geojson.py
```

Salida esperada:
```
Total features en archivos: 8051
Propiedades insertadas: 8051
Errores: 0
```

### Paso 5: Cargar puntos de interes (2,801 registros)

```bash
docker exec -it geoinformatica-backend python scripts/cargar_servicios.py
```

Salida esperada:
```
Total POIs insertados: 2801
```

### Paso 6: Verificar la carga

```bash
docker exec geoinformatica-db psql -U postgres -d inmobiliaria_db -c "SELECT (SELECT COUNT(*) FROM comunas) as comunas, (SELECT COUNT(*) FROM propiedades) as propiedades, (SELECT COUNT(*) FROM puntos_interes) as pois;"
```

Resultado esperado:
```
 comunas | propiedades | pois
---------+-------------+------
       4 |        8051 | 2801
```

## Acceso a la Aplicacion

| Servicio | URL |
|----------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Documentacion API (Swagger) | http://localhost:8000/docs |
| Base de Datos | localhost:5432 |

### Credenciales de Base de Datos

```
Host: localhost
Puerto: 5432
Base de datos: inmobiliaria_db
Usuario: postgres
Contraseña: geo_pass
```

## Datos Cargados

### Propiedades por Comuna

| Comuna | Propiedades |
|--------|-------------|
| Ñuñoa | 2,655 |
| Estacion Central | 2,099 |
| Santiago | 1,800 |
| La Reina | 1,497 |
| **Total** | **8,051** |

### Puntos de Interes

| Tipo | Cantidad |
|------|----------|
| Educacion (colegios, universidades) | 1,277 |
| Comercio y servicios | 1,270 |
| Metro (estaciones) | 120 |
| Salud (centros medicos) | 94 |
| Seguridad (cuarteles, bomberos) | 40 |
| **Total** | **2,801** |

## Comandos Utiles

### Ver logs en tiempo real

```bash
# Todos los servicios
docker compose logs -f

# Solo backend
docker compose logs -f backend

# Solo frontend
docker compose logs -f frontend
```

### Reiniciar un servicio

```bash
docker compose restart backend
docker compose restart frontend
```

### Acceder a PostgreSQL

```bash
docker exec -it geoinformatica-db psql -U postgres -d inmobiliaria_db
```

### Detener todos los servicios

```bash
docker compose down
```

### Eliminar todo y empezar de cero

```bash
# Detener y eliminar volumenes (borra la base de datos)
docker compose down -v

# Levantar nuevamente
docker compose up -d --build
```

## Solucion de Problemas

### Error: "Port already in use"

Otro servicio esta usando el puerto. Opciones:
1. Detener el servicio que usa el puerto
2. Cambiar el puerto en `docker-compose.yml`

### Error: "Cannot connect to database"

```bash
# Verificar que la BD este corriendo
docker compose ps db

# Ver logs de la BD
docker compose logs db
```

### Error: "No such file or directory: datos_nuevos"

Verificar que el directorio existe y contiene los archivos GeoJSON:
```bash
ls datos_nuevos/DATOS_FILTRADOS/
```

### Contenedor no inicia

```bash
# Ver logs del contenedor
docker compose logs <servicio>

# Forzar reconstruccion
docker compose up -d --build --force-recreate
```

## Estructura de Datos

### Archivos GeoJSON de Propiedades

Ubicacion: `datos_nuevos/DATOS_FILTRADOS/`

- `casas_nunoa.geojson`
- `casas_la_reina.geojson`
- `casas_Santiago.geojson`
- `casas_estacon_central.geojson`
- `departamentos_nunoa.geojson`
- `departamentos_la_reina.geojson`
- `departamentos_Santiago.geojson`
- `departamentos_estacion_central.geojson`

### Archivos GeoJSON de Servicios

Ubicacion: `autocorrelacion_espacial/semana2_caracteristicas_espaciales/datos_normalizados/`

- `Estaciones_metro_Santiago.geojson`
- `establecimientos_educacion_escolar.geojson`
- `establecimientos_educacion_superior.geojson`
- `puntos_medicos_farmacias_hospitales_filtrados.geojson`
- `servicios_filtrados.geojson`
- `tiendas_filtradas.geojson`
- Y mas...

## Modelos de Machine Learning

### Modelo de Prediccion de Precios (Random Forest)

- **R² Score**: 0.914
- **Features**: 16 (superficie, dormitorios, baños, distancias, comuna)
- **Archivo**: `geo-proyect-backend/modelos/modelo_rf_optimizado.pkl`

### Modelo de Satisfaccion (LightGBM)

- **R² Score**: 0.87
- **Features**: 42 (fisicas, derivadas, distancias, comunas)
- **Archivo**: `geo-proyect-backend/modelos/modelo_satisfaccion_venta.pkl`

## Endpoints Principales de la API

| Metodo | Endpoint | Descripcion |
|--------|----------|-------------|
| GET | `/api/v1/health` | Estado del sistema |
| GET | `/api/v1/propiedades` | Listar propiedades |
| GET | `/api/v1/comunas/stats` | Estadisticas por comuna |
| POST | `/api/v1/prediccion/precio` | Predecir precio |
| POST | `/api/v1/satisfaccion/predecir` | Predecir satisfaccion |
| POST | `/api/v1/ml/recomendaciones` | Obtener recomendaciones |

## Licencia

MIT License - Proyecto educativo

## Autor

Felipe Baeza - Proyecto Geoinformatica
