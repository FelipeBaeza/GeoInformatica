# Guia Rapida de Deployment

Guia paso a paso para levantar el proyecto con Docker Compose y migrar los datos.

## Prerequisitos

- Docker Desktop instalado y ejecutandose
- Puertos libres: 3000, 5432, 8000

## Deployment en 6 Pasos

### 1. Levantar servicios

```bash
cd GeoInformatica
docker compose up -d --build
```

Esperar ~2-3 minutos hasta que todos esten `healthy`:

```bash
docker compose ps
```

### 2. Habilitar PostGIS

```bash
docker exec geoinformatica-db psql -U postgres -d inmobiliaria_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"
docker exec geoinformatica-db psql -U postgres -d inmobiliaria_db -c "CREATE EXTENSION IF NOT EXISTS postgis_topology;"
```

### 3. Insertar comunas

```bash
docker exec geoinformatica-db psql -U postgres -d inmobiliaria_db -c "INSERT INTO comunas (nombre) VALUES ('Santiago'), ('Ñuñoa'), ('La Reina'), ('Estación Central') ON CONFLICT (nombre) DO NOTHING;"
```

### 4. Cargar propiedades

```bash
docker exec -it geoinformatica-backend python scripts/cargar_propiedades_geojson.py
```

Resultado: 8,051 propiedades insertadas

### 5. Cargar puntos de interes

```bash
docker exec -it geoinformatica-backend python scripts/cargar_servicios.py
```

Resultado: 2,801 POIs insertados

### 6. Verificar

```bash
docker exec geoinformatica-db psql -U postgres -d inmobiliaria_db -c "SELECT (SELECT COUNT(*) FROM comunas) as comunas, (SELECT COUNT(*) FROM propiedades) as propiedades, (SELECT COUNT(*) FROM puntos_interes) as pois;"
```

Resultado esperado:
```
 comunas | propiedades | pois
---------+-------------+------
       4 |        8051 | 2801
```

## URLs de Acceso

| Servicio | URL |
|----------|-----|
| Frontend | http://localhost:3000 |
| Backend API | http://localhost:8000 |
| Swagger Docs | http://localhost:8000/docs |

## Comandos Comunes

```bash
# Ver logs
docker compose logs -f

# Reiniciar servicio
docker compose restart backend

# Detener todo
docker compose down

# Eliminar y recrear (borra datos)
docker compose down -v
docker compose up -d --build
```

## Credenciales BD

```
Host: localhost:5432
DB: inmobiliaria_db
User: postgres
Pass: geo_pass
```
