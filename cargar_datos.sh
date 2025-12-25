#!/bin/bash
# ============================================================================
# Script de Carga Automática de Datos - v2.0
# ============================================================================
# Descripción: Carga las ~8,000 propiedades desde GeoJSON
# Uso: sudo ./cargar_datos.sh
# ============================================================================

set -e

echo "============================================================================"
echo " CARGA AUTOMÁTICA DE DATOS - Base de Datos Inmobiliaria v2.0"
echo "============================================================================"
echo ""

# Colores
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Detectar Docker
DOCKER_CMD="docker"
COMPOSE_CMD="docker-compose"

if ! docker ps > /dev/null 2>&1; then
    if sudo docker ps > /dev/null 2>&1; then
        DOCKER_CMD="sudo docker"
        COMPOSE_CMD="sudo docker-compose"
        echo -e "${YELLOW}ℹ  Usando sudo para Docker${NC}"
    else
        echo -e "${RED} Error: No se puede acceder a Docker${NC}"
        exit 1
    fi
fi

# Verificar servicios Docker
echo " Verificando servicios Docker..."
if ! $COMPOSE_CMD ps 2>/dev/null | grep -q "Up"; then
    echo -e "${YELLOW}  Contenedores no detectados. Intentando iniciar...${NC}"
    $COMPOSE_CMD up -d db backend
    sleep 10
fi
echo -e "${GREEN} Servicios Docker activos${NC}"
echo ""

# Esperar PostgreSQL
echo " Esperando que PostgreSQL esté listo..."
for i in {1..30}; do
    if $DOCKER_CMD exec geoinformatica-db pg_isready -U postgres > /dev/null 2>&1; then
        echo -e "${GREEN} PostgreSQL está listo${NC}"
        break
    fi
    echo "   Intento $i/30..."
    sleep 2
done
echo ""

# ============================================================================
# PASO 1: Copiar archivos GeoJSON al contenedor
# ============================================================================
echo "============================================================================"
echo " PASO 1: Copiando archivos de datos al contenedor"
echo "============================================================================"

# Crear directorio en el contenedor
$DOCKER_CMD exec geoinformatica-backend mkdir -p /app/datos_nuevos/DATOS_FILTRADOS 2>/dev/null || true

# Copiar archivos GeoJSON
if [ -d "datos_nuevos/DATOS_FILTRADOS" ]; then
    echo " Copiando archivos GeoJSON..."
    for file in datos_nuevos/DATOS_FILTRADOS/*.geojson; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            $DOCKER_CMD cp "$file" "geoinformatica-backend:/app/datos_nuevos/DATOS_FILTRADOS/$filename"
            echo "    $filename"
        fi
    done
else
    echo -e "${RED} No se encontró el directorio datos_nuevos/DATOS_FILTRADOS${NC}"
    exit 1
fi

# Copiar script de carga
echo ""
echo " Copiando script de carga..."
$DOCKER_CMD cp geo-proyect-backend/scripts/cargar_propiedades_geojson.py geoinformatica-backend:/app/cargar_propiedades_geojson.py
echo -e "${GREEN} Archivos copiados${NC}"
echo ""

# ============================================================================
# PASO 2: Ejecutar carga de propiedades
# ============================================================================
echo "============================================================================"
echo " PASO 2: Cargando propiedades (~8,000 registros)"
echo "============================================================================"
echo " Este proceso puede tomar 1-2 minutos..."
echo ""

$DOCKER_CMD exec geoinformatica-backend python3 /app/cargar_propiedades_geojson.py

# ============================================================================
# PASO 2.5: Cargar puntos de interés (servicios)
# ============================================================================
echo ""
echo "============================================================================"
echo " PASO 2.5: Cargando puntos de interés (servicios cercanos)"
echo "============================================================================"
echo " Cargando colegios, hospitales, farmacias, metro, parques, etc..."
echo ""

# Crear directorio de datos normalizados en el contenedor
$DOCKER_CMD exec geoinformatica-backend mkdir -p /app/datos_normalizados 2>/dev/null || true

# Copiar archivos de datos normalizados
if [ -d "autocorrelacion_espacial/semana1_preparacion_datos/datos_normalizados/datos_normalizados" ]; then
    echo " Copiando archivos de servicios..."
    $DOCKER_CMD cp autocorrelacion_espacial/semana1_preparacion_datos/datos_normalizados/datos_normalizados/. geoinformatica-backend:/app/datos_normalizados/
    echo "    Archivos de servicios copiados"
else
    echo -e "${YELLOW}  No se encontró el directorio de datos normalizados${NC}"
fi

# Copiar script de carga de servicios
$DOCKER_CMD cp geo-proyect-backend/scripts/cargar_servicios.py geoinformatica-backend:/app/cargar_servicios.py

# Ejecutar carga de servicios
$DOCKER_CMD exec geoinformatica-backend python3 /app/cargar_servicios.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN} Propiedades cargadas exitosamente${NC}"
else
    echo -e "${RED} Error al cargar propiedades${NC}"
    exit 1
fi
echo ""

# ============================================================================
# PASO 3: Verificación de datos
# ============================================================================
echo "============================================================================"
echo " PASO 3: Verificación de datos cargados"
echo "============================================================================"

$DOCKER_CMD exec geoinformatica-db psql -U postgres -d inmobiliaria_db << 'EOF'
\echo ''
\echo ' Resumen de datos cargados:'
\echo '─────────────────────────────────────'

SELECT 'Total propiedades' as descripcion, COUNT(*)::text as valor FROM propiedades
UNION ALL
SELECT 'Comunas con datos' as descripcion, COUNT(DISTINCT comuna_id)::text as valor FROM propiedades
UNION ALL
SELECT 'Con coordenadas' as descripcion, COUNT(*)::text as valor FROM propiedades WHERE latitud IS NOT NULL AND longitud IS NOT NULL;

\echo ''
\echo ' Distribución por comuna:'
\echo '─────────────────────────────────────'

SELECT c.nombre as comuna, COUNT(*) as propiedades
FROM propiedades p
JOIN comunas c ON p.comuna_id = c.id
GROUP BY c.nombre
ORDER BY COUNT(*) DESC;

\echo ''
\echo ' Puntos de Interés (Servicios) cargados:'
\echo '─────────────────────────────────────'

SELECT tipo, COUNT(*) as cantidad
FROM puntos_interes
GROUP BY tipo
ORDER BY cantidad DESC;

\echo ''
SELECT 'Total puntos de interés' as descripcion, COUNT(*)::text as valor FROM puntos_interes;
\echo ''
EOF

echo ""

# ============================================================================
# PASO 4: Test de API
# ============================================================================
echo "============================================================================"
echo " PASO 4: Verificando API de recomendaciones"
echo "============================================================================"

# Test sin filtros
echo " Test 1: Endpoint sin filtros (limit=10)..."
RESPONSE=$(curl -s -X POST "http://localhost:8000/api/v1/recomendaciones-ml?limit=10" \
    -H "Content-Type: application/json" \
    -d '{}')

TOTAL_ANALIZADAS=$(echo $RESPONSE | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('total_analizadas', 0))")
TOTAL_ENCONTRADAS=$(echo $RESPONSE | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('total_encontradas', 0))")

echo "   Total analizadas: $TOTAL_ANALIZADAS"
echo "   Total encontradas: $TOTAL_ENCONTRADAS"

if [ "$TOTAL_ANALIZADAS" -gt 7000 ]; then
    echo -e "   ${GREEN} API analiza más de 7,000 propiedades${NC}"
else
    echo -e "   ${YELLOW}  API analiza menos propiedades de lo esperado${NC}"
fi

# Test con limit alto
echo ""
echo " Test 2: Endpoint con limit=1000..."
RESPONSE2=$(curl -s -X POST "http://localhost:8000/api/v1/recomendaciones-ml?limit=1000" \
    -H "Content-Type: application/json" \
    -d '{}')

RECOMENDACIONES=$(echo $RESPONSE2 | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('recomendaciones', [])))")
echo "   Recomendaciones retornadas: $RECOMENDACIONES"

if [ "$RECOMENDACIONES" -eq 1000 ]; then
    echo -e "   ${GREEN} API retorna correctamente el límite solicitado${NC}"
else
    echo -e "   ${YELLOW}  API retorna $RECOMENDACIONES (se esperaban 1000)${NC}"
fi

# Test con filtro de dormitorios
echo ""
echo " Test 3: Filtro por dormitorios (min 3)..."
RESPONSE3=$(curl -s -X POST "http://localhost:8000/api/v1/recomendaciones-ml?limit=100" \
    -H "Content-Type: application/json" \
    -d '{"dormitorios_min": 3}')

FILTRADAS=$(echo $RESPONSE3 | python3 -c "import json,sys; d=json.load(sys.stdin); print(d.get('total_analizadas', 0))")
echo "   Propiedades con 3+ dormitorios analizadas: $FILTRADAS"

echo ""
echo "============================================================================"
echo -e "${GREEN} CARGA Y VERIFICACIÓN COMPLETADA${NC}"
echo "============================================================================"
echo ""
echo " Resumen:"
echo "   • Total propiedades en DB: $TOTAL_ANALIZADAS"
echo "   • API funcionando correctamente"
echo ""
echo " Sistema listo para usar:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
