#!/bin/bash
# ============================================================================
# Script de Carga Automática de Datos
# ============================================================================
# Descripción: Carga todas las tablas con datos iniciales
# Uso: ./cargar_datos.sh
# ============================================================================

set -e  # Salir si hay algún error

echo "============================================================================"
echo "🚀 CARGA AUTOMÁTICA DE DATOS - Base de Datos Inmobiliaria"
echo "============================================================================"
echo ""

# Colores para output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Verificar que Docker Compose esté corriendo
echo "📋 Verificando servicios Docker..."
if ! docker-compose ps | grep -q "Up"; then
    echo -e "${RED}❌ Error: Los contenedores no están corriendo${NC}"
    echo "   Ejecutar primero: docker-compose up -d"
    exit 1
fi
echo -e "${GREEN}✅ Servicios Docker activos${NC}"
echo ""

# Esperar que la base de datos esté lista
echo "⏳ Esperando que PostgreSQL esté listo..."
for i in {1..30}; do
    if docker exec geoinformatica-db pg_isready -U postgres > /dev/null 2>&1; then
        echo -e "${GREEN}✅ PostgreSQL está listo${NC}"
        break
    fi
    echo "   Intento $i/30..."
    sleep 2
done
echo ""

# Paso 1: Cargar estructura base (solo comunas desde SQL)
echo "============================================================================"
echo "📍 PASO 1: Cargando comunas (32 comunas de Santiago)"
echo "============================================================================"
docker exec -i geoinformatica-db psql -U postgres -d inmobiliaria_db << 'EOF'
INSERT INTO comunas (id, nombre) VALUES
(1, 'Cerrillos'), (2, 'Cerro Navia'), (3, 'Conchalí'), (4, 'El Bosque'),
(5, 'Estación Central'), (6, 'Huechuraba'), (7, 'Independencia'), (8, 'La Cisterna'),
(9, 'La Florida'), (10, 'La Granja'), (11, 'La Pintana'), (12, 'La Reina'),
(13, 'Las Condes'), (14, 'Lo Barnechea'), (15, 'Lo Espejo'), (16, 'Lo Prado'),
(17, 'Macul'), (18, 'Maipú'), (19, 'Ñuñoa'), (20, 'Pedro Aguirre Cerda'),
(21, 'Peñalolén'), (22, 'Providencia'), (23, 'Pudahuel'), (24, 'Quilicura'),
(25, 'Quinta Normal'), (26, 'Recoleta'), (27, 'Renca'), (28, 'San Joaquín'),
(29, 'San Miguel'), (30, 'San Ramón'), (31, 'Santiago'), (32, 'Vitacura')
ON CONFLICT (id) DO NOTHING;
SELECT COUNT(*) as comunas_cargadas FROM comunas;
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Comunas cargadas exitosamente${NC}"
else
    echo -e "${RED}❌ Error al cargar comunas${NC}"
    exit 1
fi
echo ""

# Paso 2: Verificar archivos necesarios
echo "============================================================================"
echo "📂 PASO 2: Verificando archivos de datos"
echo "============================================================================"

FILES_NEEDED=(
    "/app/clean_alquiler_02_11_2023cc.csv"
    "/app/datos_normalizados"
    "/tmp/grilla_con_densidades.geojson"
    "/app/cargar_propiedades_csv.py"
    "/app/cargar_grilla_densidades.py"
)

ALL_FILES_PRESENT=true
for file in "${FILES_NEEDED[@]}"; do
    if docker exec geoinformatica-backend test -e "$file" 2>/dev/null; then
        echo -e "${GREEN}✅${NC} $file"
    else
        echo -e "${RED}❌${NC} $file (NO ENCONTRADO)"
        ALL_FILES_PRESENT=false
    fi
done

if [ "$ALL_FILES_PRESENT" = false ]; then
    echo ""
    echo -e "${YELLOW}⚠️  Archivos faltantes detectados${NC}"
    echo "   Copiando archivos necesarios al contenedor..."
    
    docker cp clean_alquiler_02_11_2023cc.csv geoinformatica-backend:/app/ 2>/dev/null || \
        echo -e "${RED}   ❌ No se pudo copiar CSV${NC}"
    
    docker cp autocorrelacion_espacial/semana1_preparacion_datos/datos_normalizados geoinformatica-backend:/app/ 2>/dev/null || \
        echo -e "${RED}   ❌ No se pudieron copiar datos normalizados${NC}"
    
    docker cp autocorrelacion_espacial/semana2_caracteristicas_espaciales/features/grilla_con_densidades.geojson geoinformatica-backend:/tmp/ 2>/dev/null || \
        echo -e "${RED}   ❌ No se pudo copiar grilla${NC}"
    
    docker cp geo-proyect-backend/scripts/cargar_propiedades_csv.py geoinformatica-backend:/app/ 2>/dev/null || \
        echo -e "${RED}   ❌ No se pudo copiar script de propiedades${NC}"
    
    docker cp geo-proyect-backend/scripts/cargar_grilla_densidades.py geoinformatica-backend:/app/ 2>/dev/null || \
        echo -e "${RED}   ❌ No se pudo copiar script de grilla${NC}"
    
    echo -e "${GREEN}✅ Archivos copiados${NC}"
fi
echo ""

# Paso 3: Cargar propiedades
echo "============================================================================"
echo "🏠 PASO 3: Cargando propiedades (1,623 registros esperados)"
echo "============================================================================"
echo "⏳ Este proceso puede tomar 2-3 minutos..."
echo ""

docker exec geoinformatica-backend python3 /app/cargar_propiedades_csv.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Propiedades cargadas exitosamente${NC}"
else
    echo -e "${RED}❌ Error al cargar propiedades${NC}"
    exit 1
fi
echo ""

# Paso 4: Cargar grilla espacial
echo "============================================================================"
echo "🗺️  PASO 4: Cargando grilla espacial (3,149 puntos esperados)"
echo "============================================================================"
echo "⏳ Este proceso puede tomar 1-2 minutos..."
echo ""

docker exec geoinformatica-backend python3 /app/cargar_grilla_densidades.py

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Grilla espacial cargada exitosamente${NC}"
else
    echo -e "${RED}❌ Error al cargar grilla${NC}"
    exit 1
fi
echo ""

# Verificación final
echo "============================================================================"
echo "📊 VERIFICACIÓN FINAL"
echo "============================================================================"

docker exec geoinformatica-db psql -U postgres -d inmobiliaria_db << 'EOF'
\echo ''
\echo '📋 Resumen de datos cargados:'
\echo '─────────────────────────────────────'

SELECT 
    'comunas' as tabla,
    COUNT(*)::text as registros,
    '' as info_adicional
FROM comunas

UNION ALL

SELECT 
    'propiedades' as tabla,
    COUNT(*)::text as registros,
    'Precio promedio: $' || ROUND(AVG(precio)::numeric, 0)::text as info_adicional
FROM propiedades

UNION ALL

SELECT 
    'grilla_espacial' as tabla,
    COUNT(*)::text as registros,
    'Densidad promedio: ' || ROUND(AVG(dens_total_600m_km2)::numeric, 2)::text || ' serv/km²' as info_adicional
FROM grilla_espacial;

\echo ''
EOF

echo ""
echo "============================================================================"
echo -e "${GREEN}✅ CARGA COMPLETADA EXITOSAMENTE${NC}"
echo "============================================================================"
echo ""
echo "📊 Datos disponibles:"
echo "   • 32 comunas de Santiago"
echo "   • 1,623 propiedades reales"
echo "   • 3,149 puntos de grilla con características espaciales"
echo ""
echo "🚀 Sistema listo para usar"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo ""
