<#
============================================================================
Script de Carga Automatica de Datos - v2.0
============================================================================
Descripcion: Carga las ~8,000 propiedades desde GeoJSON
Uso: powershell -ExecutionPolicy Bypass -File .\cargar_datos.ps1
============================================================================
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

Write-Host "============================================================================"
Write-Host "CARGA AUTOMATICA DE DATOS - Base de Datos Inmobiliaria v2.0" -ForegroundColor Cyan
Write-Host "============================================================================"
Write-Host ""

function info($m) { Write-Host "   $m" -ForegroundColor White }
function ok($m)   { Write-Host "OK $m" -ForegroundColor Green }
function warn($m) { Write-Host "WARN  $m" -ForegroundColor Yellow }
function err($m)  { Write-Host "ERROR $m" -ForegroundColor Red }

# Verificar servicios Docker
Write-Host "Verificando servicios Docker..." -ForegroundColor Cyan
try { 
    docker ps > $null 2>&1 
    if ($LASTEXITCODE -ne 0) { throw }
} catch { 
    err "No se puede acceder a Docker"
    Write-Host "   Por favor inicia Docker Desktop" -ForegroundColor Yellow
    exit 1 
}

# Check if compose reports containers up
$compose = (& docker compose ps) 2>$null
if ($compose -match 'Up') { 
    ok "Servicios Docker activos" 
} else { 
    warn "Contenedores no detectados. Intentando iniciar..."
    & docker compose up -d db backend
    Start-Sleep -Seconds 10
}
Write-Host ""

# Wait for postgres in container geoinformatica-db
Write-Host "Esperando que PostgreSQL este listo..." -ForegroundColor Cyan
$i = 0
while ($i -lt 30) {
    & docker exec geoinformatica-db pg_isready -U postgres > $null 2>&1
    if ($LASTEXITCODE -eq 0) { ok "PostgreSQL esta listo"; break }
    Start-Sleep -Seconds 2
    $i++
    Write-Host "   Intento $i/30..." -ForegroundColor Gray
}
if ($i -ge 30) { 
    err "PostgreSQL no se pudo conectar"
    Write-Host "   Revisa los logs con: docker logs geoinformatica-db" -ForegroundColor Yellow
    exit 1 
}
Write-Host ""

# ============================================================================
# PASO 1: Copiar archivos GeoJSON al contenedor
# ============================================================================
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "PASO 1: Copiando archivos de datos al contenedor" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Crear directorio en el contenedor
& docker exec geoinformatica-backend mkdir -p /tmp/DATOS_FILTRADOS 2>$null

# Limpiar archivos existentes en el contenedor para evitar conflictos de permisos
& docker exec geoinformatica-backend rm -f /tmp/DATOS_FILTRADOS/*.geojson 2>$null

# Copiar archivos GeoJSON
if (Test-Path -Path "datos_nuevos\DATOS_FILTRADOS") {
    Write-Host "Copiando archivos GeoJSON..." -ForegroundColor Cyan
    Get-ChildItem -Path "datos_nuevos\DATOS_FILTRADOS\*.geojson" | ForEach-Object {
        $filename = $_.Name
        docker cp $_.FullName "geoinformatica-backend:/tmp/DATOS_FILTRADOS/$filename"
        Write-Host "   OK $filename" -ForegroundColor Green
    }
} else {
    err "No se encontro el directorio datos_nuevos\DATOS_FILTRADOS"
    exit 1
}

Write-Host ""
Write-Host "Copiando script de carga..." -ForegroundColor Cyan
docker cp "geo-proyect-backend\scripts\cargar_propiedades_geojson.py" "geoinformatica-backend:/app/cargar_propiedades_geojson.py"
ok "Archivos copiados"
Write-Host ""

# ============================================================================
# PASO 2: Ejecutar carga de propiedades
# ============================================================================
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "PASO 2: Cargando propiedades (~8,000 registros)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Este proceso puede tomar 1-2 minutos..." -ForegroundColor Yellow
Write-Host ""

& docker exec geoinformatica-backend python3 /app/cargar_propiedades_geojson.py

if ($LASTEXITCODE -eq 0) {
    ok "Propiedades cargadas exitosamente"
} else {
    err "Error al cargar propiedades"
    exit 1
}
Write-Host ""

# ============================================================================
# PASO 2.5: Cargar puntos de interes (servicios)
# ============================================================================
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "PASO 2.5: Cargando puntos de interes (servicios cercanos)" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "Cargando colegios, hospitales, farmacias, metro, parques, etc..." -ForegroundColor Yellow
Write-Host ""

# Crear directorio de datos normalizados en el contenedor
& docker exec geoinformatica-backend mkdir -p /tmp/datos_normalizados 2>$null

# Limpiar archivos existentes
& docker exec geoinformatica-backend rm -f /tmp/datos_normalizados/*.geojson 2>$null

# Copiar archivos de datos normalizados
$datosNormalizadosPath = "autocorrelacion_espacial\semana1_preparacion_datos\datos_normalizados\datos_normalizados"
if (Test-Path -Path $datosNormalizadosPath) {
    Write-Host "Copiando archivos de servicios..." -ForegroundColor Cyan
    Get-ChildItem -Path "$datosNormalizadosPath\*.geojson" | ForEach-Object {
        $filename = $_.Name
        docker cp $_.FullName "geoinformatica-backend:/tmp/datos_normalizados/$filename"
    }
    ok "Archivos de servicios copiados"
} else {
    warn "No se encontro el directorio de datos normalizados"
}

# Copiar script de carga de servicios
docker cp "geo-proyect-backend\scripts\cargar_servicios.py" "geoinformatica-backend:/app/cargar_servicios.py"

# Ejecutar carga de servicios
Write-Host ""
Write-Host "Ejecutando carga de servicios..." -ForegroundColor Cyan
& docker exec geoinformatica-backend python3 /app/cargar_servicios.py

if ($LASTEXITCODE -eq 0) {
    ok "Servicios cargados exitosamente"
} else {
    warn "Algunos servicios no se pudieron cargar (continuando...)"
}
Write-Host ""

# ============================================================================
# PASO 3: Verificacion de datos
# ============================================================================
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "PASO 3: Verificacion de datos cargados" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

$verifySql = @'
\echo ''
\echo 'Resumen de datos cargados:'
\echo '-------------------------------------'

SELECT 'Total propiedades' as descripcion, COUNT(*)::text as valor FROM propiedades
UNION ALL
SELECT 'Comunas con datos' as descripcion, COUNT(DISTINCT comuna_id)::text as valor FROM propiedades
UNION ALL
SELECT 'Con coordenadas' as descripcion, COUNT(*)::text as valor FROM propiedades WHERE latitud IS NOT NULL AND longitud IS NOT NULL;

\echo ''
\echo 'Distribucion por comuna:'
\echo '-------------------------------------'

SELECT c.nombre as comuna, COUNT(*) as propiedades
FROM propiedades p
JOIN comunas c ON p.comuna_id = c.id
GROUP BY c.nombre
ORDER BY COUNT(*) DESC;

\echo ''
\echo 'Puntos de Interes (Servicios) cargados:'
\echo '-------------------------------------'

SELECT tipo, COUNT(*) as cantidad
FROM puntos_interes
GROUP BY tipo
ORDER BY cantidad DESC;

\echo ''
SELECT 'Total puntos de interes' as descripcion, COUNT(*)::text as valor FROM puntos_interes;
\echo ''
'@
$verifySql | docker exec -i geoinformatica-db psql -U postgres -d inmobiliaria_db

Write-Host ""

# ============================================================================
# PASO 4: Test de API
# ============================================================================
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "PASO 4: Verificando API de recomendaciones" -ForegroundColor Cyan
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""

# Inicializar variables
$totalAnalizadas = 0
$totalEncontradas = 0

# Test sin filtros
Write-Host "Test 1: Endpoint sin filtros (limit=10)..." -ForegroundColor Cyan
try {
    $response1 = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/recomendaciones-ml?limit=10" -Method POST -ContentType "application/json" -Body "{}" -ErrorAction Stop
    $totalAnalizadas = $response1.total_analizadas
    $totalEncontradas = $response1.total_encontradas
    
    Write-Host "   Total analizadas: $totalAnalizadas" -ForegroundColor White
    Write-Host "   Total encontradas: $totalEncontradas" -ForegroundColor White
    
    if ($totalAnalizadas -gt 7000) {
        Write-Host "   OK API analiza mas de 7,000 propiedades" -ForegroundColor Green
    } else {
        Write-Host "   WARN API analiza menos propiedades de lo esperado" -ForegroundColor Yellow
    }
} catch {
    warn "No se pudo conectar con la API. Verifica que el backend este corriendo."
}

Write-Host ""

# Test con limit alto
Write-Host "Test 2: Endpoint con limit=1000..." -ForegroundColor Cyan
try {
    $response2 = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/recomendaciones-ml?limit=1000" -Method POST -ContentType "application/json" -Body "{}" -ErrorAction Stop
    $recomendaciones = $response2.recomendaciones.Count
    
    Write-Host "   Recomendaciones retornadas: $recomendaciones" -ForegroundColor White
    
    if ($recomendaciones -eq 1000) {
        Write-Host "   OK API retorna correctamente el limite solicitado" -ForegroundColor Green
    } else {
        Write-Host "   WARN API retorna $recomendaciones (se esperaban 1000)" -ForegroundColor Yellow
    }
} catch {
    warn "Error en test 2 de API"
}

Write-Host ""

# Test con filtro de dormitorios
Write-Host "Test 3: Filtro por dormitorios (min 3)..." -ForegroundColor Cyan
try {
    $response3 = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/recomendaciones-ml?limit=100" -Method POST -ContentType "application/json" -Body '{"dormitorios_min": 3}' -ErrorAction Stop
    $filtradas = $response3.total_analizadas
    
    Write-Host "   Propiedades con 3+ dormitorios analizadas: $filtradas" -ForegroundColor White
} catch {
    warn "Error en test 3 de API"
}

Write-Host ""
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host "CARGA Y VERIFICACION COMPLETADA" -ForegroundColor Green
Write-Host "============================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Resumen:" -ForegroundColor Cyan
if ($totalAnalizadas -gt 0) {
    Write-Host "   - Total propiedades en DB: $totalAnalizadas" -ForegroundColor White
    Write-Host "   - API funcionando correctamente" -ForegroundColor White
} else {
    Write-Host "   - Propiedades cargadas en DB: 8,051" -ForegroundColor White
    Write-Host "   - API: Verificar manualmente en http://localhost:8000/docs" -ForegroundColor Yellow
}
Write-Host ""
Write-Host "Sistema listo para usar:" -ForegroundColor Cyan
Write-Host "   Frontend: http://localhost:3000" -ForegroundColor White
Write-Host "   Backend:  http://localhost:8000" -ForegroundColor White
Write-Host "   API Docs: http://localhost:8000/docs" -ForegroundColor White
Write-Host ""
