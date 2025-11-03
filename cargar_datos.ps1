<#
  PowerShell loader script (ASCII-only) for Windows
  - Run from project root:
      powershell -ExecutionPolicy Bypass -File .\cargar_datos.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function info($m) { Write-Host "[INFO]  $m" }
function ok($m)   { Write-Host "[OK]    $m" }
function warn($m) { Write-Host "[WARN]  $m" }
function err($m)  { Write-Host "[ERROR] $m" }

info "Starting loader script"

# Check docker
try { docker ps > $null 2>&1 } catch { err "Docker not accessible. Start Docker Desktop."; exit 1 }
ok "Docker is accessible"

# Check if compose reports containers up
$compose = (& docker compose ps) 2>$null
if ($compose -match 'Up') { ok "Docker compose containers appear up" } else { warn "Compose does not show containers as Up. Run docker compose up -d in geo-proyect-backend if needed." }

# Wait for postgres in container geoinformatica-db
info "Waiting for postgres in container 'geoinformatica-db' (up to ~60s)"
$i = 0
while ($i -lt 30) {
    & docker exec geoinformatica-db pg_isready -U postgres > $null 2>&1
    if ($LASTEXITCODE -eq 0) { ok "Postgres is ready"; break }
    Start-Sleep -Seconds 2
    $i++
    Write-Host "  try $i/30..."
}
if ($i -ge 30) { err "Postgres did not become ready. Check docker logs geoinformatica-db"; exit 1 }

# Load comunas: use comunas.sql if present, else use embedded SQL
info "Loading comunas"
if (Test-Path -Path .\comunas.sql) {
    Get-Content .\comunas.sql -Raw | docker exec -i geoinformatica-db psql -U postgres -d inmobiliaria_db
} else {
    $sql = @'
INSERT INTO comunas (id, nombre) VALUES
(1, ''Cerrillos''),(2, ''Cerro Navia''),(3, ''Conchali''),(4, ''El Bosque''),
(5, ''Estacion Central''),(6, ''Huechuraba''),(7, ''Independencia''),(8, ''La Cisterna''),
(9, ''La Florida''),(10, ''La Granja''),(11, ''La Pintana''),(12, ''La Reina''),
(13, ''Las Condes''),(14, ''Lo Barnechea''),(15, ''Lo Espejo''),(16, ''Lo Prado''),
(17, ''Macul''),(18, ''Maipu''),(19, ''Nunoa''),(20, ''Pedro Aguirre Cerda''),
(21, ''Penalolen''),(22, ''Providencia''),(23, ''Pudahuel''),(24, ''Quilicura''),
(25, ''Quinta Normal''),(26, ''Recoleta''),(27, ''Renca''),(28, ''San Joaquin''),
(29, ''San Miguel''),(30, ''San Ramon''),(31, ''Santiago''),(32, ''Vitacura'')
ON CONFLICT (id) DO NOTHING;
'@
    $sql | docker exec -i geoinformatica-db psql -U postgres -d inmobiliaria_db
}
if ($LASTEXITCODE -ne 0) { err "Failed to run comunas SQL"; exit 1 }
ok "Comunas loaded"

# Files to check and copy into backend container
$needed = @(
    @{host='clean_alquiler_02_11_2023cc.csv'; container='/app/clean_alquiler_02_11_2023cc.csv'},
    @{host='autocorrelacion_espacial\semana1_preparacion_datos\datos_normalizados'; container='/app/datos_normalizados'},
    @{host='autocorrelacion_espacial\semana2_caracteristicas_espaciales\features\grilla_con_densidades.geojson'; container='/tmp/grilla_con_densidades.geojson'},
    @{host='geo-proyect-backend\scripts\cargar_propiedades_csv.py'; container='/app/cargar_propiedades_csv.py'},
    @{host='geo-proyect-backend\scripts\cargar_grilla_densidades.py'; container='/app/cargar_grilla_densidades.py'}
)

# Ensure files are present inside the backend container; copy from host when needed
$allGood = $true
foreach ($item in $needed) {
    # check existence inside container
    & docker exec geoinformatica-backend test -e $($item.container) > $null 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  [IN_CONTAINER] $($item.container)"
        continue
    }

    # not present in container; attempt to copy from host
    $hostPath = Join-Path -Path $PWD -ChildPath $item.host
    if (Test-Path $hostPath) {
        try {
            docker cp $hostPath "geoinformatica-backend:$($item.container)"
            ok "Copied $($item.host) -> $($item.container)"
        } catch {
            warn "Failed to copy $($item.host): $($_.Exception.Message)"
            $allGood = $false
        }
    } else {
        warn "Missing on host and container: $($item.host)"
        $allGood = $false
    }
}

if (-not $allGood) { warn "Some required files are missing or failed to copy. Check messages above." }

function Run-PythonInContainer($scriptPathInContainer) {
    info "Running $scriptPathInContainer in geoinformatica-backend"
    & docker exec geoinformatica-backend python3 $scriptPathInContainer
    if ($LASTEXITCODE -ne 0) { & docker exec geoinformatica-backend python $scriptPathInContainer }
    return $LASTEXITCODE
}

info "Running cargar_propiedades_csv.py"
$rc = Run-PythonInContainer '/app/cargar_propiedades_csv.py'
if ($rc -ne 0) { warn "cargar_propiedades_csv.py exited with code $rc — check backend logs for details" } else { ok "Propiedades loaded" }

info "Running cargar_grilla_densidades.py"
$rc = Run-PythonInContainer '/app/cargar_grilla_densidades.py'
if ($rc -ne 0) { warn "cargar_grilla_densidades.py exited with code $rc — check backend logs for details" } else { ok "Grid loaded" }

# Final checks
$summarySql = @'
SELECT 'comunas' as tabla, COUNT(*)::text as registros, '' as info_adicional FROM comunas
UNION ALL
SELECT 'propiedades' as tabla, COUNT(*)::text as registros, 'Precio promedio: $' || ROUND(AVG(precio)::numeric, 0)::text as info_adicional FROM propiedades
UNION ALL
SELECT 'grilla_espacial' as tabla, COUNT(*)::text as registros, 'Densidad promedio: ' || ROUND(AVG(dens_total_600m_km2)::numeric, 2)::text || ' serv/km2' as info_adicional FROM grilla_espacial;
'@
$summarySql | docker exec -i geoinformatica-db psql -U postgres -d inmobiliaria_db

ok "Load completed"
Write-Host "Frontend: http://localhost:3000    Backend: http://localhost:8000"
