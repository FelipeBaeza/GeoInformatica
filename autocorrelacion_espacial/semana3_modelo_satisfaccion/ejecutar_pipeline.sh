#!/bin/bash
# Script para ejecutar el pipeline completo de Semana 3
# ====================================================

set -e  # Salir si hay error

WORKSPACE="/home/felipe/Documentos/GeoInformatica"
SEMANA3="$WORKSPACE/autocorrelacion_espacial/semana3_modelo_satisfaccion"

echo "=============================================="
echo "  PIPELINE SEMANA 3 - Modelo de Satisfacción"
echo "=============================================="
echo ""

# Verificar prerequisitos
echo " Verificando prerequisitos..."

if [ ! -f "$WORKSPACE/clean_alquiler_02_11_2023cc.csv" ]; then
    echo " ERROR: No se encontró clean_alquiler_02_11_2023cc.csv"
    exit 1
fi

GRILLA="$WORKSPACE/autocorrelacion_espacial/semana2_caracteristicas_espaciales/features/grilla_con_densidades.geojson"
if [ ! -f "$GRILLA" ]; then
    echo " ERROR: No se encontró grilla_con_densidades.geojson"
    echo "   Ejecuta primero los scripts de semana2"
    exit 1
fi

echo "✓ Archivos base encontrados"
echo ""

# Verificar Python y dependencias
echo " Verificando dependencias Python..."
python3 -c "import geopandas, pandas, sklearn" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "  Faltan dependencias. Instalando..."
    pip3 install --user -q geopandas pandas numpy scikit-learn matplotlib seaborn scipy tqdm
    echo "✓ Dependencias instaladas"
fi

cd "$WORKSPACE"

# Paso 1: Integración
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "1️  INTEGRACIÓN DE DATOS ESPACIALES"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 "$SEMANA3/scripts/01_integrar_datos.py"

if [ ! -f "$SEMANA3/data/propiedades_con_factores_espaciales.csv" ]; then
    echo " ERROR: No se generó el archivo de integración"
    exit 1
fi

# Paso 2: Modelo Baseline
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "2️  MODELO RANDOM FOREST BASELINE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 "$SEMANA3/scripts/02_modelo_satisfaccion.py"

# Paso 3: Autocorrelación
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "3️  ANÁLISIS DE AUTOCORRELACIÓN ESPACIAL"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 "$SEMANA3/scripts/03_autocorrelacion_residuos.py"

# Paso 4: Validación
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "4️  VALIDACIÓN ESTADÍSTICA (Test Permutación)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 "$SEMANA3/scripts/04_validar_autocorrelacion.py"

# Leer resultado del test
PVALUE=$(python3 -c "import pandas as pd; df=pd.read_csv('$SEMANA3/resultados/test_autocorrelacion.csv'); print(df['p_value'].iloc[0])")
SIGNIFICATIVO=$(python3 -c "import pandas as pd; df=pd.read_csv('$SEMANA3/resultados/test_autocorrelacion.csv'); print(df['significativo'].iloc[0])")

echo ""
echo " Resultado del test:"
echo "   P-value: $PVALUE"

# Paso 5: GWRF (solo si es significativo)
if [ "$SIGNIFICATIVO" = "True" ]; then
    echo "     Autocorrelación SIGNIFICATIVA → Ejecutando GWRF"
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "5️  GWRF + STACKING"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    python3 "$SEMANA3/scripts/05_implementar_gwrf.py"
else
    echo "    Autocorrelación NO significativa → GWRF opcional"
    echo ""
    read -p "¿Ejecutar GWRF de todas formas? (s/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Ss]$ ]]; then
        echo ""
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        echo "5️  GWRF + STACKING (opcional)"
        echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        python3 "$SEMANA3/scripts/05_implementar_gwrf.py"
    fi
fi

# Paso 6: Modelo Mejorado (usa idx_habitabilidad_global)
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "6️  MODELO SATISFACCIÓN MEJORADO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Activar venv si existe
if [ -d "$SEMANA3/venv_viz" ]; then
    source "$SEMANA3/venv_viz/bin/activate"
fi

python3 "$SEMANA3/scripts/06_modelo_satisfaccion_mejorado.py"

# Paso 7: Visualizaciones Completas
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "7️  VISUALIZACIONES CARTOGRÁFICAS"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 "$SEMANA3/scripts/08_visualizaciones_cartograficas.py"

# Desactivar venv
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate 2>/dev/null || true
fi

# Resumen final
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo " PIPELINE COMPLETADO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo " Archivos generados en:"
echo "   • $SEMANA3/data/"
echo "   • $SEMANA3/resultados/"
echo "   • $SEMANA3/resultados/modelo_mejorado/"
echo "   • $SEMANA3/graficos/"
echo ""
echo " Para ver resultados:"
echo "   cat $SEMANA3/resultados/metricas_modelo.csv"
echo "   cat $SEMANA3/resultados/test_autocorrelacion.csv"
echo "   cat $SEMANA3/resultados/modelo_mejorado/metricas_modelo_mejorado.json"
echo ""
echo " Visualizaciones:"
echo "   • 3 mapas temáticos (PNG)"
echo "   • 5 gráficos estadísticos (PNG)"
echo "   • 1 mapa interactivo (HTML)"
echo ""
