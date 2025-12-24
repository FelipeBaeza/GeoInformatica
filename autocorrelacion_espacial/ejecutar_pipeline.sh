#!/bin/bash
# =============================================================================
# PIPELINE TERRAMATCH - Ejecución Rápida
# =============================================================================
# 
# Uso:
#   ./ejecutar_pipeline.sh         # Pipeline completo
#   ./ejecutar_pipeline.sh 1       # Solo Semana 1
#   ./ejecutar_pipeline.sh 2       # Solo Semana 2
#   ./ejecutar_pipeline.sh 3       # Solo Semana 3
#   ./ejecutar_pipeline.sh 5       # Semanas 2 y 3
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║           TERRAMATCH - Pipeline Geoespacial                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Verificar Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 no encontrado"
    exit 1
fi

# Ejecutar pipeline con opción
if [ -n "$1" ]; then
    echo "$1" | python3 ejecutar_pipeline_completo.py
else
    python3 ejecutar_pipeline_completo.py
fi
