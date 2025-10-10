"""Herramienta CLI para generar un GeoJSON de comunas con buffer adicional."""
from pathlib import Path
import sys

import geopandas as gpd

DEFAULT_INPUT = Path("base_maestra/comunas.geojson")
OUTPUT_FILE = Path("salida/comunas_buffer.geojson")
GEODETIC_CRS = "EPSG:4326"
METRIC_CRS = "EPSG:32719"


def prompt_distance(message: str) -> float:
    while True:
        raw = input(message).strip()
        if not raw:
            print("Ingresa un valor numerico positivo.")
            continue
        try:
            value = float(raw)
        except ValueError:
            print("El valor debe ser numerico.")
            continue
        if value <= 0:
            print("El valor debe ser mayor que cero.")
            continue
        return value


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    input_path = (base_dir / DEFAULT_INPUT).resolve()
    output_path = (base_dir / OUTPUT_FILE).resolve()

    print("=== Expansor de comunas ===")
    print(f"Archivo de entrada: {input_path}")
    print(f"Archivo de salida: {output_path}")

    if not input_path.exists():
        print(f"No se encontro el archivo: {input_path}")
        sys.exit(1)

    distance_m = prompt_distance(
        "Ingresa la distancia de buffer en metros (ej. 500): "
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        gdf = gpd.read_file(input_path)
    except Exception as exc:
        print(f"Error al leer el archivo: {exc}")
        sys.exit(1)

    if gdf.crs is None:
        print(
            "Advertencia: el archivo no tiene CRS. Se asume EPSG:4326 antes de reproyectar"
        )
        gdf = gdf.set_crs(GEODETIC_CRS)

    try:
        gdf_metric = gdf.to_crs(METRIC_CRS)
    except Exception as exc:
        print(f"No se pudo reproyectar a {METRIC_CRS}: {exc}")
        sys.exit(1)

    buffered = gdf_metric.copy()
    buffered["geometry"] = buffered.geometry.buffer(distance_m, join_style=2)

    try:
        buffered = buffered.to_crs(GEODETIC_CRS)
    except Exception as exc:
        print(f"No se pudo volver a {GEODETIC_CRS}: {exc}")
        sys.exit(1)

    try:
        buffered.to_file(output_path, driver="GeoJSON")
    except Exception as exc:
        print(f"No se pudo escribir el GeoJSON: {exc}")
        sys.exit(1)

    print(f"Archivo generado: {output_path}")
    print(
        "Recuerda que el buffer es aproximado porque se proyecto a UTM zona 19S\n"
        "Si tu area cubre varias zonas UTM evalua usar una proyeccion diferente."
    )


if __name__ == "__main__":
    main()
