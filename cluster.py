# ===========================
# 1. Importar librerías
# ===========================
import os
import sys
import geopandas as gpd
import matplotlib.pyplot as plt
import libpysal
from esda.moran import Moran
from esda.moran import Moran_Local
import mapclassify
import pandas as pd

# ===========================
# 2. Cargar datos
# ===========================
# Leer shapefile
# Allow passing the shapefile path as first CLI arg, default to `base_maestra.shp`
shp_path = sys.argv[1] if len(sys.argv) > 1 else "base_maestra.shp"

# If the .shx index is missing GDAL can recreate it by setting this config option
# (pyogrio/GDAL respect the environment variable).
os.environ.setdefault("SHAPE_RESTORE_SHX", "YES")

if not os.path.exists(shp_path):
    base_prefix = os.path.splitext(os.path.basename(shp_path))[0].lower()
    # show helpful diagnostics including other shapefile-like files in cwd
    siblings = [f for f in os.listdir(".") if f.lower().startswith(base_prefix)]
    # try to find a subdirectory with that prefix containing the .shp
    candidate = None
    for entry in os.listdir("."):
        if os.path.isdir(entry) and entry.lower().startswith(base_prefix):
            possible = os.path.join(entry, f"{base_prefix}.shp")
            if os.path.exists(possible):
                candidate = possible
                break

    if candidate:
        print(f"No se encontró '{shp_path}' en cwd, usando shapefile encontrado en: {candidate}")
        shp_path = candidate
    else:
        raise FileNotFoundError(
            f"Shapefile not found: '{shp_path}'.\nFiles with same prefix in cwd: {siblings}\n"
            "Place the full set of shapefile components (.shp, .shx, .dbf, etc.) in this folder, or pass the path as an argument."
        )

gdf = gpd.read_file(shp_path)

# Revisar CRS
print("CRS original:", gdf.crs)

# Si el GeoDataFrame no tiene CRS, permitir que el usuario lo proporcione
# como segundo argumento de la línea de comandos, o asumir WGS84 (EPSG:4326)
if gdf.crs is None:
    if len(sys.argv) > 2:
        provided_crs = sys.argv[2]
        print(f"No se encontró CRS. Estableciendo CRS a: {provided_crs}")
        gdf.set_crs(provided_crs, inplace=True, allow_override=True)
    else:
        default_crs = "EPSG:4326"
        print(f"No se encontró CRS. Asignando CRS por defecto: {default_crs}.")
        print("Si esto es incorrecto, vuelva a ejecutar el script con el CRS correcto como segundo argumento, por ejemplo:\n  python cluster.py base_maestra.shp EPSG:32719")
        gdf.set_crs(default_crs, inplace=True, allow_override=True)

# Reproyectar a UTM (zona 19S para RM, EPSG:32719)
try:
    gdf = gdf.to_crs(epsg=32719)
except Exception as e:
    raise RuntimeError(f"Error al reproyectar geometrías: {e}\nRevise que el CRS original sea correcto y que las geometrías sean válidas.")

# ===========================
# 3. Crear variable precio_m2
# ===========================
# Verificar que existan las columnas necesarias
required_cols = ["total_uf", "t_constr"]
missing = [c for c in required_cols if c not in gdf.columns]
if missing:
    # Si el usuario pasó un CSV con atributos como tercer argumento, intentar unirlo
    if len(sys.argv) > 3 and os.path.exists(sys.argv[3]):
        attrs_path = sys.argv[3]
        print(f"Columnas faltantes {missing}. Intentando leer atributos desde: {attrs_path}")
        df_attr = pd.read_csv(attrs_path)
        # Intentar unir por columna en común
        common = list(set(gdf.columns).intersection(df_attr.columns))
        if common:
            key = common[0]
            print(f"Uniendo por columna común: {key}")
            gdf = gdf.merge(df_attr, on=key, how="left")
        else:
            # Intentar unir por índice si CSV tiene columna 'index' o 'idx'
            if "index" in df_attr.columns:
                df_attr = df_attr.set_index("index")
                gdf = gdf.join(df_attr, how="left")
            elif df_attr.shape[0] == gdf.shape[0] and set(required_cols).issubset(df_attr.columns):
                print("CSV tiene las columnas requeridas y el mismo número de filas; asignando directamente.")
                for c in required_cols:
                    gdf[c] = df_attr[c].values
            else:
                raise RuntimeError(
                    f"No se pudo unir atributos automáticamente. \nCSV columnas: {list(df_attr.columns)}\n" \
                    "Proporcione un CSV con una columna de unión que exista en el shapefile, o un CSV con las columnas requeridas y el mismo número de filas."
                )

    else:
        raise RuntimeError(
            f"Faltan columnas requeridas en el shapefile: {missing}.\nColumnas disponibles: {list(gdf.columns)}\n"
            "Soluciones:\n  - Proporcione un shapefile que incluya las columnas 'total_uf' y 't_constr'.\n"
            "  - O pase un CSV de atributos como tercer argumento para unir:\n    python cluster.py base_maestra.shp EPSG:4326 atributos.csv"
        )

gdf["precio_m2"] = gdf["total_uf"] / gdf["t_constr"]

# Eliminar registros con 0 o NaN en superficie
gdf = gdf[gdf["t_constr"] > 0].dropna(subset=["precio_m2"])

# ===========================
# 4. Matriz de pesos espaciales
# ===========================
# Usamos 8 vecinos más cercanos
w = libpysal.weights.KNN.from_dataframe(gdf, k=8)
w.transform = "R"  # normalizar filas

# ===========================
# 5. Moran Global
# ===========================
y = gdf["precio_m2"].values
moran = Moran(y, w)
print(f"Moran’s I: {moran.I:.3f}, p-value: {moran.p_sim:.4f}")

# ===========================
# 6. Moran Local (LISA)
# ===========================
lisa = Moran_Local(y, w)

# Guardar clusters en el GeoDataFrame
gdf["lisa_cluster"] = lisa.q
gdf["lisa_sig"] = lisa.p_sim < 0.05  # significancia 95%

# ===========================
# 7. Mapa de clusters LISA
# ===========================
fig, ax = plt.subplots(1, 1, figsize=(10, 8))

# Colores: 1=HH, 2=LH, 3=LL, 4=HL
cluster_labels = {
    1: "Alto-Alto (Hotspot)",
    2: "Bajo-Alto",
    3: "Bajo-Bajo (Coldspot)",
    4: "Alto-Bajo"
}

gdf.assign(
    cl=gdf["lisa_cluster"].map(cluster_labels)
).plot(
    column="cl", categorical=True, legend=True,
    cmap="Set1", linewidth=0.1, edgecolor="grey", ax=ax
)

ax.set_title("Clusters LISA de precio por m²", fontsize=14)
plt.show()
