# Breve descripción:
"""Este script realiza un análisis espacial de autocorrelación sobre una variable georreferenciada (por ejemplo, 
precio, índice o tasa) para detectar patrones territoriales. Primero prepara los datos geográficos
(verifica/reproyecta el sistema de coordenadas y calcula la variable de interés). Luego construye una matriz de vecindad
(p. ej. k-nearest neighbors) que define qué unidades son “vecinas” y cuánto se influyen entre sí. Con esa matriz calcula
el Índice de Moran Global, que responde si existe autocorrelación espacial general (valores altos y p pequeños indican 
agrupamiento espacial significativo). Para localizar dónde están esos agrupamientos se aplica LISA (indicadores locales),
que clasifica cada unidad como hotspot (alto rodeado de altos), coldspot (bajo rodeado de bajos) o outlier (valor distinto a su entorno).
El resultado final incluye estadísticas (Moran I + p‑valor) y una clasificación espacial por unidad que se puede mapear 
para identificar zonas de concentración, vacíos o anomalías con implicancias prácticas para planificación, segmentación de 
mercados o priorización de intervenciones.
"""

# ===========================
# 1. Librerías
# ===========================
import geopandas as gpd
import matplotlib.pyplot as plt
import libpysal
from esda.moran import Moran, Moran_Local
import numpy as np
import pandas as pd
import unicodedata
import re

# ===========================
# 2. Cargar datos
# ===========================
gdf = gpd.read_file("base_maestra_comunas_filtradas.geojson")

# Revisar CRS y reproyectar a metros (UTM zona 19S, EPSG:32719)
if gdf.crs is None:
    gdf = gdf.set_crs("EPSG:4326")
gdf = gdf.to_crs(epsg=32719)

# ===========================
# 3. Filtrar comunas de interés
# ===========================
target_comunas = ["La Reina", "Ñuñoa", "Santiago", "Estación Central"]

def normalize_name(s: str) -> str:
    """Normaliza nombre de comuna (acentos, mayúsculas, espacios)."""
    if s is None: return ""
    s = str(s).strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    s = re.sub(r"[^0-9a-z\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

gdf["comuna_norm"] = gdf["comuna"].apply(normalize_name)
target_norm = [normalize_name(x) for x in target_comunas]

# Correcciones comunes
gdf["comuna_norm"] = gdf["comuna_norm"].replace({
    "estacion centra": "estacion central",
    "estacioncentral": "estacion central"
})

gdf = gdf[gdf["comuna_norm"].isin(target_norm)].copy()
print("Registros filtrados:", len(gdf))

# ===========================
# 4. Variable precio_m2
# ===========================
gdf = gdf[gdf["t_constr"] > 0].copy()
gdf["precio_m2"] = gdf["total_uf"] / gdf["t_constr"]

# Winsorización simple para evitar outliers extremos
lower, upper = gdf["precio_m2"].quantile([0.01, 0.99])
gdf["precio_m2_w"] = gdf["precio_m2"].clip(lower, upper)

# ===========================
# 5. Moran Global
# ===========================
w = libpysal.weights.KNN.from_dataframe(gdf, k=8)
w.transform = "R"

moran = Moran(gdf["precio_m2_w"], w)
print(f"Moran’s I: {moran.I:.3f}, p-value: {moran.p_sim:.4f}")

# ===========================
# 6. LISA (Moran Local)
# ===========================
lisa = Moran_Local(gdf["precio_m2_w"], w)

# Etiquetas de clusters
cluster_labels = {
    1: "Alto-Alto (Hotspot)",
    2: "Bajo-Alto",
    3: "Bajo-Bajo (Coldspot)",
    4: "Alto-Bajo"
}
gdf["cluster"] = lisa.q
gdf["sig"] = lisa.p_sim < 0.05
gdf["cluster_label"] = gdf.apply(
    lambda row: cluster_labels.get(row["cluster"], "No significativo") if row["sig"] else "No significativo",
    axis=1
)

# ===========================
# 7. Mapa de clusters LISA
# ===========================
colors = {
    "Alto-Alto (Hotspot)": "red",
    "Bajo-Bajo (Coldspot)": "blue",
    "Alto-Bajo": "pink",
    "Bajo-Alto": "lightblue",
    "No significativo": "lightgrey"
}

fig, ax = plt.subplots(1, 1, figsize=(10, 8))
for cat, color in colors.items():
    subset = gdf[gdf["cluster_label"] == cat]
    subset.plot(ax=ax, color=color, edgecolor="black", linewidth=0.2, label=cat)

ax.set_title("Clusters LISA: Precio UF/m²", fontsize=14)
ax.axis("off")
ax.legend(title="Cluster", loc="lower left")
plt.show()
