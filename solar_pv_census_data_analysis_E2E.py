# %% [markdown]
# # Setup, Dataset Fetch and Pre-processing to Enable End to End Analysis in a Single Notebook

# %%
# install git python package to fetch our repository for required data fetching and preprocessing scripts (cant rely on classmates having git cli installed)
# give preference to uv but use pip when not installed
import shutil
import os
import time
import random

uv_path = shutil.which("uv")
if uv_path is not None:
    print("Installing GitPython via uv...")
    !uv pip install GitPython
else:
    print("Installing GitPython via pip...")
    !pip install GitPython

local_path = "pv_solar_analysis/"
# delete directory if it exists to ensure fresh clone and easier notebook re-runs
if os.path.exists(local_path):
    shutil.rmtree(local_path, ignore_errors=True)

# %%
# fetch requirements and pre-processing notebooks from github
from git import Repo

# Replace with your repository URL and desired local path
repo_url = "https://github.com/avega17/CCOM6994_PV_Solar_Project.git"
# NOTE: adjust local_path as needed

repo_name = repo_url.split("/")[-1].replace(".git", "")
if not os.path.exists(local_path):
    os.mkdir(local_path)

try:
    Repo.clone_from(repo_url, local_path)
    # remove the .git folder to avoid confusion and conflict during team development
    shutil.rmtree(f"{local_path}/.git", ignore_errors=True)
    print(f"Repository cloned successfully to {local_path}")

except Exception as e:
    print(f"Error cloning repository: {e}")

# %%
# install requirements
if uv_path is not None:
    print("Installing requirements via uv...")
    !uv pip install -r {local_path}requirements.txt
else:
    print("Installing requirements via pip...")
    !pip install -r {local_path}requirements.txt

# %% [markdown]
# <div align="center">
# 
# ## Universidad de Puerto Rico Río Piedras
# 
# ## Departamento de Ciencia de Cómputos
# 
# ## CCOM6994: Herramientas Computacionales para el Análisis de Datos
# 
# </div>
# 
# <div align="center">
# 
# ------------------------------------------------------------------------------
# 
# # Analisis de Instalaciones de Paneles Solares Fotovoltaicos (FV) en EEUU y sus territorios:  
# 
# ## Distribución de Cubiertas de Suelo y Uso de Tierra de las instalacions  
# ## y Pruebas Estadisticas entre las Poblaciones con y sin presencia de sistemas FV  
# ## agrupadas por divisones del Censo 2020 
# </div>
# 
# -----------------------------------------------------------------------------------
# 
# #### Miembros del Equipo:
# 
# - Alejandro S. Vega Nogales
# - Francheska I. Lebrón López
# - Nicole M. Ramírez Mulero
# - Luis M. Fontán Rodríguez
# 
# 

# %%
import pandas as pd
import numpy as np
import duckdb

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

import geopandas as gpd
from shapely import wkt, wkb
from shapely.geometry import Point, box
from dotenv import load_dotenv

from IPython.display import display, HTML, clear_output
import ipywidgets as widgets
from ipywidgets import interact
from ipywidgets import HBox, VBox, Button, Text, Output, Layout

# %%
# configuracion de variables del archivo .env que descargamos de github

load_dotenv(dotenv_path=os.path.join(local_path,'.env'))

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)

# Plotting configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

DB_PATH = os.getenv('DEMO_DB_PATH', 'db/pv_project.duckdb')
DATASET_URI = os.getenv('CONSOLIDATED_PV_DATASET_FILE', 'https://eo-pv-elt.work/geoparquet/ccom6994_pv_dataset.parquet')
GEOM_TYPE = ''

# %% [markdown]
# ---
# 
# ## 📥 Tarea 0: Descargar nuestro conjunto e Intro Breve a Datos Geoespaciales
# ### Acerca de Nuestro Conjunto de Datos
# 
# Estamos trabajando con un **conjunto de datos consolidado global de paneles solares (PV)** que incluye:
# - Localizaciones y metadatos de instalaciones FV alrededor del mundo provenientes de publicaciones científicas y datos abiertos
# - Coordenadas geográficas y límites de polígonos de las instalaciones
# - Metadatos de instalación (área, capacidad, fechas de instalación)
# - Estos serán filtrados espacialmente para centrarse en los EE.UU. y sus territorios para nuestro análisis.
# 
# ### Formato de Datos: GeoParquet
# 
# **GeoParquet** es un formato eficiente y nativo en la nube para datos geoespaciales:
# - Almacenamiento columnar y compresión eficiente
# - Metadatos espaciales integrados (CRS, cajas delimitadoras)
# - Funciona con pandas, GeoPandas, DuckDB y otras herramientas
# - Almacenado en **Cloudflare R2** (compatible con S3; 5GB gratis mensuales)
# 
# ### Estrategia de Lectura
# 
# Usaremos **pandas** para leer el archivo parquet via https URL:
# 1. Leer el archivo parquet en un DataFrame de pandas
# 2. Convertir cadenas de geometría WKT o WKB a objetos Shapely que GeoPandas reconoce
# 3. Crear un GeoDataFrame con el sistema de referencia de coordenadas (CRS) adecuado

# %%
# Fetch and Filter
t1 = time.time()
pv_df = pd.read_parquet(DATASET_URI, engine='pyarrow')
t2 = time.time()
print(f"Datos se descargaron en {t2 - t1:.2f} segundos.")
global_count = len(pv_df)
print(f"En total tenemos {global_count:,} instalaciones fotovoltaicas del conjunto de datos global.")

# Display count by dataset
print("\nConteos por Fuente de los Datos:")
print(pv_df.groupby('dataset_name').size())

# Display basic info
print(f"\n📊 Conjunto de Datos:")
print(f"   Dimensiones: {pv_df.shape}")
print(f"   Espacio en Memoria: {pv_df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
print(f"\n   Columnas: {pv_df.dtypes}")

# Summary statistics for numeric columns
print("\n📈 Estadísticas Resumidas:\n")
display(pv_df.describe())

# %%
pv_df.sample(5)

# %% [markdown]
# **Columnas Claves:**
# 
# - `geometry` contiene strings WKT (*Well-Known Text*) que definen geometrías (Puntos, Líneas, Polígonos)
# - `centroid_lon`, `centroid_lat`: En datos geoespaciales, las geometrías se definen por coordenadas geográficas. Los centroides representan el "centro" de estas geometrías.
# - `area_m2` muestra el tamaño del panel en metros cuadrados
# - `h3_index_8` contiene el índice espacial H3 (lo exploraremos más adelante)
# - `installation_date` indica cuándo se instaló el panel solar
# - `capacity_kw` muestra la capacidad de generación nominal en kilovatios (kW)

# %% [markdown]
# ## GeoPandas, Geometrías, y Datos Geoespaciales
# 
# Como indica el nombre, GeoPandas extiende pandas para manejar datos geoespaciales de manera eficiente.  
# 
# La estructura de datos principal son el *GeoDataFrame* y la *GeoSeries*: clases derivadas de pandas DataFrame y Series que almacenan geometrías y tienen métodos para operaciones espaciales:
# - Data Members para: area, delimitación geométrica, centroide, etc
# - Funciones y relaciones espaciales: distancia, intersección, contención, unión espacial, etc
# - Manejo de sistemas de referencia de coordenadas (CRS): El manejo adecuado de CRS es crucial en datos geoespaciales para asegurar que las coordenadas y geometrías se interpreten correctamente.
# <!-- GeoPandas facilita la transformación entre diferentes CRS. -->

# %% [markdown]
# En la siguiente celda realizamos un pre-procesamiento requerido para convertir geometrías de su representación en almacénamiento (WKT o WKB) a representaciones de geometrías en memoria que GeoPandas puede utilizar.

# %%
# load WKT geometries as shapely objects
if pv_df['geometry'].dtype == 'object' and isinstance(pv_df['geometry'].iloc[0], str):
    pv_df['geometry'] = pv_df['geometry'].apply(wkt.loads)
    GEOM_TYPE = 'WKT'
# load WKB geometries as shapely objects
elif pv_df['geometry'].dtype == 'object' and isinstance(pv_df['geometry'].iloc[0], bytes):
    pv_df['geometry'] = pv_df['geometry'].apply(wkb.loads)
    GEOM_TYPE = 'WKB'

pv_gdf = gpd.GeoDataFrame(pv_df, geometry='geometry', crs='EPSG:4326')

# %% [markdown]
# #### Visualización de Muestras Aleatorias de Geometrías 

# %%
# Filter for Polygons only to avoid issues with MultiPolygons in simple plot
poly_subset = pv_gdf[pv_gdf.geometry.type == 'Polygon']
sample_polys = poly_subset.sample(n=6, random_state=42)

# create a simple button to re-run the plot function interactively
button = widgets.Button(description="Visualizar Muestra", layout=Layout(width='200px'))
output = widgets.Output()

def plot_poly_geoms(sample_polys):
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, (i, row) in enumerate(sample_polys.iterrows()):
        ax = axes[idx]
        # Plot the geometry
        gpd.GeoSeries([row.geometry]).plot(ax=ax, color='skyblue', edgecolor='black', alpha=0.6)
        display_name = row['dataset_name'].split('_')[0] + '_'.join(row['dataset_name'].split('_')[3::])
        ax.set_title(f"Fuente: {display_name}...\nArea: {row['area_m2']:.0f} m²")
        ax.axis('off')

    plt.tight_layout()
    # Save to a specific directory to avoid cluttering root
    # plt.savefig('/Volumes/Expanse/repos/ice-mELT_ducklake/notebooks/01_polygon_samples.png', dpi=150, bbox_inches='tight')
    # print("💾 Saved plot: 01_polygon_samples.png")
    plt.show()

def on_button_clicked(b):
    clear_output(wait=True)
    sample_polys = poly_subset.sample(n=6, random_state=random.randint(0,10000))
    plot_poly_geoms(sample_polys)
    display(button)

button.on_click(on_button_clicked)

with output:
    plot_poly_geoms(sample_polys)
    display(button)

display(output)

# %% [markdown]
# ### Manipulación de GeoDataFrames con indices
# 
# Además de las operaciones que hereda de Pandas como `.iloc` y `.loc`, GeoPandas permite la indexación basada en coordenadas espaciales usando `.cx` como demostramos a continuación limitando nuestro conjunto de datos a varias areas de interés (AOI) en los Estados Unidos y sus territorios.

# %%
# Collection of bounding boxes covering USA (including PR,HI,AK,Guam+USVI)
# Format: (xmin/west, ymin/south, xmax/east, ymax/north) in WGS84 degrees
AOI_bboxes = [
    os.getenv('CONUS_BBOX', '-125.0,24.0,-66.5,49.5'),  # CONUS
    os.getenv('ALASKA_BBOX', '-170.0,51.0,-130.0,72.0'),  # Alaska
    os.getenv('HAWAII_BBOX', '-161.0,18.5,-154.5,23.0'),  # Hawaii
    os.getenv('GUAM_BBOX', '144.5,13.2,145.0,13.6'),  # Guam
    os.getenv('USVI_BBOX', '-65.1,17.6,-64.5,18.6'),   # US Virgin Islands
    os.getenv('PUERTO_RICO_BBOX', '-67.3,17.9,-65.2,18.5')  # Puerto Rico
]
# Parse AOI bounding boxes into list of float tuples
AOI_bboxes = [ tuple(map(float, bbox_str.split(','))) for bbox_str in AOI_bboxes ]


t1 = time.time()
# Coordinate based indexer to select by intersection with bounding box: https://geopandas.org/en/stable/docs/user_guide/indexing.html
# apply indexing via all bboxes and concatenate into single gdf
pv_df = pd.concat([pv_gdf.cx[bbox[0]:bbox[2], bbox[1]:bbox[3]] for bbox in AOI_bboxes])

# handle geometry column for gdf since we converted to pandas during concat
# pv_df['geometry'] = pv_df['geometry'].apply(wkt.loads) if GEOM_TYPE == 'WKT' else pv_df['geometry'].apply(wkb.loads)

# use global geographic CRS for now; TODO: ID relevant CRS's for CONUS, caribbean, and pacific regions
pv_gdf = gpd.GeoDataFrame(pv_df, geometry='geometry', crs='EPSG:4326')
t2 = time.time()
# Filter by Bounding Box
print(f"Utilizamos indexación con coordenadas para eliminar {global_count - len(pv_gdf):,} instalaciones fuera de nuestras {len(AOI_bboxes)} AOI's en {t2 - t1:.2f} segundos...")

# %% [markdown]
# #### Vamos a darle un vistazo a nuestro nuevo subconjunto mediante un geo-scatter plot interactivo utilizando `plotly`:

# %%
# crea instancia de un scatter plot con un mapa base interactivo
fig = px.scatter_map(
    pv_gdf,
    lat='centroid_lat',
    lon='centroid_lon',
    zoom=2,
    height=800,
    color='dataset_name',
    opacity=0.45
)

# plot multiple AOIs in global map 
for AOI_BBOX in AOI_bboxes:

    # add bbox as line trace
    fig.add_trace(
        px.scatter_map(
            pd.DataFrame({
                'lon': [AOI_BBOX[0], AOI_BBOX[2], AOI_BBOX[2], AOI_BBOX[0], AOI_BBOX[0]],
                'lat': [AOI_BBOX[1], AOI_BBOX[1], AOI_BBOX[3], AOI_BBOX[3], AOI_BBOX[1]]
            }),
            lat='lat',
            lon='lon'
        ).data[0]
    )

    # update bbox line style
    fig.data[-1].update(
        mode='lines',
        line=dict(color='red', width=2),
        showlegend=False
    )

# show final map with AOI bounding boxes
fig.update_layout(title="Localizaciones de Instalaciones Fotovoltaicas en el Conjunto de Datos Consolidado con AOI's Destacadas")
# config map projection and other geo config
fig.update_geos(projection_type="orthographic")
# use pre-configred mapbox style
# fig.update_layout(mapbox_style="satellite-streets") 
fig.show()

# %% [markdown]
# #### Deduplicación de instalaciones fotovoltaicas basadas en geometrías
# 
# Ya que estamos trabajando con un conjunto de datos consolidado de múltiples fuentes, es posible que existan instalaciones fotovoltaicas duplicadas o muy cercanas entre sí. Para asegurar la integridad de nuestro análisis, implementaremos una estrategia de deduplicación basada en la proximidad espacial.

# %%
# Check for exact duplicates in geometry
initial_count = len(pv_gdf)
print(f"   Conteo de filas inicial: {initial_count:,}")

# see here for recommended approach for deduplication: https://geopandas.org/en/latest/docs/user_guide/how_to.html
pv_gdf['geometry'] = pv_gdf['geometry'].normalize()
# Note: This can be slow for large datasets as it compares every geometry
pv_gdf_dedup = pv_gdf.drop_duplicates(subset=['geometry'], inplace=False, ignore_index=True)
num_duplicates = initial_count - len(pv_gdf_dedup)  


print(f"   Duplicados con geometrías idénticas: {num_duplicates:,}")

# pv_gdf_dedup = pv_gdf[~duplicates_mask].copy()
print(f"   Conteo después de la deduplicación con GeoPandas: {len(pv_gdf_dedup):,}")

# remove invalid geometries before saving
og_len = len(pv_gdf_dedup)
pv_gdf_dedup = pv_gdf_dedup[pv_gdf_dedup.is_valid].copy()
print(f"   Eliminamos {og_len - len(pv_gdf_dedup):,} geometrías inválidas antes de guardar.")

# %% [markdown]
# ### Ahora almacenamos el conjunto de datos en DuckDB, una base de datos en un solo archivo para el resto del pre-procesamiento fuera de esta libreta

# %%
# Connect to persistent DB file
con_persistent = duckdb.connect(DB_PATH)
# install spatial and httpfs extensions 
con_persistent.execute("INSTALL spatial; LOAD spatial;")

# Convert geometry to WKB for DuckDB storage 
pv_gdf_dedup['geometry'] = pv_gdf_dedup['geometry'].apply(lambda geom: geom.wkb)

# process reading WKB/WKT geometry into Geometry data type for duckdb
# save as WKT for compatibility and easier retrieval wi
con_persistent.execute("""
    CREATE OR REPLACE TABLE processed_pv_data AS
        SELECT ST_GeomFromWKB(geometry) as geometry, * EXCLUDE (geometry) FROM pv_gdf_dedup
    """)
print("   ✅ Data saved to table 'processed_pv_data'")
# delete dataframes so far since we'll load the final processed data from duckdb in next steps
del pv_gdf_dedup
del pv_gdf
del pv_df

# %% [markdown]
# ## ⚙️ 1: Pre-procesamiento para relacionar nuestro conjunto de datos con los Datos Geográficos y Demográficos
# 
# Uno de los aspectos más poderosos de los datos geoespaciales es la capacidad de relacionar diferentes conjuntos de datos basados en su proximidad espacial. En esta sección, realizaremos un pre-procesamiento para relacionar nuestro conjunto de datos de instalaciones fotovoltaicas con las fuentes de las variables geográficas y demográficas que utilizaremos en nuestro análisis estadístico a continuación.
# 
# Estas fuentes incluyen:
# 1. Censo de los EE.UU. 2020 a nivel de estado, county, y census tracto  
# 
# 2. Cubiertas de Suelo con resolución de 10 metros derivadas del ESA WorldCover 2020 (distribuido por [Overture Maps](https://docs.overturemaps.org/blog/2024/05/16/land-cover/))  
# 
# 3. Clasificaciones del Uso de Tierra obtenidas a través de OpenStreetMap (OSM) y procesadas por [Overture Maps](https://docs.overturemaps.org/guides/base/)

# %% [markdown]
# ## 📚 Conceptual Framework: Socioeconomic Factors & Solar Adoption
# 
# Research suggests several socioeconomic factors influence solar panel adoption:
# 
# | Factor | Census Variable Groups | Hypothesis |
# |--------|----------------------|------------|
# | **Income** | B19013 (Median HH Income) | Higher income → higher adoption (upfront costs) |
# | **Poverty** | B17001 (Poverty Status) | Higher poverty → lower adoption |
# | **Housing** | B25077 (Home Value), B25003 (Tenure) | Homeowners more likely to install solar |
# | **Education** | B15003 (Educational Attainment) | Higher education → more awareness |
# | **Demographics** | B01003 (Population), B03002 (Race/Ethnicity) | Adoption patterns may vary by demographics |

# %%
# run our script for reverse geocoding and attaching the census state, county, and tract columns we need to match with census data during our statistical analysis
!python {local_path}notebooks/02_geocoding_census_geographies.py

# %%
# preview census-enriched PV data from our local database
print("📥 Cargando datos enriquecidos con identificadores del Censo desde la base de datos...")

con = duckdb.connect(DB_PATH)
con.execute("INSTALL spatial; LOAD spatial;")

# Query the census-enriched PV data
pv_census_df = con.execute("""
    SELECT 
        ST_AsText(geometry) as geometry,
        * EXCLUDE (geometry)
    FROM census_enriched_pv_data
""").df()

print(f"✅ Cargamos {len(pv_census_df):,} instalaciones FV con identificadores del Censo.")
print(f"\n📊 Columnas añadidas por el script de geocodificación:")
census_cols = ['country_code', 'rg_state', 'rg_county', 'STATE_FIPS', 'STATE_ABBR', 'STATE_GEOID', 
               'COUNTY_FIPS', 'COUNTY_GEOID', 'COUNTY_NAME', 'TRACT_CODE', 'TRACT_GEOID', 'TRACT_NAME']
print(f"   {census_cols}")

# Show sample of enriched data
print(f"\n📋 Muestra de datos enriquecidos:")
display(pv_census_df[['dataset_name', 'centroid_lat', 'centroid_lon', 'STATE_ABBR', 'COUNTY_NAME', 'TRACT_GEOID']].sample(5))

# Show counts by state
print(f"\n📊 Distribución de instalaciones FV por estado:")
state_counts = pv_census_df.groupby('STATE_ABBR').size().sort_values(ascending=False)
display(state_counts.head(10))

con.close()


# %% [markdown]
# ## 🌍 What is Land Use-Land Cover (LULC)?
# 
# **Land Cover** refers to the physical and biological cover of the Earth's surface,
# including vegetation, water, bare soil, and artificial structures. It answers the
# question: **"What is physically present on the ground?"**
# 
# ### Common Land Cover Classes
# | Class | Description | Examples |
# |-------|-------------|----------|
# | **Forest** | Tree-dominated areas | Deciduous, coniferous, mixed forests |
# | **Shrub** | Woody vegetation <5m tall | Chaparral, scrubland |
# | **Grass** | Herbaceous vegetation | Prairies, pastures, lawns |
# | **Crop** | Agricultural land | Farms, orchards, vineyards |
# | **Urban** | Built-up areas | Cities, roads, parking lots |
# | **Barren** | Minimal vegetation | Deserts, rock outcrops, beaches |
# | **Wetland** | Water-saturated areas | Marshes, swamps, bogs |
# | **Water** | Open water bodies | Lakes, rivers, oceans |
# 
# 
# <!-- ## 🎯 Why Land Cover Matters for Solar PV Analysis
# 
# Understanding land cover context around solar installations helps us:
# 
# 1. **Site Characterization**: What land types host solar farms?
# 2. **Land Use Change**: Are panels replacing cropland, forest, or developed areas?
# 3. **Environmental Impact**: Assess ecological footprint of solar development
# 4. **Policy Analysis**: Identify patterns in permitting across land types
# 5. **Predictive Modeling**: Which land cover types are most likely to have future solar? -->
# <!-- --- -->
# 
# ## 📊 Overture Maps Land Cover Schema
# 
# Overture Maps provides vector-based land cover with the following structure:
# 
# | Column | Type | Description |
# |--------|------|-------------|
# | `id` | VARCHAR | Unique feature identifier |
# | `geometry` | POLYGON | Feature boundary |
# | `subtype` | VARCHAR | Land cover class (forest, urban, crop, etc.) |
# | `cartography` | STRUCT | Rendering hints (min_zoom, max_zoom) |
# | `sources` | ARRAY | Data provenance |
# 
# **Available subtypes**: `barren`, `crop`, `forest`, `grass`, `mangrove`, `moss`, 
# `shrub`, `snow`, `urban`, `wetland`

# %%

!python {local_path}notebooks/04_overture_land_cover_fetch.py

# %%
# preview enhanced data

# %% [markdown]
# ## 🏘️ What is Land Use?
# 
# While **Land Cover** describes *what* is physically on the ground, **Land Use** 
# describes *how* humans use that land. It answers the question: 
# **"What is this land being used for?"**
# 
# ### Common Land Use Categories
# 
# | Category | Description | Solar Relevance |
# |----------|-------------|-----------------|
# | **Residential** | Housing areas | Rooftop solar potential |
# | **Commercial** | Retail, offices | Large flat roofs |
# | **Industrial** | Manufacturing, warehouses | Ground-mount opportunities |
# | **Agricultural** | Farms, ranches | Agrivoltaics potential |
# | **Recreation** | Parks, sports facilities | Limited development |
# | **Protected** | Conservation areas | Generally restricted |
# | **Developed** | General built-up areas | Mixed potential |
# 
# <!-- ---
# 
# ## 🎯 Why Land Use Matters for Solar PV Analysis
# 
# Land use data helps us understand:
# 
# 1. **Zoning Compatibility**: Is solar allowed in this land use zone?
# 2. **Development Patterns**: Which land use types attract solar investment?
# 3. **Dual-Use Opportunities**: Agrivoltaics (solar + farming), floating solar, etc.
# 4. **Policy Implications**: How do land use policies affect solar adoption?
# 5. **Future Projections**: Where might new solar development occur?
# 
# --- -->
# 
# ## 📊 Overture Maps Land Use Schema
# 
# Overture Maps provides vector-based land use with a hierarchical classification:
# 
# | Column | Type | Description |
# |--------|------|-------------|
# | `id` | VARCHAR | Unique feature identifier |
# | `geometry` | POLYGON | Feature boundary |
# | `subtype` | VARCHAR | Primary category (residential, agriculture, etc.) |
# | `class` | VARCHAR | Detailed sub-category (farmland, retail, etc.) |
# | `names` | STRUCT | Feature names (if available) |
# | `sources` | ARRAY | Data provenance |

# %%
!python {local_path}notebooks/05_overture_land_use_fetch.py

# %%
# preview what we've added

# %% [markdown]
# ## 📊 3: Análisis Exploratorio de Datos (EDA), Normalidad, y 

# %% [markdown]
# ## 📚 References & Documentation
# 
# - [censusdis Introduction](https://censusdis.readthedocs.io/en/latest/intro.html)
# - [Census API Datasets](https://api.census.gov/data/2020.html)
# - [Exploring Variables](https://censusdis.readthedocs.io/en/latest/nb/Exploring%20Variables.html)
# - [Data With Geometry](https://censusdis.readthedocs.io/en/latest/nb/Data%20With%20Geometry.html)
# - [ipywidgets Widget List](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html)


