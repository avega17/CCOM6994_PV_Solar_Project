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
# types of geometries in our data
geom_types = pv_gdf.geometry.geom_type.value_counts()
print("Tipos de geometrías en nuestros datos:")
display(geom_types)

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
print(f"   Eliminamos {og_len - len(pv_gdf_dedup):,} geometrías inválidas antes de guardar.\n")
print(f"Distribución de los datos por fuente después de la limpieza:")
display(pv_gdf_dedup['dataset_name'].value_counts())

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
print("   ✅ Almacenamos nuestros datos en la tabla 'processed_pv_data' de DuckDB")
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
# !python {local_path}notebooks/04_overture_land_cover_fetch.py

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
# !python {local_path}notebooks/05_overture_land_use_fetch.py

# %%
# preview what we've added

# %% [markdown]
# ## 📊 3: Análisis Exploratorio de Datos (EDA), Normalidad, y Análisis de Poder Estadístico
# 
# En esta sección realizaremos:
# 1. Descarga de variables del Censo para **todos** los census tracts de EEUU
# 2. Unión con nuestros datos de instalaciones FV para crear grupos de comparación
# 3. Análisis exploratorio de las distribuciones de variables
# 4. Pruebas de normalidad (Shapiro-Wilk, Kolmogorov-Smirnov)
# 5. Análisis de poder estadístico para determinar tamaños de muestra adecuados

# %%
# Import additional libraries for statistical analysis
import scipy.stats as stats
import statsmodels.stats.power as smp
import statsmodels.api as sm
import censusdis.data as ced
import censusdis.maps as cem
from censusdis.datasets import ACS5
from censusdis.states import ALL_STATES_AND_DC, NAMES_FROM_IDS, ABBREVIATIONS_FROM_IDS

# %% [markdown]
# ### 3.0 Análisis de Normalidad de Variables del Dataset PV
# 
# Antes de proceder con el análisis del Censo, verificamos la distribución de las variables 
# clave en nuestro dataset de instalaciones fotovoltaicas:
# - `area_m2`: Área de la instalación en metros cuadrados derivada de las geometrías
# - `source_area_m2`: Área reportada por la fuente original
# - `capacity_mw`: Capacidad de generación en megavatios
# 
# Esta información es útil para entender la naturaleza de nuestros datos y seleccionar 
# métodos estadísticos apropiados.

# %%
# Normality analysis for PV installation variables
print("📊 Análisis de Normalidad: Variables del Dataset PV")
print("=" * 80)

pv_analysis_vars = ['area_m2', 'source_area_m2', 'capacity_mw']

# First, check data availability
print("\n📋 Disponibilidad de datos:")
for var in pv_analysis_vars:
    if var in pv_census_df.columns:
        non_null = pv_census_df[var].notna().sum()
        pct = non_null / len(pv_census_df) * 100
        print(f"   {var}: {non_null:,} valores válidos ({pct:.1f}%)")
    else:
        print(f"   {var}: ⚠️ No disponible en el dataset")

# Q-Q Plots for PV variables
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

for idx, var in enumerate(pv_analysis_vars):
    ax = axes[idx]
    if var in pv_census_df.columns:
        data = pv_census_df[var].dropna()
        if len(data) > 0:
            # Use sample for visualization (max 5000 points for Q-Q plot)
            sample_data = data.sample(n=min(5000, len(data)), random_state=42)
            stats.probplot(sample_data, dist="norm", plot=ax)
            ax.set_title(f'Q-Q Plot: {var}\n(n={len(data):,})', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Sin datos', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{var}: Sin datos')
    else:
        ax.text(0.5, 0.5, 'Variable no disponible', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{var}')

plt.tight_layout()
plt.suptitle('Q-Q Plots: Variables del Dataset PV (vs Distribución Normal)', y=1.02, fontsize=12)
plt.show()

# %%
# Statistical normality tests for PV variables
print("\n📊 Pruebas de Normalidad para Variables PV")
print("=" * 80)
print(f"{'Variable':<20} {'N':<12} {'Shapiro-Wilk p':<18} {'K-S p':<18} {'Normal?':<10} {'Recomendación'}")
print("-" * 100)

pv_normality_results = []

for var in pv_analysis_vars:
    if var in pv_census_df.columns:
        data = pv_census_df[var].dropna()
        n = len(data)
        
        if n > 0:
            # Shapiro-Wilk (sample for large datasets)
            sample_size = min(5000, n)
            sample = data.sample(n=sample_size, random_state=42)
            shapiro_stat, shapiro_p = stats.shapiro(sample)
            
            # K-S test (standardize first)
            standardized = (data - data.mean()) / data.std()
            ks_stat, ks_p = stats.kstest(standardized, 'norm')
            
            is_normal = (shapiro_p > 0.05 and ks_p > 0.05)
            recommendation = "Paramétrica" if is_normal else "No paramétrica o transformar"
            
            pv_normality_results.append({
                'variable': var,
                'n': n,
                'shapiro_p': shapiro_p,
                'ks_p': ks_p,
                'is_normal': is_normal
            })
            
            print(f"{var:<20} {n:<12,} p={shapiro_p:.2e}  {'✓' if shapiro_p > 0.05 else '✗'}    p={ks_p:.2e}  {'✓' if ks_p > 0.05 else '✗'}    {'Sí' if is_normal else 'No':<10} {recommendation}")
        else:
            print(f"{var:<20} {'N/A':<12} {'Sin datos':<18} {'Sin datos':<18} {'N/A':<10}")
    else:
        print(f"{var:<20} {'N/A':<12} {'No disponible':<18} {'No disponible':<18} {'N/A':<10}")

print("-" * 100)
print("\n💡 Interpretación:")
print("   - Las variables de área y capacidad típicamente siguen distribuciones log-normales o de cola pesada")
print("   - Para análisis paramétricos, considerar transformación logarítmica: log(x + 1)")
print("   - Para comparaciones de grupos, usar pruebas no paramétricas (Mann-Whitney U, Kruskal-Wallis)")

# %%
# Distribution histograms with log scale option
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

for idx, var in enumerate(pv_analysis_vars):
    # Top row: Original scale
    ax_orig = axes[0, idx]
    # Bottom row: Log scale
    ax_log = axes[1, idx]
    
    if var in pv_census_df.columns:
        data = pv_census_df[var].dropna()
        data_positive = data[data > 0]  # Log requires positive values
        
        if len(data) > 0:
            # Original scale histogram
            sns.histplot(data, kde=True, ax=ax_orig, color='steelblue', alpha=0.7)
            ax_orig.set_title(f'{var}\n(Escala Original)', fontsize=10)
            ax_orig.axvline(data.mean(), color='red', linestyle='--', label=f'Media: {data.mean():.2e}')
            ax_orig.axvline(data.median(), color='green', linestyle=':', label=f'Mediana: {data.median():.2e}')
            ax_orig.legend(fontsize=8)
            
            # Log scale histogram (for positive values)
            if len(data_positive) > 0:
                log_data = np.log10(data_positive)
                sns.histplot(log_data, kde=True, ax=ax_log, color='darkgreen', alpha=0.7)
                ax_log.set_title(f'log10({var})\n(Escala Logarítmica)', fontsize=10)
                ax_log.set_xlabel(f'log10({var})')
            else:
                ax_log.text(0.5, 0.5, 'Sin valores positivos', ha='center', va='center', transform=ax_log.transAxes)
        else:
            ax_orig.text(0.5, 0.5, 'Sin datos', ha='center', va='center', transform=ax_orig.transAxes)
            ax_log.text(0.5, 0.5, 'Sin datos', ha='center', va='center', transform=ax_log.transAxes)
    else:
        ax_orig.text(0.5, 0.5, 'No disponible', ha='center', va='center', transform=ax_orig.transAxes)
        ax_log.text(0.5, 0.5, 'No disponible', ha='center', va='center', transform=ax_log.transAxes)

plt.tight_layout()
plt.suptitle('Distribución de Variables PV: Escala Original vs Logarítmica', y=1.02, fontsize=12)
plt.show()

print("\n📊 Estadísticas descriptivas de variables PV:")
pv_stats = pv_census_df[pv_analysis_vars].describe()
print(pv_stats.to_string())

# %% [markdown]
# ### 3.1 Descarga de Variables del Censo para Todos los Census Tracts
# 
# Utilizamos la API del Censo de EEUU vía `censusdis` para obtener variables demográficas y socioeconómicas:
# 
# | Variable | Código | Descripción |
# |----------|--------|-------------|
# | total_population | B01003_001E | Población total |
# | median_household_income | B19013_001E | Ingreso mediano del hogar |
# | population_below_poverty | B17001_002E | Población bajo nivel de pobreza |
# | median_home_value | B25077_001E | Valor mediano de vivienda |
# | pct_unemployment | DP03_0009PE | % desempleo |
# | pct_bachelors_or_higher | DP02_0068PE | % con bachillerato o más |
# | pct_hispanic | DP05_0071PE | % población hispana |

# %%
# Define census variables to fetch
CENSUS_YEAR = 2022
ACS_DATASET = "acs/acs5"
ACS_PROFILE = "acs/acs5/profile"

# Detailed table variables
VARS_DETAILED = {
    "B01003_001E": "total_population",
    "B19013_001E": "median_household_income",
    "B17001_002E": "population_below_poverty",
    "B25077_001E": "median_home_value",
    "B25001_001E": "total_housing_units",  # For solar adoption rate calculation
}

# Profile variables (percentages)
VARS_PROFILE = {
    "DP03_0009PE": "pct_unemployment",
    "DP02_0068PE": "pct_bachelors_or_higher",
    "DP05_0071PE": "pct_hispanic",
}

# Get unique states from our PV dataset
pv_states = pv_census_df['STATE_FIPS'].unique().tolist()
print(f"📊 Estados con instalaciones FV en nuestro dataset: {len(pv_states)}")
print(f"   {[ABBREVIATIONS_FROM_IDS.get(s, s) for s in pv_states]}")

# NOTE: Some territories (Guam=66, Virgin Islands=78) do not have ACS5 tract data
# Puerto Rico (72) does have ACS5 tract data
TERRITORIES_NO_ACS5 = {"66", "78"}  # GU, VI

# %%
# Fetch census data for all tracts in states with PV installations
print("📥 Descargando variables del Censo para todos los census tracts...")
print("   Esto puede tomar unos minutos dependiendo de la conexión...")
print(f"   ⚠️ Territorios sin datos ACS5: {[ABBREVIATIONS_FROM_IDS.get(s, s) for s in TERRITORIES_NO_ACS5]}\n")

all_tracts_detailed = []
all_tracts_profile = []
skipped_territories = []

t1 = time.time()
for state_fips in pv_states:
    state_name = NAMES_FROM_IDS.get(state_fips, state_fips)
    
    # Skip territories without ACS5 tract data
    if state_fips in TERRITORIES_NO_ACS5:
        print(f"   ⏭️  Omitiendo {state_name} (sin datos ACS5 para tracts)")
        skipped_territories.append(state_fips)
        continue
    
    print(f"   Descargando datos para {state_name}...", end=" ")
    
    try:
        # Fetch detailed variables with geometry
        df_det = ced.download(
            ACS_DATASET,
            CENSUS_YEAR,
            ["NAME", "GEO_ID"] + list(VARS_DETAILED.keys()),
            state=state_fips,
            county="*",
            tract="*",
            with_geometry=True
        )
        
        # Fetch profile variables (no geometry needed)
        df_prof = ced.download(
            ACS_PROFILE,
            CENSUS_YEAR,
            list(VARS_PROFILE.keys()),
            state=state_fips,
            county="*",
            tract="*",
            with_geometry=False
        )
        
        all_tracts_detailed.append(df_det)
        all_tracts_profile.append(df_prof)
        print(f"✓ ({len(df_det):,} tracts)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        skipped_territories.append(state_fips)

t2 = time.time()
print(f"\n✅ Descarga completada en {t2-t1:.1f} segundos.")
if skipped_territories:
    print(f"⚠️  Territorios omitidos: {[ABBREVIATIONS_FROM_IDS.get(s, s) for s in skipped_territories]}")

# %%
# Combine all states into single dataframes
census_detailed_gdf = gpd.GeoDataFrame(pd.concat(all_tracts_detailed, ignore_index=True))
census_profile_df = pd.concat(all_tracts_profile, ignore_index=True)

# Merge detailed and profile data
census_all_tracts = census_detailed_gdf.merge(
    census_profile_df,
    on=["STATE", "COUNTY", "TRACT"],
    how="left"
)

# Rename columns to human-friendly English names
rename_map = {**VARS_DETAILED, **VARS_PROFILE}
census_all_tracts = census_all_tracts.rename(columns=rename_map)

# Create TRACT_GEOID for joining with PV data (STATE + COUNTY + TRACT)
census_all_tracts['TRACT_GEOID'] = census_all_tracts['STATE'] + census_all_tracts['COUNTY'] + census_all_tracts['TRACT']

print(f"📊 Total de census tracts descargados: {len(census_all_tracts):,}")
print(f"   Columnas: {list(census_all_tracts.columns)}")
display(census_all_tracts.head())

# %% [markdown]
# ### 3.2 Creación del DataFrame de Análisis
# 
# Unimos los datos del Censo con nuestro conteo de instalaciones FV por tract para crear:
# - `n_solar`: Cantidad de instalaciones FV en el tract
# - `area_km2`: Área del tract en km²
# - `solar_density`: Densidad de instalaciones (n_solar / area_km2) - útil para análisis espacial
# - `solar_adoption_rate`: Instalaciones por cada 1,000 unidades de vivienda - **mejor métrica para comparaciones**
# - `solar_capacity_per_capita`: Capacidad MW por cada 1,000 residentes - mide intensidad de generación
# - `log_solar_count`: Transformación logarítmica log(n_solar + 1) - ver nota abajo
# - `has_solar`: Variable binaria (1 si tiene instalaciones, 0 si no)
# 
# > **Nota**: `solar_adoption_rate` es preferible a `solar_density` porque normaliza por oportunidad
# > (unidades de vivienda) en lugar de área, haciéndola comparable entre contextos urbanos y rurales.
# 
# ---
# 
# #### 📐 Sobre la Transformación Logarítmica (`log_solar_count`)
# 
# La transformación `log(n + 1)` (también llamada *log1p*) es una técnica estadística común para:
# 
# 1. **Manejar ceros**: A diferencia de `log(n)`, `log(n + 1)` está definida para n=0 (resulta en 0)
# 
# 2. **Reducir asimetría (skewness)**: Las distribuciones de conteos suelen tener "cola derecha" 
#    (muchos valores bajos, pocos valores muy altos). La transformación logarítmica comprime 
#    los valores altos y expande los bajos, aproximando a una distribución más simétrica.
# 
# 3. **Estabilizar varianza**: En datos de conteo, la varianza suele aumentar con la media.
#    La transformación log estabiliza esta varianza (homocedasticidad). [???]
# 
# 4. **Interpretación multiplicativa**: En regresión, coeficientes con log se interpretan como
#    cambios porcentuales: "un aumento de 1 unidad en X se asocia con un aumento de β% en el conteo"
# 
# **Cuándo usarla:**
# - En regresiones lineales donde la variable dependiente es un conteo
# - Cuando los residuos muestran heterocedasticidad [?????]
# - Para visualizaciones donde los valores extremos comprimen el resto
# 
# **Referencias:**
# - [Log Transformation in Statistics](https://en.wikipedia.org/wiki/Data_transformation_(statistics)#Log_transformation)
# - [When to Use Log Transforms](https://stats.stackexchange.com/questions/18844/when-and-why-should-you-take-the-log-of-a-distribution)

# %%
# Count solar installations per tract and aggregate capacity
solar_counts = pv_census_df.groupby('TRACT_GEOID').size().reset_index(name='n_solar')

# Aggregate total capacity per tract (if capacity_mw is available)
if 'capacity_mw' in pv_census_df.columns:
    solar_capacity = pv_census_df.groupby('TRACT_GEOID')['capacity_mw'].sum().reset_index(name='total_capacity_mw')
    solar_counts = solar_counts.merge(solar_capacity, on='TRACT_GEOID', how='left')
    solar_counts['total_capacity_mw'] = solar_counts['total_capacity_mw'].fillna(0)
    print(f"📊 Tracts con instalaciones FV: {len(solar_counts):,}")
    print(f"   Capacidad total agregada: {solar_counts['total_capacity_mw'].sum():,.2f} MW")
else:
    solar_counts['total_capacity_mw'] = 0
    print(f"📊 Tracts con instalaciones FV: {len(solar_counts):,}")
    print(f"   ⚠️ capacity_mw no disponible - solar_capacity_per_capita será 0")

# Merge with census data (left join to keep all tracts)
analysis_df = census_all_tracts.merge(
    solar_counts,
    on='TRACT_GEOID',
    how='left'
)

# Fill NaN for tracts without solar installations
analysis_df['n_solar'] = analysis_df['n_solar'].fillna(0).astype(int)

# Calculate area in km² using appropriate projection (Albers Equal Area for CONUS)
# First ensure we have geometry and proper CRS
analysis_gdf = gpd.GeoDataFrame(analysis_df, geometry='geometry', crs='EPSG:4269')
analysis_gdf = analysis_gdf.to_crs(epsg=5070)  # NAD83 / Conus Albers for area calculation

# Calculate area; 1km2 = 1,000,000 m2
analysis_gdf['area_km2'] = analysis_gdf.geometry.area / 1_000_000

# Calculate solar density and binary indicator
analysis_gdf['solar_density'] = analysis_gdf['n_solar'] / analysis_gdf['area_km2']
analysis_gdf['has_solar'] = (analysis_gdf['n_solar'] > 0).astype(int)

# Calculate solar adoption rate (installations per 1,000 housing units)
# This is a better metric than solar_density because it normalizes by opportunity (housing stock)
# rather than area, making it comparable across urban and rural contexts
analysis_gdf['solar_adoption_rate'] = np.where(
    analysis_gdf['total_housing_units'] > 0,
    (analysis_gdf['n_solar'] / analysis_gdf['total_housing_units']) * 1000,
    0
)

# Calculate solar capacity per capita (MW per 1,000 residents)
# This captures both the quantity AND size of installations
# Useful for understanding total generation potential relative to population
if 'total_capacity_mw' in analysis_gdf.columns:
    analysis_gdf['solar_capacity_per_capita'] = np.where(
        analysis_gdf['total_population'] > 0,
        (analysis_gdf['total_capacity_mw'] / analysis_gdf['total_population']) * 1000,
        0
    )
else:
    analysis_gdf['solar_capacity_per_capita'] = 0

# Log-transformed solar count: log(n + 1)
# This transformation:
# - Handles zeros (log(0+1) = 0)
# - Reduces skewness in count distributions
# - Stabilizes variance for regression analysis
# - Enables multiplicative interpretation in models
analysis_gdf['log_solar_count'] = np.log1p(analysis_gdf['n_solar'])

# Convert back to WGS84 for visualization
analysis_gdf = analysis_gdf.to_crs(epsg=4326)

print(f"\n📊 DataFrame de Análisis Final:")
print(f"   Total tracts: {len(analysis_gdf):,}")
print(f"   Tracts CON instalaciones FV: {analysis_gdf['has_solar'].sum():,}")
print(f"   Tracts SIN instalaciones FV: {(~analysis_gdf['has_solar'].astype(bool)).sum():,}")

# Show summary statistics for all derived variables
print("\n📈 Estadísticas de variables derivadas:")
derived_vars = ['n_solar', 'log_solar_count', 'area_km2', 'solar_density', 'solar_adoption_rate', 'solar_capacity_per_capita']
derived_stats = analysis_gdf[derived_vars].describe()
print(derived_stats.to_string())

# Check for potential issues with solar_density (very small values)
print("\n🔍 Diagnóstico de métricas para tracts CON instalaciones FV:")
with_solar = analysis_gdf[analysis_gdf['has_solar'] == 1]
print(f"   solar_density:           {with_solar['solar_density'].min():.2e} to {with_solar['solar_density'].max():.4f} (inst/km²)")
print(f"   solar_adoption_rate:     {with_solar['solar_adoption_rate'].min():.2f} to {with_solar['solar_adoption_rate'].max():.2f} (per 1,000 housing units)")
print(f"   solar_capacity_per_capita: {with_solar['solar_capacity_per_capita'].min():.4f} to {with_solar['solar_capacity_per_capita'].max():.4f} (MW per 1,000 pop)")
print(f"   log_solar_count:         {with_solar['log_solar_count'].min():.4f} to {with_solar['log_solar_count'].max():.4f}")
print(f"\n   Medianas (tracts con FV):")
print(f"   - solar_adoption_rate:     {with_solar['solar_adoption_rate'].median():.2f} per 1,000 units")
print(f"   - solar_capacity_per_capita: {with_solar['solar_capacity_per_capita'].median():.4f} MW per 1,000 pop")
print(f"   - log_solar_count:         {with_solar['log_solar_count'].median():.4f}")

# Show summary statistics with better formatting
display(analysis_gdf[['total_population', 'median_household_income', 'population_below_poverty', 
                       'median_home_value', 'total_housing_units', 'pct_unemployment', 'pct_bachelors_or_higher', 
                       'pct_hispanic', 'n_solar', 'solar_adoption_rate', 'solar_capacity_per_capita', 'log_solar_count']].describe())

# %%
# Handle missing values - drop rows with missing census data for analysis
# Note: Census API returns -666666666 for missing/suppressed data
CENSUS_MISSING_VALUE = -666666666

analysis_vars = ['total_population', 'median_household_income', 'population_below_poverty',
                 'median_home_value', 'total_housing_units', 'pct_unemployment', 'pct_bachelors_or_higher', 'pct_hispanic']

print("📊 Valores faltantes en variables del Censo:")
for var in analysis_vars:
    missing_count = ((analysis_gdf[var].isna()) | (analysis_gdf[var] == CENSUS_MISSING_VALUE)).sum()
    pct = missing_count / len(analysis_gdf) * 100
    print(f"   {var}: {missing_count:,} ({pct:.2f}%)")

# Replace census missing values with NaN
for var in analysis_vars:
    analysis_gdf.loc[analysis_gdf[var] == CENSUS_MISSING_VALUE, var] = np.nan

# Create clean analysis dataset (drop rows with any missing values in analysis variables)
analysis_clean = analysis_gdf.dropna(subset=analysis_vars).copy()
print(f"\n✅ Dataset limpio para análisis: {len(analysis_clean):,} tracts")
print(f"   Eliminados por datos faltantes: {len(analysis_gdf) - len(analysis_clean):,}")

# %% [markdown]
# ### 3.3 Visualización de Distribuciones y Correlaciones

# %%
# Distribution plots for key variables
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, var in enumerate(analysis_vars):
    ax = axes[idx]
    data = analysis_clean[var].dropna()
    
    sns.histplot(data, kde=True, ax=ax, color='steelblue', alpha=0.7)
    ax.set_title(f'Distribución: {var}', fontsize=10)
    ax.set_xlabel('')
    
    # Add mean and median lines
    ax.axvline(data.mean(), color='red', linestyle='--', label=f'Media: {data.mean():.2f}')
    ax.axvline(data.median(), color='green', linestyle=':', label=f'Mediana: {data.median():.2f}')
    ax.legend(fontsize=8)

# Remove empty subplot
axes[-1].axis('off')
plt.tight_layout()
plt.suptitle('Distribución de Variables del Censo', y=1.02, fontsize=14)
plt.show()

# %%
# Correlation matrix
plt.figure(figsize=(10, 8))
corr_vars = analysis_vars + ['n_solar', 'solar_density', 'has_solar']
corr_matrix = analysis_clean[corr_vars].corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f',
            square=True, linewidths=0.5)
plt.title('Matriz de Correlaciones: Variables del Censo e Instalaciones FV')
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 3.4 Pruebas de Normalidad
# 
# Antes de seleccionar las pruebas estadísticas apropiadas, debemos verificar si nuestras variables siguen una distribución normal. Utilizamos:
# 
# 1. **Q-Q Plots**: Visualización gráfica
# 2. **Shapiro-Wilk Test**: Prueba estadística (mejor para n < 5000)
# 3. **Kolmogorov-Smirnov Test**: Prueba estadística (para muestras grandes)
# 
# **Interpretación de p-value:**
# - p > 0.05: No rechazamos H₀ → Los datos podrían ser normales
# - p ≤ 0.05: Rechazamos H₀ → Los datos NO son normales

# %%
# Q-Q Plots for normality assessment
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, var in enumerate(analysis_vars):
    ax = axes[idx]
    data = analysis_clean[var].dropna()
    
    # Use a sample if data is too large for visualization
    sample_data = data.sample(n=min(1000, len(data)), random_state=42)
    
    stats.probplot(sample_data, dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot: {var}', fontsize=10)

axes[-1].axis('off')
plt.tight_layout()
plt.suptitle('Q-Q Plots para Evaluación de Normalidad', y=1.02, fontsize=14)
plt.show()

# %%
# Statistical normality tests
print("📊 Pruebas de Normalidad para Variables del Censo")
print("=" * 80)
print(f"{'Variable':<30} {'Shapiro-Wilk':<20} {'Kolmogorov-Smirnov':<20} {'Normal?'}")
print("-" * 80)

normality_results = []

for var in analysis_vars:
    data = analysis_clean[var].dropna()
    
    # Shapiro-Wilk test (use sample for large datasets)
    sample_size = min(5000, len(data))
    sample = data.sample(n=sample_size, random_state=42)
    shapiro_stat, shapiro_p = stats.shapiro(sample)
    
    # Kolmogorov-Smirnov test
    # Standardize data for K-S test against standard normal
    standardized = (data - data.mean()) / data.std()
    ks_stat, ks_p = stats.kstest(standardized, 'norm')
    
    is_normal = "Sí" if (shapiro_p > 0.05 and ks_p > 0.05) else "No"
    
    normality_results.append({
        'variable': var,
        'shapiro_stat': shapiro_stat,
        'shapiro_p': shapiro_p,
        'ks_stat': ks_stat,
        'ks_p': ks_p,
        'is_normal': is_normal == "Sí"
    })
    
    print(f"{var:<30} p={shapiro_p:.4f}  {'✓' if shapiro_p > 0.05 else '✗'}     p={ks_p:.4f}  {'✓' if ks_p > 0.05 else '✗'}      {is_normal}")

print("-" * 80)
print("Nota: ✓ indica p > 0.05 (no se rechaza normalidad), ✗ indica p ≤ 0.05 (se rechaza normalidad)")

normality_df = pd.DataFrame(normality_results)

# %% [markdown]
# ### 3.5 Análisis de Poder Estadístico
# 
# El análisis de poder nos ayuda a determinar si tenemos suficientes observaciones para detectar efectos estadísticamente significativos.
# 
# **Conceptos clave:**
# - **Effect Size (Cohen's d)**: Magnitud del efecto (pequeño=0.2, mediano=0.5, grande=0.8)
# - **Alpha (α)**: Probabilidad de error Tipo I (falso positivo) - típicamente 0.05
# - **Power (1-β)**: Probabilidad de detectar un efecto verdadero - típicamente 0.80
# - **Sample Size (n)**: Número de observaciones necesarias

# %%
# Power Analysis
print("📊 Análisis de Poder Estadístico")
print("=" * 60)

# Get sample sizes for our two groups
n_with_solar = analysis_clean['has_solar'].sum()
n_without_solar = len(analysis_clean) - n_with_solar
print(f"\nTamaños de muestra actuales:")
print(f"   Tracts CON instalaciones FV (n₁): {n_with_solar:,}")
print(f"   Tracts SIN instalaciones FV (n₂): {n_without_solar:,}")
print(f"   Ratio n₂/n₁: {n_without_solar/n_with_solar:.2f}")

# Calculate required sample sizes for different effect sizes
effect_sizes = [0.2, 0.5, 0.8]
alpha = 0.05
power = 0.80

print(f"\n📐 Tamaño de muestra requerido por grupo (α={alpha}, power={power}):")
print("-" * 60)
for es in effect_sizes:
    required_n = smp.tt_ind_solve_power(
        effect_size=es, 
        nobs1=None, 
        alpha=alpha, 
        power=power, 
        ratio=1, 
        alternative='two-sided'
    )
    size_label = "pequeño" if es == 0.2 else ("mediano" if es == 0.5 else "grande")
    meets_req = "✓" if min(n_with_solar, n_without_solar) >= required_n else "✗"
    print(f"   Effect size {es} ({size_label}): n = {round(required_n):,} por grupo {meets_req}")

# Calculate actual power with our sample sizes
print(f"\n📈 Poder estadístico con nuestros tamaños de muestra actuales:")
print("-" * 60)
for es in effect_sizes:
    actual_power = smp.tt_ind_solve_power(
        effect_size=es,
        nobs1=n_with_solar,
        alpha=alpha,
        power=None,
        ratio=n_without_solar/n_with_solar,
        alternative='two-sided'
    )
    size_label = "pequeño" if es == 0.2 else ("mediano" if es == 0.5 else "grande")
    print(f"   Effect size {es} ({size_label}): power = {actual_power:.4f} ({actual_power*100:.1f}%)")

# %%
# Power curves visualization
fig, ax = plt.subplots(figsize=(10, 6))

effect_sizes_range = np.array([0.2, 0.5, 0.8, 1.0])
sample_sizes_range = np.arange(10, 500, 10)

power_obj = smp.TTestIndPower()
power_obj.plot_power(dep_var='nobs', nobs=sample_sizes_range, effect_size=effect_sizes_range, ax=ax)

# Add vertical line for our smallest group
ax.axvline(x=min(n_with_solar, n_without_solar), color='red', linestyle='--', 
           label=f'Nuestro n mínimo: {min(n_with_solar, n_without_solar):,}')
ax.axhline(y=0.8, color='gray', linestyle=':', alpha=0.7, label='Power = 0.80')

ax.set_xlabel("Tamaño de Muestra por Grupo")
ax.set_ylabel("Poder Estadístico")
ax.set_title("Curvas de Poder para Diferentes Tamaños de Efecto")
ax.legend(loc='lower right')
ax.set_xlim(0, 500)
plt.tight_layout()
plt.show()

print("\n💡 Interpretación:")
print("   Con nuestros tamaños de muestra, tenemos poder estadístico suficiente")
print("   para detectar efectos pequeños (d=0.2) con alta confiabilidad.")

# %% [markdown]
# ---
# 
# ## 📈 4: Pruebas Estadísticas de Diferencias entre Grupos
# 
# Ahora que hemos verificado la normalidad (o falta de ella) de nuestras variables y confirmado que tenemos poder estadístico suficiente, realizamos las pruebas para determinar si existen diferencias significativas entre:
# 
# - **Grupo 1**: Census tracts CON instalaciones fotovoltaicas
# - **Grupo 2**: Census tracts SIN instalaciones fotovoltaicas
# 
# ### Selección de Prueba
# - Si los datos son **normales**: t-test independiente
# - Si los datos **NO son normales**: Mann-Whitney U (no paramétrica)
# 
# Basado en nuestros resultados de normalidad, utilizaremos **Mann-Whitney U** para la mayoría de las variables.

# %%
# Compare groups: with vs without solar installations
print("📊 Comparación de Grupos: Tracts con vs sin Instalaciones FV")
print("=" * 80)

# Split data into two groups
group_with_solar = analysis_clean[analysis_clean['has_solar'] == 1]
group_without_solar = analysis_clean[analysis_clean['has_solar'] == 0]

print(f"\nEstadísticas descriptivas por grupo:")
print("-" * 80)

comparison_results = []

for var in analysis_vars:
    data_with = group_with_solar[var].dropna()
    data_without = group_without_solar[var].dropna()
    
    # Calculate means and medians
    mean_with = data_with.mean()
    mean_without = data_without.mean()
    median_with = data_with.median()
    median_without = data_without.median()
    
    # Determine which test to use based on normality
    var_is_normal = normality_df[normality_df['variable'] == var]['is_normal'].values[0]
    
    if var_is_normal:
        # Use independent t-test
        stat, p_value = stats.ttest_ind(data_with, data_without)
        test_name = "t-test"
    else:
        # Use Mann-Whitney U test
        stat, p_value = stats.mannwhitneyu(data_with, data_without, alternative='two-sided')
        test_name = "Mann-Whitney U"
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt(((len(data_with)-1)*data_with.std()**2 + (len(data_without)-1)*data_without.std()**2) / 
                         (len(data_with) + len(data_without) - 2))
    cohens_d = (mean_with - mean_without) / pooled_std if pooled_std > 0 else 0
    
    sig = "***" if p_value < 0.001 else ("**" if p_value < 0.01 else ("*" if p_value < 0.05 else ""))
    
    comparison_results.append({
        'variable': var,
        'mean_with_solar': mean_with,
        'mean_without_solar': mean_without,
        'median_with_solar': median_with,
        'median_without_solar': median_without,
        'test': test_name,
        'statistic': stat,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'significant': p_value < 0.05
    })

# Display results as DataFrame
results_df = pd.DataFrame(comparison_results)
print("\n📋 Resultados de las Pruebas Estadísticas:")
display(results_df[['variable', 'mean_with_solar', 'mean_without_solar', 'test', 'p_value', 'cohens_d', 'significant']])

# %%
# Visualize group comparisons
fig, axes = plt.subplots(2, 4, figsize=(16, 10))
axes = axes.flatten()

for idx, var in enumerate(analysis_vars):
    ax = axes[idx]
    
    # Prepare data for boxplot
    data_with = group_with_solar[var].dropna()
    data_without = group_without_solar[var].dropna()
    
    # Create boxplot
    box_data = [data_without, data_with]
    bp = ax.boxplot(box_data, tick_labels=['Sin FV', 'Con FV'], patch_artist=True)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor('lightcoral')
    bp['boxes'][1].set_facecolor('lightgreen')
    
    # Get p-value for title
    p_val = results_df[results_df['variable'] == var]['p_value'].values[0]
    sig_marker = "***" if p_val < 0.001 else ("**" if p_val < 0.01 else ("*" if p_val < 0.05 else "n.s."))
    
    ax.set_title(f'{var}\np = {p_val:.4f} {sig_marker}', fontsize=10)
    ax.set_ylabel('Valor')

axes[-1].axis('off')
plt.tight_layout()
plt.suptitle('Comparación de Variables del Censo: Tracts Con vs Sin Instalaciones FV\n(* p<0.05, ** p<0.01, *** p<0.001, n.s. = no significativo)', 
             y=1.02, fontsize=12)
plt.show()

# %%
# Summary of significant differences
print("\n📊 Resumen de Diferencias Significativas (p < 0.05):")
print("=" * 60)

sig_results = results_df[results_df['significant']]
if len(sig_results) > 0:
    for _, row in sig_results.iterrows():
        direction = "mayor" if row['mean_with_solar'] > row['mean_without_solar'] else "menor"
        effect_size = "pequeño" if abs(row['cohens_d']) < 0.5 else ("mediano" if abs(row['cohens_d']) < 0.8 else "grande")
        print(f"\n   📍 {row['variable']}:")
        print(f"      - Los tracts CON instalaciones FV tienen un valor {direction}")
        print(f"      - Media con FV: {row['mean_with_solar']:.2f} vs sin FV: {row['mean_without_solar']:.2f}")
        print(f"      - Tamaño del efecto (Cohen's d): {row['cohens_d']:.3f} ({effect_size})")
        print(f"      - p-value: {row['p_value']:E}")
else:
    print("   No se encontraron diferencias estadísticamente significativas.")

print("\n💡 Nota: Estas diferencias sugieren patrones en la adopción de energía solar")
print("   en relación con características socioeconómicas de los census tracts.")

# %% [markdown]
# ### 4.2 Modelo de Regresión Logística (Opcional)
# 
# Complementamos el análisis con un modelo de regresión logística para predecir la presencia de instalaciones FV basándonos en las variables del Censo.

# %%
import statsmodels.formula.api as smf

# Prepare data for logistic regression (drop any remaining NaN)
logit_data = analysis_clean[['has_solar'] + analysis_vars].dropna()

# Standardize predictors for better coefficient interpretation
for var in analysis_vars:
    logit_data[f'{var}_std'] = (logit_data[var] - logit_data[var].mean()) / logit_data[var].std()

# Build formula with standardized variables
std_vars = [f'{var}_std' for var in analysis_vars]
formula = f"has_solar ~ {' + '.join(std_vars)}"

print("📊 Modelo de Regresión Logística")
print("=" * 60)
print(f"   Variable dependiente: has_solar (1 = tiene instalaciones FV, 0 = no tiene)")
print(f"   Variables independientes: {analysis_vars}")
print(f"   Observaciones: {len(logit_data):,}")

# Fit logistic regression model
try:
    model = smf.logit(formula=formula, data=logit_data).fit(disp=0)
    print(model.summary())
except Exception as e:
    print(f"⚠️ Error al ajustar el modelo: {e}")
    print("   Esto puede ocurrir si hay multicolinealidad o separación perfecta en los datos.")

# %% [markdown]
# ---
# 
# ## 🎯 5: Conclusiones y Próximos Pasos
# 
# ### Hallazgos Principales:
# 
# 1. **Distribución de datos**: La mayoría de las variables del Censo NO siguen una distribución normal, justificando el uso de pruebas no paramétricas (Mann-Whitney U).
# 
# 2. **Poder estadístico**: Con nuestros tamaños de muestra, tenemos poder suficiente para detectar efectos pequeños a grandes.
# 
# 3. **Diferencias significativas**: [Se actualizará basado en resultados]
# 
# 4. **Métrica de adopción**: `solar_adoption_rate` (instalaciones por 1,000 unidades de vivienda) es una métrica más interpretable que `solar_density` para comparar adopción entre tracts urbanos y rurales.
# 
# ### Próximos Pasos:
# 
# - [ ] Integrar análisis de Land Cover (LULC) cuando esté disponible
# - [ ] Análisis temporal de adopción de energía solar
# - [ ] Modelos predictivos más sofisticados
# 
# ### 🔬 Trabajo Futuro: Imputación de `capacity_mw`
# 
# La variable `capacity_mw` (capacidad de generación en megavatios) es un excelente candidato para **imputación o derivación** basada en:
# 
# 1. **Área del panel (`area_m2`)**: Existe una relación física directa entre área y capacidad
#    - Regla general: ~150-200 W/m² para paneles cristalinos típicos
#    - Fórmula aproximada: `capacity_kw ≈ area_m2 * 0.15 a 0.20`
# 
# 2. **Ángulo de inclinación del panel**: Si está disponible, mejora significativamente la precisión
#    - Paneles con inclinación óptima (latitud ± 15°) maximizan generación
# 
# 3. **Modelos físicos de energía existentes**:
#    - **PVWatts** (NREL): Modelo estándar de la industria para estimación de generación
#    - **SAM** (System Advisor Model): Modelo más detallado que incluye degradación y pérdidas
#    - Parámetros: radiación solar local, temperatura, eficiencia del panel, orientación
# 
# Esta imputación permitiría análisis más completos sobre la capacidad de generación agregada por tract.

# %% [markdown]
# ## 📚 Referencias y Documentación
# 
# ### Censusdis y API del Censo
# - [censusdis Introduction](https://censusdis.readthedocs.io/en/latest/intro.html)
# - [Census API Datasets](https://api.census.gov/data/2020.html)
# - [Exploring Variables](https://censusdis.readthedocs.io/en/latest/nb/Exploring%20Variables.html)
# - [Data With Geometry](https://censusdis.readthedocs.io/en/latest/nb/Data%20With%20Geometry.html)
# - [Column Labels (Human-friendly names)](https://github.com/censusdis/censusdis/blob/main/notebooks/Column%20Labels.ipynb)
# - [Population Density](https://github.com/censusdis/censusdis/blob/main/notebooks/Population%20Density.ipynb)
# 
# ### Análisis Estadístico
# - [SciPy Stats Documentation](https://docs.scipy.org/doc/scipy/reference/stats.html)
# - [Statsmodels Power Analysis](https://www.statsmodels.org/stable/stats.html#power-and-sample-size-calculations)
# - [Mann-Whitney U Test](https://en.wikipedia.org/wiki/Mann%E2%80%93Whitney_U_test)
# 
# ### Visualización
# - [ipywidgets Widget List](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html)
# - [Plotly Express Documentation](https://plotly.com/python/plotly-express/)
# 
# 


