# CCOM6994_PV_Solar_Project

> Exploratory Data Analysis of Solar Panel Installations and Geophysical/Sociodemographic Variables

This project investigates the spatial correlations between photovoltaic (PV) solar panel installations and their surrounding geophysical and sociodemographic characteristics across the United States and Puerto Rico. By integrating multiple open datasets—including solar panel location data, census demographics, land use/land cover classifications, and climate reanalysis data—we aim to establish the groundwork for identifying and understanding the key factors that are associated with spatial proximity to solar energy infrastructure.

## Project Goals

- **Spatial EDA**: Perform exploratory data analysis on PV installation locations enriched with census, land cover, and climate variables
- **Correlation Identification**: Discover relationships between solar panel density and sociodemographic indicators (income, housing, population)
- **Geophysical Analysis**: Analyze how land use patterns, solar irradiance, and temperature influence PV installation distribution
- **Data Pipeline Development**: Build reproducible workflows using [GeoPandas](https://geopandas.org/) and [DuckDB](https://geo.rocks/post/duckdb_geospatial/) for geospatial data processing

---

## Setup

### Prerequisites

- Python 3.10+
- Git

### Installing Dependencies

We recommend using **[uv](https://docs.astral.sh/uv/)** for package management—a fast, modern Python package installer written in Rust. However, traditional `pip` workflows are also supported.

#### Option 1: Using uv (Recommended)

**Why uv?**
- ⚡ **Speed**: Rust implementation, multi-threaded processing, and optimized metadata handling means much faster package resolution and installation than pip
- 💾 **Disk Efficiency**: Uses a global cache to avoid duplicate package storage across projects
- 🔧 **All-in-One**: Manages Python versions, virtual environments, and dependencies
- 🔄 **Legacy pip Compatible**: Works as a drop-in replacement with `uv pip` commands

**Install uv:**

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Or via Homebrew (macOS)
brew install uv

# Or via Winget (Windows)
winget install -e --id astral-sh.uv
```

**Set up the project:**

```bash
# Clone and navigate to the project
git clone https://github.com/avega17/CCOM6994_PV_Solar_Project
cd CCOM6994_PV_Solar_Project

# Create uv project but avoid main.py, README.md, and .gitignore generation
uv init --python=3.11 --bare
# Use uv's pyproject.toml support to manage dependencies: https://pydevtools.com/blog/requirementstxt-vs-pyprojecttoml/
uv sync
# Activate the virtual environment so dependencies like jupyter and other external tools are available
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

#### Option 2: Using pip

```bash
# Clone and navigate to the project
git clone https://github.com/avega17/CCOM6994_PV_Solar_Project
cd CCOM6994_PV_Solar_Project

# Create virtual environment with uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -r requirements.txt
```

### Jupyter Notebook Setup

If running notebooks in VS Code, ensure `ipykernel` is installed:

```bash
uv add ipykernel  # or: pip install ipykernel
```

Then [select the `.venv` Python interpreter as your notebook kernel](https://code.visualstudio.com/docs/datascience/jupyter-notebooks#_setting-up-your-environment).

---

## Data Sources

This project integrates multiple open data sources to enable comprehensive spatial analysis:

### 1. PV Solar Panel Locations (Scientific DOI Datasets)

Georeferenced solar panel installation data derived from satellite imagery and machine learning models. The list of datasets and their metadata can be found in `ingest/doi_manifest_usa.json`. These datasets provide polygon geometries and metadata for individual PV installations across the United States.

- **Source**: Scientific Research Publications
- **Format**: GeoJSON, GeoParquet
- **Key Fields**: Installation geometry, surface area (where available), installation date (where available), generation capacity (where available)

### 2. U.S. Census Data

Sociodemographic variables at multiple geographic levels (state, county, tract) via the Census Bureau API and `censusdis` Python library.

- **Source**: [U.S. Census Bureau](https://www.census.gov/)
- **Years**: 2020 and 2010 Decennial Census, American Community Survey (ACS), UACE (Urban Area Geographies and statistics)
- **Variables**: Population demographics, household income, housing characteristics, education levels, rural-urban classification
- **Access**: Via `censusdis` library for streamlined API queries; direct URL fetching of specific datasets as needed
- **Geographic Hierarchy**: State, County, Census Tract boundaries (fetched from official TIGER/Line shapefiles)

### 3. Overture Maps (Land Use & Land Cover)

Open-source geospatial data providing land use classifications and administrative boundaries.

- **Source**: [Overture Maps Foundation](https://overturemaps.org/)
- **Themes Used**:
  - `divisions`: Administrative boundaries (country, region, county, locality)
  - `base/land_use`: Land use polygons (residential, commercial, industrial, agricultural)
  - `base/land_cover`: Physical land cover classifications
- **Access**: Direct S3 queries via DuckDB

### 4. ERA5 Climate Reanalysis (Historical Weather)

Global climate reanalysis dataset providing historical temperature and solar irradiance data.

- **Source**: [Copernicus Climate Data Store / ECMWF republished by Google BigQuery Public Datasets](https://console.cloud.google.com/marketplace/product/bigquery-public-data/arco-era5)
- **Variables** [see available variables](https://github.com/google-research/arco-era5/?tab=readme-ov-file#analysis-ready-data)
  - Surface solar radiation downwards (SSRD)
  - 2-meter temperature (T2M)
  - fraction of Cloud cover
- **Resolution**: Hourly data, Daily aggregates, 0.25° x 0.25° harmonized spatial resolution
- **Total Coverage**: 1940–present

---

## Project Structure

```text
CCOM6994_PV_Solar_Project/
├── dataflows/          # Hamilton DAG definitions for ingest and consolidation pipelines
├── db/                 # DuckDB database and GeoParquet exports
├── ingest/             # Data ingestion manifests and configs
├── notebooks/          # Jupyter notebooks for EDA and analysis
│   ├── 01_geopandas_spatial_data_intro.ipynb
│   ├── 02_geocoding_fetch_geometries.ipynb
│   ├── 03_census_data_demo.ipynb
│   ├── 04_overture_land_cover_fetch.ipynb
│   └── 05_overture_land_use_fetch.ipynb
├── pyproject.toml      # Project configuration and dependencies
├── requirements.txt    # Pip-compatible dependency list
└── README.md
```

---

## References

- [uv Documentation](https://docs.astral.sh/uv/)
- [GeoPandas User Guide](https://geopandas.org/)
- [DuckDB Spatial Extension](https://duckdb.org/docs/extensions/spatial.html)
- [Overture Maps Schema](https://docs.overturemaps.org/)
- [censusdis Documentation](https://censusdis.readthedocs.io/)
