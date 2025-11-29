# %% [markdown]
# # Overture Maps: Fetching Land Cover Data
# 
# **CCOM 6994: Data Analysis Tools - Final Project**
# 
# This notebook fetches **Land Cover** data from the **Overture Maps Foundation** dataset
# using the [`overturemaestro`](https://kraina-ai.github.io/overturemaestro/) library for
# GeoPandas-friendly data access with built-in caching.
# 
# ---
# 
# ## 🌍 What is Land Use-Land Cover (LULC)?
# 
# **Land Cover** refers to the physical and biological cover of the Earth's surface,
# including vegetation, water, bare soil, and artificial structures. It answers the
# question: *"What is physically present on the ground?"*
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
# ### How is Land Cover Data Derived?
# 
# Land cover maps are typically derived from:
# 
# 1. **Satellite Imagery**: Landsat, Sentinel-2, MODIS provide multispectral data
# 2. **Machine Learning Classification**: Random forests, neural networks classify pixels
# 3. **Ground Truth Validation**: Field surveys verify accuracy
# 4. **Temporal Analysis**: Multi-date imagery captures seasonal changes
# 
# Popular land cover datasets include:
# - **NLCD** (National Land Cover Database) - 30m resolution, USA
# - **ESA WorldCover** - 10m resolution, global
# - **Copernicus CORINE** - 100m resolution, Europe
# - **Overture Maps** - Vector-based, derived from OpenStreetMap and other sources
# 
# ---
# 
# ## 🎯 Why Land Cover Matters for Solar PV Analysis
# 
# Understanding land cover context around solar installations helps us:
# 
# 1. **Site Characterization**: What land types host solar farms?
# 2. **Land Use Change**: Are panels replacing cropland, forest, or developed areas?
# 3. **Environmental Impact**: Assess ecological footprint of solar development
# 4. **Policy Analysis**: Identify patterns in permitting across land types
# 5. **Predictive Modeling**: Which land cover types are most likely to have future solar?
# 
# ---
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
# 
# ---

# %% [markdown]
# ## 🔧 Setup: Import Libraries

# %%
import os
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt

# OvertureMaestro for Overture Maps data fetching
import overturemaestro as om
from overturemaestro.advanced_functions import get_all_possible_column_names

# Census geometry reader
import censusdis.maps as cem

from dotenv import load_dotenv
from tqdm.auto import tqdm

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Geometry column does not contain geometry')

# Load environment variables
load_dotenv(dotenv_path=Path('../.env'))

# Configure display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(f"✅ OvertureMaestro version: {om.__version__}")
print(f"✅ Latest Overture release: {om.get_newest_release_version()}")

# %% [markdown]
# ## ⚙️ Configuration
# 
# Adjust these settings based on your hardware and processing needs.
# 
# **Recommended settings for different hardware:**
# 
# | Hardware | MAX_WORKERS | DUCKDB_THREADS | DUCKDB_MEMORY |
# |----------|-------------|----------------|---------------|
# | Laptop (4 cores, 16GB) | 4 | 4 | 12GB |
# | Desktop (8 cores, 32GB) | 6-8 | 8 | 24GB |
# | Workstation (16 cores, 128GB) | 10-12 | 16 | 100GB |
# 
# **Environment Variables:**
# Create a `.env` file in the project root with these settings:
# ```
# MAX_WORKERS=12
# DUCKDB_THREADS=16
# DUCKDB_MEMORY_LIMIT=100GB
# PROJECT_DB=../db/pv_project.ddb
# ```

# %%
# === PROCESSING CONFIGURATION ===

# Database path (relative to notebooks/ directory)
DB_PATH = os.getenv('PROJECT_DB', '../db/pv_project.ddb')

# === PARALLEL PROCESSING SETTINGS ===
# Set based on your hardware: 16 cores = try 10-12 workers
# Too many workers can cause memory issues or API throttling
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '8'))

# === DUCKDB PERFORMANCE SETTINGS ===
# Threads: Set to your physical core count (not hyperthreads)
DUCKDB_THREADS = int(os.getenv('DUCKDB_THREADS', '8'))

# Memory limit: Set to ~80% of available RAM
# Examples: '12GB' (laptop), '24GB' (desktop), '100GB' (workstation)
DUCKDB_MEMORY_LIMIT = os.getenv('DUCKDB_MEMORY_LIMIT', '16GB')

# Enable Parquet metadata cache for faster repeated queries
DUCKDB_PARQUET_CACHE = os.getenv('DUCKDB_PARQUET_CACHE', 'true').lower() == 'true'

# States to process (None = all states with PV installations)
# For testing, use a subset: ['CA', 'AZ', 'NV']
STATES_TO_PROCESS = None  

# Save interval: commit to DB every N counties to avoid data loss
SAVE_INTERVAL = 10

print(f"📊 Configuration:")
print(f"   Database: {DB_PATH}")
print(f"   Max parallel workers: {MAX_WORKERS}")
print(f"   DuckDB threads: {DUCKDB_THREADS}")
print(f"   DuckDB memory limit: {DUCKDB_MEMORY_LIMIT}")
print(f"   Parquet metadata cache: {DUCKDB_PARQUET_CACHE}")
print(f"   States filter: {STATES_TO_PROCESS or 'All'}")

# %% [markdown]
# ## 📥 Task 1: Load Census-Enriched PV Data
# 
# We load our PV dataset that has been enriched with Census identifiers in notebook 02.
# This gives us the list of counties that contain solar installations.

# %%
print(f"📂 Connecting to database: {DB_PATH}")

con = duckdb.connect(DB_PATH, read_only=True)

# === DUCKDB PERFORMANCE CONFIGURATION ===
# Install and load spatial extension
con.execute("INSTALL spatial; LOAD spatial;")

# Set thread count for parallel query execution
con.execute(f"SET threads = {DUCKDB_THREADS};")

# Set memory limit (important for large spatial operations)
con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}';")

# Enable Parquet metadata cache for faster repeated access to Overture files
if DUCKDB_PARQUET_CACHE:
    con.execute("SET parquet_metadata_cache = true;")
    con.execute("SET enable_http_metadata_cache = true;")
    
# Optimize for our workload
con.execute("SET enable_progress_bar = true;")

print(f"✅ DuckDB configured:")
print(f"   Threads: {DUCKDB_THREADS}")
print(f"   Memory limit: {DUCKDB_MEMORY_LIMIT}")
print(f"   Parquet cache: {DUCKDB_PARQUET_CACHE}")

# List available tables
tables = con.execute("SHOW TABLES").fetchall()
print("📋 Available tables:")
for t in tables:
    try:
        count = con.execute(f"SELECT COUNT(*) FROM \"{t[0]}\"").fetchone()[0]
        print(f"   - {t[0]}: {count:,} rows")
    except:
        print(f"   - {t[0]}: (error reading)")

# %%
# Load the census-enriched PV data
print("\n📥 Loading census_enriched_pv_data...")
pv_df = con.execute("SELECT * FROM census_enriched_pv_data").df()
pv_df['geometry'] = pv_df['geometry'].apply(wkt.loads)
pv_gdf = gpd.GeoDataFrame(pv_df, geometry='geometry', crs='EPSG:4326')

print(f"✅ Loaded {len(pv_gdf):,} PV installations")
print(f"   States: {pv_gdf['STATE_ABBR'].nunique()}")
print(f"   Counties: {pv_gdf['COUNTY_GEOID'].nunique()}")

con.close()

# %% [markdown]
# ## 🗺️ Task 2: Get County Geometries
# 
# We fetch actual county polygon geometries from Census Bureau shapefiles.
# These polygons are used as precise filters when querying Overture Maps - 
# much more accurate than simple bounding boxes.

# %%
# Get unique counties from our PV data
pv_counties_info = pv_gdf[['STATE_FIPS', 'COUNTY_FIPS', 'COUNTY_GEOID', 'COUNTY_NAME', 'STATE_ABBR']].drop_duplicates()
print(f"📊 PV installations span {len(pv_counties_info)} unique counties")

# Show top states by county count
print(f"\n📊 Counties with PV by state (top 10):")
print(pv_counties_info.groupby('STATE_ABBR').size().sort_values(ascending=False).head(10))

# %%
# Fetch county geometries from Census Bureau
print("\n🗺️ Fetching county geometries from Census Bureau...")
reader = cem.ShapeReader(year=2020)
county_bounds_gdf = reader.read_cb_shapefile(
    shapefile_scope="us", 
    geography="county", 
    crs="EPSG:4326"
)
print(f"✅ Loaded {len(county_bounds_gdf):,} county geometries")

# Filter to only counties with PV installations
pv_county_geoids = set(pv_counties_info['COUNTY_GEOID'].tolist())
pv_county_bounds = county_bounds_gdf[county_bounds_gdf['GEOID'].isin(pv_county_geoids)].copy()
print(f"✅ Filtered to {len(pv_county_bounds):,} counties with PV installations")

# %% [markdown]
# ## 🌲 Task 3: Explore Land Cover Schema
# 
# Before fetching, let's understand the available land cover categories.

# %%
print("🌲 Available Land Cover subtypes in Overture Maps:")
land_cover_cols = get_all_possible_column_names(theme="base", type="land_cover")
for col in land_cover_cols:
    subtype = col.split('|')[-1]
    print(f"   • {subtype}")

# %% [markdown]
# ## 🧪 Task 4: Test with a Single County
# 
# Before running the full batch, let's verify the workflow with one county.

# %%
# Pick a test county
test_county = pv_county_bounds.iloc[0]
test_geoid = test_county['GEOID']
test_name = test_county['NAME']
test_state = test_county['STATEFP']
test_geom = test_county.geometry

# Count PV in this county
test_pv_count = len(pv_gdf[pv_gdf['COUNTY_GEOID'] == test_geoid])

print(f"🧪 Testing with: {test_name} County (GEOID: {test_geoid})")
print(f"   State FIPS: {test_state}")
print(f"   PV installations: {test_pv_count}")
print(f"   Bounds: {test_geom.bounds}")

# %%
# Fetch Land Cover for test county
print(f"\n🌲 Fetching Land Cover for {test_name} County...")

t1 = time.time()
test_lc = om.convert_geometry_to_geodataframe(
    theme="base",
    type="land_cover",
    geometry_filter=test_geom,
    columns_to_download=["id", "subtype", "geometry"],
)
t2 = time.time()

print(f"✅ Fetched {len(test_lc):,} land cover features in {t2-t1:.1f}s")
print(f"\n📊 Land cover distribution:")
print(test_lc['subtype'].value_counts())

# %% [markdown]
# ## ⚡ Task 5: Parallel Batch Processing
# 
# We use Python's `ThreadPoolExecutor` to fetch data for multiple counties in parallel.
# OvertureMaestro handles caching, so repeated runs are fast.
# 
# ### Why ThreadPoolExecutor?
# - OvertureMaestro operations are I/O-bound (network requests to S3)
# - Threads work well for I/O-bound tasks (GIL is released during I/O)
# - Simpler than multiprocessing, avoids serialization overhead
# 
# ### Memory Considerations
# - Each worker holds one county's data in memory
# - With 8 workers and ~5MB per county avg, expect ~40MB concurrent usage
# - We batch-save to DuckDB to limit memory growth

# %%
def fetch_land_cover_for_county(county_row):
    """
    Fetch land cover data for a single county.
    Returns tuple: (county_geoid, state_fips, geodataframe or None, error or None)
    """
    geoid = county_row['GEOID']
    state_fips = county_row['STATEFP']
    geom = county_row.geometry
    
    try:
        lc_gdf = om.convert_geometry_to_geodataframe(
            theme="base",
            type="land_cover",
            geometry_filter=geom,
            columns_to_download=["id", "subtype", "geometry"],
        )
        
        if len(lc_gdf) > 0:
            lc_gdf = lc_gdf.reset_index()  # id becomes a column
            lc_gdf['county_geoid'] = geoid
            lc_gdf['state_fips'] = state_fips
            lc_gdf['geometry'] = lc_gdf['geometry'].apply(lambda g: g.wkt)
            return (geoid, state_fips, lc_gdf, None)
        else:
            return (geoid, state_fips, None, None)
            
    except Exception as e:
        return (geoid, state_fips, None, str(e))

# %%
# Filter states if configured
if STATES_TO_PROCESS:
    counties_to_process = pv_county_bounds[pv_county_bounds['STATEFP'].isin(
        [s if len(s) == 2 else f"0{s}" for s in STATES_TO_PROCESS]
    )]
else:
    counties_to_process = pv_county_bounds

print(f"🔄 Processing {len(counties_to_process)} counties with {MAX_WORKERS} parallel workers")

# Prepare county data for parallel processing
county_records = [row for _, row in counties_to_process.iterrows()]

# %%
# Initialize results storage
all_results = []
errors = []
processed_count = 0

# Open DB connection for writing with optimized settings
con = duckdb.connect(DB_PATH)

# Apply full performance configuration for batch processing
con.execute("INSTALL spatial; LOAD spatial;")
con.execute(f"SET threads = {DUCKDB_THREADS};")
con.execute(f"SET memory_limit = '{DUCKDB_MEMORY_LIMIT}';")
if DUCKDB_PARQUET_CACHE:
    con.execute("SET parquet_metadata_cache = true;")

# Optimize for batch inserts
con.execute("SET enable_progress_bar = false;")  # Avoid conflicts with tqdm

# Create output table
con.execute("DROP TABLE IF EXISTS overture_land_cover")
con.execute("""
    CREATE TABLE overture_land_cover (
        id VARCHAR,
        geometry VARCHAR,
        subtype VARCHAR,
        county_geoid VARCHAR,
        state_fips VARCHAR
    )
""")

print("✅ Created overture_land_cover table")

# %%
# Parallel processing with progress bar
print(f"\n🚀 Starting parallel fetch with {MAX_WORKERS} workers...")
t_start = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks
    future_to_county = {
        executor.submit(fetch_land_cover_for_county, county): county['GEOID']
        for county in county_records
    }
    
    # Process completed tasks with progress bar
    with tqdm(total=len(county_records), desc="Fetching Land Cover") as pbar:
        for future in as_completed(future_to_county):
            geoid = future_to_county[future]
            
            try:
                county_geoid, state_fips, result_df, error = future.result()
                
                if error:
                    errors.append({'county_geoid': county_geoid, 'error': error})
                elif result_df is not None:
                    all_results.append(result_df)
                
                processed_count += 1
                
                # Periodic save to database
                if len(all_results) >= SAVE_INTERVAL:
                    batch_df = pd.concat(all_results, ignore_index=True)
                    con.execute("INSERT INTO overture_land_cover SELECT * FROM batch_df")
                    pbar.set_postfix({'saved': con.execute("SELECT COUNT(*) FROM overture_land_cover").fetchone()[0]})
                    all_results = []
                    
            except Exception as e:
                errors.append({'county_geoid': geoid, 'error': str(e)})
            
            pbar.update(1)

# Save remaining results
if all_results:
    batch_df = pd.concat(all_results, ignore_index=True)
    con.execute("INSERT INTO overture_land_cover SELECT * FROM batch_df")

t_end = time.time()

# %%
# Processing summary
total_features = con.execute("SELECT COUNT(*) FROM overture_land_cover").fetchone()[0]
unique_counties = con.execute("SELECT COUNT(DISTINCT county_geoid) FROM overture_land_cover").fetchone()[0]

print(f"\n✅ Parallel processing complete!")
print(f"   Total time: {(t_end - t_start)/60:.1f} minutes")
print(f"   Counties processed: {processed_count}")
print(f"   Counties with data: {unique_counties}")
print(f"   Total features: {total_features:,}")
print(f"   Errors: {len(errors)}")

if errors:
    print(f"\n⚠️ Counties with errors:")
    for e in errors[:5]:
        print(f"   {e['county_geoid']}: {e['error'][:50]}...")

# %% [markdown]
# ## 📊 Task 6: Data Summary and Verification

# %%
print("📊 Land Cover Summary by Subtype:")
summary = con.execute("""
    SELECT 
        subtype, 
        COUNT(*) as feature_count,
        COUNT(DISTINCT county_geoid) as county_count
    FROM overture_land_cover 
    GROUP BY subtype 
    ORDER BY feature_count DESC
""").df()
print(summary.to_string(index=False))

# %%
print("\n📊 Top 10 Counties by Feature Count:")
top_counties = con.execute("""
    SELECT 
        county_geoid,
        state_fips,
        COUNT(*) as feature_count
    FROM overture_land_cover 
    GROUP BY county_geoid, state_fips
    ORDER BY feature_count DESC
    LIMIT 10
""").df()
print(top_counties.to_string(index=False))

# %% [markdown]
# ## 🗄️ Task 7: Create Spatial Index
# 
# Adding a spatial index improves query performance for spatial operations.

# %%
print("🗄️ Creating spatial index...")
try:
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_land_cover_geom 
        ON overture_land_cover USING RTREE (ST_GeomFromText(geometry))
    """)
    print("✅ Spatial index created")
except Exception as e:
    print(f"⚠️ Could not create spatial index: {e}")

# %%
# Final verification
final_count = con.execute("SELECT COUNT(*) FROM overture_land_cover").fetchone()[0]
print(f"\n📈 Final table size: {final_count:,} land cover features")

con.close()
print("✅ Database connection closed")

# %% [markdown]
# ## 📝 Summary
# 
# This notebook demonstrated:
# 
# 1. **Land Cover Fundamentals**: Physical surface classification for Earth observation
# 2. **Overture Maps Integration**: Using overturemaestro for GeoPandas-friendly access
# 3. **Parallel Processing**: ThreadPoolExecutor for I/O-bound operations
# 4. **Efficient Storage**: Batch saves to DuckDB with spatial indexing
# 
# ### Performance Notes
# 
# | Configuration | Expected Time |
# |---------------|---------------|
# | Sequential (1 worker) | ~6 hours for 1,071 counties |
# | 4 workers | ~1.5 hours |
# | 8 workers | ~45-60 minutes |
# | 16 workers | ~30-45 minutes (may hit API limits) |
# 
# **Tip**: OvertureMaestro caches downloaded data, so re-runs are much faster!
# 
# ### Next Steps
# → Continue to **Notebook 05** for Land Use data
# → Use the data in analysis notebooks for PV-LULC correlation studies


