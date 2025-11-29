# %% [markdown]
# # Overture Maps: Fetching Land Use Data
# 
# **CCOM 6994: Data Analysis Tools - Final Project**
# 
# This notebook fetches **Land Use** data from the **Overture Maps Foundation** dataset.
# It follows the same parallel processing pattern as the Land Cover notebook.
# 
# ---
# 
# ## 🏘️ What is Land Use?
# 
# While **Land Cover** describes *what* is physically on the ground, **Land Use** 
# describes *how* humans use that land. It answers the question: 
# *"What is this land being used for?"*
# 
# ### Land Cover vs Land Use
# 
# | Aspect | Land Cover | Land Use |
# |--------|------------|----------|
# | **Definition** | Physical surface materials | Human activity/purpose |
# | **Example** | Trees, grass, concrete | Forest reserve, park, parking lot |
# | **Detection** | Remote sensing (satellite) | Surveys, zoning, permits |
# | **Change Rate** | Slow (years) | Can be fast (rezoning) |
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
# ---
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
# ---
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
# 
# ### Land Use Hierarchy Examples
# 
# | Subtype | Class Examples |
# |---------|----------------|
# | `residential` | residential, garages |
# | `agriculture` | farmland, meadow, farmyard, animal_keeping |
# | `developed` | industrial, retail, commercial, brownfield |
# | `recreation` | pitch, playground, track |
# | `managed` | grass, flowerbed |
# | `protected` | nature_reserve, national_park |
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
# Same configuration pattern as Land Cover notebook.
# Adjust based on your hardware capabilities.
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
# For 16 cores/32 threads, try 10-12 workers
# Land Use has more features per county, so slightly fewer workers may be better
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
# For testing: ['CA', 'AZ', 'NV']
STATES_TO_PROCESS = None

# Save interval: commit to DB every N counties
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
# We reuse the same county list from Land Cover processing.

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
# Fetch county polygon geometries for precise Overture filtering.

# %%
# Get unique counties from our PV data
pv_counties_info = pv_gdf[['STATE_FIPS', 'COUNTY_FIPS', 'COUNTY_GEOID', 'COUNTY_NAME', 'STATE_ABBR']].drop_duplicates()
print(f"📊 PV installations span {len(pv_counties_info)} unique counties")

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
# ## 🏘️ Task 3: Explore Land Use Schema
# 
# Land Use has a two-level hierarchy: `subtype` and `class`.

# %%
print("🏘️ Land Use subtypes (top level):")
land_use_cols_l1 = get_all_possible_column_names(theme="base", type="land_use", hierarchy_depth=1)
for col in land_use_cols_l1:
    subtype = col.split('|')[-1]
    print(f"   • {subtype}")

# %%
print("\n🏘️ Land Use classes (full hierarchy, sample):")
land_use_cols_full = get_all_possible_column_names(theme="base", type="land_use")
print(f"   Total categories: {len(land_use_cols_full)}")

# Group by subtype
from collections import defaultdict
hierarchy = defaultdict(list)
for col in land_use_cols_full:
    parts = col.split('|')
    if len(parts) >= 4:
        subtype = parts[2]
        cls = parts[3]
        if cls not in hierarchy[subtype]:
            hierarchy[subtype].append(cls)

for subtype in sorted(hierarchy.keys())[:5]:
    print(f"\n   {subtype}:")
    for cls in hierarchy[subtype][:5]:
        print(f"      - {cls}")
    if len(hierarchy[subtype]) > 5:
        print(f"      ... and {len(hierarchy[subtype]) - 5} more")

# %% [markdown]
# ## 🧪 Task 4: Test with a Single County
# 
# Verify the Land Use fetch workflow before parallel processing.

# %%
# Pick a test county
test_county = pv_county_bounds.iloc[0]
test_geoid = test_county['GEOID']
test_name = test_county['NAME']
test_state = test_county['STATEFP']
test_geom = test_county.geometry

print(f"🧪 Testing with: {test_name} County (GEOID: {test_geoid})")
print(f"   State FIPS: {test_state}")
print(f"   Bounds: {test_geom.bounds}")

# %%
# Fetch Land Use for test county
print(f"\n🏘️ Fetching Land Use for {test_name} County...")

t1 = time.time()
test_lu = om.convert_geometry_to_geodataframe(
    theme="base",
    type="land_use",
    geometry_filter=test_geom,
    columns_to_download=["id", "subtype", "class", "geometry"],
)
t2 = time.time()

print(f"✅ Fetched {len(test_lu):,} land use features in {t2-t1:.1f}s")

if len(test_lu) > 0:
    print(f"\n📊 Land use subtype distribution:")
    print(test_lu['subtype'].value_counts().head(10))
    
    print(f"\n📊 Land use class distribution (top 10):")
    print(test_lu['class'].value_counts().head(10))

# %% [markdown]
# ## ⚡ Task 5: Parallel Batch Processing
# 
# Same parallel pattern as Land Cover, but fetching Land Use data.
# 
# ### Notes on Land Use Data Volume
# - Land Use typically has **more features** per county than Land Cover
# - Urban areas may have 10,000+ features (individual parcels)
# - Consider reducing `MAX_WORKERS` if memory becomes an issue

# %%
def fetch_land_use_for_county(county_row):
    """
    Fetch land use data for a single county.
    Returns tuple: (county_geoid, state_fips, geodataframe or None, error or None)
    """
    geoid = county_row['GEOID']
    state_fips = county_row['STATEFP']
    geom = county_row.geometry
    
    try:
        lu_gdf = om.convert_geometry_to_geodataframe(
            theme="base",
            type="land_use",
            geometry_filter=geom,
            columns_to_download=["id", "subtype", "class", "geometry"],
        )
        
        if len(lu_gdf) > 0:
            lu_gdf = lu_gdf.reset_index()  # id becomes a column
            lu_gdf['county_geoid'] = geoid
            lu_gdf['state_fips'] = state_fips
            lu_gdf['geometry'] = lu_gdf['geometry'].apply(lambda g: g.wkt)
            return (geoid, state_fips, lu_gdf, None)
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
con.execute("DROP TABLE IF EXISTS overture_land_use")
con.execute("""
    CREATE TABLE overture_land_use (
        id VARCHAR,
        geometry VARCHAR,
        subtype VARCHAR,
        class VARCHAR,
        county_geoid VARCHAR,
        state_fips VARCHAR
    )
""")

print("✅ Created overture_land_use table")

# %%
# Parallel processing with progress bar
print(f"\n🚀 Starting parallel fetch with {MAX_WORKERS} workers...")
t_start = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks
    future_to_county = {
        executor.submit(fetch_land_use_for_county, county): county['GEOID']
        for county in county_records
    }
    
    # Process completed tasks with progress bar
    with tqdm(total=len(county_records), desc="Fetching Land Use") as pbar:
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
                    con.execute("INSERT INTO overture_land_use SELECT * FROM batch_df")
                    pbar.set_postfix({'saved': con.execute("SELECT COUNT(*) FROM overture_land_use").fetchone()[0]})
                    all_results = []
                    
            except Exception as e:
                errors.append({'county_geoid': geoid, 'error': str(e)})
            
            pbar.update(1)

# Save remaining results
if all_results:
    batch_df = pd.concat(all_results, ignore_index=True)
    con.execute("INSERT INTO overture_land_use SELECT * FROM batch_df")

t_end = time.time()

# %%
# Processing summary
total_features = con.execute("SELECT COUNT(*) FROM overture_land_use").fetchone()[0]
unique_counties = con.execute("SELECT COUNT(DISTINCT county_geoid) FROM overture_land_use").fetchone()[0]

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
print("📊 Land Use Summary by Subtype:")
summary = con.execute("""
    SELECT 
        subtype, 
        COUNT(*) as feature_count,
        COUNT(DISTINCT county_geoid) as county_count
    FROM overture_land_use 
    GROUP BY subtype 
    ORDER BY feature_count DESC
""").df()
print(summary.to_string(index=False))

# %%
print("\n📊 Top Land Use Classes:")
class_summary = con.execute("""
    SELECT 
        subtype,
        class,
        COUNT(*) as feature_count
    FROM overture_land_use 
    GROUP BY subtype, class 
    ORDER BY feature_count DESC
    LIMIT 15
""").df()
print(class_summary.to_string(index=False))

# %%
print("\n📊 Top 10 Counties by Feature Count:")
top_counties = con.execute("""
    SELECT 
        county_geoid,
        state_fips,
        COUNT(*) as feature_count
    FROM overture_land_use 
    GROUP BY county_geoid, state_fips
    ORDER BY feature_count DESC
    LIMIT 10
""").df()
print(top_counties.to_string(index=False))

# %% [markdown]
# ## 🗄️ Task 7: Create Spatial Index
# 
# Add spatial index for efficient spatial queries.

# %%
print("🗄️ Creating spatial index...")
try:
    con.execute("""
        CREATE INDEX IF NOT EXISTS idx_land_use_geom 
        ON overture_land_use USING RTREE (ST_GeomFromText(geometry))
    """)
    print("✅ Spatial index created")
except Exception as e:
    print(f"⚠️ Could not create spatial index: {e}")

# %%
# Final verification
final_count = con.execute("SELECT COUNT(*) FROM overture_land_use").fetchone()[0]
print(f"\n📈 Final table size: {final_count:,} land use features")

con.close()
print("✅ Database connection closed")

# %% [markdown]
# ## 🔗 Combining Land Cover and Land Use
# 
# With both datasets now in DuckDB, you can perform combined analyses:
# 
# ```sql
# -- Example: Find land use types under land cover = 'urban'
# SELECT 
#     lu.subtype as land_use,
#     lu.class as land_use_class,
#     COUNT(*) as overlap_count
# FROM overture_land_cover lc
# JOIN overture_land_use lu 
#     ON ST_Intersects(ST_GeomFromText(lc.geometry), ST_GeomFromText(lu.geometry))
# WHERE lc.subtype = 'urban'
# GROUP BY lu.subtype, lu.class
# ORDER BY overlap_count DESC;
# ```

# %% [markdown]
# ## 📝 Summary
# 
# This notebook demonstrated:
# 
# 1. **Land Use Concepts**: Human activity classification vs physical cover
# 2. **Hierarchical Schema**: Subtype → Class classification in Overture
# 3. **Parallel Processing**: Efficient multi-threaded data fetching
# 4. **Data Persistence**: Batch saves to DuckDB with spatial indexing
# 
# ### Combined LULC Analysis Potential
# 
# With both Land Cover (physical) and Land Use (human activity) data, you can:
# 
# - **Cross-tabulate**: What land uses occur on agricultural land cover?
# - **Change Detection**: Compare official land use vs actual land cover
# - **Solar Siting**: Identify optimal land use/cover combinations for solar
# - **Policy Analysis**: How do land use regulations affect solar deployment?
# 
# ### Performance Notes
# 
# | Configuration | Land Cover | Land Use | Total |
# |---------------|------------|----------|-------|
# | 8 workers | ~45 min | ~60 min | ~2 hours |
# | 12 workers | ~30 min | ~45 min | ~1.5 hours |
# 
# **Tip**: Land Use has more features than Land Cover, so expect longer processing!
# 
# ### Next Steps
# → Proceed to analysis notebooks for PV-LULC correlation studies
# → Spatial join PV installations with LULC features
