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
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, as_completed

import duckdb
import pandas as pd
import geopandas as gpd
from shapely import wkt
import matplotlib.pyplot as plt

# OvertureMaestro for Overture Maps data fetching
import overturemaestro as om
try:
    from overturemaestro.advanced_functions import get_all_possible_column_names
except ImportError:
    # Fallback for older versions
    from overturemaestro.advanced_functions.wide_form import _get_all_possible_column_names as get_all_possible_column_names

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
DB_PATH = os.getenv('PROJECT_DB', '../db/pv_project.duckdb')

# === PARALLEL PROCESSING SETTINGS ===
# For 16 cores/32 threads, try 10-12 workers
# Land Use has more features per tract, so slightly fewer workers may be better
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

# Save interval: commit to DB every N tracts (increased for tract-level processing)
SAVE_INTERVAL = 50

print(f"📊 Configuration:")
print(f"   Database: {DB_PATH}")
print(f"   Processing level: Tract")
print(f"   Max parallel workers: {MAX_WORKERS}")
print(f"   DuckDB threads: {DUCKDB_THREADS}")
print(f"   DuckDB memory limit: {DUCKDB_MEMORY_LIMIT}")
print(f"   Parquet metadata cache: {DUCKDB_PARQUET_CACHE}")
print(f"   States filter: {STATES_TO_PROCESS or 'All'}")

# %% [markdown]
# ## 📥 Task 1: Load Census-Enriched PV Data
# 
# We load tract information from the census-enriched PV data.
# This is more efficient than county-level processing since:
# 1. Smaller query regions = faster Overture API responses
# 2. Only fetch data for tracts that actually have PV installations
# 3. Better caching behavior in OvertureMaestro

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
# Load the census-enriched PV data (REQUIRED - must run Notebook 02 first)
print("\n📥 Loading census_enriched_pv_data...")

# Verify the required table exists
tables = con.execute("SHOW TABLES").fetchall()
table_names = [t[0] for t in tables]

if 'census_enriched_pv_data' not in table_names:
    con.close()
    raise FileNotFoundError(
        "❌ Required table 'census_enriched_pv_data' not found in database.\n"
        "   Please run Notebook 02 (02_geocoding_census_geographies.ipynb) first to create this table.\n"
        f"   Database path: {DB_PATH}\n"
        f"   Available tables: {table_names}"
    )

pv_df = con.execute("SELECT * FROM census_enriched_pv_data").df()
pv_df['geometry'] = pv_df['geometry'].apply(wkt.loads)
pv_gdf = gpd.GeoDataFrame(pv_df, geometry='geometry', crs='EPSG:4326')

# Extract tract information
pv_tracts_info = pv_gdf[['STATE_FIPS', 'COUNTY_FIPS', 'TRACT_GEOID', 'STATE_ABBR']].drop_duplicates()
pv_per_tract = pv_gdf.groupby('TRACT_GEOID').size().to_dict()

print(f"✅ Loaded {len(pv_gdf):,} PV installations")
print(f"   States: {pv_gdf['STATE_ABBR'].nunique()}")
print(f"   Counties: {pv_gdf['COUNTY_GEOID'].nunique()}")
print(f"   Tracts: {len(pv_tracts_info):,}")
print(f"   Avg PV per tract: {len(pv_gdf)/len(pv_tracts_info):.1f}")

con.close()

# %% [markdown]
# ## 🗺️ Task 2: Get Tract Geometries
# 
# Fetch tract polygon geometries for precise Overture filtering.
# We fetch by state to optimize memory usage and API calls.

# %%
# Get unique states from our PV data
unique_states = pv_tracts_info['STATE_FIPS'].unique().tolist()
print(f"📊 PV installations span {len(unique_states)} states")

# Filter states if configured
if STATES_TO_PROCESS:
    state_fips_filter = [s.zfill(2) for s in STATES_TO_PROCESS]
    unique_states = [s for s in unique_states if s in state_fips_filter]
    print(f"   Filtered to {len(unique_states)} states: {unique_states}")

# %%
# Fetch tract geometries from Census Bureau (by state for efficiency)
print("\n🗺️ Fetching tract geometries from Census Bureau...")
reader = cem.ShapeReader(year=2020)

tract_bounds_list = []
for state_fips in tqdm(unique_states, desc="Loading state tracts"):
    try:
        state_tracts = reader.read_cb_shapefile(
            shapefile_scope=state_fips,
            geography="tract",
            crs="EPSG:4326"
        )
        tract_bounds_list.append(state_tracts)
    except Exception as e:
        print(f"   ⚠️ Could not load tracts for state {state_fips}: {e}")

# Combine all tracts
all_tract_bounds = pd.concat(tract_bounds_list, ignore_index=True)
all_tract_bounds = gpd.GeoDataFrame(all_tract_bounds, geometry='geometry', crs='EPSG:4326')
print(f"✅ Loaded {len(all_tract_bounds):,} tract geometries")

# Filter to only tracts with PV installations
pv_tract_geoids = set(pv_tracts_info['TRACT_GEOID'].tolist())
pv_tract_bounds = all_tract_bounds[all_tract_bounds['GEOID'].isin(pv_tract_geoids)].copy()
print(f"✅ Filtered to {len(pv_tract_bounds):,} tracts with PV installations")

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
# ## 🧪 Task 4: Test with a Single Tract
# 
# Verify the Land Use fetch workflow before parallel processing.

# %%
# Pick a test tract
test_tract = pv_tract_bounds.iloc[0]
test_geoid = test_tract['GEOID']
test_name = test_tract.get('NAME', test_geoid)
test_state = test_tract['STATEFP']
test_geom = test_tract.geometry

print(f"🧪 Testing with: Tract {test_geoid}")
print(f"   State FIPS: {test_state}")
print(f"   Bounds: {test_geom.bounds}")
print(f"   PV installations in tract: {pv_per_tract.get(test_geoid, 0)}")

# %%
# Fetch Land Use for test tract
print(f"\n🏘️ Fetching Land Use for tract {test_geoid}...")

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
# - Land Use typically has **more features** per tract than Land Cover
# - Urban tracts may have 1,000+ features (individual parcels)
# - Consider reducing `MAX_WORKERS` if memory becomes an issue
# 
# ### Improvements over county-level processing:
# 1. **Tract-level processing**: More granular, better caching
# 2. **Proper progress tracking**: Shows overall progress, not per-thread
# 3. **Thread-safe result collection**: Uses lock for safe concurrent writes

# %%
def fetch_land_use_for_tract(args):
    """
    Fetch land use data for a single tract and match to PV installations.
    
    Args:
        args: Tuple of (tract_geoid, tract_geometry, pv_subset_gdf)
        
    Returns:
        Tuple of (tract_geoid, list_of_results, error_message or None)
        Each result is a dict with pv_unified_id, lu_id, lu_subtype, lu_class
    """
    tract_geoid, tract_geom, pv_subset = args
    
    try:
        # Fetch land use for this tract
        lu_gdf = om.convert_geometry_to_geodataframe(
            theme="base",
            type="land_use",
            geometry_filter=tract_geom,
            columns_to_download=["id", "subtype", "class", "geometry"],
        )
        
        if len(lu_gdf) == 0:
            return (tract_geoid, [], None)
        
        # Reset index to get id as column
        lu_gdf = lu_gdf.reset_index()
        
        # Match each PV to its containing land use feature
        results = []
        for _, pv_row in pv_subset.iterrows():
            pv_point = pv_row.geometry.centroid
            
            # Find land use that contains this PV
            for _, lu_row in lu_gdf.iterrows():
                if lu_row.geometry.contains(pv_point):
                    results.append({
                        'pv_unified_id': pv_row['unified_id'],
                        'lu_id': lu_row['id'],
                        'lu_subtype': lu_row['subtype'],
                        'lu_class': lu_row.get('class', None)
                    })
                    break  # Each PV gets one land use match
        
        return (tract_geoid, results, None)
        
    except Exception as e:
        return (tract_geoid, [], str(e))

# %%
# Prepare tract tasks
print(f"\n📊 Preparing tract processing tasks...")

tract_tasks = []
for _, tract_row in pv_tract_bounds.iterrows():
    tract_geoid = tract_row['GEOID']
    tract_geom = tract_row.geometry
    pv_subset = pv_gdf[pv_gdf['TRACT_GEOID'] == tract_geoid]
    
    if len(pv_subset) > 0:
        tract_tasks.append((tract_geoid, tract_geom, pv_subset))

print(f"✅ Prepared {len(tract_tasks)} tract tasks")
print(f"   Total PV installations to process: {sum(len(t[2]) for t in tract_tasks):,}")

# %%
# Initialize results storage with thread-safe lock
all_results = []
errors = []
results_lock = Lock()

# Track progress
total_tracts = len(tract_tasks)
total_pv_matched = 0

print(f"\n🚀 Starting parallel fetch with {MAX_WORKERS} workers...")
print(f"   Processing {total_tracts} tracts...")
t_start = time.time()

# Use ThreadPoolExecutor with proper progress tracking
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks
    futures = {executor.submit(fetch_land_use_for_tract, task): task[0] for task in tract_tasks}
    
    # Process completed tasks with progress bar
    with tqdm(total=total_tracts, desc="Processing tracts", unit="tract") as pbar:
        for future in as_completed(futures):
            tract_geoid = futures[future]
            
            try:
                result_tract_id, results, error = future.result()
                
                with results_lock:
                    if error:
                        errors.append({'tract_geoid': result_tract_id, 'error': error})
                    elif results:
                        all_results.extend(results)
                        total_pv_matched += len(results)
                
                # Update progress bar with current stats
                pbar.set_postfix({
                    'matched': total_pv_matched,
                    'errors': len(errors)
                }, refresh=True)
                    
            except Exception as e:
                with results_lock:
                    errors.append({'tract_geoid': tract_geoid, 'error': str(e)})
            
            pbar.update(1)

t_end = time.time()

# %%
# Create results DataFrame
if all_results:
    lu_enrichment_df = pd.DataFrame(all_results)
    print(f"\n💾 Collected {len(lu_enrichment_df):,} PV-Land Use matches")
else:
    lu_enrichment_df = pd.DataFrame(columns=['pv_unified_id', 'lu_id', 'lu_subtype', 'lu_class'])
    print(f"\n⚠️ No land use matches found")

# Processing summary
print(f"\n✅ Parallel processing complete!")
print(f"   Total time: {(t_end - t_start)/60:.1f} minutes")
print(f"   Tracts processed: {total_tracts}")
print(f"   PV installations matched: {len(lu_enrichment_df):,}")
print(f"   Errors: {len(errors)}")

if len(lu_enrichment_df) > 0:
    print(f"\n📊 Land Use Distribution:")
    print(lu_enrichment_df['lu_subtype'].value_counts())

if errors:
    print(f"\n⚠️ Tracts with errors (first 5):")
    for e in errors[:5]:
        print(f"   {e['tract_geoid']}: {e['error'][:80]}...")

# %% [markdown]
# ## 📊 Task 6: Enrich PV Dataset with Land Use Data

# %%
print("\n🔗 Merging land use attributes into PV dataset...")

# Open DB connection
con = duckdb.connect(DB_PATH)
con.execute("INSTALL spatial; LOAD spatial;")
con.execute(f"SET threads={DUCKDB_THREADS};")

# Load existing PV data
pv_data = con.execute("SELECT * FROM census_enriched_pv_data").df()
print(f"   Loaded {len(pv_data):,} PV installations from database")

# Merge land use data
pv_enriched = pv_data.merge(
    lu_enrichment_df,
    left_on='unified_id',
    right_on='pv_unified_id',
    how='left'
)

# Drop the redundant pv_unified_id column
if 'pv_unified_id' in pv_enriched.columns:
    pv_enriched = pv_enriched.drop(columns=['pv_unified_id'])

print(f"   ✅ Merged land use data: {pv_enriched['lu_id'].notna().sum():,} PVs matched")

# Summary statistics
print("\n📊 Land Use Summary by Subtype:")
summary = pv_enriched['lu_subtype'].value_counts()
print(summary.to_string())

print(f"\n📊 Coverage: {pv_enriched['lu_subtype'].notna().sum() / len(pv_enriched) * 100:.1f}% of PV installations have land use data")

# %%
print("\n📊 Top Land Use Classes:")
class_summary = pv_enriched.groupby(['lu_subtype', 'lu_class']).size().sort_values(ascending=False).head(15)
print(class_summary.to_string())

# %% [markdown]
# ## 🗄️ Task 7: Save Enriched Dataset
# 
# Update the database with land use enriched PV data.

# %%
print("\n💾 Saving enriched dataset to database...")

# Drop old table and create new one with land use columns
con.execute("DROP TABLE IF EXISTS lu_enriched_pv_data")

# Register the enriched dataframe
con.register('pv_enriched_temp', pv_enriched)

# Get column names, excluding geometry for proper handling
non_geom_cols = [col for col in pv_enriched.columns if col != 'geometry']
cols_sql = ', '.join(non_geom_cols)

# Create new table with geometry properly handled
con.execute(f"""
    CREATE TABLE lu_enriched_pv_data AS
    SELECT 
        {cols_sql},
        ST_GeomFromText(geometry) as geometry
    FROM pv_enriched_temp
""")

final_count = con.execute("SELECT COUNT(*) FROM lu_enriched_pv_data").fetchone()[0]
print(f"   ✅ Saved {final_count:,} records to lu_enriched_pv_data table")

# Verify land use columns
lu_count = con.execute("SELECT COUNT(*) FROM lu_enriched_pv_data WHERE lu_id IS NOT NULL").fetchone()[0]
print(f"   ✅ {lu_count:,} records have land use data ({lu_count/final_count*100:.1f}%)")

con.close()
print("\n✅ Database connection closed")

# %% [markdown]
# ## 🔗 Combining Land Cover and Land Use
# 
# With both Land Cover (notebook 04) and Land Use (this notebook) enrichments,
# you can combine them for comprehensive LULC analysis:
# 
# ```python
# # Load both enriched datasets
# lc_df = con.execute("SELECT unified_id, lc_id, lc_subtype FROM lc_enriched_pv_data").df()
# lu_df = con.execute("SELECT unified_id, lu_id, lu_subtype, lu_class FROM lu_enriched_pv_data").df()
# 
# # Combine LULC attributes
# lulc_df = lc_df.merge(lu_df, on='unified_id', how='outer')
# ```

# %% [markdown]
# ## 📝 Summary
# 
# This notebook demonstrated:
# 
# 1. **Land Use Concepts**: Human activity classification vs physical cover
# 2. **Hierarchical Schema**: Subtype → Class classification in Overture
# 3. **Tract-Level Processing**: More efficient than county-level (smaller queries, better caching)
# 4. **Parallel Processing**: ThreadPoolExecutor with proper progress tracking
# 5. **Efficient Storage**: Land use IDs stored as attributes in PV dataset (no separate geometry table)
# 
# ### Key Improvements
# 
# - **Tract-level queries**: Only fetch data for ~10K tracts with PV installations
# - **Better progress tracking**: Shows overall progress, not per-thread
# - **Thread-safe collection**: Uses Lock for concurrent result aggregation
# - **No geometry duplication**: Only IDs are stored; source geometries can be fetched from Overture when needed
# 
# ### Output Table Schema
# 
# The `lu_enriched_pv_data` table extends `census_enriched_pv_data` with:
# - `lu_id`: Overture Maps Land Use feature ID
# - `lu_subtype`: Land use type (residential, agriculture, developed, etc.)
# - `lu_class`: Detailed land use class (farmland, retail, etc.)
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
# | Configuration | Expected Time |
# |---------------|---------------|
# | Sequential (1 worker) | ~4-5 hours for ~10K tracts |
# | 4 workers | ~1-1.5 hours |
# | 8 workers | ~30-45 minutes |
# | 16 workers | ~20-30 minutes (may hit API limits) |
# 
# **Tip**: OvertureMaestro caches downloaded data, so re-runs are much faster!
# 
# ### Next Steps
# → Combine lc_enriched_pv_data and lu_enriched_pv_data for full LULC analysis
# → Use combined data in the main E2E notebook for PV-LULC correlation studies
