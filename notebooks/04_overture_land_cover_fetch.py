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
# - **Overture Maps** - Vector-based, [derived from ESA's 2020 WorldCover](https://docs.overturemaps.org/blog/2024/05/16/land-cover/) that uses Sentinel-2 Imagery
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
from threading import Lock

import duckdb
import pandas as pd
import geopandas as gpd
from shapely import wkt, wkb
import matplotlib.pyplot as plt

# OvertureMaestro for Overture Maps data fetching
import overturemaestro as om
from overturemaestro.advanced_functions.wide_form import _get_all_possible_column_names

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
# ### Key Change: Tract-Level Processing
# 
# This notebook now processes at the **census tract level** instead of county level:
# - Only fetches land cover for tracts that have PV installations
# - More efficient: smaller query areas = faster responses
# - Better caching: tract-level queries are more cacheable

# %%
# === PROCESSING CONFIGURATION ===

# Database path (relative to notebooks/ directory)
DB_PATH = os.getenv('PROJECT_DB', '../db/pv_project.duckdb')

# Overture Maps parameters
OVERTURE_RELEASE = os.getenv('OVERTURE_RELEASE', '2025-11-19.0')

# Parallel processing settings
# Set based on your hardware: 16 cores = try 8-12 workers
# Too many workers can cause memory issues or API throttling
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '8'))

# States to process (None = all states with PV installations)
# For testing, use a subset: ['CA', 'AZ', 'NV']
STATES_TO_PROCESS = None  

# Save interval: commit to DB every N tracts to avoid data loss
SAVE_INTERVAL = 50

# DuckDB thread configuration (for multi-threaded operations)
DUCKDB_THREADS = int(os.getenv('DUCKDB_THREADS', '8'))

print(f"📊 Configuration:")
print(f"   Database: {DB_PATH}")
print(f"   Max parallel workers: {MAX_WORKERS}")
print(f"   DuckDB threads: {DUCKDB_THREADS}")
print(f"   States filter: {STATES_TO_PROCESS or 'All'}")
print(f"   Save interval: every {SAVE_INTERVAL} tracts")

# %% [markdown]
# ## 📥 Task 1: Load Census-Enriched PV Data
# 
# We load our PV dataset that has been enriched with Census identifiers in notebook 02.
# This gives us the list of **tracts** (not counties) that contain solar installations.

# %%
print(f"📂 Connecting to database: {DB_PATH}")

con = duckdb.connect(DB_PATH, read_only=True)
con.execute("INSTALL spatial; LOAD spatial;")
con.execute(f"SET threads={DUCKDB_THREADS};")

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

# Handle geometry - check if it's WKT string or already geometry
if 'geometry' in pv_df.columns:
    if isinstance(pv_df['geometry'].iloc[0], str):
        pv_df['geometry'] = pv_df['geometry'].apply(wkt.loads)
    elif isinstance(pv_df['geometry'].iloc[0], bytes):
        pv_df['geometry'] = pv_df['geometry'].apply(wkb.loads)

pv_gdf = gpd.GeoDataFrame(pv_df, geometry='geometry', crs='EPSG:4326')

print(f"✅ Loaded {len(pv_gdf):,} PV installations")
print(f"   States: {pv_gdf['STATE_ABBR'].nunique()}")
print(f"   Counties: {pv_gdf['COUNTY_GEOID'].nunique()}")
print(f"   Tracts: {pv_gdf['TRACT_GEOID'].nunique()}")

con.close()

# %% [markdown]
# ## 🗺️ Task 2: Get Tract Geometries
# 
# We fetch **tract** polygon geometries from Census Bureau shapefiles.
# Only tracts that have PV installations will be processed.

# %%
# Get unique tracts from our PV data
pv_tracts_info = pv_gdf[['STATE_FIPS', 'COUNTY_FIPS', 'TRACT_GEOID', 'STATE_ABBR']].drop_duplicates()
print(f"📊 PV installations span {len(pv_tracts_info)} unique tracts")

# Count PV per tract
pv_per_tract = pv_gdf.groupby('TRACT_GEOID').size()
print(f"\n📊 PV installations per tract:")
print(f"   Min: {pv_per_tract.min()}")
print(f"   Max: {pv_per_tract.max()}")
print(f"   Median: {pv_per_tract.median():.0f}")
print(f"   Mean: {pv_per_tract.mean():.1f}")

# Show top states by tract count
print(f"\n📊 Tracts with PV by state (top 10):")
print(pv_tracts_info.groupby('STATE_ABBR').size().sort_values(ascending=False).head(10))

# %%
# Fetch tract geometries from Census Bureau
# We need to fetch by state since Census shapefiles are organized that way
print("\n🗺️ Fetching tract geometries from Census Bureau...")

# Get unique state FIPS codes
state_fips_list = pv_tracts_info['STATE_FIPS'].unique().tolist()

# Filter states if configured
if STATES_TO_PROCESS:
    state_fips_list = [s for s in state_fips_list if s in STATES_TO_PROCESS or s.lstrip('0') in STATES_TO_PROCESS]
    print(f"   Filtering to states: {STATES_TO_PROCESS}")

print(f"   Fetching tracts for {len(state_fips_list)} states...")

reader = cem.ShapeReader(year=2020)
all_tract_bounds = []

for state_fips in tqdm(state_fips_list, desc="Loading tract geometries"):
    try:
        state_tracts = reader.read_cb_shapefile(
            shapefile_scope=state_fips,
            geography="tract",
            crs="EPSG:4326"
        )
        all_tract_bounds.append(state_tracts)
    except Exception as e:
        print(f"   ⚠️ Could not load tracts for state {state_fips}: {e}")

tract_bounds_gdf = gpd.GeoDataFrame(pd.concat(all_tract_bounds, ignore_index=True))
print(f"✅ Loaded {len(tract_bounds_gdf):,} tract geometries")

# Filter to only tracts with PV installations
pv_tract_geoids = set(pv_tracts_info['TRACT_GEOID'].tolist())
pv_tract_bounds = tract_bounds_gdf[tract_bounds_gdf['GEOID'].isin(pv_tract_geoids)].copy()
print(f"✅ Filtered to {len(pv_tract_bounds):,} tracts with PV installations")

# %% [markdown]
# ## 🌲 Task 3: Explore Land Cover Schema
# 
# Before fetching, let's understand the available land cover categories.

# %%
print("🌲 Available Land Cover subtypes in Overture Maps:")
try:
    land_cover_cols = _get_all_possible_column_names(
        theme="base", 
        type="land_cover", 
        release_version=OVERTURE_RELEASE, 
        hierarchy_columns=['subtype']
    )
    print(land_cover_cols.head(15))
except Exception as e:
    print(f"   Could not fetch schema: {e}")
    print("   Known subtypes: barren, crop, forest, grass, mangrove, moss, shrub, snow, urban, wetland")

# %% [markdown]
# ## 🔧 Task 4: Define Land Cover Fetch Function
# 
# This function fetches land cover for a single tract and matches it to PV installations.
# 
# ### Key Optimizations:
# 1. **Tract-level queries**: Smaller areas = faster responses
# 2. **Direct PV intersection**: Only saves land cover that intersects with PV
# 3. **No geometry storage**: Only stores land cover ID and subtype

# %%
def fetch_land_cover_for_tract(args):
    """
    Fetch land cover data for a single tract and find dominant land cover for PVs.
    
    Args:
        args: tuple of (tract_geoid, tract_geometry, pv_gdf_subset)
        
    Returns:
        tuple: (tract_geoid, list of result dicts, error or None)
        
    Each result dict contains:
        - pv_unified_id: PV installation identifier
        - lc_id: Overture Land Cover feature ID
        - lc_subtype: Land cover type (crop, forest, urban, etc.)
    """
    tract_geoid, tract_geom, pv_subset = args
    
    try:
        # Fetch Land Cover for this tract
        lc_gdf = om.convert_geometry_to_geodataframe(
            theme="base",
            type="land_cover",
            geometry_filter=tract_geom,
            columns_to_download=["id", "subtype", "geometry"],
        )
        
        if lc_gdf is None or len(lc_gdf) == 0:
            return (tract_geoid, [], None)

        # Rename 'id' to 'lc_id' to avoid conflicts
        lc_gdf = lc_gdf.reset_index()
        if 'id' in lc_gdf.columns:
            lc_gdf = lc_gdf.rename(columns={'id': 'lc_id'})
        
        # Ensure CRS matches
        if lc_gdf.crs != pv_subset.crs:
            lc_gdf = lc_gdf.to_crs(pv_subset.crs)

        # Find dominant land cover for each PV installation
        results = []
        
        for _, pv_row in pv_subset.iterrows():
            pv_geom = pv_row.geometry
            pv_id = pv_row.get('unified_id', pv_row.name)
            
            if pv_geom is None or pv_geom.is_empty:
                continue
            
            best_lc_id = None
            best_lc_subtype = None
            max_area = 0.0
            
            # Check each land cover feature for intersection
            for _, lc_row in lc_gdf.iterrows():
                lc_geom = lc_row.geometry
                if lc_geom is None or lc_geom.is_empty:
                    continue
                    
                if pv_geom.intersects(lc_geom):
                    try:
                        intersection = pv_geom.intersection(lc_geom)
                        area = intersection.area
                        
                        if area > max_area:
                            max_area = area
                            best_lc_id = lc_row.get('lc_id', lc_row.get('id'))
                            best_lc_subtype = lc_row['subtype']
                    except Exception:
                        # Skip invalid geometries
                        continue
            
            if best_lc_id is not None:
                results.append({
                    'pv_unified_id': str(pv_id),
                    'lc_id': best_lc_id,
                    'lc_subtype': best_lc_subtype
                })
        
        return (tract_geoid, results, None)
            
    except Exception as e:
        return (tract_geoid, [], str(e))

# %% [markdown]
# ## 🧪 Task 5: Test with a Single Tract
# 
# Before running the full batch, let's verify the workflow with one tract.

# %%
# Pick a test tract (one with a reasonable number of PVs)
test_tract_geoid = pv_per_tract.idxmax()  # Tract with most PVs for good test
test_tract_row = pv_tract_bounds[pv_tract_bounds['GEOID'] == test_tract_geoid].iloc[0]
test_tract_geom = test_tract_row.geometry
test_pv_subset = pv_gdf[pv_gdf['TRACT_GEOID'] == test_tract_geoid]

print(f"🧪 Testing with tract: {test_tract_geoid}")
print(f"   PV installations: {len(test_pv_subset)}")
print(f"   Bounds: {test_tract_geom.bounds}")

# %%
# Fetch Land Cover for test tract
print(f"\n🌲 Fetching Land Cover for test tract...")

t1 = time.time()
test_args = (test_tract_geoid, test_tract_geom, test_pv_subset)
tract_id, test_results, test_error = fetch_land_cover_for_tract(test_args)
t2 = time.time()

if test_error:
    print(f"⚠️ Error: {test_error}")
elif len(test_results) > 0:
    test_df = pd.DataFrame(test_results)
    print(f"✅ Matched {len(test_df):,} PV installations in {t2-t1:.1f}s")
    print(f"\n📊 Land cover distribution:")
    print(test_df['lc_subtype'].value_counts())
else:
    print(f"⚠️ No land cover matches found")

# %% [markdown]
# ## ⚡ Task 6: Parallel Batch Processing
# 
# We use Python's `ThreadPoolExecutor` to fetch data for multiple tracts in parallel.
# 
# ### Improvements over previous version:
# 1. **Tract-level processing**: More granular, better caching
# 2. **Proper progress tracking**: Shows overall progress, not per-thread
# 3. **Thread-safe result collection**: Uses lock for safe concurrent writes

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
    futures = {executor.submit(fetch_land_cover_for_tract, task): task[0] for task in tract_tasks}
    
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
    lc_enrichment_df = pd.DataFrame(all_results)
    print(f"\n💾 Collected {len(lc_enrichment_df):,} PV-Land Cover matches")
else:
    lc_enrichment_df = pd.DataFrame(columns=['pv_unified_id', 'lc_id', 'lc_subtype'])
    print(f"\n⚠️ No land cover matches found")

# Processing summary
print(f"\n✅ Parallel processing complete!")
print(f"   Total time: {(t_end - t_start)/60:.1f} minutes")
print(f"   Tracts processed: {total_tracts}")
print(f"   PV installations matched: {len(lc_enrichment_df):,}")
print(f"   Errors: {len(errors)}")

if len(lc_enrichment_df) > 0:
    print(f"\n📊 Land Cover Distribution:")
    print(lc_enrichment_df['lc_subtype'].value_counts())

if errors:
    print(f"\n⚠️ Tracts with errors (first 5):")
    for e in errors[:5]:
        print(f"   {e['tract_geoid']}: {e['error'][:80]}...")

# %% [markdown]
# ## 📊 Task 7: Enrich PV Dataset with Land Cover Data

# %%
print("\n🔗 Merging land cover attributes into PV dataset...")

# Open DB connection
con = duckdb.connect(DB_PATH)
con.execute("INSTALL spatial; LOAD spatial;")
con.execute(f"SET threads={DUCKDB_THREADS};")

# Load existing PV data
pv_data = con.execute("SELECT * FROM census_enriched_pv_data").df()
print(f"   Loaded {len(pv_data):,} PV installations from database")

# Merge land cover data
pv_enriched = pv_data.merge(
    lc_enrichment_df,
    left_on='unified_id',
    right_on='pv_unified_id',
    how='left'
)

# Drop the redundant pv_unified_id column
if 'pv_unified_id' in pv_enriched.columns:
    pv_enriched = pv_enriched.drop(columns=['pv_unified_id'])

print(f"   ✅ Merged land cover data: {pv_enriched['lc_id'].notna().sum():,} PVs matched")

# Summary statistics
print("\n📊 Land Cover Summary by Subtype:")
summary = pv_enriched['lc_subtype'].value_counts()
print(summary.to_string())

print(f"\n📊 Coverage: {pv_enriched['lc_subtype'].notna().sum() / len(pv_enriched) * 100:.1f}% of PV installations have land cover data")

# %% [markdown]
# ## 🗄️ Task 7: Save Enriched Dataset
# 
# Update the database with land cover enriched PV data.

# %%
print("\n💾 Saving enriched dataset to database...")

# Drop old table and create new one with land cover columns
con.execute("DROP TABLE IF EXISTS lc_enriched_pv_data")

# Register the enriched dataframe
con.register('pv_enriched_temp', pv_enriched)

# Get column names, excluding geometry for proper handling
non_geom_cols = [col for col in pv_enriched.columns if col != 'geometry']
cols_sql = ', '.join(non_geom_cols)

# Create new table with geometry properly handled
con.execute(f"""
    CREATE TABLE lc_enriched_pv_data AS
    SELECT 
        {cols_sql},
        ST_GeomFromText(geometry) as geometry
    FROM pv_enriched_temp
""")

final_count = con.execute("SELECT COUNT(*) FROM lc_enriched_pv_data").fetchone()[0]
print(f"   ✅ Saved {final_count:,} records to lc_enriched_pv_data table")

# Verify land cover columns
lc_count = con.execute("SELECT COUNT(*) FROM lc_enriched_pv_data WHERE lc_id IS NOT NULL").fetchone()[0]
print(f"   ✅ {lc_count:,} records have land cover data ({lc_count/final_count*100:.1f}%)")

con.close()
print("\n✅ Database connection closed")

# %% [markdown]
# ## 📝 Summary
# 
# This notebook demonstrated:
# 
# 1. **Land Cover Fundamentals**: Physical surface classification for Earth observation
# 2. **Overture Maps Integration**: Using overturemaestro for GeoPandas-friendly access
# 3. **Tract-Level Processing**: More efficient than county-level (smaller queries, better caching)
# 4. **Parallel Processing**: ThreadPoolExecutor with proper progress tracking
# 5. **Efficient Storage**: Land cover IDs stored as attributes in PV dataset (no separate geometry table)
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
# The `lc_enriched_pv_data` table extends `census_enriched_pv_data` with:
# - `lc_id`: Overture Maps Land Cover feature ID
# - `lc_subtype`: Land cover type (crop, forest, urban, grass, etc.)
# 
# ### Performance Notes
# 
# | Configuration | Expected Time |
# |---------------|---------------|
# | Sequential (1 worker) | ~4 hours for ~10K tracts |
# | 4 workers | ~1 hour |
# | 8 workers | ~30-45 minutes |
# | 16 workers | ~20-30 minutes (may hit API limits) |
# 
# **Tip**: OvertureMaestro caches downloaded data, so re-runs are much faster!
# 
# ### Next Steps
# → Continue to **Notebook 05** for Land Use data
# → Use `lc_enriched_pv_data` table for PV-LULC correlation analysis
