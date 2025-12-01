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

# %%
# === PROCESSING CONFIGURATION ===

# Database path (relative to notebooks/ directory)
DB_PATH = os.getenv('PROJECT_DB', '../db/pv_project.ddb')

# Overture Maps parameters
OVERTURE_RELEASE = os.getenv('OVERTURE_RELEASE', '2025-11-19.0')


# Parallel processing settings
# Set based on your hardware: 16 cores = try 8-12 workers
# Too many workers can cause memory issues or API throttling
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '8'))

# States to process (None = all states with PV installations)
# For testing, use a subset: ['CA', 'AZ', 'NV']
STATES_TO_PROCESS = None  

# Save interval: commit to DB every N counties to avoid data loss
SAVE_INTERVAL = 10

# DuckDB thread configuration (for multi-threaded operations)
DUCKDB_THREADS = int(os.getenv('DUCKDB_THREADS', '8'))

print(f"📊 Configuration:")
print(f"   Database: {DB_PATH}")
print(f"   Max parallel workers: {MAX_WORKERS}")
print(f"   DuckDB threads: {DUCKDB_THREADS}")
print(f"   States filter: {STATES_TO_PROCESS or 'All'}")

# %% [markdown]
# ## 📥 Task 1: Load Census-Enriched PV Data
# 
# We load our PV dataset that has been enriched with Census identifiers in notebook 02.
# This gives us the list of counties that contain solar installations.

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
pv_df = con.execute("SELECT ST_AsText(geometry) as geometry, * EXCLUDE (geometry) FROM census_enriched_pv_data").df()
pv_df['geometry'] = pv_df['geometry'].apply(wkt.loads)
pv_gdf = gpd.GeoDataFrame(pv_df, geometry='geometry', crs='EPSG:4326')

print(f"✅ Loaded {len(pv_gdf):,} PV installations")
print(f"   States: {pv_gdf['STATE_ABBR'].nunique()}")
print(f"   Counties: {pv_gdf['COUNTY_GEOID'].nunique()}")

con.close()

# %%
pv_gdf.geometry.dtype

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
land_cover_df = _get_all_possible_column_names(theme="base", type="land_cover", release_version=OVERTURE_RELEASE, hierarchy_columns=['subtype'])
# display all land cover subtypes from fetched df
land_cover_df.head()

# %%
def fetch_land_cover_for_county(args):
    """
    Fetch land cover data for a single county and find dominant land cover for PVs.
    args: (county_row, pv_gdf_county)
    Returns tuple: (county_geoid, state_fips, dataframe or None, error or None)
    
    Returns a DataFrame with columns:
    - pv_unified_id: PV installation identifier
    - lc_id: Overture Land Cover feature ID
    - lc_subtype: Land cover type (crop, forest, urban, etc.)
    """
    county_row, pv_gdf_county = args
    geoid = county_row['GEOID']
    state_fips = county_row['STATEFP']
    geom = county_row.geometry
    
    try:
        # Fetch Land Cover (no geometry needed, just IDs and attributes)
        lc_gdf = om.convert_geometry_to_geodataframe(
            theme="base",
            type="land_cover",
            geometry_filter=geom,
            columns_to_download=["id", "subtype", "geometry"],  # Need geometry for intersection calc
        )
        
        if lc_gdf is None or len(lc_gdf) == 0:
            return (geoid, state_fips, None, None)

        # Ensure CRS matches (Overture is EPSG:4326)
        if lc_gdf.crs != pv_gdf_county.crs:
            lc_gdf = lc_gdf.to_crs(pv_gdf_county.crs)

        # Spatial Join to find intersections
        joined = gpd.sjoin(pv_gdf_county, lc_gdf, how='left', predicate='intersects')
        
        # Filter out non-matches
        joined = joined.dropna(subset=['index_right'])
        
        if len(joined) == 0:
             return (geoid, state_fips, None, None)

        # Calculate intersection area for each match to find the dominant land cover
        final_rows = []
        
        # Identify the ID column
        id_col = 'unified_id' if 'unified_id' in pv_gdf_county.columns else pv_gdf_county.index.name or 'index'
        
        for pv_idx, pv_row in pv_gdf_county.iterrows():
            # Get PV ID
            pv_id = pv_row[id_col] if id_col in pv_row else pv_idx
            
            # Get potential matches from sjoin result
            if pv_idx not in joined.index:
                continue
                
            matches = joined.loc[[pv_idx]]
            
            best_lc_id = None
            best_lc_subtype = None
            max_area = -1.0
            pv_geom = pv_row.geometry
            
            for _, match in matches.iterrows():
                lc_idx = match['index_right']
                lc_row = lc_gdf.loc[lc_idx]
                lc_geom = lc_row.geometry
                
                # Calculate intersection area
                intersection = pv_geom.intersection(lc_geom)
                area = intersection.area
                
                if area > max_area:
                    max_area = area
                    best_lc_id = lc_row['id']
                    best_lc_subtype = lc_row['subtype']
            
            if best_lc_id is not None:
                final_rows.append({
                    'pv_unified_id': str(pv_id),
                    'lc_id': best_lc_id,
                    'lc_subtype': best_lc_subtype
                })
        
        if len(final_rows) > 0:
            return (geoid, state_fips, pd.DataFrame(final_rows), None)
        else:
            return (geoid, state_fips, None, None)
            
    except Exception as e:
        return (geoid, state_fips, None, str(e))

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
# Prepare args
test_pv_subset = pv_gdf[pv_gdf['COUNTY_GEOID'] == test_geoid]
test_args = (test_county, test_pv_subset)

_, _, test_result, test_error = fetch_land_cover_for_county(test_args)
t2 = time.time()

if test_result is not None:
    print(f"✅ Fetched {len(test_result):,} PV-Land Cover matches in {t2-t1:.1f}s")
    print(f"\n📊 Land cover distribution:")
    print(test_result['lc_subtype'].value_counts())
else:
    print(f"⚠️ No data found or error: {test_error}")

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
# Filter states if configured
if STATES_TO_PROCESS:
    counties_to_process = pv_county_bounds[pv_county_bounds['STATEFP'].isin(
        [s if len(s) == 2 else f"0{s}" for s in STATES_TO_PROCESS]
    )]
else:
    counties_to_process = pv_county_bounds

print(f"🔄 Processing {len(counties_to_process)} counties with {MAX_WORKERS} parallel workers")

# Prepare county data for parallel processing
# We pair each county with its PV installations
county_records = []
for _, row in counties_to_process.iterrows():
    geoid = row['GEOID']
    pv_subset = pv_gdf[pv_gdf['COUNTY_GEOID'] == geoid]
    if len(pv_subset) > 0:
        county_records.append((row, pv_subset))

print(f"🔄 Prepared {len(county_records)} tasks (counties with PV)")

# %%
# Initialize results storage
all_results = []
errors = []
processed_count = 0

print("📊 Land cover enrichment will be stored as new columns in census_enriched_pv_data")
print("   Columns to add: lc_id, lc_subtype")

# %%
# Parallel processing with progress bar
print(f"\n🚀 Starting parallel fetch with {MAX_WORKERS} workers...")
t_start = time.time()

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
    # Submit all tasks
    future_to_county = {
        executor.submit(fetch_land_cover_for_county, args): args[0]['GEOID']
        for args in county_records
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
                pbar.set_postfix({'matches': sum(len(df) for df in all_results)})
                    
            except Exception as e:
                errors.append({'county_geoid': geoid, 'error': str(e)})
            
            pbar.update(1)

# Consolidate all land cover matches
if all_results:
    lc_enrichment_df = pd.concat(all_results, ignore_index=True)
    print(f"\n💾 Collected {len(lc_enrichment_df):,} PV-Land Cover matches")
else:
    lc_enrichment_df = pd.DataFrame(columns=['pv_unified_id', 'lc_id', 'lc_subtype'])
    print(f"\n⚠️ No land cover matches found")

t_end = time.time()

# %%
# Processing summary
print(f"\n✅ Parallel processing complete!")
print(f"   Total time: {(t_end - t_start)/60:.1f} minutes")
print(f"   Counties processed: {processed_count}")
print(f"   PV installations matched: {len(lc_enrichment_df):,}")
print(f"   Errors: {len(errors)}")

if errors:
    print(f"\n⚠️ Counties with errors:")
    for e in errors[:5]:
        print(f"   {e['county_geoid']}: {e['error'][:50]}...")

# %% [markdown]
# ## 📊 Task 6: Enrich PV Dataset with Land Cover Data

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

# Create new table with geometry preserved
con.execute("""
    CREATE TABLE lc_enriched_pv_data AS
    SELECT 
        *,
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
# 3. **Parallel Processing**: ThreadPoolExecutor for I/O-bound operations
# 4. **Efficient Storage**: Land cover IDs stored as attributes in PV dataset (no separate geometry table)
# 
# ### Key Changes for Space Efficiency
# 
# - **No separate land cover table**: Land cover IDs and subtypes are stored as columns in the PV dataset
# - **No geometry duplication**: Only IDs are stored; source geometries can be fetched from Overture when needed
# - **Optimized joins**: Spatial intersection calculated during fetching, only dominant land cover per PV saved
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
# | Sequential (1 worker) | ~6 hours for 1,071 counties |
# | 4 workers | ~1.5 hours |
# | 8 workers | ~45-60 minutes |
# | 16 workers | ~30-45 minutes (may hit API limits) |
# 
# **Tip**: OvertureMaestro caches downloaded data, so re-runs are much faster!
# 
# ### Next Steps
# → Continue to **Notebook 05** for Land Use data
# → Use `lc_enriched_pv_data` table for PV-LULC correlation analysis
