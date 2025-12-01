# %% [markdown]
# # PV Dataset Consolidation & Standardization
# 
# **CCOM 6994: Data Analysis Tools - Final Project**
# 
# This notebook consolidates multiple raw Solar PV datasets from our S3 data lake into a single, 
# standardized, and spatially deduplicated GeoParquet file that serves as the foundation for all 
# downstream analysis notebooks (01-05).
# 
# ---
# 
# ## 🎯 Objectives
# 
# 1. **Fetch Raw Data**: Retrieve `raw_*` GeoParquet files from data lake via HTTPS.
# 2. **Parallel Processing**: Use `ThreadPoolExecutor` to process datasets concurrently.
# 3. **Standardize Schema**: Apply consistent column naming and handle missing values via COALESCE logic.
# 4. **Enrich Geometries**: Calculate area (m²), centroids (lat/lon), and H3 spatial indices.
# 5. **Consolidate**: Merge all datasets into a unified GeoDataFrame.
# 6. **Spatial Deduplication**: Remove overlapping installations using H3 hexagonal indexing.
# 7. **Export to Data Lake**: Write final dataset to `https://eo-pv-elt.work/geoparquet/ccom6994_pv_dataset.parquet`.
# 
# ---
# 
# ## 🛠️ Technology Stack
# 
# - **Pandas/GeoPandas**: Data manipulation and spatial operations (with native HTTPS support)
# - **H3**: Uber's hierarchical hexagonal spatial indexing
# - **ThreadPoolExecutor**: Parallel I/O for faster processing
# 
# ---

# %% [markdown]
# ## 🔧 Setup: Import Libraries

# %%
import os
import json
import time
import warnings
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import geopandas as gpd
import h3.api.basic_int as h3
from shapely import wkt, wkb
from dotenv import load_dotenv
from tqdm.auto import tqdm

# Suppress warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# Load environment variables
load_dotenv()

# Configure display
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print("✅ Libraries loaded successfully")

# track notebook execution time
nb_start_time = time.time()

# %% [markdown]
# ## ⚙️ Configuration

# %%
# Data Lake Configuration (HTTPS)
DATA_LAKE_BASE_URL = "https://eo-pv-elt.work/geoparquet/"

# Primary output: Data lake (required for downstream notebooks)
OUTPUT_URL = f"{DATA_LAKE_BASE_URL}ccom6994_pv_dataset.parquet"

# S3 output path for R2 export (when credentials available)
S3_OUTPUT_PATH = "s3://eo-pv-lakehouse/geoparquet/ccom6994_pv_dataset.parquet"

# Local export directory
LOCAL_OUTPUT_DIR = Path("../data")
LOCAL_OUTPUT_PATH = LOCAL_OUTPUT_DIR / "ccom6994_pv_dataset.parquet"

# Export control
SKIP_LOCAL_EXPORT = os.getenv('SKIP_PARQUET_EXPORT', 'false').lower() == 'true'

# Processing Configuration
MAX_WORKERS = int(os.getenv('MAX_WORKERS', '8'))
H3_RESOLUTION = 12  # Standard resolution for deduplication
SKIP_DEDUP_ON_INGEST = os.getenv('SKIP_DEDUP_ON_INGEST', 'true').lower() == 'true'
DEFAULT_CRS = "EPSG:4326"
PROJECTED_CRS = "EPSG:3857"  # Web Mercator for area calculations

# Schema Standardization Mapping
# Target Column: [List of potential source columns]
STANDARDIZED_FIELDS = {
    'unified_id': ['fid', 'sol_id', 'GID_0', 'polygon_id', 'unique_id', 'osm_id', 'FID_PV_202', 'arrayID', 'FID', 'eia_id'],
    'source_area_m2': ['Shape_Area', 'Area', 'area_sqm', 'panels.area', 'area', 'totArea', 'area_meters', 'p_area'],
    'capacity_mw': ['capacity_mw', 'power', 'capacity', 'capMW', 'p_cap_dc'],
    'install_date': ['install_date', 'installation_date', 'Date', 'instYr', 'p_year'],
}

# Final Output Columns
FINAL_COLUMNS = [
    'dataset_name', 
    'geometry', 
    'centroid_lon', 
    'centroid_lat', 
    'area_m2', 
    'processed_at', 
    'unified_id', 
    'source_area_m2', 
    'capacity_mw', 
    'install_date',
    f'h3_index_{H3_RESOLUTION}'
]

print(f"📊 Configuration:")
print(f"   Source: {DATA_LAKE_BASE_URL}")
print(f"   Data Lake URL: {OUTPUT_URL}")
print(f"   S3 Output: {S3_OUTPUT_PATH}")
print(f"   Local Export: {LOCAL_OUTPUT_PATH}")
print(f"   Skip Local: {SKIP_LOCAL_EXPORT}")
print(f"   Skip Dedup on Ingest: {SKIP_DEDUP_ON_INGEST}")
print(f"   H3 Resolution: {H3_RESOLUTION}")
print(f"   Parallel Workers: {MAX_WORKERS}")

# %% [markdown]
# ## 🛠️ Helper Functions

# %%
def get_storage_options():
    """Get storage options for pandas read_parquet with HTTPS URLs.
    
    For HTTPS URLs, no storage options are needed - pandas handles them natively.
    This function is kept for consistency with the workflow structure.
    """
    print("   Using HTTPS (no credentials needed for public data lake)")
    return None

def get_s3_storage_options():
    """Get S3 storage options for R2 export using s3fs.
    
    Returns dict with S3 credentials if R2 env vars are set, otherwise None.
    """
    access_key = os.getenv('R2_ACCESS_KEY_ID')
    secret_key = os.getenv('R2_SECRET_KEY')
    endpoint = os.getenv('R2_S3_ENDPOINT', 'e833ac2d32c62bcff5e4b72c74e5351d.r2.cloudflarestorage.com')
    
    if access_key and secret_key:
        print(f"   Using R2 S3 credentials (endpoint: {endpoint})")
        return {
            'key': access_key,
            'secret': secret_key,
            'client_kwargs': {
                'region_name': 'auto',  # R2 uses 'auto' as region
                'endpoint_url': f'https://{endpoint}'
            }
        }
    else:
        return None

def calculate_geometry_stats(gdf):
    """Calculate area and centroids using projected CRS."""
    # Project to Web Mercator for accurate metric calculations
    gdf_proj = gdf.to_crs(PROJECTED_CRS)
    
    # Calculate Area
    gdf['area_m2'] = gdf_proj.geometry.area
    
    # Calculate Centroids (in WGS84)
    centroids_proj = gdf_proj.geometry.centroid
    centroids_wgs84 = centroids_proj.to_crs(DEFAULT_CRS)
    
    gdf['centroid_lon'] = centroids_wgs84.x
    gdf['centroid_lat'] = centroids_wgs84.y
    
    return gdf

def assign_h3_index(gdf, resolution):
    """Assign H3 index to each geometry centroid.
    
    Uses h3.latlng_to_cell (current API) instead of deprecated geo_to_h3.
    See: https://uber.github.io/h3-py/api_quick.html
    """
    def get_h3(lat, lon):
        try:
            # Use latlng_to_cell (current API) - returns int
            return h3.latlng_to_cell(lat, lon, resolution)
        except:
            return None
            
    gdf[f'h3_index_{resolution}'] = gdf.apply(
        lambda row: get_h3(row['centroid_lat'], row['centroid_lon']), 
        axis=1
    )
    return gdf

def enforce_schema(gdf):
    """Enforce consistent data types across all columns."""
    # String columns
    gdf['dataset_name'] = gdf['dataset_name'].astype(str)
    if 'unified_id' in gdf.columns:
        gdf['unified_id'] = gdf['unified_id'].astype(str)
    
    # Float columns
    for col in ['centroid_lon', 'centroid_lat', 'area_m2', 'source_area_m2', 'capacity_mw']:
        if col in gdf.columns:
            gdf[col] = pd.to_numeric(gdf[col], errors='coerce')
    
    # Timestamp columns
    if 'processed_at' in gdf.columns:
        gdf['processed_at'] = pd.to_datetime(gdf['processed_at'], errors='coerce')
    
    # DateTime columns (install_date)
    if 'install_date' in gdf.columns:
        gdf['install_date'] = pd.to_datetime(gdf['install_date'], errors='coerce')
    
    # Integer columns (H3 index)
    h3_col = f'h3_index_{H3_RESOLUTION}'
    if h3_col in gdf.columns:
        gdf[h3_col] = pd.to_numeric(gdf[h3_col], errors='coerce').astype('Int64')  # Nullable integer
    
    return gdf

def standardize_schema(gdf, dataset_name):
    """Rename columns to standard schema and fill missing fields."""
    # 1. Add metadata
    gdf['dataset_name'] = dataset_name
    gdf['processed_at'] = pd.Timestamp.now()
    
    # 2. Coalesce standardized fields
    for target_col, candidates in STANDARDIZED_FIELDS.items():
        # Find the first candidate that exists in the dataframe
        found = False
        for candidate in candidates:
            if candidate in gdf.columns:
                gdf[target_col] = gdf[candidate]
                found = True
                break
        
        # If no candidate found, create empty column
        if not found:
            gdf[target_col] = None
            
    # 3. Ensure all final columns exist
    for col in FINAL_COLUMNS:
        if col not in gdf.columns:
            gdf[col] = None
    
    # 4. Enforce consistent data types
    gdf = enforce_schema(gdf)
            
    # 5. Return only selected columns
    return gdf[FINAL_COLUMNS]

def process_dataset(url, storage_options):
    """
    Process a single raw dataset: Fetch -> Standardize -> Enrich -> Return
    
    Args:
        url: HTTPS URL to raw GeoParquet file
        storage_options: Storage options for pandas (None for HTTPS)
        
    Returns:
        Tuple of (dataset_name, standardized_gdf, basic_stats, error)
    """
    dataset_name = Path(url).stem.replace('raw_', '')
    
    try:
        # 1. Read Parquet using pandas (HTTPS URLs work natively)
        if storage_options:
            df = pd.read_parquet(url, storage_options=storage_options)
        else:
            df = pd.read_parquet(url)
    
        # 2. Convert to GeoDataFrame (handle WKB/WKT geometry formats)
        if 'geometry' in df.columns:
            # handle either WKT or WKB formats
            if df['geometry'].dtype == 'object' and len(df) > 0:
                first_val = df['geometry'].iloc[0]
                if isinstance(first_val, str):
                    # WKT text format (common in GeoDataFrames)
                    df['geometry'] = df['geometry'].apply(wkt.loads)
                elif isinstance(first_val, bytes):
                    # Assume WKB binary format (common in raw data)
                    df['geometry'] = df['geometry'].apply(wkb.loads)
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs=DEFAULT_CRS)
        else:
            raise ValueError("No geometry column found in dataset")
        
        # 3. Basic pre-processing stats (column list + describe)
        original_rows = len(gdf)
        original_columns = list(gdf.columns)
        
        # Filter invalid geometries
        gdf = gdf[gdf.geometry.notna() & gdf.geometry.is_valid].copy()
        
        # 4. Calculate Geometry Stats (Area, Centroids)
        gdf = calculate_geometry_stats(gdf)
        
        # 5. Assign H3 Spatial Index
        gdf = assign_h3_index(gdf, H3_RESOLUTION)
        
        # 6. Standardize Schema (COALESCE logic for unified columns)
        gdf_std = standardize_schema(gdf, dataset_name)
        
        # 7. Return basic stats (detailed missingness analysis happens after consolidation)
        stats = {
            'dataset': dataset_name,
            'original_rows': original_rows,
            'final_rows': len(gdf_std),
            'original_columns': original_columns,
            'invalid_geoms_removed': original_rows - len(gdf)
        }
        
        return dataset_name, gdf_std, stats, None
        
    except Exception as e:
        return dataset_name, None, None, str(e)

# %% [markdown]
# ## 📋 Task 1: Load Dataset Manifest & Build HTTPS URLs
# 
# We read from `doi_manifest.json` to get the list of datasets, then construct 
# HTTPS URLs using the pattern `raw_{dataset_key}.parquet`.

# %%
# Load manifest file
MANIFEST_PATH = Path(os.getenv('DATASET_MANIFEST_PATH', '../ingest/doi_manifest_usa.json'))
print(f"📋 Loading dataset manifest from {MANIFEST_PATH.name}...")

with open(MANIFEST_PATH, 'r') as f:
    manifest = json.load(f)

# Filter to datasets where skip=false
active_datasets = {
    key: meta for key, meta in manifest.items() 
    if not meta.get('skip', False)
}

# Build HTTPS URLs for raw parquet files
dataset_paths = [
    f"{DATA_LAKE_BASE_URL}raw_{dataset_key}.parquet" 
    for dataset_key in active_datasets.keys()
]

print(f"   Found {len(dataset_paths)} active datasets in manifest:")
for dataset_key in active_datasets.keys():
    print(f"   - {dataset_key}")

print(f"\n📍 HTTPS URLs to fetch:")
for path in dataset_paths:
    print(f"   {path}")

# %% [markdown]
# ## 🚀 Task 2: Parallel Fetch & Standardize
# 
# Process all datasets in parallel using `ThreadPoolExecutor`.
# Each worker:
# 1. Reads a raw GeoParquet from S3
# 2. Standardizes the schema (column renaming via COALESCE)
# 3. Calculates geometry stats (area, centroids)
# 4. Assigns H3 spatial indices
# 5. Returns a standardized GeoDataFrame

# %%
_, gdf, tmp, tmp2 = process_dataset(dataset_paths[1], storage_options=None)

# %%
gdf.head()

# %%
len(gdf)

# %%
len(gdf['unified_id'].unique())

# %%
tmp

# %%
# Get storage options (None for HTTPS URLs)
storage_options = get_storage_options()

processed_dfs = []
dataset_stats = []
errors = []

print(f"\n🚀 Starting parallel processing with {MAX_WORKERS} workers...")
t_start = time.time()


with ThreadPoolExecutor(max_workers=len(dataset_paths)) as executor:
    # Submit all tasks
    future_to_path = {
        executor.submit(process_dataset, path, storage_options): path 
        for path in dataset_paths
    }
    
    # Process results as they complete
    with tqdm(total=len(dataset_paths), desc="Processing Datasets") as pbar:
        for future in as_completed(future_to_path):
            ds_name, gdf, stats, error = future.result()
            
            if error:
                errors.append({'dataset': ds_name, 'error': error})
                print(f"   ❌ Error: {ds_name} - {error}")
            else:
                processed_dfs.append(gdf)
                dataset_stats.append(stats)
                print(f"   ✅ {ds_name}: {stats['final_rows']:,} rows")
                
            pbar.update(1)

t_end = time.time()
print(f"\n⏱️  Processing time: {t_end - t_start:.2f} seconds")

if errors:
    print(f"\n⚠️  {len(errors)} dataset(s) failed:")
    for e in errors:
        print(f"   - {e['dataset']}: {e['error']}")

# %% [markdown]
# ## 📊 Task 3: Dataset Ingestion Summary
# 
# Review the raw datasets we've loaded. Each dataset's basic statistics:
# - Original row count
# - Final row count (after filtering invalid geometries)
# - Column names from the raw source

# %%
if dataset_stats:
    print("\n📊 Dataset Ingestion Summary:")
    stats_df = pd.DataFrame(dataset_stats)
    print(stats_df[['dataset', 'original_rows', 'final_rows', 'invalid_geoms_removed']].to_string(index=False))
    
    print("\n📋 Sample: Original column names per dataset:")
    for stat in dataset_stats[:3]:  # Show first 3 as examples
        print(f"\n{stat['dataset']}:")
        print(f"  {', '.join(stat['original_columns'][:10])}...")
else:
    print("\n❌ No datasets were successfully processed.")
    raise SystemExit(1)

# %% [markdown]
# ## 🔗 Task 4: Consolidation
# 
# Merge all standardized datasets into a single unified GeoDataFrame.
# All datasets now share the same schema and coordinate reference system.
# Generate unique IDs for each installation based on consolidated index.

# %%
print("\n🔗 Consolidating all datasets...")
consolidated_gdf = pd.concat(processed_dfs, ignore_index=True)

# Generate unified IDs based on dataset name and index
consolidated_gdf['unified_id'] = (
    consolidated_gdf['dataset_name'].astype(str) + '_' + 
    consolidated_gdf.index.astype(str).str.zfill(8)
)
print(f"   ✅ Generated {len(consolidated_gdf):,} unique IDs")

# Enforce schema one final time after concatenation
consolidated_gdf = enforce_schema(consolidated_gdf)

total_rows = len(consolidated_gdf)
print(f"   ✅ Consolidated: {total_rows:,} total installations")
print(f"   📊 Datasets: {consolidated_gdf['dataset_name'].nunique()}")
print(f"\n   Dataset distribution:")
print(consolidated_gdf['dataset_name'].value_counts().to_string())

# %% [markdown]
# ## 📈 Task 5: Consolidated Data Quality Analysis
# 
# **Now** we perform detailed data quality analysis on the consolidated dataset:
# - Missing value percentages for each column
# - Descriptive statistics for numeric columns
# - Temporal coverage (install dates)

# %%
print("\n📈 Consolidated Data Quality Analysis:")

# Missing value analysis
print("\n1️⃣ Missing Values by Column:")
missing_pct = (consolidated_gdf.isna().sum() / len(consolidated_gdf) * 100).sort_values(ascending=False)
for col, pct in missing_pct.items():
    if pct > 0:
        print(f"   {col}: {pct:.1f}%")

# Numeric column statistics
print("\n2️⃣ Numeric Column Statistics:")
numeric_cols = ['area_m2', 'centroid_lat', 'centroid_lon', 'source_area_m2', 'capacity_mw']
print(consolidated_gdf[numeric_cols].describe().to_string())

# Temporal analysis (if install_date available)
if 'install_date' in consolidated_gdf.columns and consolidated_gdf['install_date'].notna().any():
    print("\n3️⃣ Temporal Coverage (Install Dates):")
    valid_dates = consolidated_gdf['install_date'].dropna()
    # Filter out obviously invalid dates (before 1990 or in future)
    valid_dates = valid_dates[(valid_dates.dt.year >= 1990) & (valid_dates.dt.year <= 2030)]
    print(f"   Records with valid dates: {len(valid_dates):,} ({len(valid_dates)/len(consolidated_gdf)*100:.1f}%)")
    if len(valid_dates) > 0:
        try:
            print(f"   Date range: {valid_dates.min()} to {valid_dates.max()}")
        except Exception as e:
            print(f"   Unable to calculate date range: {e}")

# %% [markdown]
# ## ✂️ Task 6: Spatial Deduplication (Optional)
# 
# Remove overlapping installations using H3 hexagonal spatial indexing.
# 
# **Strategy:**
# 1. Sort by `area_m2` (descending) to prioritize larger/more detailed polygons
# 2. Drop duplicates based on H3 index (resolution 12 ≈ 0.0003 km²)
# 3. Keep the first occurrence (largest polygon) for each hex
# 
# **Why H3?**
# - Consistent spatial binning across datasets
# - Handles edge cases near dataset boundaries
# - Fast lookup and deduplication
# 
# **Note:** By default, deduplication is SKIPPED during consolidation (SKIP_DEDUP_ON_INGEST=true).
# This allows downstream notebooks to filter to their region of interest first, then deduplicate.
# This is more efficient for analysis focused on specific regions (e.g., US-only analysis).

# %%
if SKIP_DEDUP_ON_INGEST:
    print(f"\n✂️ Deduplication: SKIPPED (SKIP_DEDUP_ON_INGEST=true)")
    print(f"   Deduplication will be performed in downstream notebooks after regional filtering")
    print(f"   Final dataset: {total_rows:,} installations (includes global data)")
    deduped_gdf = consolidated_gdf
else:
    print(f"\n✂️ Deduplicating using H3 index (resolution {H3_RESOLUTION})...")
    
    # Sort by area (largest first)
    consolidated_gdf = consolidated_gdf.sort_values('area_m2', ascending=False)
    
    # Drop duplicates on H3 index
    h3_col = f'h3_index_{H3_RESOLUTION}'
    deduped_gdf = consolidated_gdf.drop_duplicates(subset=[h3_col], keep='first')
    
    dupes_removed = total_rows - len(deduped_gdf)
    print(f"   ✅ Removed {dupes_removed:,} duplicates ({dupes_removed/total_rows*100:.1f}%)")
    print(f"   📊 Final dataset: {len(deduped_gdf):,} unique installations")

# %% [markdown]
# ## 💾 Task 7: Export Dataset
# 
# Save the final consolidated and deduplicated dataset.
# 
# **Export Priority:**
# 1. **S3 (R2)**: If R2 credentials available, export to `s3://eo-pv-lakehouse/geoparquet/ccom6994_pv_dataset.parquet`
# 2. **Local**: Always export to `../data/ccom6994_pv_dataset.parquet` (unless SKIP_PARQUET_EXPORT=true)
# 
# The HTTPS URL `https://eo-pv-elt.work/geoparquet/ccom6994_pv_dataset.parquet` is the canonical 
# read-only path for downstream notebooks.

# %%
print(f"\n💾 Exporting consolidated dataset...")

# Try S3 export first (if R2 credentials available)
r2_s3_endpoint = os.getenv('R2_S3_ENDPOINT', 'e833ac2d32c62bcff5e4b72c74e5351d.r2.cloudflarestorage.com')
r2_access_key = os.getenv('R2_ACCESS_KEY_ID', '')
r2_secret_key = os.getenv('R2_SECRET_KEY', '')
s3_url_style = 'path'

export_to_s3 = True if r2_access_key and r2_secret_key else False
export_gdf = consolidated_gdf.copy() if SKIP_DEDUP_ON_INGEST else deduped_gdf.copy()
# set geometry from shapely to WKB for compatibility
export_gdf['geometry'] = export_gdf['geometry'].apply(lambda geom: geom.wkt)

if export_to_s3:
    print(f"\n1️⃣ Attempting S3 export with DuckDB: {S3_OUTPUT_PATH}")
    # use memory db for export
    import duckdb
    t_s3_start = time.time()
    try:
        duckdb_conn = duckdb.connect(database=':memory:')
        # set R2 s3 credentials
        duckdb_conn.execute(f"set s3_access_key_id='{r2_access_key}';")
        duckdb_conn.execute(f"set s3_secret_access_key='{r2_secret_key}';")
        duckdb_conn.execute(f"set s3_url_style='{s3_url_style}';")
        duckdb_conn.execute(f"set s3_endpoint='{r2_s3_endpoint}';")
        duckdb_conn.execute(f"""
            CREATE TABLE pv_data AS 
            SELECT * FROM export_gdf
        """)
        duckdb_conn.execute(f"""
            COPY pv_data TO '{S3_OUTPUT_PATH}' 
            (FORMAT PARQUET, COMPRESSION 'zstd', COMPRESSION_LEVEL 12);
        """)
        t_s3_end = time.time()
        print(f"   ✅ S3 export complete ({t_s3_end - t_s3_start:.1f}s)")
        exported_to_s3 = True
    except Exception as e:
        print(f"   ❌ S3 export failed: {e}")
        exported_to_s3 = False
else:
    print(f"\n1️⃣ S3 export: Skipped (no R2 credentials found)")
    print(f"   To enable S3 export, set: R2_ACCESS_KEY_ID, R2_SECRET_KEY")

# Local export (always, unless explicitly skipped)
if not SKIP_LOCAL_EXPORT:
    print(f"\n2️⃣ Saving local copy: {LOCAL_OUTPUT_PATH}")
    t_local_start = time.time()
    LOCAL_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    export_gdf.to_parquet(LOCAL_OUTPUT_PATH, compression='snappy')
    t_local_end = time.time()
    
    file_size = LOCAL_OUTPUT_PATH.stat().st_size / (1024 * 1024)
    print(f"   ✅ Local export complete ({file_size:.2f} MB, {t_local_end - t_local_start:.1f}s)")
else:
    print(f"\n2️⃣ Local export: Skipped (SKIP_PARQUET_EXPORT=true)")

# Summary
if not exported_to_s3 and SKIP_LOCAL_EXPORT:
    print(f"\n⚠️  WARNING: No exports completed!")
elif exported_to_s3:
    print(f"\n✅ Export complete! Dataset accessible at: {OUTPUT_URL}")
else:
    print(f"\n📝 Next step: Upload {LOCAL_OUTPUT_PATH} to {OUTPUT_URL}")

print("\n✅ Dataset consolidation complete!")
print(f"   Final size: {len(export_gdf):,} installations")
print(f"   Columns: {len(export_gdf.columns)}")
if SKIP_DEDUP_ON_INGEST:
    print(f"   Note: Includes duplicates across global datasets")
    print(f"   Deduplication should be performed after regional filtering")
if exported_to_s3:
    print(f"   S3 output: {S3_OUTPUT_PATH}")
if not SKIP_LOCAL_EXPORT:
    print(f"   Local output: {LOCAL_OUTPUT_PATH}")
print(f"   HTTPS URL: {OUTPUT_URL}")

# %%
nb_end_time = time.time()
print(f"\n⏱️  Total notebook execution time: {nb_end_time - nb_start_time:.2f} seconds")


