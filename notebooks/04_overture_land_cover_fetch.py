# %% [markdown]
# # Overture Maps: Fetching Land Cover Data (Batch Processing)
# 
# **CCOM 6994: Data Analysis Tools - Final Project**
# 
# This notebook fetches **Land Cover** data from the **Overture Maps Foundation** dataset.
# To handle the large data volume efficiently, we:
# 1.  Identify **Census Counties** that contain Solar Panels (from our PV dataset).
# 2.  Process data **State by State**.
# 3.  Fetch Overture data for the State's "PV Counties" bounding box.
# 4.  Spatially filter to keep only land cover features intersecting these counties.
# 5.  Save the results to **DuckDB**.
# 
# ### 🛠️ Tools Used
# -   **DuckDB**: Spatial SQL engine for efficient querying and filtering.
# -   **GeoPandas**: For spatial joins and geometry handling.
# -   **Lonboard**: For high-performance interactive map visualization.
# 
# ---

# %% [markdown]
# ## 🔧 Setup: Import Libraries

# %%
import os
import time
import duckdb
import pandas as pd
import geopandas as gpd
from shapely import wkt
from shapely.geometry import box
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# Try importing lonboard for visualization
try:
    from lonboard import Map, viz
    LONBOARD_AVAILABLE = True
except ImportError:
    print("⚠️ lonboard not installed. Visualization will be limited.")
    LONBOARD_AVAILABLE = False

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.getcwd()), '.env'))

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# %% [markdown]
# ## 📥 Task 1: Load Data from DuckDB
# 
# We need two datasets:
# 1.  **PV Data**: `processed_pv_data` (Points/Polygons of solar panels)
# 2.  **Census Counties**: `census_acs5_county_2020` (Polygons of US Counties)

# %%
DB_PATH = os.getenv('PROJECT_DB', 'db/pv_project.ddb')
print(f"📂 Connecting to database: {DB_PATH}")

con = duckdb.connect(DB_PATH)
con.execute("INSTALL spatial; LOAD spatial;")
con.execute("INSTALL httpfs; LOAD httpfs;")
con.execute("SET s3_region='us-west-2';") # Overture bucket region

# 1. Load PV Data (we only need geometry for the join)
print("   Loading PV Data...")
pv_df = con.execute("SELECT geometry FROM processed_pv_data").df()
pv_df['geometry'] = pv_df['geometry'].apply(wkt.loads)
pv_gdf = gpd.GeoDataFrame(pv_df, geometry='geometry', crs='EPSG:4326')
print(f"   ✅ Loaded {len(pv_gdf):,} PV installations.")

# 2. Load Census Counties
print("   Loading Census Counties...")
# We use the 2020 table which now has GEOID
counties_df = con.execute("SELECT * FROM census_acs5_county_2020").df()
counties_df['geometry'] = counties_df['geometry'].apply(wkt.loads)
counties_gdf = gpd.GeoDataFrame(counties_df, geometry='geometry', crs='EPSG:4326')
print(f"   ✅ Loaded {len(counties_gdf):,} Counties.")

# %% [markdown]
# ## 🔗 Task 2: Identify "PV Counties"
# 
# We only want Land Cover data for counties that actually have solar panels.
# We perform a **Spatial Join** to find these counties.

# %%
print("🔗 Identifying Counties with PV installations...")

# Spatial Join: Counties containing PV points
# We use 'inner' join to keep only matching counties
# Note: This assumes PV data are points or small polygons inside counties
pv_counties = gpd.sjoin(counties_gdf, pv_gdf, how='inner', predicate='intersects')

# Drop duplicates using GEOID (which is now guaranteed to exist)
if 'GEOID' in pv_counties.columns:
    pv_counties = pv_counties.drop_duplicates(subset=['GEOID'])
else:
    # Fallback if GEOID is missing (should not happen with new Notebook 3)
    print("⚠️ GEOID not found, using STATE+COUNTY fallback.")
    pv_counties = pv_counties.drop_duplicates(subset=['STATE', 'COUNTY'])

print(f"   ✅ Found {len(pv_counties):,} counties containing solar panels.")
print(f"   States involved: {pv_counties['STATE'].unique().tolist()}")

# %% [markdown]
# ## 🔄 Task 3: Batch Processing by State
# 
# We will iterate through each State present in our "PV Counties".
# For each state:
# 1.  Get the bounding box of all relevant counties.
# 2.  Fetch Overture Land Cover data for that bbox.
# 3.  Filter to keep only features intersecting the counties.
# 4.  Append to DuckDB.

# %%
# Initialize output table in DuckDB
con.execute("DROP TABLE IF EXISTS overture_land_cover")
con.execute("""
    CREATE TABLE overture_land_cover (
        id VARCHAR,
        subtype VARCHAR,
        geometry GEOMETRY,
        state_fips VARCHAR
    )
""")

# Overture Configuration
OVERTURE_RELEASE = "2025-11-19.0"
S3_PATH = f"s3://overturemaps-us-west-2/release/{OVERTURE_RELEASE}/theme=base/type=land_cover/*"

# Get unique states to process
states_to_process = pv_counties['STATE'].unique()

print(f"🔄 Starting batch processing for {len(states_to_process)} states...")

for state_fips in states_to_process:
    print(f"\n   📍 Processing State FIPS: {state_fips}")
    
    # 1. Filter counties for this state
    state_counties = pv_counties[pv_counties['STATE'] == state_fips]
    
    # 2. Calculate Bounding Box for these counties
    # (minx, miny, maxx, maxy)
    bounds = state_counties.total_bounds
    bbox_str = f"{bounds[0]}, {bounds[1]}, {bounds[2]}, {bounds[3]}"
    print(f"      BBox: {bbox_str}")
    
    # 3. Create a temporary table for the counties geometry (for spatial filtering)
    # We convert to WKT for DuckDB
    state_counties_wkt = state_counties[['geometry']].copy()
    state_counties_wkt['geometry'] = state_counties_wkt['geometry'].apply(lambda x: x.wkt)
    
    con.execute("CREATE OR REPLACE TABLE current_state_counties AS SELECT * FROM state_counties_wkt")
    con.execute("ALTER TABLE current_state_counties ALTER geometry TYPE GEOMETRY USING ST_GeomFromText(geometry)")
    
    # 4. Fetch and Filter Overture Data
    # We use a subquery to read from S3 with BBox filter
    # Then we join with our counties table
    query = f"""
        INSERT INTO overture_land_cover
        SELECT 
            lc.id,
            lc.subtype,
            lc.geometry,
            '{state_fips}' as state_fips
        FROM (
            SELECT 
                id, 
                subtype, 
                ST_GeomFromWKB(geometry) as geometry
            FROM read_parquet('{S3_PATH}')
            WHERE bbox.xmin > {bounds[0]} 
              AND bbox.xmax < {bounds[2]}
              AND bbox.ymin > {bounds[1]} 
              AND bbox.ymax < {bounds[3]}
        ) lc
        JOIN current_state_counties c
        ON ST_Intersects(lc.geometry, c.geometry)
    """
    
    t1 = time.time()
    con.execute(query)
    t2 = time.time()
    
    # Check how many rows inserted
    count = con.execute(f"SELECT COUNT(*) FROM overture_land_cover WHERE state_fips = '{state_fips}'").fetchone()[0]
    print(f"      ✅ Fetched & Filtered {count:,} features in {t2 - t1:.2f}s")

print("\n✅ Batch processing complete.")

# %% [markdown]
# ## 📊 Task 4: Visualization with Lonboard
# 
# Let's visualize the results for a sample area (e.g., one state or a subset) using `lonboard`.

# %%
if LONBOARD_AVAILABLE:
    print("📊 Visualizing with Lonboard...")
    
    # Fetch a subset for visualization (e.g., first 100k rows or a specific state)
    # Let's pick the state with the most PV counties
    top_state = pv_counties['STATE'].mode()[0]
    print(f"   Visualizing State FIPS: {top_state}")
    
    viz_df = con.execute(f"SELECT * FROM overture_land_cover WHERE state_fips = '{top_state}' LIMIT 50000").df()
    
    if not viz_df.empty:
        # Convert to GeoDataFrame
        from shapely import wkb
        
        try:
            viz_df['geometry'] = viz_df['geometry'].apply(lambda x: wkb.loads(bytes(x)))
        except Exception:
             # Fallback if it's already string or something else
            viz_df['geometry'] = viz_df['geometry'].apply(lambda x: wkt.loads(str(x)))
            
        viz_gdf = gpd.GeoDataFrame(viz_df, geometry='geometry', crs='EPSG:4326')
        
        # Create Map
        layer = viz.viz(viz_gdf, get_fill_color=[0, 128, 255, 100])
        display(layer)
        
    else:
        print("   ⚠️ No data found for visualization.")
else:
    print("⚠️ Lonboard not available. Skipping visualization.")

# %% [markdown]
# ## 💾 Task 5: Verify and Cleanup
# 
# Verify the final table size and close the connection.

# %%
total_count = con.execute("SELECT COUNT(*) FROM overture_land_cover").fetchone()[0]
print(f"\n📈 Total Land Cover features saved: {total_count:,}")

con.close()
print("✅ Database connection closed.")
