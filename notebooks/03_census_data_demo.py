# %% [markdown]
# # Notebook 3: Census Data Integration with censusdis
# 
# **CCOM 6994: Data Analysis Tools - Final Project**  
# **Course: Análisis de Datos de Paneles Solares**
# 
# ---
# 
# ## 🎯 Learning Objectives
# 
# By the end of this notebook, you will be able to:
# 
# 1.  **Discover Census Datasets**: Programmatically list available datasets and variables using the Census API.
# 2.  **Fetch Demographic Data**: Retrieve ACS 5-Year Estimates for the entire US and Puerto Rico.
# 3.  **Visualize Spatial Data**: Create choropleth maps to analyze income distribution.
# 4.  **Persist Data**: Save census data to DuckDB with proper identifiers (`GEOID`) for integration.
# 
# ## 📚 References & Documentation
# 
# -   [censusdis Introduction](https://censusdis.readthedocs.io/en/latest/intro.html)
# -   [Census API Datasets (2020)](https://api.census.gov/data/2020.html)
# -   [Exploring Variables](https://censusdis.readthedocs.io/en/latest/nb/Exploring%20Variables.html)
# -   [Data With Geometry](https://censusdis.readthedocs.io/en/latest/nb/Data%20With%20Geometry.html)
# 
# ---

# %% [markdown]
# ## 🔧 Setup: Import Libraries

# %%
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import duckdb
from dotenv import load_dotenv

# censusdis imports
import censusdis.data as ced
import censusdis.states as states

# Load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.getcwd()), '.env'))

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)
plt.style.use('seaborn-v0_8-darkgrid')

print("✅ Libraries imported successfully.")

# %% [markdown]
# ---
# 
# ## 🔍 Part 1: Programmatic Data Discovery
# 
# Instead of hardcoding dataset names and variables, let's explore what's available via the Census API.
# 
# ### 1.1 Listing Available Datasets (2020)
# 
# The Census API provides a list of datasets for each year. We can fetch this table directly.

# %%
YEAR = 2020
DATASETS_URL = f"https://api.census.gov/data/{YEAR}.html"

print(f"🔍 Fetching available datasets for {YEAR} from: {DATASETS_URL}")

try:
    # Read HTML tables from the URL
    tables = pd.read_html(DATASETS_URL)
    datasets_df = tables[0] # The first table usually contains the dataset list
    
    # Clean up the dataframe
    # The table structure varies, but usually has Title, Description, Dataset Name, etc.
    print(f"   Found {len(datasets_df)} datasets.")
    
    # Filter for ACS datasets as an example
    acs_datasets = datasets_df[datasets_df['Title'].astype(str).str.contains('American Community Survey', case=False, na=False)]
    print("\n📊 ACS Datasets found:")
    display(acs_datasets[['Title', 'Description']].head())
    
except Exception as e:
    print(f"⚠️ Could not fetch datasets programmatically: {e}")
    print("   Falling back to manual selection.")

# %%
# manageable with data wrangler extension in vscode
display(tables[0])

# %% [markdown]
# ### 1.2 Discovering Variables
# 
# Once we choose a dataset (e.g., `acs/acs5`), we can fetch its variables.
# URL format: `https://api.census.gov/data/{YEAR}/acs/acs5/variables.json` (or .html)

# %%
DATASET_NAME = 'acs/acs5'
VARIABLES_URL = f"https://api.census.gov/data/{YEAR}/{DATASET_NAME}/variables.json"

print(f"🔍 Fetching variables for '{DATASET_NAME}' from: {VARIABLES_URL}")

try:
    # Fetch JSON directly into a DataFrame
    variables_df = pd.read_json(VARIABLES_URL)
    
    # Transpose because the JSON is often column-oriented with variable names as keys
    # Or sometimes it's a list of rows. Let's inspect.
    # Actually, the variables.json usually returns a dictionary where keys are variable names.
    # pd.read_json might read it with variable names as columns. Let's transpose.
    variables_df = variables_df.T
    
    # Reset index to make variable name a column
    variables_df.index.name = 'name'
    variables_df = variables_df.reset_index()
    
    print(f"   Found {len(variables_df):,} variables.")
    
    # Search for "Median Household Income"
    search_term = "Median Household Income"
    print(f"\n🔎 Searching for '{search_term}'...")
    
    matches = variables_df[
        variables_df['label'].astype(str).str.contains(search_term, case=False, na=False)
    ]
    
    display(matches[['name', 'label', 'concept']].head())
    
    # Let's also find Total Population
    pop_matches = variables_df[
        variables_df['label'].astype(str).str.contains("Total", case=False) & 
        (variables_df['concept'] == "TOTAL POPULATION")
    ]
    # display(pop_matches.head())

except Exception as e:
    print(f"⚠️ Could not fetch variables programmatically: {e}")

# %% [markdown]
# ---
# 
# ## 📥 Part 2: Fetching Census Data (US + Puerto Rico)
# 
# Based on our discovery, we select:
# -   **Dataset**: `acs/acs5`
# -   **Vintage**: `2020`
# -   **Variables**:
#     -   `B01003_001E`: Total Population
#     -   `B19013_001E`: Median Household Income
# 
# We will fetch data for **all US States** and **Counties**.
# 
# **CRITICAL**: We must ensure we have a `GEOID` column (FIPS code) to join with other datasets later.

# %%
VARIABLES = {
    'B01003_001E': 'Total_Population',
    'B19013_001E': 'Median_Household_Income'
}

# %% [markdown]
# ### 2.1 Fetching State-Level Data

# %%
print("📥 Fetching State-Level Data (US + PR)...")

gdf_state = ced.download(
    dataset=DATASET_NAME,
    vintage=YEAR,
    download_variables=list(VARIABLES.keys()),
    state='*',
    with_geometry=True
)

# Rename columns
gdf_state = gdf_state.rename(columns=VARIABLES)

# Ensure GEOID exists (for State, it's usually the 'STATE' column, but let's be explicit)
if 'GEO_ID' in gdf_state.columns:
    gdf_state['GEOID'] = gdf_state['GEO_ID'].str.split('US').str[-1] # Extract FIPS from full GEO_ID
elif 'STATE' in gdf_state.columns:
    gdf_state['GEOID'] = gdf_state['STATE']

print(f"✅ Fetched {len(gdf_state)} states. GEOID example: {gdf_state['GEOID'].iloc[0]}")
display(gdf_state.head(3))

# %% [markdown]
# ### 2.2 Fetching County-Level Data

# %%
print("📥 Fetching County-Level Data (All US Counties)...")

gdf_county = ced.download(
    dataset=DATASET_NAME,
    vintage=YEAR,
    download_variables=list(VARIABLES.keys()),
    state='*',
    county='*',
    with_geometry=True
)

# Rename columns
gdf_county = gdf_county.rename(columns=VARIABLES)

# Construct GEOID for Counties (State FIPS + County FIPS)
# censusdis usually returns 'STATE' and 'COUNTY' columns
if 'GEO_ID' in gdf_county.columns:
     gdf_county['GEOID'] = gdf_county['GEO_ID'].str.split('US').str[-1]
else:
    # Manual construction if GEO_ID is missing
    gdf_county['GEOID'] = gdf_county['STATE'] + gdf_county['COUNTY']

print(f"✅ Fetched {len(gdf_county):,} counties. GEOID example: {gdf_county['GEOID'].iloc[0]}")
display(gdf_county.head(3))

# %% [markdown]
# ---
# 
# ## 🗺️ Part 3: Visualizing Spatial Data
# 
# We'll visualize the data to verify it looks correct.

# %%
print("🗺️ Visualizing Median Household Income (State Level)...")

# Filter for contiguous US for better plotting
contiguous_us = gdf_state[~gdf_state['STATE'].isin(['02', '15', '72'])].copy()

fig, ax = plt.subplots(figsize=(15, 10))
contiguous_us.plot(
    column='Median_Household_Income',
    cmap='viridis',
    legend=True,
    legend_kwds={'label': "Median Household Income ($)", 'orientation': "horizontal"},
    edgecolor='white',
    linewidth=0.5,
    ax=ax
)
ax.set_title(f"Median Household Income by State ({YEAR} ACS5)", fontsize=16)
ax.axis('off')
plt.show()

# %% [markdown]
# ---
# 
# ## 💾 Part 4: Saving Data to DuckDB
# 
# We save the data to DuckDB, ensuring the `GEOID` column is preserved.
# 
# Tables:
# -   `census_acs5_state_2020`
# -   `census_acs5_county_2020`

# %%
print("💾 Saving Census Data to DuckDB...")

DB_PATH = os.getenv('PROJECT_DB', 'db/pv_project.ddb')
con = duckdb.connect(DB_PATH)
con.execute("INSTALL spatial; LOAD spatial;")

# 1. Save State Data
state_save = gdf_state.copy()
state_save['geometry'] = state_save['geometry'].apply(lambda x: x.wkt)
con.execute(f"CREATE OR REPLACE TABLE census_acs5_state_{YEAR} AS SELECT * FROM state_save")
print(f"   ✅ Saved 'census_acs5_state_{YEAR}'")

# 2. Save County Data
county_save = gdf_county.copy()
county_save['geometry'] = county_save['geometry'].apply(lambda x: x.wkt)
con.execute(f"CREATE OR REPLACE TABLE census_acs5_county_{YEAR} AS SELECT * FROM county_save")
print(f"   ✅ Saved 'census_acs5_county_{YEAR}'")

# Verify
tables = con.execute("SHOW TABLES").fetchall()
print(f"   Tables in DB: {[t[0] for t in tables]}")

# Check GEOID in saved table
check = con.execute(f"SELECT GEOID FROM census_acs5_county_{YEAR} LIMIT 1").fetchone()
print(f"   Verified GEOID in DB: {check[0]}")

con.close()
print("\n🎉 Notebook complete!")


