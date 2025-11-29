# %% [markdown]
# # Notebook 3: Interactive Census Data Exploration for Solar PV Analysis
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
# 1. **Explore Census datasets interactively** using `ipywidgets` to select datasets and variables dynamically.
# 2. **Query at granular levels** (county, census tract) instead of state-level summaries.
# 3. **Focus on PV-relevant geographies** by using identifiers from our enriched PV dataset.
# 4. **Perform correlation analysis** between socioeconomic variables and solar PV adoption.
# 5. **Identify sampling criteria** for finding similar counties/tracts for comparative analysis.
# 
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
# 
# ## 📚 References & Documentation
# 
# - [censusdis Introduction](https://censusdis.readthedocs.io/en/latest/intro.html)
# - [Census API Datasets](https://api.census.gov/data/2020.html)
# - [Exploring Variables](https://censusdis.readthedocs.io/en/latest/nb/Exploring%20Variables.html)
# - [Data With Geometry](https://censusdis.readthedocs.io/en/latest/nb/Data%20With%20Geometry.html)
# - [ipywidgets Widget List](https://ipywidgets.readthedocs.io/en/latest/examples/Widget%20List.html)
# 
# ---

# %% [markdown]
# ## 🔧 Setup: Import Libraries

# %%
import os
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
from shapely import wkt
from scipy import stats

from dotenv import load_dotenv

# censusdis imports
import censusdis.data as ced
import censusdis.maps as cem
import censusdis.geography as cgeo
import censusdis.values as cev
from censusdis import states

# Interactive widgets
import ipywidgets as widgets
from IPython.display import display, clear_output

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', message='Geometry column does not contain geometry')

# Load environment variables
load_dotenv(dotenv_path=Path('../.env'))

# Configure display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 100)
plt.style.use('seaborn-v0_8-darkgrid')

print("✅ Libraries imported successfully.")

# %% [markdown]
# ---
# 
# ## 📥 Part 1: Load PV-Enriched Data from DuckDB
# 
# Instead of analyzing all counties nationwide, we focus on **counties and tracts 
# that actually contain solar PV installations** from our enriched dataset.

# %%
DB_PATH = os.getenv('PROJECT_DB', '../db/pv_project.ddb')
CENSUS_API_KEY = os.getenv('CENSUS_API_KEY', None)

print(f"📂 Connecting to database: {DB_PATH}")
print(f"🔑 Census API Key: {'Configured' if CENSUS_API_KEY else 'Not set (rate limits may apply)'}")

con = duckdb.connect(DB_PATH, read_only=True)
con.execute("INSTALL spatial; LOAD spatial;")

# List available tables
tables = con.execute("SHOW TABLES").fetchall()
print("\n📋 Available tables:")
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
print(f"   Census Tracts: {pv_gdf['TRACT_GEOID'].nunique()}")

con.close()

# %%
# Get unique geographic identifiers from our PV data
pv_counties = pv_gdf[['STATE_FIPS', 'STATE_ABBR', 'COUNTY_GEOID', 'COUNTY_NAME']].drop_duplicates()
pv_tracts = pv_gdf[['STATE_FIPS', 'COUNTY_GEOID', 'TRACT_GEOID']].drop_duplicates()

print(f"\n📊 PV Coverage Summary:")
print(f"   Unique counties with PV: {len(pv_counties)}")
print(f"   Unique tracts with PV: {len(pv_tracts)}")

# Count PV installations per county
pv_per_county = pv_gdf.groupby(['COUNTY_GEOID', 'COUNTY_NAME', 'STATE_ABBR']).size().reset_index(name='pv_count')
print(f"\n🏆 Top 10 Counties by PV Count:")
display(pv_per_county.nlargest(10, 'pv_count'))

# %% [markdown]
# ---
# 
# ## 🔍 Part 2: Interactive Census Dataset & Variable Discovery
# 
# We'll build interactive widgets to explore Census datasets and their variables
# without hardcoding anything.

# %% [markdown]
# ### 2.1 Discover Available Datasets
# 
# The Census API provides a catalog of datasets at: 
# `https://api.census.gov/data/{YEAR}.html`

# %%
# Supported years for our analysis
SUPPORTED_YEARS = [2020, 2021, 2022, 2023]

# Common ACS datasets for socioeconomic analysis
RECOMMENDED_DATASETS = {
    'acs/acs5': 'American Community Survey 5-Year Estimates (most complete)',
    'acs/acs5/profile': 'ACS 5-Year Data Profiles (summary percentages)',
    'acs/acs5/subject': 'ACS 5-Year Subject Tables (topic-focused)',
    'pep/population': 'Population Estimates Program (yearly estimates)',
}

print("📚 Recommended Census Datasets for Socioeconomic Analysis:")
for ds, desc in RECOMMENDED_DATASETS.items():
    print(f"   • {ds}: {desc}")

# %% [markdown]
# ### 2.2 Curated Variable Groups for Solar Analysis
# 
# Rather than navigating thousands of variables, we define **relevant variable groups**
# based on solar adoption research.

# %%
# Curated variable groups relevant to solar PV analysis
SOLAR_RELEVANT_VARIABLES = {
    'Income & Poverty': {
        'B19013_001E': 'Median Household Income',
        'B17001_002E': 'Population Below Poverty Level',
        'B19301_001E': 'Per Capita Income',
    },
    'Housing': {
        'B25077_001E': 'Median Home Value',
        'B25003_002E': 'Owner-Occupied Housing Units',
        'B25003_003E': 'Renter-Occupied Housing Units',
        'B25024_002E': 'Single-Family Detached Homes',
    },
    'Demographics': {
        'B01003_001E': 'Total Population',
        'B01002_001E': 'Median Age',
        'B03002_003E': 'White Alone (Not Hispanic)',
        'B03002_012E': 'Hispanic or Latino',
    },
    'Education': {
        'B15003_022E': "Bachelor's Degree",
        'B15003_023E': "Master's Degree",
        'B15003_025E': 'Doctoral Degree',
    },
    'Employment': {
        'B23025_002E': 'In Labor Force',
        'B23025_005E': 'Unemployed',
    },
}

# Flatten for easy access
ALL_VARIABLES = {}
for category, vars_dict in SOLAR_RELEVANT_VARIABLES.items():
    for var_code, var_name in vars_dict.items():
        ALL_VARIABLES[var_code] = f"[{category}] {var_name}"

print(f"📊 Curated {len(ALL_VARIABLES)} variables across {len(SOLAR_RELEVANT_VARIABLES)} categories")

# %% [markdown]
# ### 2.3 Interactive Variable Explorer with ipywidgets

# %%
# Create interactive widgets for dataset and variable selection
year_dropdown = widgets.Dropdown(
    options=SUPPORTED_YEARS,
    value=2020,
    description='Census Year:',
    style={'description_width': '100px'}
)

dataset_dropdown = widgets.Dropdown(
    options=list(RECOMMENDED_DATASETS.keys()),
    value='acs/acs5',
    description='Dataset:',
    style={'description_width': '100px'}
)

category_dropdown = widgets.Dropdown(
    options=['All'] + list(SOLAR_RELEVANT_VARIABLES.keys()),
    value='All',
    description='Category:',
    style={'description_width': '100px'}
)

# Multi-select for variables
variable_select = widgets.SelectMultiple(
    options=list(ALL_VARIABLES.items()),
    value=[list(ALL_VARIABLES.keys())[0]],  # Default to first variable
    description='Variables:',
    rows=10,
    style={'description_width': '100px'},
    layout=widgets.Layout(width='500px')
)

# Geography level selector
geography_dropdown = widgets.Dropdown(
    options=[
        ('County (recommended for overview)', 'county'),
        ('Census Tract (most granular)', 'tract'),
    ],
    value='county',
    description='Geography:',
    style={'description_width': '100px'}
)

# Output area for dynamic content
output_area = widgets.Output()

def update_variables(*args):
    """Update variable list based on selected category."""
    category = category_dropdown.value
    if category == 'All':
        options = list(ALL_VARIABLES.items())
    else:
        cat_vars = SOLAR_RELEVANT_VARIABLES.get(category, {})
        options = [(code, f"{name}") for code, name in cat_vars.items()]
    variable_select.options = options
    if options:
        variable_select.value = [options[0][0]]

category_dropdown.observe(update_variables, names='value')

# Display the widgets
print("🎛️ Interactive Census Variable Selector")
print("=" * 50)
display(widgets.VBox([
    widgets.HBox([year_dropdown, dataset_dropdown]),
    widgets.HBox([category_dropdown, geography_dropdown]),
    widgets.Label("Select variables (Ctrl+click for multiple):"),
    variable_select,
]))

# %% [markdown]
# ---
# 
# ## 📊 Part 3: Fetch Census Data for PV Counties
# 
# Now we fetch the selected census variables **only for counties/tracts that 
# contain PV installations**.

# %%
def fetch_census_data_for_pv_areas(
    year: int,
    dataset: str,
    variables: list,
    geography: str = 'county',
    state_fips_list: list = None
) -> gpd.GeoDataFrame:
    """
    Fetch census data for areas with PV installations.
    
    Parameters:
    -----------
    year : int
        Census year (2020, 2021, etc.)
    dataset : str
        Census dataset (e.g., 'acs/acs5')
    variables : list
        List of variable codes to fetch
    geography : str
        'county' or 'tract'
    state_fips_list : list
        List of state FIPS codes to query
        
    Returns:
    --------
    gpd.GeoDataFrame with census data and geometries
    """
    if state_fips_list is None:
        state_fips_list = pv_counties['STATE_FIPS'].unique().tolist()
    
    # Always include NAME for labeling
    vars_to_fetch = ['NAME'] + [v for v in variables if v != 'NAME']
    
    print(f"📥 Fetching {len(vars_to_fetch)} variables for {len(state_fips_list)} states at {geography} level...")
    
    all_data = []
    
    for state_fips in state_fips_list:
        try:
            if geography == 'county':
                gdf = ced.download(
                    dataset=dataset,
                    vintage=year,
                    download_variables=vars_to_fetch,
                    state=state_fips,
                    county='*',
                    with_geometry=True,
                    set_to_nan=cev.ALL_SPECIAL_VALUES,
                    api_key=CENSUS_API_KEY
                )
                # Construct GEOID for joining
                gdf['GEOID'] = gdf['STATE'] + gdf['COUNTY']
            else:  # tract
                gdf = ced.download(
                    dataset=dataset,
                    vintage=year,
                    download_variables=vars_to_fetch,
                    state=state_fips,
                    tract='*',
                    with_geometry=True,
                    set_to_nan=cev.ALL_SPECIAL_VALUES,
                    api_key=CENSUS_API_KEY
                )
                # Construct GEOID for joining
                gdf['GEOID'] = gdf['STATE'] + gdf['COUNTY'] + gdf['TRACT']
            
            all_data.append(gdf)
        except Exception as e:
            print(f"   ⚠️ Error fetching {state_fips}: {e}")
    
    if not all_data:
        raise ValueError("No data fetched!")
    
    result = pd.concat(all_data, ignore_index=True)
    result = gpd.GeoDataFrame(result, geometry='geometry', crs='EPSG:4326')
    
    print(f"✅ Fetched {len(result):,} {geography} records")
    
    return result

# %%
# Fetch census data using current widget selections
selected_vars = list(variable_select.value)
selected_year = year_dropdown.value
selected_dataset = dataset_dropdown.value
selected_geography = geography_dropdown.value

print(f"🔍 Fetching data with current selections:")
print(f"   Year: {selected_year}")
print(f"   Dataset: {selected_dataset}")
print(f"   Geography: {selected_geography}")
print(f"   Variables: {selected_vars}")

# Fetch the data
census_gdf = fetch_census_data_for_pv_areas(
    year=selected_year,
    dataset=selected_dataset,
    variables=selected_vars,
    geography=selected_geography
)

display(census_gdf.head())

# %% [markdown]
# ---
# 
# ## 🔗 Part 4: Join Census Data with PV Counts
# 
# We merge census socioeconomic data with our PV installation counts to 
# analyze correlations.

# %%
# Count PV installations per geographic unit
if selected_geography == 'county':
    pv_counts = pv_gdf.groupby('COUNTY_GEOID').size().reset_index(name='pv_count')
    pv_counts = pv_counts.rename(columns={'COUNTY_GEOID': 'GEOID'})
else:  # tract
    pv_counts = pv_gdf.groupby('TRACT_GEOID').size().reset_index(name='pv_count')
    pv_counts = pv_counts.rename(columns={'TRACT_GEOID': 'GEOID'})

print(f"📊 PV counts aggregated to {selected_geography} level: {len(pv_counts):,} units with PV")

# %%
# Merge census data with PV counts
census_pv = census_gdf.merge(pv_counts, on='GEOID', how='left')
census_pv['pv_count'] = census_pv['pv_count'].fillna(0).astype(int)
census_pv['has_pv'] = (census_pv['pv_count'] > 0).astype(int)

# Calculate area and density
census_pv_projected = census_pv.to_crs(epsg=5070)  # Albers Equal Area for US
census_pv['area_km2'] = census_pv_projected.geometry.area / 1_000_000
census_pv['pv_density'] = census_pv['pv_count'] / census_pv['area_km2']

print(f"\n📊 Census-PV Merged Dataset Summary:")
print(f"   Total {selected_geography}s: {len(census_pv):,}")
print(f"   {selected_geography.title()}s with PV: {census_pv['has_pv'].sum():,} ({100*census_pv['has_pv'].mean():.1f}%)")
print(f"   Total PV installations: {census_pv['pv_count'].sum():,}")

display(census_pv.head())

# %% [markdown]
# ---
# 
# ## 📈 Part 5: Exploratory Data Analysis (EDA)
# 
# Now we perform EDA similar to the California analysis but for ALL our PV-relevant areas.

# %% [markdown]
# ### 5.1 Distribution Analysis

# %%
# Get numeric columns for analysis (exclude IDs and geometry)
numeric_cols = census_pv.select_dtypes(include=[np.number]).columns.tolist()
analysis_cols = [c for c in numeric_cols if c not in ['pv_count', 'has_pv', 'pv_density', 'area_km2'] 
                 and not c.endswith('_GEOID') and c not in ['STATE', 'COUNTY', 'TRACT']]

print(f"📊 Numeric columns available for analysis: {analysis_cols}")

# %%
# Plot distributions of key variables
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

plot_vars = analysis_cols[:6] if len(analysis_cols) >= 6 else analysis_cols

for i, var in enumerate(plot_vars):
    if i < len(axes):
        ax = axes[i]
        data = census_pv[var].dropna()
        if len(data) > 0:
            sns.histplot(data, kde=True, ax=ax, color='steelblue')
            ax.set_title(ALL_VARIABLES.get(var, var), fontsize=10)
            ax.set_xlabel('')
        else:
            ax.set_title(f'{var} (no data)')

# Hide unused subplots
for j in range(len(plot_vars), len(axes)):
    axes[j].set_visible(False)

plt.suptitle(f'Distribution of Census Variables ({selected_year} {selected_geography.title()} Level)', fontsize=14)
plt.tight_layout()
plt.show()

# %% [markdown]
# ### 5.2 Correlation Analysis: Socioeconomic Factors vs PV Adoption

# %%
# Calculate correlations with PV density
corr_cols = analysis_cols + ['pv_count', 'pv_density', 'has_pv']
corr_data = census_pv[corr_cols].dropna()

if len(corr_data) > 10:
    corr_matrix = corr_data.corr()
    
    # Extract correlations with PV metrics
    pv_correlations = corr_matrix[['pv_count', 'pv_density', 'has_pv']].drop(['pv_count', 'pv_density', 'has_pv'])
    
    # Plot heatmap
    fig, ax = plt.subplots(figsize=(10, max(6, len(pv_correlations) * 0.4)))
    
    # Create labels with variable names
    y_labels = [ALL_VARIABLES.get(v, v) for v in pv_correlations.index]
    
    sns.heatmap(
        pv_correlations, 
        annot=True, 
        cmap='RdBu_r', 
        center=0,
        fmt='.2f',
        yticklabels=y_labels,
        ax=ax
    )
    ax.set_title(f'Correlation: Census Variables vs PV Adoption ({selected_year})', fontsize=14)
    plt.tight_layout()
    plt.show()
    
    # Print top correlations
    print("\n🔍 Top Correlations with PV Density:")
    top_corr = pv_correlations['pv_density'].abs().sort_values(ascending=False)
    for var, corr_val in top_corr.head(5).items():
        actual_corr = pv_correlations.loc[var, 'pv_density']
        direction = "+" if actual_corr > 0 else "-"
        print(f"   {direction} {ALL_VARIABLES.get(var, var)}: {actual_corr:.3f}")
else:
    print("⚠️ Not enough data for correlation analysis")

# %% [markdown]
# ### 5.3 Scatter Plots: Key Relationships

# %%
# Create scatter plots for key relationships
key_vars = [v for v in analysis_cols if v in ['B19013_001E', 'B17001_002E', 'B25077_001E', 'B15003_022E']][:4]

if len(key_vars) > 0:
    fig, axes = plt.subplots(1, len(key_vars), figsize=(5*len(key_vars), 5))
    if len(key_vars) == 1:
        axes = [axes]
    
    for ax, var in zip(axes, key_vars):
        data = census_pv[[var, 'pv_density']].dropna()
        if len(data) > 5:
            sns.regplot(
                data=data,
                x=var,
                y='pv_density',
                ax=ax,
                scatter_kws={'alpha': 0.3},
                line_kws={'color': 'red'}
            )
            
            # Calculate correlation
            corr = data[var].corr(data['pv_density'])
            ax.set_title(f'{ALL_VARIABLES.get(var, var)}\nr = {corr:.3f}', fontsize=10)
            ax.set_xlabel('')
            ax.set_ylabel('PV Density (per km²)')
    
    plt.suptitle('Socioeconomic Factors vs Solar PV Density', fontsize=14)
    plt.tight_layout()
    plt.show()

# %% [markdown]
# ### 5.4 Comparison: Areas With vs Without Solar PV

# %%
# Statistical comparison: areas with PV vs without PV
print("📊 Statistical Comparison: Areas With vs Without Solar PV")
print("=" * 60)

comparison_results = []

for var in analysis_cols:
    with_pv = census_pv[census_pv['has_pv'] == 1][var].dropna()
    without_pv = census_pv[census_pv['has_pv'] == 0][var].dropna()
    
    if len(with_pv) > 5 and len(without_pv) > 5:
        # Mann-Whitney U test (non-parametric)
        stat, p_value = stats.mannwhitneyu(with_pv, without_pv, alternative='two-sided')
        
        comparison_results.append({
            'Variable': ALL_VARIABLES.get(var, var),
            'Code': var,
            'With_PV_Median': with_pv.median(),
            'Without_PV_Median': without_pv.median(),
            'Difference_%': 100 * (with_pv.median() - without_pv.median()) / without_pv.median() if without_pv.median() != 0 else np.nan,
            'P_Value': p_value,
            'Significant': 'Yes' if p_value < 0.05 else 'No'
        })

comparison_df = pd.DataFrame(comparison_results)
if len(comparison_df) > 0:
    display(comparison_df.sort_values('P_Value'))

# %%
# Box plots for significant differences
if len(comparison_df) > 0:
    sig_vars = comparison_df[comparison_df['Significant'] == 'Yes']['Code'].tolist()[:4]
    
    if sig_vars:
        fig, axes = plt.subplots(1, len(sig_vars), figsize=(4*len(sig_vars), 5))
        if len(sig_vars) == 1:
            axes = [axes]
        
        for ax, var in zip(axes, sig_vars):
            sns.boxplot(
                data=census_pv,
                x='has_pv',
                y=var,
                ax=ax,
                palette=['lightcoral', 'lightgreen']
            )
            ax.set_xticklabels(['No Solar PV', 'Has Solar PV'])
            ax.set_title(ALL_VARIABLES.get(var, var), fontsize=10)
            ax.set_xlabel('')
        
        plt.suptitle('Significant Differences: Areas With vs Without Solar PV', fontsize=14)
        plt.tight_layout()
        plt.show()

# %% [markdown]
# ---
# 
# ## 🗺️ Part 6: Spatial Visualization
# 
# Map the census variables and PV distribution.

# %%
# Choropleth map of PV density
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# Get contiguous US only for cleaner visualization
contiguous_states = [s for s in states.ALL_STATES_AND_DC if s not in ['02', '15', '72', '78', '66', '69', '60']]
census_pv_contiguous = census_pv[census_pv['STATE'].isin(contiguous_states)].copy()

if len(census_pv_contiguous) > 0:
    # Map 1: PV Density
    ax1 = axes[0]
    census_pv_contiguous.plot(
        column='pv_density',
        cmap='YlOrRd',
        legend=True,
        legend_kwds={'label': 'PV Density (per km²)', 'shrink': 0.6},
        ax=ax1,
        edgecolor='white',
        linewidth=0.1
    )
    ax1.set_title(f'Solar PV Installation Density by {selected_geography.title()}', fontsize=14)
    ax1.axis('off')
    
    # Map 2: Key census variable (if available)
    ax2 = axes[1]
    if 'B19013_001E' in census_pv_contiguous.columns:
        map_var = 'B19013_001E'
        map_label = 'Median Household Income ($)'
    elif len(analysis_cols) > 0:
        map_var = analysis_cols[0]
        map_label = ALL_VARIABLES.get(map_var, map_var)
    else:
        map_var = None
    
    if map_var:
        census_pv_contiguous.plot(
            column=map_var,
            cmap='viridis',
            legend=True,
            legend_kwds={'label': map_label, 'shrink': 0.6},
            ax=ax2,
            edgecolor='white',
            linewidth=0.1
        )
        ax2.set_title(map_label, fontsize=14)
        ax2.axis('off')

plt.tight_layout()
plt.show()

# %% [markdown]
# ---
# 
# ## 🎯 Part 7: Identifying Similar Areas for Sampling
# 
# To support comparative analysis, we identify counties/tracts with **similar 
# socioeconomic profiles** that could serve as comparison groups.

# %%
def calculate_similarity_score(df, target_geoid, feature_cols):
    """
    Calculate similarity scores between a target area and all other areas.
    Uses normalized Euclidean distance on selected features.
    """
    # Get target row
    target = df[df['GEOID'] == target_geoid]
    if len(target) == 0:
        return None
    
    target_values = target[feature_cols].values[0]
    
    # Normalize features
    normalized_df = df[feature_cols].copy()
    for col in feature_cols:
        col_min = normalized_df[col].min()
        col_max = normalized_df[col].max()
        if col_max > col_min:
            normalized_df[col] = (normalized_df[col] - col_min) / (col_max - col_min)
    
    target_normalized = normalized_df.loc[target.index[0]].values
    
    # Calculate Euclidean distance
    distances = np.sqrt(((normalized_df.values - target_normalized) ** 2).sum(axis=1))
    
    result = df[['GEOID', 'NAME', 'pv_count', 'pv_density']].copy()
    result['similarity_distance'] = distances
    result['similarity_rank'] = result['similarity_distance'].rank()
    
    return result.sort_values('similarity_distance')

# %%
# Example: Find areas similar to the county with highest PV density
if len(analysis_cols) >= 2:
    # Get top PV county
    top_pv_area = census_pv.nlargest(1, 'pv_count').iloc[0]
    print(f"🎯 Finding areas similar to: {top_pv_area['NAME']}")
    print(f"   PV Count: {top_pv_area['pv_count']}")
    print(f"   GEOID: {top_pv_area['GEOID']}")
    
    # Calculate similarity using available features
    similarity_features = [c for c in analysis_cols if census_pv[c].notna().sum() > len(census_pv) * 0.5][:5]
    
    if len(similarity_features) >= 2:
        similar_areas = calculate_similarity_score(
            census_pv.dropna(subset=similarity_features),
            top_pv_area['GEOID'],
            similarity_features
        )
        
        if similar_areas is not None:
            print(f"\n📊 Top 10 Most Similar Areas (based on {len(similarity_features)} features):")
            display(similar_areas.head(10))
            
            print(f"\n📊 Similar Areas WITHOUT PV (potential comparison group):")
            similar_no_pv = similar_areas[similar_areas['pv_count'] == 0]
            display(similar_no_pv.head(10))

# %% [markdown]
# ---
# 
# ## 💾 Part 8: Save Enriched Data to DuckDB
# 
# Save the census-enriched PV analysis dataset for use in other notebooks.

# %%
# Prepare data for saving
save_df = census_pv.copy()
save_df['geometry'] = save_df['geometry'].apply(lambda g: g.wkt if g else None)

# Connect and save
con = duckdb.connect(DB_PATH)
con.execute("INSTALL spatial; LOAD spatial;")

table_name = f'census_pv_{selected_geography}_{selected_year}'
con.execute(f"DROP TABLE IF EXISTS {table_name}")
con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM save_df")

row_count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
print(f"✅ Saved '{table_name}' with {row_count:,} rows")

# Show all tables
tables = con.execute("SHOW TABLES").fetchall()
print(f"\n📋 Current tables in database:")
for t in tables:
    count = con.execute(f"SELECT COUNT(*) FROM \"{t[0]}\"").fetchone()[0]
    print(f"   - {t[0]}: {count:,} rows")

con.close()

# %% [markdown]
# ---
# 
# ## 📝 Summary & Next Steps
# 
# ### Key Findings
# 
# This interactive analysis allows us to:
# 
# 1. **Focus on PV-relevant areas**: Analyze only counties/tracts with solar installations
# 2. **Dynamic variable selection**: Explore any census variable interactively
# 3. **Statistical testing**: Identify significant differences between areas with/without PV
# 4. **Similarity matching**: Find comparable areas for quasi-experimental analysis
# 
# ### Research Directions
# 
# Based on this framework, potential next steps include:
# 
# - **Propensity Score Matching**: Match PV and non-PV areas on socioeconomic factors
# - **Time Series Analysis**: Compare 2010 vs 2020 census to track demographic changes
# - **Spatial Autocorrelation**: Analyze clustering patterns in PV adoption
# - **Machine Learning**: Predict PV adoption probability based on census features
# 
# ### Variables to Explore Further
# 
# Based on initial correlations, consider exploring:
# - Housing stock age (older homes may have different adoption patterns)
# - Education levels (environmental awareness)
# - Income quintiles (affordability analysis)
# - Racial/ethnic composition (equity analysis)

# %%
print("🎉 Notebook complete!")
print(f"\n📊 Analysis Summary:")
print(f"   Census Year: {selected_year}")
print(f"   Dataset: {selected_dataset}")
print(f"   Geography Level: {selected_geography}")
print(f"   Variables Analyzed: {len(selected_vars)}")
print(f"   Areas Analyzed: {len(census_pv):,}")
print(f"   Areas with PV: {census_pv['has_pv'].sum():,}")


