# %%
!pip install pandas pyarrow
!pip install censusdis
!pip install pandas geopandas shapely censusdis pyarrow

# %%
# -----------------------------------------
# Analisis de Datos
# Francheska Lebron
# Luis Fontan
# -----------------------------------------

# %%
import pandas as pd
import censusdis.data as ced

# -----------------------------------------
#Lectura data (archivo .parquet)
# -----------------------------------------

file_path = 'ccom6994_pv_dataset.parquet'
df = pd.read_parquet(file_path)
df.head()

# %%
import geopandas as gpd

# -----------------------------------------
# Convertir datos de placas solares (archivo .parquet) a GeoDataFrame (data mundial)
# -----------------------------------------

gdf_solar = gpd.GeoDataFrame(df, geometry=gpd.GeoSeries.from_wkt(df["geometry"]) if df["geometry"].dtype == "object" else df["geometry"])

# -----------------------------------------
# Asegurar CRS (suponiendo WGS84)
# -----------------------------------------
gdf_solar = gdf_solar.set_crs(epsg=4326)

# %%
import censusdis.data as ced
import pandas as pd

# -----------------------------------------
# Definir dataset ACS y variables
# -----------------------------------------
DATASET = "acs/acs5"
PROFILE = "acs/acs5/profile"
YEAR = 2022

# -----------------------------------------
# Variables ACS detalladas
# -----------------------------------------

vars_detailed = [
    "NAME",
    "B01003_001E",   # poblacion total
    "B19013_001E",   # ingreso mediano
    "B17001_002E",   # bajo pobreza
    "B25077_001E",   # valor mediano vivienda
]
# -----------------------------------------
# Variables ACS Profile
# -----------------------------------------

vars_profile = [
    "DP03_0009PE",   # % desempleo
    "DP02_0068PE",   # % bachillerato o más
    "DP05_0071PE",   # % hispano
]



# %%
# -----------------------------------------
# Descargar data (solo California)
# -----------------------------------------
gdf_prof = ced.download(
    PROFILE,
    YEAR,
    vars_profile,
    state="06",
    county="*",
    tract="*",
    with_geometry=False
)

gdf_det = ced.download(
    DATASET,
    YEAR,
    vars_detailed,
    state="06",      # California
    county="*",
    tract="*",
    with_geometry=True
)

# -----------------------------------------
# Renombrar geometría
# -----------------------------------------

gdf_det = gdf_det.rename(columns={"geometry": "geometry_census"})
gdf_det = gdf_det.set_geometry("geometry_census")




# %%

# -----------------------------------------
# Unir ambas tablas
# -----------------------------------------
gdf = gdf_det.merge(
    gdf_prof,
    on=["STATE", "COUNTY", "TRACT"],
    how="left"
)
9
# -----------------------------------------
# Renombrar columnas
# -----------------------------------------
gdf = gdf.rename(columns={
    "B01003_001E": "poblacion_total",
    "B19013_001E": "ingreso_medio_hogar",
    "B17001_002E": "poblacion_bajo_pobreza",
    "B25077_001E": "valor_medio_vivienda",
    "DP03_0009PE": "porc_desempleo",
    "DP02_0068PE": "porc_educ_bachillerato",
    "DP05_0071PE": "porc_hispano"
})

# -----------------------------------------
# Asegurar CRS
# -----------------------------------------
gdf = gdf.set_crs(epsg=4269).to_crs(epsg=4326)



# %%
from shapely.geometry import Point

# -----------------------------------------

#hacer Join Geoespacial  de Datos de Placas Solares y del Censo
# -----------------------------------------

gdf_points = gdf_solar.copy()
gdf_points["geometry"] = [
    Point(lon, lat)
    for lon, lat in zip(gdf_points["centroid_lon"], gdf_points["centroid_lat"])
]
gdf_points = gdf_points.set_crs(epsg=4326)

gdf_california = gpd.sjoin(
    gdf_points,
    gdf,
    how="left",
    predicate="within"
)


gdf_california

# %%
gdf_california_full = gdf_california[gdf_california["STATE"] == "06"].copy()


# %%
len(gdf_california_full.geometry)

# %%
# -----------------------------------------
#Contar cantidad de placas solares por County
# -----------------------------------------

conteo = (
    gdf_california_full
    .groupby(["STATE", "COUNTY","TRACT"], as_index=False)
    .size()
    .rename(columns={"size": "n_plantas"})
)

# -----------------------------------------
# Unir conteo al geodataframe
# -----------------------------------------

gdf_california_full = gdf.merge(
    conteo,
    on=["STATE", "COUNTY", "TRACT"],
    how="left"
).fillna({"n_plantas": 0})


# %%
# -----------------------------------------
# Proyectar counties a CRS adecuado para área en California
# -----------------------------------------

gdf_california_full = gdf_california_full.to_crs(epsg=3310)

# -----------------------------------------
# Calcular área del tract en km²
# -----------------------------------------

gdf_california_full["area_km2"] = gdf_california_full["geometry_census"].area / 1_000_000

# -----------------------------------------
# Calcular densidad de plantas solares
# -----------------------------------------

gdf_california_full["densidad_solar"] = (
    gdf_california_full["n_plantas"] / gdf_california_full["area_km2"]
)

# -----------------------------------------
# Variable binaria: presencia o ausencia de plantas solares
# -----------------------------------------

gdf_california_full["tiene_solar"] = (gdf_california_full["n_plantas"] > 0).astype(int)


# %%
gdf_california_full.n_plantas


# %%
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
# -----------------------------------------
# Creación de Scatterplot y correlación
# -----------------------------------------

vars_to_plot = [
    ("ingreso_medio_hogar", "Ingreso mediano del hogar"),
    ("poblacion_bajo_pobreza", "Población bajo pobreza"),
    ("porc_hispano", "% población hispana"),
    ("porc_desempleo", "% desempleo"),
    ("porc_educ_bachillerato", "% bachillerato o más"),
    ("valor_medio_vivienda", "Valor mediano de vivienda"),
]



for var, label in vars_to_plot:
    plt.figure(figsize=(8, 6))

    sns.regplot(
        data=gdf_california_full,
        x=var,
        y="densidad_solar",
        scatter_kws={"alpha": 0.6},
        line_kws={"color": "red"}
    )

    # -----------------------------------------
    # Correlación Pearson
    corr = gdf_california_full[[var, "densidad_solar"]].corr().iloc[0, 1]

    plt.title(f"Densidad solar vs {label}\nCorrelación: {corr:.3f}", fontsize=14)
    plt.xlabel(label, fontsize=12)
    plt.ylabel("Densidad de plantas solares (por km²)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.show()


# %%
# -----------------------------------------
#Mapa de Tract de California y distribución de cantidad de infraetsructura solar
# -----------------------------------------

gdf_california_full.plot(
    column="n_plantas",
    figsize=(12,8),
    legend=True,
    cmap="viridis"
)
plt.title("Placas solares por tract")
plt.show()

# %%
# -----------------------------------------
#Identificando que tract tiene mayor cantidad de placas solares
# -----------------------------------------
fila_max = gdf_california_full.loc[gdf_california_full["n_plantas"].idxmax()]



# %%
# -----------------------------------------
#matriz de correlación entre todas las variables
# -----------------------------------------

plt.figure(figsize=(8,6))
sns.heatmap(
    gdf_california_full[["ingreso_medio_hogar","poblacion_bajo_pobreza",
               "valor_medio_vivienda","porc_desempleo","porc_educ_bachillerato",
                        "porc_hispano", "densidad_solar","n_plantas","tiene_solar" ]].corr(),
    annot=True,
    cmap="coolwarm"
)
plt.title("Matriz de correlaciones")
plt.show()


# %%
gdf_california_full.info()


# %%
# -----------------------------------------
#Cantidad de datos missing
# -----------------------------------------

gdf_california_full.isna().sum()


# %%
# -----------------------------------------
#Observando si las variables se comportan normalmente
# -----------------------------------------

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.histplot(gdf_california_full["ingreso_medio_hogar"], kde=True)
plt.title("Distribución del ingreso medio del hogar")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(gdf_california_full["poblacion_bajo_pobreza"], kde=True)
plt.title("Distribución poblacion_bajo_pobreza")
plt.show()

plt.figure(figsize=(8,5))
sns.histplot(gdf_california_full["porc_educ_bachillerato"], kde=True)
plt.title("Distribución % educ_bachillerato")
plt.show()


# %%
# -----------------------------------------
#Prueba para comprobar que no se comportan normalemnte las variables
# -----------------------------------------

import scipy.stats as stats
import matplotlib.pyplot as plt

stats.probplot(gdf_california_full["ingreso_medio_hogar"].dropna(), dist="norm", plot=plt)
plt.title("Q-Q Plot - Ingreso Medio del Hogar")
plt.show()

stats.probplot(gdf_california_full["poblacion_bajo_pobreza"].dropna(), dist="norm", plot=plt)
plt.title("Q-Q Plot - poblacion_bajo_pobreza")
plt.show()

stats.probplot(gdf_california_full["porc_educ_bachillerato"].dropna(), dist="norm", plot=plt)
plt.title("Q-Q Plot - porc_educ_bachillerato")
plt.show()


# %%
from scipy.stats import mannwhitneyu
# -----------------------------------------
#¿Los tracts que tienen plantas solares son socioeconómicamente distintos a los que no tienen?
# -----------------------------------------



# Lista de variables que quieres evaluar
variables = [
    "ingreso_medio_hogar",
    "poblacion_bajo_pobreza",
    "valor_medio_vivienda",
    "porc_educ_bachillerato",
    "porc_hispano",
    "porc_desempleo"
]

# Eliminar NAs solo una vez en todas las variables
df = gdf_california_full.dropna(subset=variables)

resultados = []

for var in variables:

    group_yes = df[df["tiene_solar"] == 1][var]
    group_no  = df[df["tiene_solar"] == 0][var]

    stat, p = mannwhitneyu(group_yes, group_no)

    resultados.append({
        "variable": var,
        "U_statistic": stat,
        "p_value": p
    })

# Convertir a DataFrame
tabla_resultados = pd.DataFrame(resultados)
tabla_resultados



# %%
vars_interes = [
    "ingreso_medio_hogar",
    "poblacion_bajo_pobreza",
    "valor_medio_vivienda",
    "porc_educ_bachillerato",
    "porc_hispano",
    "porc_desempleo"
]

medianas = (
    gdf_california_full
    .groupby("tiene_solar")[vars_interes]
    .mean()
    .rename(index={0: "sin_solar", 1: "con_solar"})
)

medianas


# %%
import statsmodels.formula.api as smf
#---------------------------------
# Modelo Logistico
#------------------------------

model = smf.logit(
    formula="tiene_solar ~ ingreso_medio_hogar + poblacion_bajo_pobreza + valor_medio_vivienda + porc_educ_bachillerato + porc_hispano + porc_desempleo",
    data=gdf_california_full
).fit()

print(model.summary())



