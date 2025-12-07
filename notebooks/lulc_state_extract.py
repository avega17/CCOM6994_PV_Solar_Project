#!/usr/bin/env python
"""
One-state LULC pre-processing pipeline.

- Accepts a US state (FIPS or USPS abbr) and fetches Overture land cover + land use
  clipped to that state bbox, then matches to PV installations in that state.
- Persists aggregated LULC attributes per PV (arrays of IDs/subtypes/classes) into DuckDB
  for reuse by the E2E analysis script/notebook.

Usage:
    python notebooks/lulc_state_extract.py --state CA
    python notebooks/lulc_state_extract.py --state 06

Env (.env):
    PROJECT_DB: path to duckdb file (default ../db/pv_project.duckdb)
    OVERTURE_RELEASE: e.g., 2025-11-19.0
    DUCKDB_THREADS, DUCKDB_MEMORY_LIMIT, DUCKDB_PARQUET_CACHE (optional)
"""
import argparse
import os
import sys
from pathlib import Path

import duckdb
import geopandas as gpd
import overturemaestro as om
import pandas as pd
from dotenv import load_dotenv
from shapely.geometry import box
import censusdis.maps as cem

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def norm_state_arg(state: str) -> tuple[str, str]:
    """Normalize a state argument to (fips, abbr-like) strings."""
    s = state.strip()
    if len(s) == 2 and s.isdigit():
        return s.zfill(2), None
    if len(s) == 2:
        return None, s.upper()
    raise ValueError("State must be FIPS (2 digits) or USPS abbreviation (2 letters)")


def ensure_state_geoms(con: duckdb.DuckDBPyConnection, state_fips: str) -> gpd.GeoDataFrame:
    """Cache state + county geometries in DuckDB if missing; return state geom."""
    con.execute("INSTALL spatial; LOAD spatial;")
    # Create tables if absent
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS state_boundaries AS
        SELECT * FROM (SELECT '' AS state_fips, '' AS state_abbr, ST_GeomFromText('POINT (0 0)') AS geometry) WHERE 1=0;
        """
    )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS county_boundaries AS
        SELECT * FROM (SELECT '' AS state_fips, '' AS county_fips, ST_GeomFromText('POINT (0 0)') AS geometry) WHERE 1=0;
        """
    )

    # Check if state exists
    exists = con.execute(
        "SELECT COUNT(*) FROM state_boundaries WHERE state_fips = ?",
        [state_fips],
    ).fetchone()[0]

    if exists:
        state_row = con.execute(
            "SELECT state_fips, state_abbr, ST_AsText(geometry) AS geometry FROM state_boundaries WHERE state_fips = ?",
            [state_fips],
        ).fetchdf()
        from shapely import wkt as wkt_parser
        state_row["geometry"] = state_row["geometry"].apply(wkt_parser.loads)
        return gpd.GeoDataFrame(state_row, geometry="geometry", crs="EPSG:4326")

    # Fetch via censusdis
    print(f"  Fetching Census geometries for state {state_fips}...")
    reader = cem.ShapeReader(year=2020)
    try:
        # Fetch all states (scope="us") then filter
        all_states_gdf = reader.read_cb_shapefile(shapefile_scope="us", geography="state", crs="EPSG:4326")
        state_gdf = all_states_gdf[all_states_gdf["STATEFP"] == state_fips].copy()
        
        if state_gdf.empty:
            raise ValueError(f"State FIPS {state_fips} not found in Census data")

        # Fetch counties for the specific state
        # Fetch all counties (scope="us") then filter to avoid missing state-specific files
        all_counties_gdf = reader.read_cb_shapefile(shapefile_scope="us", geography="county", crs="EPSG:4326")
        county_gdf = all_counties_gdf[all_counties_gdf["STATEFP"] == state_fips].copy()
    except Exception as e:
        print(f"  ⚠️  Failed to fetch Census geometries: {e}")
        print(f"  This may be a temporary network issue. Please retry later.")
        raise

    state_gdf = gpd.GeoDataFrame(state_gdf, geometry="geometry", crs="EPSG:4326")
    county_gdf = gpd.GeoDataFrame(county_gdf, geometry="geometry", crs="EPSG:4326")

    # Persist - convert geometry to WKT strings for DuckDB registration
    state_save_df = state_gdf.copy()
    state_save_df['state_fips'] = state_fips
    state_save_df['state_abbr'] = state_gdf["STUSPS"] if "STUSPS" in state_gdf.columns else None
    state_save_df['geom_wkt'] = state_save_df['geometry'].apply(lambda g: g.wkt)
    con.register("_state_df", state_save_df[["state_fips", "state_abbr", "geom_wkt"]])
    con.execute(
        """
        INSERT INTO state_boundaries
        SELECT state_fips, COALESCE(state_abbr, '') AS state_abbr, ST_GeomFromText(geom_wkt) AS geometry
        FROM _state_df
        """
    )

    county_save_df = county_gdf.copy()
    county_save_df['state_fips'] = county_gdf["STATEFP"]
    county_save_df['county_fips'] = county_gdf["COUNTYFP"]
    county_save_df['geom_wkt'] = county_save_df['geometry'].apply(lambda g: g.wkt)
    con.register("_county_df", county_save_df[["state_fips", "county_fips", "geom_wkt"]])
    con.execute(
        """
        INSERT INTO county_boundaries
        SELECT state_fips, county_fips, ST_GeomFromText(geom_wkt) AS geometry
        FROM _county_df
        """
    )

    return gpd.GeoDataFrame(state_gdf.assign(state_fips=state_fips), geometry="geometry", crs="EPSG:4326")


def fetch_overture_duckdb(con: duckdb.DuckDBPyConnection, layer: str, bbox: tuple, release: str) -> None:
    """Fetch overture data directly into DuckDB temp tables using S3 queries.
    
    Args:
        con: DuckDB connection
        layer: 'land_cover' or 'land_use'
        bbox: (minx, miny, maxx, maxy) bounding box
        release: Overture release version
    """
    # Construct S3 path
    s3_path = f"s3://overturemaps-us-west-2/release/{release}/theme=base/type={layer}/*"
    
    # Build column list based on layer type
    # land_cover: id, subtype, geometry
    # land_use: id, subtype, geometry (we don't need class)
    if layer == "land_cover":
        cols = "id, subtype, geometry"
        table_name = "lc_geom"
    else:
        cols = "id, subtype, geometry"
        table_name = "lu_geom"
    
    minx, miny, maxx, maxy = bbox
    
    # Query directly from S3 with bbox filter
    print(f"  Fetching {layer} from S3 (bbox filter)...")
    query = f"""
        CREATE OR REPLACE TEMP TABLE {table_name} AS
        SELECT {cols}
        FROM read_parquet('{s3_path}')
        WHERE bbox.xmin <= {maxx}
          AND bbox.xmax >= {minx}
          AND bbox.ymin <= {maxy}
          AND bbox.ymax >= {miny}
    """
    con.execute(query)
    
    # Get count
    count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
    print(f"    Loaded {count:,} {layer} features into temp table")


def array_merge_sql(col_name: str) -> str:
    return f"array_agg(DISTINCT {col_name}) FILTER (WHERE {col_name} IS NOT NULL)"


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="One-state LULC pre-processing (Overture → DuckDB)")
    parser.add_argument("--state", required=True, help="State FIPS (06) or USPS abbr (CA)")
    parser.add_argument("--db", default=None, help="Path to DuckDB file (default from PROJECT_DB)")
    parser.add_argument("--release", default=None, help="Overture release (default from env or latest)")
    args = parser.parse_args()

    # Get repo root (parent of notebooks/) - this is the cloned repo root
    repo_root = Path(__file__).resolve().parent.parent
    load_dotenv(dotenv_path=repo_root / ".env")

    # Resolve DB path: prefer CLI arg, then DEMO_DB_PATH (relative safe), then PROJECT_DB, else fallback
    def _resolve_db(path_str: str | None) -> str:
        if not path_str:
            return ""
        p = Path(path_str)
        if p.is_absolute():
            return str(p)
        # Relative path: resolve against repo root
        resolved = (repo_root / p).resolve()
        return str(resolved)

    # Try multiple locations in priority order:
    # 1. CLI argument (--db)
    # 2. Environment variable DEMO_DB_PATH or PROJECT_DB (resolved relative to repo root)
    # 3. repo_root/db/pv_project.duckdb (PREFERRED - inside cloned repo)
    # 4. current_working_directory/db/pv_project.duckdb
    # 5. Fallback: create in repo_root/db/ (inside cloned repo)
    
    env_db = os.getenv("DEMO_DB_PATH") or os.getenv("PROJECT_DB")
    
    # Primary location: inside the cloned repo (pv_solar_analysis/db/)
    primary_db = repo_root / "db" / "pv_project.duckdb"
    
    candidate_paths = [
        _resolve_db(args.db) if args.db else None,
        _resolve_db(env_db) if env_db else None,
        primary_db,  # Inside cloned repo - PREFERRED
        Path.cwd() / "db" / "pv_project.duckdb",
    ]
    
    # Find first existing database
    db_path = None
    for candidate in candidate_paths:
        if candidate and Path(candidate).exists():
            db_path = str(Path(candidate).resolve())
            break
    
    # If no existing DB found, use primary location and create directory
    if not db_path:
        primary_db.parent.mkdir(parents=True, exist_ok=True)
        db_path = str(primary_db)
        print(f"⚠️  No existing database found, will create at: {db_path}")
    
    print(f"Using database: {db_path}")
    
    overture_release = args.release or os.getenv("OVERTURE_RELEASE", om.get_newest_release_version())
    threads = int(os.getenv("DUCKDB_THREADS", "8"))
    mem_limit = os.getenv("DUCKDB_MEMORY_LIMIT", "16GB")
    parquet_cache = os.getenv("DUCKDB_PARQUET_CACHE", "true").lower() == "true"

    state_fips, state_abbr = norm_state_arg(args.state)

    con = duckdb.connect(db_path)
    con.execute("INSTALL spatial; LOAD spatial;")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET threads = {threads};")
    con.execute(f"SET memory_limit = '{mem_limit}';")
    con.execute("SET s3_region='us-west-2';")  # Overture bucket region
    con.execute("SET s3_url_style='path';")    # Required for S3 access
    if parquet_cache:
        con.execute("SET parquet_metadata_cache = true;")
        con.execute("SET enable_http_metadata_cache = true;")

    # Load PV data for the state with TRACT_GEOID for optimized bbox fetching
    pv_query = """
        SELECT unified_id, STATE_ABBR, STATE_FIPS, TRACT_GEOID, ST_AsText(geometry) AS geometry
        FROM census_enriched_pv_data
        WHERE STATE_FIPS = ?
    """
    pv_df = con.execute(pv_query, [state_fips] if state_fips else [None]).fetchdf()
    if pv_df.empty and state_abbr:
        pv_df = con.execute(
            """
            SELECT unified_id, STATE_ABBR, STATE_FIPS, TRACT_GEOID, ST_AsText(geometry) AS geometry
            FROM census_enriched_pv_data
            WHERE STATE_ABBR = ?
            """,
            [state_abbr],
        ).fetchdf()
    if pv_df.empty:
        print(f"No PV rows for state {args.state}; nothing to do.")
        sys.exit(0)

    # Parse WKT geometry strings
    from shapely import wkt as wkt_parser
    pv_df["geometry"] = pv_df["geometry"].apply(wkt_parser.loads)
    pv_gdf = gpd.GeoDataFrame(pv_df, geometry="geometry", crs="EPSG:4326")

    # State geometry (cached)
    state_fips_val = pv_gdf["STATE_FIPS"].iloc[0]
    state_geom_gdf = ensure_state_geoms(con, state_fips_val)
    
    # Get census tract geometries for PV installations to minimize LULC fetch area
    unique_tract_geoids = pv_gdf["TRACT_GEOID"].unique().tolist()
    print(f"Fetching {len(unique_tract_geoids)} census tract geometries for optimized LULC extraction...")
    
    # Fetch tract geometries from Census API
    reader = cem.ShapeReader(year=2020)
    all_tracts = reader.read_cb_shapefile(shapefile_scope="us", geography="tract", crs="EPSG:4326")
    state_tracts = all_tracts[all_tracts["GEOID"].isin(unique_tract_geoids)].copy()
    
    # Use union of tract geometries bbox instead of full state bbox
    tract_union = state_tracts.geometry.union_all()
    optimized_bbox = box(*tract_union.bounds)
    
    print(f"Fetching Overture land cover/use for {len(unique_tract_geoids)} census tracts in state {state_fips_val} (release {overture_release})")
    print(f"  Optimized bbox area reduction: {(1 - optimized_bbox.area / box(*state_geom_gdf.geometry.union_all().bounds).area) * 100:.1f}%")

    # Register PV geometries in DuckDB first
    pv_state_df = pv_gdf[["unified_id", "STATE_ABBR", "STATE_FIPS"]].copy()
    pv_state_df['wkt'] = pv_gdf.geometry.apply(lambda g: g.wkt)
    con.register("pv_state", pv_state_df)
    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE pv_state_geom AS
        SELECT unified_id, STATE_ABBR, STATE_FIPS, ST_GeomFromText(wkt) AS geom
        FROM pv_state
        """
    )

    # Fetch LULC data using optimized census tract bbox (much smaller area than full state)
    optimized_bbox_tuple = optimized_bbox.bounds  # (minx, miny, maxx, maxy)
    fetch_overture_duckdb(con, "land_cover", optimized_bbox_tuple, overture_release)
    fetch_overture_duckdb(con, "land_use", optimized_bbox_tuple, overture_release)

    # Check if tables were created and have data
    lc_count = con.execute("SELECT COUNT(*) FROM lc_geom").fetchone()[0]
    lu_count = con.execute("SELECT COUNT(*) FROM lu_geom").fetchone()[0]
    
    print(f"  Performing spatial joins...")
    
    # Aggregate matches
    con.execute("CREATE OR REPLACE TEMP TABLE lc_matches AS SELECT * FROM (SELECT 1 WHERE 1=0)")
    con.execute("CREATE OR REPLACE TEMP TABLE lu_matches AS SELECT * FROM (SELECT 1 WHERE 1=0)")

    if lc_count > 0:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE lc_matches AS
            SELECT p.unified_id,
                   {array_merge_sql('lc.id')} AS lc_ids,
                   {array_merge_sql('lc.subtype')} AS lc_subtypes
            FROM pv_state_geom p
            JOIN lc_geom lc ON ST_Intersects(p.geom, lc.geometry)
            GROUP BY 1
            """
        )
    
    if lu_count > 0:
        con.execute(
            f"""
            CREATE OR REPLACE TEMP TABLE lu_matches AS
            SELECT p.unified_id,
                   {array_merge_sql('lu.id')} AS lu_ids,
                   {array_merge_sql('lu.subtype')} AS lu_subtypes
            FROM pv_state_geom p
            JOIN lu_geom lu ON ST_Intersects(p.geom, lu.geometry)
            GROUP BY 1
            """
        )
    
    # Track which columns exist for dynamic merge
    lc_has_id = lc_count > 0
    lc_has_subtype = lc_count > 0
    lu_has_id = lu_count > 0
    lu_has_subtype = lu_count > 0

    # Merge and persist - build dynamic column list based on what exists
    lc_cols_select = []
    if lc_has_id:
        lc_cols_select.append("lc.lc_ids")
    else:
        lc_cols_select.append("NULL::VARCHAR[] AS lc_ids")
    
    if lc_has_subtype:
        lc_cols_select.append("lc.lc_subtypes")
    else:
        lc_cols_select.append("NULL::VARCHAR[] AS lc_subtypes")
    
    lu_cols_select = []
    if lu_has_id:
        lu_cols_select.append("lu.lu_ids")
    else:
        lu_cols_select.append("NULL::VARCHAR[] AS lu_ids")
    
    if lu_has_subtype:
        lu_cols_select.append("lu.lu_subtypes")
    else:
        lu_cols_select.append("NULL::VARCHAR[] AS lu_subtypes")

    con.execute(
        """
        CREATE TABLE IF NOT EXISTS lulc_enriched_pv_data AS
        SELECT unified_id, STATE_ABBR, STATE_FIPS, NULL::VARCHAR[] AS lc_ids, NULL::VARCHAR[] AS lc_subtypes,
               NULL::VARCHAR[] AS lu_ids, NULL::VARCHAR[] AS lu_subtypes,
               geom AS geometry
        FROM pv_state_geom WHERE 1=0;
        """
    )

    con.execute(
        "DELETE FROM lulc_enriched_pv_data WHERE STATE_FIPS = ?",
        [state_fips_val],
    )

    all_cols_select = ",\n               ".join(lc_cols_select + lu_cols_select)
    con.execute(
        f"""
        INSERT INTO lulc_enriched_pv_data
        SELECT p.unified_id,
               p.STATE_ABBR,
               p.STATE_FIPS,
               {all_cols_select},
               p.geom AS geometry
        FROM pv_state_geom p
        LEFT JOIN lc_matches lc USING (unified_id)
        LEFT JOIN lu_matches lu USING (unified_id)
        """,
    )

    total_rows = con.execute(
        "SELECT COUNT(*) FROM lulc_enriched_pv_data WHERE STATE_FIPS = ?",
        [state_fips_val],
    ).fetchone()[0]
    lc_cov = con.execute(
        "SELECT COUNT(*) FROM lulc_enriched_pv_data WHERE STATE_FIPS = ? AND lc_ids IS NOT NULL",
        [state_fips_val],
    ).fetchone()[0]
    lu_cov = con.execute(
        "SELECT COUNT(*) FROM lulc_enriched_pv_data WHERE STATE_FIPS = ? AND lu_ids IS NOT NULL",
        [state_fips_val],
    ).fetchone()[0]

    print(f"Saved {total_rows} rows for state {state_fips_val} into lulc_enriched_pv_data")
    print(f"Land cover coverage: {lc_cov}/{total_rows} ({lc_cov/total_rows*100:.1f}%)")
    print(f"Land use coverage:   {lu_cov}/{total_rows} ({lu_cov/total_rows*100:.1f}%)")

    con.close()


if __name__ == "__main__":
    main()
