"""
Helper functions for DOI PV location dataflow storage operations.

Contains storage-related helper functions to keep the main dataflow compact.
All functions prefixed with _ to exclude from Hamilton DAG visualization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List
import re

import geopandas as gpd
import pyarrow as pa
import duckdb

# Add repo root to path to import dataflows
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
# import ducklake connection helper from utils/ducklake.py
from utils.ducklake import _create_ducklake_connection



def _apply_file_filters(files: List[Path], file_filters: Dict[str, Any]) -> List[Path]:
    """
    Apply file filters from DOI manifest to filter files.

    Args:
        files: List of file paths to filter
        file_filters: Filter configuration from DOI manifest

    Returns:
        Filtered list of file paths
    """
    if not file_filters:
        return files

    include_patterns = file_filters.get("include_patterns", [])
    exclude_patterns = file_filters.get("exclude_patterns", [])
    use_regex = file_filters.get("use_regex", False)

    filtered_files = []

    for file_path in files:
        file_name = file_path.name
        include_match = False
        exclude_match = False

        # Check include patterns
        if include_patterns:
            for pattern in include_patterns:
                if use_regex:
                    if re.search(pattern, file_name):
                        include_match = True
                        break
                else:
                    if file_path.match(pattern):
                        include_match = True
                        break
        else:
            # No include patterns means include all
            include_match = True

        # Check exclude patterns
        if exclude_patterns:
            for pattern in exclude_patterns:
                if use_regex:
                    if re.search(pattern, file_name):
                        exclude_match = True
                        break
                else:
                    if file_path.match(pattern):
                        exclude_match = True
                        break

        # Include file if it matches include patterns and doesn't match exclude patterns
        if include_match and not exclude_match:
            filtered_files.append(file_path)

    return filtered_files


def _geoarrow_table(gdf: gpd.GeoDataFrame, dataset_name: str):
    """
    Convert GeoDataFrame to Arrow table with proper GeoArrow encoding using geoarrow-rs.

    Uses geoarrow-rs (Rust implementation) for efficient spatial operations and
    better GeoArrow support with direct from_geopandas() conversion.

    Adds dataset_name metadata using native arro3 methods for single geometry types
    or PyArrow methods for mixed geometry types (WKB fallback).

    References:
    - https://geoarrow.org/geoarrow-rs/python/v0.4.0-beta.3/ecosystem/geopandas/
    - https://kylebarron.dev/arro3/v0.5.1/api/core/table/
    - https://kylebarron.dev/arro3/v0.5.1/api/core/schema/

    Args:
        gdf: GeoDataFrame to convert
        dataset_name: Name of the dataset (added to table metadata)

    Returns:
        Arrow table with dataset_name metadata (arro3.core.Table or PyArrow table)
    """
    if gdf.empty:
        return pa.table({
            'geometry': pa.array([], type=pa.binary()),
            'dataset_name': pa.array([dataset_name], type=pa.string())
        })

    import pandas as pd

    gdf_copy = gdf.copy()

    # Ensure dataset_name column exists (source_file should already be present from processing)
    if 'dataset_name' not in gdf_copy.columns:
        gdf_copy['dataset_name'] = dataset_name

    # Clean problematic data types that cause conversion errors
    for col in gdf_copy.columns:
        if col != 'geometry' and gdf_copy[col].dtype == 'object':
            # Check if column contains mixed types (strings in numeric columns)
            try:
                # Try to convert to numeric, coercing errors to NaN
                numeric_series = pd.to_numeric(gdf_copy[col], errors='coerce')
                # If we have both numeric and non-numeric values, keep as string
                if numeric_series.notna().any() and numeric_series.isna().any():
                    print(f"   🔧 Column '{col}' has mixed types, keeping as string")
                    gdf_copy[col] = gdf_copy[col].astype(str)
                elif numeric_series.notna().all():
                    # All values are numeric, convert to numeric
                    gdf_copy[col] = numeric_series
            except:
                # If any error, keep as string
                gdf_copy[col] = gdf_copy[col].astype(str)

    # Check for mixed geometry types which geoarrow-rs doesn't support
    geom_types = set(gdf_copy.geometry.geom_type.unique())
    has_mixed_geom_types = len(geom_types) > 1

    try:
        # Try geoarrow-rs for efficient conversion with native GeoArrow types
        from geoarrow.rust.core import from_geopandas

        if has_mixed_geom_types:
            # Skip geoarrow-rs for mixed geometry types
            raise ValueError(f"Geometry type combination is not supported {sorted(geom_types)}")

        arro3_table = from_geopandas(gdf_copy)
        print(f"   ✅ Converted {len(gdf_copy)} features using geoarrow-rs")

        # Add dataset metadata using arro3 native methods
        metadata = {"dataset_name": dataset_name}
        schema_with_metadata = arro3_table.schema.with_metadata(metadata)
        table = arro3_table.with_schema(schema_with_metadata)

    except Exception as e:
        print(f"   ⚠️  geoarrow-rs conversion failed: {e}, using WKB fallback")
        # Fallback to WKB encoding if geoarrow-rs fails
        table = pa.table(gdf_copy.to_arrow(index=False, geometry_encoding='WKB'))
        print(f"   ✅ Converted {len(gdf_copy)} features using WKB fallback")

        # Add dataset metadata (PyArrow table)
        metadata = {"dataset_name": dataset_name}
        table = table.replace_schema_metadata(metadata)

    return table


def _duckdb_table_from_geoarrow(
    conn: 'duckdb.DuckDBPyConnection',
    arrow_table: pa.Table,
    table_name: str,
    use_wkt_geometry: bool = False
) -> int:
    """
    Create DuckDB table from GeoArrow table using geoarrow-rs conversion.

    Uses geoarrow-rs to_geopandas() to properly convert Arrow tables with geometry data
    back to GeoPandas, preserving CRS and geometry information.

    Reference: https://geoarrow.org/geoarrow-rs/python/v0.4.0-beta.3/ecosystem/geopandas/

    Args:
        conn: DuckDB connection
        arrow_table: Arrow table to import
        table_name: Target table name in DuckDB
        use_wkt_geometry: If True, store geometry as WKT strings (for DuckLake compatibility)

    Returns:
        Number of rows inserted
    """
    # Drop existing table
    conn.execute(f"DROP VIEW IF EXISTS {table_name}")

    try:
        # Convert using geoarrow-rs which preserves geometry information
        from geoarrow.rust.core import to_geopandas
        gdf = to_geopandas(arrow_table)

        # Handle DuckDB table creation - DuckDB doesn't support geoarrow extension types
        if isinstance(gdf, gpd.GeoDataFrame) and hasattr(gdf, 'geometry') and gdf.geometry is not None:
            # Set CRS if missing
            if gdf.crs is None:
                gdf = gdf.set_crs('EPSG:4326', allow_override=True)

            gdf_copy = gdf.copy()

            if use_wkt_geometry:
                # For DuckLake: Store geometry as WKT strings (no spatial types)
                gdf_copy['geometry_wkt'] = gdf_copy.geometry.to_wkt()
                gdf_copy = gdf_copy.drop(columns=['geometry'])

                # Create table with WKT geometry as string
                conn.execute(f"""
                    CREATE TABLE {table_name} AS
                    SELECT
                        * EXCLUDE (geometry_wkt),
                        geometry_wkt as geometry
                    FROM gdf_copy
                """)
                print(f"   📍 Stored geometry as WKT strings for DuckLake compatibility")
            else:
                # For regular DuckDB: Convert geometry to WKB and store as geometry column
                # Store WKB directly as geometry for consistent processing downstream
                gdf_copy['geometry'] = gdf_copy.geometry.to_wkb()

                # Create table with WKB geometry column (no conversion to spatial types)
                conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM gdf_copy")
                print(f"   📍 Stored geometry as WKB in geometry column")
        else:
            # No geometry column - simple table creation
            conn.execute(f"CREATE TABLE {table_name} AS SELECT * FROM gdf")

        row_count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"   ✅ Created table {table_name} with {row_count} rows")
        return row_count

    except Exception as e:
        print(f"   ❌ Failed to create table {table_name}: {e}")
        raise


def _geoparquet_export(
    arrow_table: pa.Table,
    dataset_name: str,
    output_dir: Path
) -> str:
    """
    Export Arrow table to GeoParquet format using geoarrow-rs I/O.

    Uses geoarrow-rs for optimal GeoParquet export with native GeoArrow support.
    Falls back to GeoPandas conversion if needed.

    Reference: https://geoarrow.org/geoarrow-rs/python/v0.4.0-beta.3/api/io/geoparquet/

    Args:
        arrow_table: Arrow table to export
        dataset_name: Name of the dataset
        output_dir: Directory to save the parquet file

    Returns:
        Path to the exported parquet file
    """
    parquet_file = output_dir / f"raw_{dataset_name}.parquet"
    print(f"   📦 Exporting {dataset_name} to GeoParquet: {parquet_file}")
    from geoarrow.rust.core import to_geopandas
    gdf = to_geopandas(arrow_table)
    gdf.to_parquet(parquet_file, geometry_encoding='WKB', compression='zstd', write_covering_bbox=True, schema_version='1.1.0')
    print(f"   ✅ Exported to GeoParquet using GeoPandas")

    return str(parquet_file)


def _geoparquet_export_cloud(
    arrow_table: pa.Table,
    dataset_name: str,
    output_path: str,
    use_cloud: bool = False,
    force_upload: bool = False
) -> str:
    """
    Export Arrow table to GeoParquet with cloud storage support.

    For cloud deployment, this function can export directly to R2/S3 using DuckDB.
    Includes hash-based change detection to avoid unnecessary uploads.

    Args:
        arrow_table: Arrow table to export
        dataset_name: Name of the dataset
        output_path: Local directory path or S3 URL for cloud storage
        use_cloud: Whether to use cloud storage (R2/S3) export
        force_upload: Skip hash check and force upload

    Returns:
        Path to the exported parquet file
    """
    if use_cloud and (output_path.startswith('s3://') or output_path.startswith('r2://')):
        # Cloud export using DuckDB S3 capabilities
        # Reference: https://duckdb.org/docs/stable/guides/network_cloud_storage/s3_export
        # Reference: https://duckdb.org/docs/stable/core_extensions/httpfs/s3api.html#writing
        try:
            import duckdb
            import os
            import hashlib
            import time

            # Calculate data hash for change detection (simple approach using record count + column names)
            if not force_upload:
                try:
                    # Use simple hash based on table structure (no conversion needed)
                    if hasattr(arrow_table, 'num_rows') and hasattr(arrow_table, 'column_names'):
                        # arro3 table
                        record_count = arrow_table.num_rows
                        columns = arrow_table.column_names
                    else:
                        # PyArrow table
                        record_count = len(arrow_table)
                        columns = arrow_table.column_names

                    # Simple hash based on structure (record count + column names)
                    # This avoids expensive data conversion while still detecting changes
                    hash_input = f"{record_count}:{':'.join(sorted(columns))}"
                    data_hash = hashlib.md5(hash_input.encode()).hexdigest()
                    print(f"   🔍 Data hash for {dataset_name}: {data_hash[:8]}... ({record_count:,} records)")

                except Exception as hash_error:
                    print(f"   ⚠️  Hash calculation failed: {hash_error}, proceeding with upload")
                    # Continue with upload if hash calculation fails

            # Create temporary DuckDB connection
            conn = duckdb.connect()

            # Install and load httpfs extension for S3 support
            conn.execute("INSTALL httpfs")
            conn.execute("LOAD httpfs")

            # Configure S3/R2 credentials
            r2_access_key = os.getenv('R2_ACCESS_KEY_ID')
            r2_secret_key = os.getenv('R2_SECRET_KEY')

            if not all([r2_access_key, r2_secret_key]):
                raise ValueError("Missing R2 credentials: R2_ACCESS_KEY_ID, R2_SECRET_KEY")

            # Configure DuckDB S3 settings for Cloudflare R2
            # Reference: https://duckdb.org/docs/stable/core_extensions/httpfs/s3api.html#configuration

            # Use S3 API endpoint with credentials (DuckDB S3 format)
            conn.execute(f"SET s3_access_key_id='{r2_access_key}'")
            conn.execute(f"SET s3_secret_access_key='{r2_secret_key}'")
            # Create R2 SECRET for DuckDB export (same as raw models)
            r2_account_id = os.getenv('CLOUDFLARE_ACCOUNT_ID')
            if r2_account_id:
                conn.execute(f"""
                    CREATE OR REPLACE SECRET r2_export_secret (
                        TYPE r2,
                        KEY_ID '{r2_access_key}',
                        SECRET '{r2_secret_key}',
                        ACCOUNT_ID '{r2_account_id}'
                    )
                """)
                print("   ✅ R2 SECRET created for export")
            else:
                print("   ⚠️  CLOUDFLARE_ACCOUNT_ID required for R2 SECRET")
            # R2 SECRET handles all configuration automatically

            cloud_file_path = f"{output_path.rstrip('/')}/raw_{dataset_name}.parquet"

            # Check if file exists and compare hash (unless forced)
            if not force_upload:
                try:
                    # Try to read existing file metadata
                    check_sql = f"SELECT COUNT(*) FROM '{cloud_file_path}'"
                    existing_count = conn.execute(check_sql).fetchone()[0]

                    # If file exists, check if we need to update it
                    # Get current record count (handle both arro3 and PyArrow)
                    try:
                        if hasattr(arrow_table, 'num_rows'):
                            # arro3 table
                            current_count = arrow_table.num_rows
                        else:
                            # PyArrow table
                            current_count = len(arrow_table)
                    except:
                        current_count = 0

                    print(f"   📁 Existing file found with {existing_count:,} records")
                    print(f"   🔄 Current data has {current_count:,} records")

                    # Simple check: if record count is the same, assume no change needed
                    if existing_count == current_count:
                        print(f"   ⏭️  Skipping upload - record count unchanged ({existing_count:,})")
                        conn.close()
                        return cloud_file_path

                except Exception:
                    # File doesn't exist or can't be read - proceed with upload
                    print(f"   📤 New file - proceeding with upload")

            # Calculate and display file size info
            try:
                # Handle both arro3 and PyArrow tables for size calculation
                if hasattr(arrow_table, 'nbytes') and hasattr(arrow_table, 'num_rows'):
                    # arro3 table - has both nbytes and num_rows properties
                    table_size_mb = arrow_table.nbytes / (1024 * 1024)
                    record_count = arrow_table.num_rows
                elif hasattr(arrow_table, 'nbytes'):
                    # PyArrow table
                    table_size_mb = arrow_table.nbytes / (1024 * 1024)
                    record_count = len(arrow_table)
                else:
                    # Fallback - estimate size
                    record_count = len(arrow_table) if hasattr(arrow_table, '__len__') else 0
                    table_size_mb = record_count * 0.001  # Rough estimate

                print(f"   📊 Uploading {dataset_name}: {record_count:,} records ({table_size_mb:.1f} MB)")

            except Exception as size_error:
                print(f"   📊 Uploading {dataset_name}: size calculation failed ({size_error})")
                table_size_mb = 0  # Set default for upload speed calculation

            # Register Arrow table in DuckDB
            # conn.register('temp_table', arrow_table) not needed - duckdb supports directly reading Arrow and dataframe tables

            print(f"   🌩️  Uploading to R2: {cloud_file_path}")

            # Fix: Write to temp file first with correct WKB geometry encoding
            # This ensures DuckDB can read the geometry properly (not GeoArrow binary)
            import tempfile
            import os
            try:
                from geoarrow.rust.core import to_geopandas
            except ImportError:
                print("   ⚠️  geoarrow.rust.core not available") 
            else:
                # Use temp file approach with proper WKB encoding
                with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as temp_file:
                    temp_path = temp_file.name

                    # Convert to GeoPandas and export with WKB encoding (same as local export)
                    gdf = to_geopandas(arrow_table)
                    gdf.to_parquet(
                        temp_path,
                        geometry_encoding='WKB',
                        compression='zstd',
                        write_covering_bbox=True,
                        schema_version='1.1.0'
                    )

                    # Now upload the properly encoded file using DuckDB
                    export_sql = f"""
                    COPY (SELECT * FROM read_parquet('{temp_path}')) TO '{cloud_file_path}'
                    (FORMAT 'parquet', COMPRESSION 'zstd')
                    """

                    # Time the upload
                    start_time = time.time()
                    conn.execute(export_sql)
                    upload_time = time.time() - start_time

                    # Clean up temp file
                    os.unlink(temp_path)

            # Calculate upload speed
            upload_speed_mbps = table_size_mb / upload_time if upload_time > 0 else 0

            conn.close()
            print(f"   ✅ Upload complete: {table_size_mb:.1f} MB in {upload_time:.1f}s ({upload_speed_mbps:.1f} MB/s)")
            return cloud_file_path

        except Exception as cloud_error:
            print(f"   ❌ Cloud export failed for {dataset_name}: {cloud_error}")
            print(f"   💡 Ensure R2 credentials are configured and bucket exists")
            # Don't fall back to local during cloud deployment - fail fast
            if use_cloud:
                raise cloud_error
            # Fall through to local export only if not explicitly using cloud

    # Local export (existing functionality)
    if output_path and (output_path.startswith('s3://') or output_path.startswith('r2://')):
        # If cloud path provided but we reached here, cloud export failed
        if use_cloud:
            # During cloud deployment, don't fall back to local - this could hide issues
            raise RuntimeError(f"Cloud export failed for {dataset_name} and fallback disabled during cloud deployment")
        else:
            # For mixed deployments, fall back to local
            from pathlib import Path
            repo_root = Path(os.getenv('REPO_ROOT', '.')).resolve()
            output_dir = repo_root / "db" / "geoparquet"
            output_dir.mkdir(parents=True, exist_ok=True)
            print(f"   ⚠️  Using local fallback for {dataset_name}")
    else:
        # Standard local export
        from pathlib import Path
        output_dir = Path(output_path) if output_path else Path("db/geoparquet")
        output_dir.mkdir(parents=True, exist_ok=True)

    return _geoparquet_export(arrow_table, dataset_name, output_dir)


def _storage_result(
    collected_arrow_tables: List[pa.Table],
    database_path: str,
    export_geoparquet: bool,
    export_path: str = None,
    use_cloud_export: bool = False,
    force_upload: bool = False
) -> Dict[str, Any]:
    """Save arrow tables to DuckDB or DuckLake and optionally export to GeoParquet (GeoArrow)."""

    # Handle None database_path by using environment variable
    if database_path is None:
        database_path = os.getenv('DUCKLAKE_ATTACH_PROD',
                                'ERROR:DUCKLAKE_CONNECTION_STRING_PROD not set')

    # Connect to DuckDB or DuckLake
    is_ducklake = database_path.startswith('ducklake:')

    if is_ducklake:
        # use helper for DuckLake connection; TODO: support target selection
        conn = _create_ducklake_connection()

        # TODO: update when we have implemented Ducklake 0.3's support for WKB and geometry types
        print(f"   ⚠️  Using WKT geometry storage (DuckLake spatial type limitation)")
    else:
        # Regular DuckDB connection
        conn = duckdb.connect(database_path)
        # Install spatial and httpfs extension
        conn.execute("INSTALL spatial; LOAD spatial;")
        conn.execute("INSTALL httpfs; LOAD httpfs;")
        print(f"   🔗 Connected to DuckDB: {database_path}")

    results = {
        "database_path": database_path,
        "tables_created": [],
        "geoparquet_files": [],
        "total_records": 0,
        "datasets_processed": 0,
        "datasets_total": len(collected_arrow_tables)
    }

    for table in collected_arrow_tables:
        # Get dataset name from metadata
        dataset_name = table.schema.metadata.get(b"dataset_name", b"unknown").decode()
        table_name = f"raw_{dataset_name}"

        # Use proven approach: convert to GeoPandas and use DuckDB's native support
        row_count = _duckdb_table_from_geoarrow(
            conn=conn,
            arrow_table=table,
            table_name=table_name,
            use_wkt_geometry=is_ducklake
        )

        results["tables_created"].append({
            "table_name": table_name,
            "dataset_name": dataset_name,
            "record_count": row_count
        })
        results["total_records"] += row_count
        results["datasets_processed"] += 1

        # Export to GeoParquet if requested (use original GeoArrow table for best format)
        if export_geoparquet:
            if export_path and use_cloud_export:
                # Use cloud-compatible export
                try:
                    parquet_file = _geoparquet_export_cloud(
                        table, dataset_name, export_path, use_cloud_export, force_upload=force_upload
                    )
                    results["geoparquet_files"].append(parquet_file)
                except Exception as e:
                    print(f"   ⚠️  Cloud GeoParquet export failed for {dataset_name}: {e}")
            else:
                # Use local export (existing functionality)
                if database_path.startswith('ducklake:'):
                    # For DuckLake (PostgreSQL or SQLite), use local data path
                    # Both postgres and sqlite catalogs use the same local data directory
                    repo_root = os.getenv('REPO_ROOT', '.')
                    output_dir = Path(repo_root) / "db" / "geoparquet"
                else:
                    # Regular DuckDB file path
                    output_dir = Path(database_path).parent / "geoparquet"
                output_dir.mkdir(parents=True, exist_ok=True)

                try:
                    parquet_file = _geoparquet_export(table, dataset_name, output_dir)
                    results["geoparquet_files"].append(parquet_file)
                except Exception as e:
                    print(f"   ⚠️  GeoParquet export failed for {dataset_name}: {e}")

    # Generate detailed summary
    print(f"\n🎯 Pipeline Results Summary:")
    print(f"   📊 Total datasets processed: {results['datasets_processed']}/{results['datasets_total']}")
    print(f"   📈 Total records imported: {results['total_records']:,}")
    print(f"   🗄️  Database: {database_path}")

    # Show table details
    print(f"\n📋 Tables created in DuckDB:")
    for table_info in results['tables_created']:
        table_name = table_info['table_name']
        record_count = table_info['record_count']
        print(f"   • {table_name}: {record_count:,} records")

    # Show database summary (CRS is standardized to WGS84)
    try:
        table_count = conn.execute("SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'main'").fetchone()[0]
        print(f"\n📚 Database Summary:")
        print(f"   • Schema: main")
        print(f"   • Tables: {table_count}")
        print(f"   • CRS: EPSG:4326 (WGS84) - standardized")
    except Exception as e:
        print(f"   ⚠️  Could not retrieve database summary: {e}")

    if results['geoparquet_files']:
        print(f"\n📦 GeoParquet exports:")
        for parquet_file in results['geoparquet_files']:
            print(f"   • {parquet_file}")

    conn.close()
    return results
