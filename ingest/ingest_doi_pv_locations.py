#!/usr/bin/env python3
"""
DOI PV Locations Ingestion Pipeline

Hamilton driver script for ingesting DOI-based PV location datasets.
Follows Hamilton best practices with separate dataflow module and driver.

Usage:
    python ingest_doi_pv_locations.py
    python ingest_doi_pv_locations.py --sequential --no-cache
    python ingest_doi_pv_locations.py --database ./custom.duckdb --max-mb 500
"""

from hamilton import driver
from hamilton.execution import executors

# Add repo root to path to import dataflows
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import os
from dotenv import load_dotenv

from dataflows.raw import doi_pv_locations

# Get repo root and manifest path from environment or use defaults
# Use relative paths as they work better than absolute paths in this environment
REPO_ROOT = str(Path(__file__).parent.parent)

# Load environment variables from .env file
load_dotenv(Path(REPO_ROOT) / ".env")
INGEST_METADATA = os.getenv('DATASET_MANIFEST_PATH', str(Path(REPO_ROOT) / "ingest" / "doi_manifest_usa.json"))
# print(f"Found Connection String: {os.getenv('DUCKLAKE_CONNECTION_STRING_PROD')}")


def create_hamilton_driver(
    config: dict,
    enable_caching: bool = True,
    use_parallel: bool = True
) -> driver.Driver:
    """
    Create Hamilton driver following documentation best practices.
    
    Args:
        config: Configuration dictionary for the pipeline
        enable_caching: Whether to enable Hamilton's built-in caching
        use_parallel: Whether to use parallel execution
        
    Returns:
        driver.Driver: Configured Hamilton driver
    """
    execution_mode = "parallel" if use_parallel else "sequential"
    cache_status = "with caching" if enable_caching else "without caching"
    print(f"🔧 Creating Hamilton driver: {execution_mode} execution {cache_status}")
    
    # Create builder with the dataflow module (following Hamilton docs pattern)
    builder = driver.Builder().with_modules(doi_pv_locations).with_config(config)
    
    # Add caching if enabled
    if enable_caching:
        cache_config = {"path": config.get("cache_path", "./.hamilton_cache")}
        if config.get("force_download", False):
            cache_config["recompute"] = True
            print("   🔄 Force download enabled - will recompute all cached nodes")
        builder = builder.with_cache(**cache_config)
        print(f"   📦 Caching enabled at: {cache_config['path']}")
    
    # Add parallel execution if enabled
    if use_parallel:
        builder = (
            builder
            .enable_dynamic_execution(allow_experimental_mode=True)
            .with_local_executor(executors.SynchronousLocalTaskExecutor())
        )
        print("   🚀 Parallel processing enabled")
    else:
        print("   📝 Sequential processing enabled")

    print(f"   🛠️  Building Hamilton driver with config: {config}"
          f" and module {doi_pv_locations}")
    dr = builder.build()
    
    return dr


def run_doi_pv_pipeline(
    database_path: str = None,
    manifest_path: str = None,
    max_mb: int = 300,
    export_geoparquet: bool = True,
    enable_caching: bool = True,
    force_download: bool = False,
    cache_path: str = "./.hamilton_cache",
    use_parallel: bool = True,
    export_path: str = None,
    use_cloud_export: bool = False,
    force_upload: bool = False,
    use_ducklake: bool = True
) -> dict:
    """
    Run the complete DOI PV locations pipeline.

    Args:
        database_path: Path to DuckDB database or DuckLake connection string
        manifest_path: Path to DOI manifest file
        max_mb: Maximum download size in MB
        export_geoparquet: Whether to export GeoParquet files
        enable_caching: Whether to enable Hamilton caching
        force_download: Force re-download (triggers cache recomputation)
        cache_path: Path to Hamilton cache directory
        use_parallel: Whether to use parallel processing
        export_path: Path for GeoParquet export (local or cloud S3/R2 path)
        use_cloud_export: Whether to use cloud export functionality
        force_upload: Force upload even if files haven't changed
        use_ducklake: Whether to use DuckLake catalog instead of regular DuckDB

    Returns:
        dict: Pipeline execution results
    """
    # Use environment variables or defaults for paths
    if database_path is None:
        if use_ducklake:
            # Use DuckLake PostgreSQL catalog (unified dev/prod)
            database_path = os.getenv('DUCKLAKE_ATTACH_PROD')
        else:
            # Use regular DuckDB database
            database_path = os.path.join(REPO_ROOT, "db", "eo_pv_data.duckdb")

    if manifest_path is None:
        manifest_path = INGEST_METADATA

    # Verify manifest file exists
    if not Path(manifest_path).exists():
        raise FileNotFoundError(f"DOI manifest not found: {manifest_path}")

    # Create configuration with execution mode and cloud support
    config = {
        "database_path": database_path,
        "manifest_path": manifest_path,
        "max_mb": max_mb,
        "export_geoparquet": export_geoparquet,
        "force_download": force_download,
        "cache_path": cache_path,
        "execution_mode": "parallel" if use_parallel else "sequential",
        "export_path": export_path,
        "use_cloud_export": use_cloud_export,
        "force_upload": force_upload,
        "use_ducklake": use_ducklake
    }
    
    # Create Hamilton driver
    dr = create_hamilton_driver(config, enable_caching=enable_caching, use_parallel=use_parallel)
    
    # Execute pipeline
    execution_mode = "parallel" if use_parallel else "sequential"
    print(f"🚀 Starting {execution_mode} DOI PV pipeline execution...")
    
    # Execute the appropriate pipeline result function based on execution mode
    result_function = "pipeline_result"  # Hamilton will resolve to the correct implementation
    result = dr.execute([result_function])

    return result[result_function]


def main():
    """Main CLI function."""

    import argparse
    
    parser = argparse.ArgumentParser(description="DOI PV Locations Ingestion Pipeline")
    parser.add_argument("--database", default=None, help=f"DuckDB database path (default: {os.getenv('DUCKLAKE_ATTACH_PROD', os.path.join(REPO_ROOT, 'db', 'pv_ingest.duckdb'))})")
    parser.add_argument("--manifest", default=None, help=f"DOI manifest file (default: {INGEST_METADATA})")
    parser.add_argument("--max-mb", type=int, default=300, help="Max download size in MB")
    parser.add_argument("--no-geoparquet", action="store_true", help="Skip GeoParquet export")
    parser.add_argument("--no-cache", action="store_true", help="Disable Hamilton caching")
    parser.add_argument("--force-download", action="store_true", help="Force re-download")
    parser.add_argument("--cache-path", default="./.hamilton_cache", help="Cache directory")
    parser.add_argument("--sequential", action="store_true", help="Use sequential processing")
    parser.add_argument("--cloud", action="store_true", help="Use cloud deployment (export to R2)")
    parser.add_argument("--export-path", default=None, help="Custom export path (local or S3/R2 URL)")
    parser.add_argument("--force-upload", action="store_true", help="Force upload even if files haven't changed")
    parser.add_argument("--no-ducklake", action="store_true", help="Use regular DuckDB instead of DuckLake catalog")
    
    args = parser.parse_args()

    print(f"Using Repo Root: {REPO_ROOT}")
    print(f"Using DOI Manifest: {INGEST_METADATA}")
    
    # Configure cloud deployment
    if args.cloud:
        export_path = args.export_path or "r2://eo-pv-lakehouse/geoparquet/"
        use_cloud_export = True
        print(f"🌩️  Cloud deployment enabled - exporting to: {export_path}")
    else:
        export_path = args.export_path
        use_cloud_export = False
        if export_path:
            print(f"📁 Custom export path: {export_path}")

    try:
        result = run_doi_pv_pipeline(
            database_path=args.database,
            manifest_path=args.manifest,
            max_mb=args.max_mb,
            export_geoparquet=not args.no_geoparquet,
            enable_caching=not args.no_cache,
            force_download=args.force_download,
            cache_path=args.cache_path,
            use_parallel=not args.sequential,
            export_path=export_path,
            use_cloud_export=use_cloud_export,
            force_upload=args.force_upload,
            use_ducklake=not args.no_ducklake
        )
        
        print("✅ Pipeline completed successfully!")
        print(f"📊 Results:")
        print(f"   📁 Database: {result['database_path']}")
        print(f"   📋 Tables created: {len(result['tables_created'])}")
        print(f"   📄 GeoParquet files: {len(result['geoparquet_files'])}")
        print(f"   📊 Total records: {result['total_records']}")
        
        if result['tables_created']:
            table_names = [table_info['table_name'] for table_info in result['tables_created']]
            table_list = '\n- '.join(table_names)
            print(f"   🗂️  Tables: \n - {table_list}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        exit(1)


if __name__ == "__main__":
    main()
