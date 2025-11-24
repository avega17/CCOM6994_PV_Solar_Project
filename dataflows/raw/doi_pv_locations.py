"""
Hamilton dataflow for scientific publication (DOI) PV location datasets.

Core sequence:
doi_metadata -> download in parallel -> load with geopandas -> convert to geoarrow -> save to duckDB (WKB) + geoparquet (GeoArrow)

Uses hybrid approach:
- GeoArrow encoding between Hamilton nodes for efficient spatial processing
- WKB conversion at DuckDB storage boundary for compatibility
- Native GeoArrow for optimal GeoParquet exports

Keeps individual datasets almost as-is, delegates schema standardization and data validation to dbt staging and downstream layers.
"""

from __future__ import annotations

import json
import tempfile
import shutil
import os
from pathlib import Path
from typing import Dict, Any, List

import geopandas as gpd
import pandas as pd
import pyarrow as pa
import duckdb
import datahugger
from dotenv import load_dotenv

# Import geoarrow-rs for efficient spatial operations
import geoarrow.rust.core as geoarrow

# Load environment variables
load_dotenv()

# Get repo root and manifest path from environment or use defaults
REPO_ROOT = os.getenv('REPO_ROOT', str(Path(__file__).parent.parent.parent))
INGEST_METADATA = os.getenv('INGEST_METADATA', os.path.join(REPO_ROOT, "data_loaders", "doi_manifest.json"))

from hamilton.function_modifiers import cache, tag, config
from hamilton.htypes import Parallelizable, Collect

# Import storage helper functions
from ._doi_pv_helpers_storage import (
    _geoarrow_table,
    _duckdb_table_from_geoarrow,
    _geoparquet_export,
    _storage_result,
    _apply_file_filters
)

# =============================================================================
# HAMILTON DATAFLOW NODES
# =============================================================================

@cache(behavior="recompute")  # Always reload to catch manifest updates
@tag(stage="metadata", data_type="config")
def doi_metadata(manifest_path: str = None) -> Dict[str, Dict[str, Any]]:
    """Load DOI dataset metadata from manifest file."""
    # Use environment variable or default
    if manifest_path is None:
        manifest_path = INGEST_METADATA

    manifest_file = Path(manifest_path)
    if not manifest_file.exists():
        raise FileNotFoundError(f"DOI manifest not found: {manifest_file}")

    with open(manifest_file, 'r') as f:
        return json.load(f)

@tag(stage="metadata", data_type="list", execution_mode="parallel")
@config.when(execution_mode="parallel")
def dataset_names__parallel(doi_metadata: Dict[str, Dict[str, Any]]) -> Parallelizable[str]:
    """Extract dataset names for parallel processing using Hamilton's Parallelizable."""
    # Filter out skipped datasets
    available_datasets = [
        name for name, metadata in doi_metadata.items()
        if not metadata.get("skip", False)
    ]
    skipped_count = len(doi_metadata) - len(available_datasets)

    print(f"Processing {len(available_datasets)} datasets in parallel")
    if skipped_count > 0:
        skipped_names = [name for name, metadata in doi_metadata.items() if metadata.get("skip", False)]
        print(f"Skipping {skipped_count} datasets marked with 'skip': true")
        print(f"   Skipped datasets: {', '.join(skipped_names)}")

    for dataset_name in available_datasets:
        yield dataset_name


@tag(stage="metadata", data_type="list", execution_mode="sequential")
@config.when(execution_mode="sequential")
def dataset_names__sequential(doi_metadata: Dict[str, Dict[str, Any]]) -> List[str]:
    """Extract dataset names for sequential processing as a simple list."""
    # Filter out skipped datasets
    available_datasets = [
        name for name, metadata in doi_metadata.items()
        if not metadata.get("skip", False)
    ]
    skipped_count = len(doi_metadata) - len(available_datasets)

    print(f"Processing {len(available_datasets)} datasets sequentially")
    if skipped_count > 0:
        skipped_names = [name for name, metadata in doi_metadata.items() if metadata.get("skip", False)]
        print(f"Skipping {skipped_count} datasets marked with 'skip': true")
        print(f"   Skipped datasets: {', '.join(skipped_names)}")

    return available_datasets

@tag(stage="download", data_type="path")
@cache(behavior="default")  # Cache download paths - let Hamilton auto-detect format for strings
@config.when(execution_mode="parallel")
def dataset_download_path__parallel(
    dataset_names: str,  # Individual dataset name from parallel processing
    doi_metadata: Dict[str, Dict[str, Any]],
    max_mb: int = 300
) -> str:
    """Download individual dataset and return path."""
    metadata = doi_metadata[dataset_names]
    
    # Create temp directory for this dataset
    download_dir = tempfile.mkdtemp(prefix=f"doi_{dataset_names}_")
    # prefer dataset-specific max_mb if provided
    if "max_mb" in metadata:
        max_mb = metadata["max_mb"]
    
    try:
        if metadata["repo"] == "github":
            # Handle GitHub URLs - distinguish between repos and raw files
            doi_url = metadata["doi"]

            if "raw.githubusercontent.com" in doi_url:
                # Raw GitHub file - download directly
                import urllib.request
                from pathlib import Path

                print(f"   📥 Downloading raw GitHub file: {doi_url}")
                filename = Path(doi_url).name
                if not filename.endswith(('.geojson', '.json', '.shp', '.gpkg', '.kml', '.gml')):
                    # If no extension, try to infer from URL or default to .geojson
                    filename = f"{dataset_names}.geojson"

                file_path = Path(download_dir) / filename
                urllib.request.urlretrieve(doi_url, file_path)
                print(f"   ✅ Downloaded {filename} to {download_dir}")

            else:
                # GitHub repository - clone it
                import subprocess
                repo_url = doi_url.replace("https://github.com/", "")
                subprocess.run([
                    "git", "clone", "--depth", "1",
                    f"https://github.com/{repo_url}.git",
                    download_dir
                ], check=True, capture_output=True)
                print(f"   ✅ Cloned repository to {download_dir}")
        
        elif metadata["repo"] == "direct_download":
            # Direct download link
            import urllib.request
            from pathlib import Path

            doi_url = metadata["doi"]
            print(f"   📥 Downloading direct link: {doi_url}")
            filename = Path(doi_url).name
            file_path = Path(download_dir) / filename
            urllib.request.urlretrieve(doi_url, file_path)
            print(f"   ✅ Downloaded {filename} to {download_dir}")

            # handle compression if specified
            if metadata["compression"] == "zip":
                import zipfile
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(download_dir)
                print(f"   ✅ Extracted zip file in {download_dir}")
                # Optionally remove zip file after extraction
                os.remove(file_path)
            elif metadata["compression"] == "tar.gz":
                import tarfile
                with tarfile.open(file_path, 'r:gz') as tar_ref:
                    tar_ref.extractall(download_dir)
                print(f"   ✅ Extracted tar.gz file in {download_dir}")
                # Optionally remove tar.gz file after extraction
                os.remove(file_path)
            elif metadata["compression"] is not None:
                raise ValueError(f"Unsupported compression type: {metadata['compression']}")
        
        elif metadata["repo"] in ("zenodo", "figshare", "osf", "huggingface"):
            # Use datahugger for DOI downloads
            datahugger.get(
                metadata["doi"],
                output_folder=download_dir,
                max_file_size=max_mb * 1024 * 1024
            )
        
        return download_dir

    except Exception as e:
        shutil.rmtree(download_dir, ignore_errors=True)
        raise RuntimeError(f"Download failed for {dataset_names}: {e}")


@config.when(execution_mode="sequential")
@tag(stage="download", data_type="path")
def dataset_download_path__sequential(
    dataset_names: List[str],  # Not used in sequential mode
    doi_metadata: Dict[str, Dict[str, Any]],
    max_mb: int = 300
) -> str:
    """Placeholder for sequential mode - actual processing happens in collected_arrow_tables__sequential."""
    # This is a placeholder since sequential processing happens all at once
    # Return empty string as this won't be used
    return ""

@tag(stage="process", data_type="geodataframe")
# Note: Caching disabled for GeoDataFrames due to potential GeoArrow serialization issues
@config.when(execution_mode="parallel")
def processed_geodataframe__parallel(
    dataset_download_path: str,  # Individual path from parallel processing
    dataset_names: str,  # Individual dataset name from parallel processing
    doi_metadata: Dict[str, Dict[str, Any]]  # DOI metadata for file filtering
) -> gpd.GeoDataFrame:
    """Load and minimally process geospatial data, keeping original schema."""
    download_path = Path(dataset_download_path)

    # Find geospatial files using file_filters if available
    metadata = doi_metadata[dataset_names]
    file_filters = metadata.get("file_filters", {})

    # Find all potential geospatial files
    geo_extensions = {'.shp', '.geojson', '.gpkg', '.kml', '.gml', '.json'}
    all_geo_files = []

    for ext in geo_extensions:
        all_geo_files.extend(download_path.rglob(f"*{ext}"))

    if not all_geo_files:
        raise ValueError(f"No geospatial files found in {download_path}")

    # Apply file filters if specified
    geo_files = _apply_file_filters(all_geo_files, file_filters)

    if not geo_files:
        print(f"   ⚠️  No files passed filters, using all found files")
        geo_files = all_geo_files

    print(f"   📁 Found {len(geo_files)} geospatial files after filtering:")
    for f in geo_files:
        print(f"      - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

    # Process all filtered files and concatenate them
    gdfs = []
    total_records = 0

    for geo_file in geo_files:
        print(f"   📄 Processing file: {geo_file.name}")
        try:
            # Load with geopandas - keep original schema
            file_gdf = gpd.read_file(geo_file)

            # Add essential metadata columns
            file_gdf['dataset_name'] = dataset_names
            file_gdf['source_file'] = str(geo_file.name)

            # Ensure valid CRS (use WGS84 if missing)
            if file_gdf.crs is None:
                file_gdf.set_crs("EPSG:4326", inplace=True)
            elif file_gdf.crs != 'EPSG:4326':
                # Reproject to WGS84 for consistency
                file_gdf = file_gdf.to_crs('EPSG:4326')

            gdfs.append(file_gdf)
            total_records += len(file_gdf)
            print(f"      ✅ {geo_file.name}: {len(file_gdf)} records")

        except Exception as e:
            print(f"      ❌ Error processing {geo_file.name}: {e}")
            continue

    if not gdfs:
        raise ValueError(f"No files could be processed successfully from {len(geo_files)} filtered files")

    # Concatenate all geodataframes
    if len(gdfs) == 1:
        gdf = gdfs[0]
    else:
        # Use pd.concat then convert back to GeoDataFrame to handle schema differences
        print(f"   🔗 Concatenating {len(gdfs)} geodataframes...")
        combined_df = pd.concat(gdfs, ignore_index=True, sort=False)

        # Convert back to GeoDataFrame and ensure geometry column is properly set
        gdf = gpd.GeoDataFrame(combined_df, geometry='geometry')

        # Ensure CRS is preserved (should be WGS84 from above)
        if gdf.crs is None:
            gdf.set_crs("EPSG:4326", inplace=True)

    print(f"   ✅ Combined dataset: {len(gdf)} total records from {len(gdfs)} files")

    # Ensure valid CRS (use WGS84 if missing)
    if gdf.crs is None:
        gdf.set_crs("EPSG:4326", inplace=True)

    return gdf


@config.when(execution_mode="sequential")
@tag(stage="process", data_type="geodataframe")
def processed_geodataframe__sequential(
    dataset_names: List[str]
) -> gpd.GeoDataFrame:
    """Placeholder for sequential mode - actual processing happens in collected_arrow_tables__sequential."""
    # Return empty geodataframe as placeholder
    return gpd.GeoDataFrame()


@tag(stage="convert", data_type="arrow")
@cache(behavior="disable")  # Disable caching for GeoArrow tables due to extension type serialization issues
@config.when(execution_mode="parallel")
def geo_arrow_table__parallel(
    processed_geodataframe: gpd.GeoDataFrame,  # Individual GDF from parallel processing
    dataset_names: str  # Individual dataset name from parallel processing
) -> pa.Table:
    """Convert geodataframe to geo arrow table using proper GeoArrow format."""
    return _geoarrow_table(processed_geodataframe, dataset_names)


@config.when(execution_mode="sequential")
@tag(stage="convert", data_type="arrow")
@cache(behavior="disable")  # Disable caching for consistency with parallel version
def geo_arrow_table__sequential(
    processed_geodataframe: gpd.GeoDataFrame,  # Use base name for consistency
    dataset_names: List[str]
) -> pa.Table:
    """Placeholder for sequential mode - actual processing happens in collected_arrow_tables__sequential."""
    # Return empty arrow table as placeholder (could use helper for consistency)
    return pa.table({})


@tag(stage="collect", data_type="arrow_list", execution_mode="parallel")
# Note: Caching disabled due to GeoArrow extension type serialization issues
@cache(behavior="disable")
@config.when(execution_mode="parallel")
def collected_arrow_tables__parallel(geo_arrow_table: Collect[pa.Table]) -> List[pa.Table]:
    """Collect all arrow tables from parallel processing."""
    return list(geo_arrow_table)


@tag(stage="collect", data_type="arrow_list", execution_mode="sequential")
# Note: Caching disabled due to GeoArrow extension type serialization issues
@cache(behavior="disable")
@config.when(execution_mode="sequential")
def collected_arrow_tables__sequential(
    dataset_names: List[str],
    doi_metadata: Dict[str, Dict[str, Any]],
    max_mb: int = 300
) -> List[pa.Table]:
    """Process all datasets sequentially and return list of arrow tables."""
    # Import Path at function level to avoid scoping issues
    from pathlib import Path
    import urllib.request
    import subprocess

    arrow_tables = []

    for dataset_name in dataset_names:
        print(f"Processing dataset: {dataset_name}")

        # Download dataset
        metadata = doi_metadata[dataset_name]
        download_dir = tempfile.mkdtemp(prefix=f"doi_{dataset_name}_")
        # prefer dataset-specific max_mb if provided
        if "max_mb" in metadata:
            max_mb = metadata["max_mb"]

        try:
            if metadata["repo"] == "github":
                # Handle GitHub URLs - distinguish between repos and raw files
                doi_url = metadata["doi"]

                if "raw.githubusercontent.com" in doi_url:
                    # Raw GitHub file - download directly
                    import urllib.request
                    from pathlib import Path

                    print(f"   📥 Downloading raw GitHub file: {doi_url}")
                    filename = Path(doi_url).name
                    if not filename.endswith(('.geojson', '.json', '.shp', '.gpkg', '.kml', '.gml')):
                        # If no extension, try to infer from URL or default to .geojson
                        filename = f"{dataset_names}.geojson"

                    file_path = Path(download_dir) / filename
                    urllib.request.urlretrieve(doi_url, file_path)
                    print(f"   ✅ Downloaded {filename} to {download_dir}")

                else:
                    # GitHub repository - clone it
                    import subprocess
                    repo_url = doi_url.replace("https://github.com/", "")
                    subprocess.run([
                        "git", "clone", "--depth", "1",
                        f"https://github.com/{repo_url}.git",
                        download_dir
                    ], check=True, capture_output=True)
                    print(f"   ✅ Cloned repository to {download_dir}")
            
            elif metadata["repo"] == "direct_download":
                # Direct download link
                import urllib.request
                from pathlib import Path

                doi_url = metadata["doi"]
                print(f"   📥 Downloading direct link: {doi_url}")
                filename = Path(doi_url).name
                file_path = Path(download_dir) / filename
                urllib.request.urlretrieve(doi_url, file_path)
                print(f"   ✅ Downloaded {filename} to {download_dir}")

                # handle compression if specified
                if metadata["compression"] == "zip":
                    import zipfile
                    with zipfile.ZipFile(file_path, 'r') as zip_ref:
                        zip_ref.extractall(download_dir)
                    print(f"   ✅ Extracted zip file in {download_dir}")
                    # Optionally remove zip file after extraction
                    os.remove(file_path)
                elif metadata["compression"] == "tar.gz":
                    import tarfile
                    with tarfile.open(file_path, 'r:gz') as tar_ref:
                        tar_ref.extractall(download_dir)
                    print(f"   ✅ Extracted tar.gz file in {download_dir}")
                    # Optionally remove tar.gz file after extraction
                    os.remove(file_path)
                elif metadata["compression"] is not None:
                    raise ValueError(f"Unsupported compression type: {metadata['compression']}")
            
            elif metadata["repo"] in ("zenodo", "figshare", "osf", "huggingface"):
                # Use datahugger for DOI downloads
                datahugger.get(
                    metadata["doi"],
                    output_folder=download_dir,
                    max_file_size=max_mb * 1024 * 1024
                )

            # Process geospatial data
            download_path = Path(download_dir)
            geo_extensions = {'.shp', '.geojson', '.gpkg', '.kml', '.gml', '.json'}
            all_geo_files = []

            for ext in geo_extensions:
                all_geo_files.extend(download_path.rglob(f"*{ext}"))

            if not all_geo_files:
                print(f"No geospatial files found for {dataset_name}")
                continue

            # Apply file filters if specified
            file_filters = metadata.get("file_filters", {})
            geo_files = _apply_file_filters(all_geo_files, file_filters)

            if not geo_files:
                print(f"   ⚠️  No files passed filters for {dataset_name}, using all found files")
                geo_files = all_geo_files

            print(f"   📁 Found {len(geo_files)} geospatial files after filtering:")
            for f in geo_files:
                print(f"      - {f.name} ({f.stat().st_size / 1024:.1f} KB)")

            # Process all filtered files and concatenate them
            gdfs = []

            for geo_file in geo_files:
                print(f"   📄 Processing file: {geo_file.name}")
                try:
                    # Read file with explicit resource management
                    file_gdf = gpd.read_file(geo_file)

                    # Add metadata
                    file_gdf['dataset_name'] = dataset_name
                    file_gdf['source_file'] = str(geo_file.name)

                    # Ensure valid CRS
                    if file_gdf.crs is None:
                        file_gdf.set_crs("EPSG:4326", inplace=True)
                    elif file_gdf.crs != 'EPSG:4326':
                        file_gdf = file_gdf.to_crs('EPSG:4326')

                    gdfs.append(file_gdf)
                    print(f"      ✅ {geo_file.name}: {len(file_gdf)} records")

                except Exception as e:
                    print(f"      ❌ Error processing {geo_file.name}: {e}")
                    continue

            if not gdfs:
                print(f"   ⚠️  No files could be processed for {dataset_name}")
                continue

            # Concatenate all geodataframes
            if len(gdfs) == 1:
                gdf = gdfs[0]
            else:
                print(f"   🔗 Concatenating {len(gdfs)} geodataframes for {dataset_name}...")
                combined_df = pd.concat(gdfs, ignore_index=True, sort=False)
                gdf = gpd.GeoDataFrame(combined_df, geometry='geometry')

                if gdf.crs is None:
                    gdf.set_crs("EPSG:4326", inplace=True)

            print(f"   ✅ Combined {dataset_name}: {len(gdf)} total records from {len(gdfs)} files")

            # Read file with explicit resource management
            try:
                # Convert to arrow table using our helper function
                table = _geoarrow_table(gdf, dataset_name)

                arrow_tables.append(table)

            finally:
                # Explicitly clean up geodataframe to free file handles
                if 'gdf' in locals():
                    del gdf

        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
        finally:
            # Clean up download directory
            shutil.rmtree(download_dir, ignore_errors=True)

    # Force garbage collection to clean up any lingering resources
    import gc
    gc.collect()

    return arrow_tables

@tag(stage="storage", data_type="result", execution_mode="parallel")
@cache(behavior="default")  # Cache final results
@config.when(execution_mode="parallel")
def pipeline_result__parallel(
    collected_arrow_tables: List[pa.Table],  # Reference base name, not suffixed version
    database_path: str = None,  # Will use DUCKLAKE_CONNECTION_STRING env var
    export_geoparquet: bool = True,
    export_path: str = None,
    use_cloud_export: bool = False
) -> Dict[str, Any]:
    """Save arrow tables to DuckLake catalog and optionally export to GeoParquet (parallel mode)."""
    return _storage_result(collected_arrow_tables, database_path, export_geoparquet, export_path, use_cloud_export)


@tag(stage="storage", data_type="result", execution_mode="sequential")
@cache(behavior="default")  # Cache final results
@config.when(execution_mode="sequential")
def pipeline_result__sequential(
    collected_arrow_tables: List[pa.Table],  # Reference base name, not suffixed version
    database_path: str = None,  # Will use DUCKLAKE_CONNECTION_STRING env var
    export_geoparquet: bool = True,
    export_path: str = None,
    use_cloud_export: bool = False
) -> Dict[str, Any]:
    """Save arrow tables to DuckLake catalog and optionally export to GeoParquet (sequential mode)."""
    return _storage_result(collected_arrow_tables, database_path, export_geoparquet, export_path, use_cloud_export)
