#!/usr/bin/env python3
"""
Generalized Hamilton DAG Visualization Tool

A flexible command-line tool for visualizing Hamilton DAGs from any pipeline module.
Supports custom configurations, output formats, and target variables.

Usage:
    python visualize_hamilton_dag.py --module hamilton_doi_pipeline --output dag.png
    python visualize_hamilton_dag.py --module my_pipeline --config config.json --final-vars var1,var2
    python visualize_hamilton_dag.py --module pipeline --list-functions
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import importlib.util
from dotenv import load_dotenv

# Ensure repo root is on sys.path so `import dataflows...` works
# Prefer REPO_ROOT from .env; fall back to two levels up from this file
load_dotenv()
REPO_ROOT = os.getenv('REPO_ROOT') or str(Path(__file__).resolve().parents[2])
sys.path.insert(0, REPO_ROOT)
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "dataflows/dag_viz", "hamilton_viz_config.json")


def load_config_file(config_path: str, pipeline_key: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from JSON file.

    Supports two formats:
    1. Direct config: {"key": "value", ...}
    2. Pipeline-based config: {"pipeline_name": {"module": "...", "config": {...}}}

    Args:
        config_path: Path to JSON configuration file
        pipeline_key: Key for specific pipeline config (if using pipeline-based format)

    Returns:
        Configuration dictionary for Hamilton driver
    """
    try:
        with open(config_path, 'r') as f:
            full_config = json.load(f)
        print(f"✅ Loaded configuration from {config_path}")

        if pipeline_key:
            # Pipeline-based format - extract specific pipeline config
            if pipeline_key in full_config:
                pipeline_config = full_config[pipeline_key]
                print(f"✅ Using pipeline config: {pipeline_key}")
                print(f"   Description: {pipeline_config.get('description', 'No description')}")
                return pipeline_config
            else:
                # Check if it's nested (like your current structure)
                for category, pipelines in full_config.items():
                    if isinstance(pipelines, dict) and pipeline_key in pipelines:
                        pipeline_config = pipelines[pipeline_key]
                        print(f"✅ Using pipeline config: {pipeline_key} (from category: {category})")
                        print(f"   Description: {pipeline_config.get('description', 'No description')}")
                        return pipeline_config

                print(f"❌ Pipeline '{pipeline_key}' not found in config file")
                print(f"Available pipelines:")
                list_available_pipelines(full_config)
                sys.exit(1)
        else:
            # Direct config format or return full config for inspection
            return full_config

    except Exception as e:
        print(f"❌ Error loading config file {config_path}: {e}")
        sys.exit(1)


def list_available_pipelines(config: Dict[str, Any]) -> None:
    """
    List all available pipeline configurations.

    Args:
        config: Full configuration dictionary
    """
    print("\n📋 Available pipeline configurations:")
    print("=" * 50)

    found_pipelines = False

    # Check for direct pipeline configs (flat structure)
    for key, value in config.items():
        if isinstance(value, dict) and 'module' in value:
            print(f"  - {key}: {value.get('description', 'No description')}")
            found_pipelines = True

    # Check for nested pipeline configs (categorized structure)
    for category, pipelines in config.items():
        if isinstance(pipelines, dict):
            category_has_pipelines = False
            for pipeline_key, pipeline_config in pipelines.items():
                if isinstance(pipeline_config, dict) and 'module' in pipeline_config:
                    if not category_has_pipelines:
                        print(f"\n  📁 {category}:")
                        category_has_pipelines = True
                        found_pipelines = True
                    print(f"    - {pipeline_key}: {pipeline_config.get('description', 'No description')}")

    if not found_pipelines:
        print("  No pipeline configurations found in the config file")

    print("=" * 50)


def load_pipeline_module(module_path: str):
    """
    Dynamically load a Python module from file path or module name.
    
    Args:
        module_path: Path to .py file or module name
        
    Returns:
        Loaded module object
    """
    try:
        if module_path.endswith('.py'):
            # Load from file path
            module_name = Path(module_path).stem
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            print(f"✅ Loaded module from file: {module_path}")
        else:
            # Load from module name
            module = importlib.import_module(module_path)
            print(f"✅ Loaded module: {module_path}")
        
        return module
    except Exception as e:
        print(f"❌ Error loading module {module_path}: {e}")
        sys.exit(1)


def create_hamilton_driver(module, config: Dict[str, Any]):
    """
    Create Hamilton driver from module and configuration.
    
    Args:
        module: Loaded pipeline module
        config: Configuration dictionary
        
    Returns:
        Hamilton driver instance
    """
    try:
        from hamilton import driver
        
        dr = driver.Builder().with_modules(module).with_config(config).build()
        print(f"✅ Created Hamilton driver with {len(dr.list_available_variables())} functions")
        return dr
    except Exception as e:
        print(f"❌ Error creating Hamilton driver: {e}")
        sys.exit(1)


def list_available_functions(dr):
    """
    List all available functions in the Hamilton DAG.

    Args:
        dr: Hamilton driver instance
    """
    try:
        # Get function names as strings and sort them
        functions = dr.list_available_variables()
        if functions and hasattr(functions[0], 'name'):
            # If these are HamiltonNode objects, extract names
            function_names = sorted([func.name for func in functions])
        else:
            # If these are already strings, sort directly
            function_names = sorted(functions)
    except Exception as e:
        # Fallback: don't sort if there are issues
        print(f"   ⚠️  Sorting failed ({e}), listing without sorting")
        functions = dr.list_available_variables()
        if functions and hasattr(functions[0], 'name'):
            function_names = [func.name for func in functions]
        else:
            function_names = functions

    print(f"\n📋 Available functions in the DAG ({len(function_names)} total):")
    print("=" * 60)

    for func_name in function_names:
        print(f"  - {func_name}")

    print("=" * 60)
    print(f"Total: {len(function_names)} functions")


def display_full_dataflow(dr, output_file: Optional[str] = None):
    """
    Display the full dataflow using Hamilton's recommended dr.display_all_functions().

    Args:
        dr: Hamilton driver instance
        output_file: Optional output file path
    """
    try:
        print(f"🎨 Creating full dataflow visualization using dr.display_all_functions()")
        if output_file:
            print(f"   Output: {output_file}")

        # Use Hamilton's recommended approach
        dr.display_all_functions(
            output_file_path=output_file,
            show_legend=True,
            orient="TB"  # Top to bottom
        )

        if output_file and Path(output_file).exists():
            file_size = Path(output_file).stat().st_size / 1024  # KB
            print(f"✅ Full dataflow visualization created!")
            print(f"   File: {output_file} ({file_size:.1f} KB)")
        else:
            print(f"✅ Full dataflow visualization displayed!")

    except Exception as e:
        print(f"❌ Error creating full dataflow visualization: {e}")


def display_lineage_analysis(dr, node_names: List[str], analysis_type: str, output_file: Optional[str] = None):
    """
    Display lineage analysis using Hamilton's lineage functions.

    Args:
        dr: Hamilton driver instance
        node_names: List of node names to analyze
        analysis_type: Type of analysis ('upstream', 'downstream', 'between')
        output_file: Optional output file path
    """
    try:
        print(f"🔍 Creating {analysis_type} lineage analysis for: {node_names}")
        if output_file:
            print(f"   Output: {output_file}")

        if analysis_type == "upstream":
            # Use *node_names to unpack the list as individual arguments
            dr.display_upstream_of(*node_names, output_file_path=output_file)
        elif analysis_type == "downstream":
            # Use *node_names to unpack the list as individual arguments
            dr.display_downstream_of(*node_names, output_file_path=output_file)
        elif analysis_type == "between" and len(node_names) == 2:
            dr.visualize_path_between(node_names[0], node_names[1], output_file_path=output_file)
        else:
            raise ValueError(f"Invalid analysis_type: {analysis_type} or incorrect number of nodes for 'between'")

        if output_file and Path(output_file).exists():
            file_size = Path(output_file).stat().st_size / 1024  # KB
            print(f"✅ {analysis_type.title()} lineage visualization created!")
            print(f"   File: {output_file} ({file_size:.1f} KB)")
        else:
            print(f"✅ {analysis_type.title()} lineage visualization displayed!")

    except Exception as e:
        print(f"❌ Error creating {analysis_type} lineage visualization: {e}")


def visualize_dag(
    dr,
    final_vars: List[str],
    output_file: str,
    format: str = "png",
    size: Optional[str] = None,
    show_legend: bool = True,
    orient: str = "TB"
):
    """
    Create DAG visualization.

    Args:
        dr: Hamilton driver instance
        final_vars: List of target variables to visualize
        output_file: Output file path
        format: Output format (png, svg, pdf, etc.)
        size: Graph size as "width,height"
        show_legend: Whether to show legend
        orient: Graph orientation (TB=top-bottom, LR=left-right)
    """
    try:
        # Prepare render kwargs - Hamilton supports format and basic graphviz options
        render_kwargs = {
            "format": format,
        }

        # Additional graphviz attributes (may not be supported by all backends)
        if size:
            render_kwargs["size"] = size

        render_kwargs.update({
            "graph_attr": {
                "rankdir": orient,
                "splines": "ortho",
                "nodesep": "0.5",
                "ranksep": "1.0"
            },
            "node_attr": {
                "shape": "box",
                "style": "rounded,filled",
                "fontname": "Arial",
                "fontsize": "10"
            },
            "edge_attr": {
                "fontname": "Arial",
                "fontsize": "8"
            }
        })
        
        print(f"🎨 Creating visualization for variables: {final_vars}")
        print(f"   Output: {output_file}")
        print(f"   Format: {format}")
        if size:
            print(f"   Size: {size}")

        # Try with full render_kwargs first, fallback to minimal if it fails
        try:
            dr.visualize_execution(
                final_vars=final_vars,
                output_file_path=output_file,
                render_kwargs=render_kwargs,
                bypass_validation=True  # Skip validation for visualization
            )
        except Exception as render_error:
            print(f"   ⚠️  Full render options failed: {render_error}")
            print(f"   🔄 Trying with minimal render options...")

            # Fallback to minimal render kwargs (no size, no styling)
            minimal_kwargs = {"format": format}

            dr.visualize_execution(
                final_vars=final_vars,
                output_file_path=output_file,
                render_kwargs=minimal_kwargs,
                bypass_validation=True  # Skip validation for visualization
            )
        
        if Path(output_file).exists():
            file_size = Path(output_file).stat().st_size / 1024  # KB
            print(f"✅ Visualization created successfully!")
            print(f"   File: {output_file} ({file_size:.1f} KB)")
        else:
            print(f"⚠️  Visualization may have failed - output file not found")
            
    except Exception as e:
        print(f"❌ Error creating visualization: {e}")
        sys.exit(1)


def main():
    """Main CLI interface for Hamilton DAG visualization."""
    parser = argparse.ArgumentParser(
        description="Visualize Hamilton DAGs with flexible configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Hamilton's recommended full dataflow visualization
  python visualize_hamilton_dag.py --config hamilton_viz_config.json --pipeline hamilton_doi_pipeline --display-all

  # List available pipelines
  python visualize_hamilton_dag.py --config hamilton_viz_config.json --list-pipelines

  # Lineage analysis - upstream dependencies
  python visualize_hamilton_dag.py --config hamilton_viz_config.json --pipeline hamilton_doi_pipeline --upstream "process_single_dataset"

  # Lineage analysis - downstream dependencies
  python visualize_hamilton_dag.py --config hamilton_viz_config.json --pipeline hamilton_doi_pipeline --downstream "dataset_metadata"

  # Path between two nodes
  python visualize_hamilton_dag.py --config hamilton_viz_config.json --pipeline hamilton_doi_pipeline --path-between "dataset_metadata,store_individual_datasets"

  # Custom execution visualization (legacy approach)
  python visualize_hamilton_dag.py --config hamilton_viz_config.json --pipeline hamilton_doi_pipeline --final-vars "store_individual_datasets" --output custom.svg --format svg

  # Traditional module-based approach
  python visualize_hamilton_dag.py --module hamilton_doi_pipeline --config-json '{"database_path": "./test.db"}' --display-all

  # List functions in a pipeline
  python visualize_hamilton_dag.py --config hamilton_viz_config.json --pipeline hamilton_doi_pipeline --list-functions
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--module", "-m",
        help="Python module name or path to .py file containing Hamilton functions"
    )

    # Configuration
    parser.add_argument(
        "--config", "-c",
        default=DEFAULT_CONFIG_PATH,
        help=f"Path to JSON configuration file for Hamilton driver (default: {DEFAULT_CONFIG_PATH})"
    )

    parser.add_argument(
        "--pipeline", "-p",
        help="Pipeline key from config file (e.g., 'hamilton_doi_pipeline')"
    )

    parser.add_argument(
        "--config-json",
        help="JSON configuration string (alternative to --config file)"
    )

    parser.add_argument(
        "--list-pipelines",
        action="store_true",
        help="List available pipeline configurations from config file and exit"
    )
    
    # Visualization options
    parser.add_argument(
        "--output", "-o",
        default="hamilton_dag.png",
        help="Output file path (default: hamilton_dag.png)"
    )
    
    parser.add_argument(
        "--final-vars",
        help="Comma-separated list of target variables to visualize (default: all leaf nodes)"
    )
    
    parser.add_argument(
        "--format", "-f",
        default="png",
        choices=["png", "svg", "pdf", "dot", "jpg"],
        help="Output format (default: png)"
    )
    
    parser.add_argument(
        "--size",
        help="Graph size as 'width,height' in inches (e.g., '20,16')"
    )
    
    parser.add_argument(
        "--orient",
        choices=["TB", "LR", "BT", "RL"],
        default="TB",
        help="Graph orientation: TB=top-bottom, LR=left-right (default: TB)"
    )
    
    # Utility options
    parser.add_argument(
        "--list-functions", "-l",
        action="store_true",
        help="List all available functions in the DAG and exit"
    )

    # Hamilton's recommended visualization methods
    parser.add_argument(
        "--display-all",
        action="store_true",
        help="Use Hamilton's dr.display_all_functions() to show full dataflow"
    )

    parser.add_argument(
        "--upstream",
        help="Show upstream dependencies of specified nodes (comma-separated)"
    )

    parser.add_argument(
        "--downstream",
        help="Show downstream dependencies of specified nodes (comma-separated)"
    )

    parser.add_argument(
        "--path-between",
        help="Show path between two nodes (format: 'node1,node2')"
    )
    
    args = parser.parse_args()

    # Handle list pipelines request
    if args.list_pipelines:
        if not args.config or not Path(args.config).exists():
            print(f"❌ Config file not found: {args.config}")
            print("   Create a config file or specify a different path with --config")
            sys.exit(1)
        full_config = load_config_file(args.config)
        list_available_pipelines(full_config)
        return

    # Load configuration and determine module
    config = {}
    module_name = args.module
    final_vars = None

    if args.config and args.pipeline:
        # Pipeline-based configuration
        if not Path(args.config).exists():
            print(f"❌ Config file not found: {args.config}")
            print("   Create a config file or specify a different path with --config")
            sys.exit(1)
        pipeline_config = load_config_file(args.config, args.pipeline)

        if 'module' in pipeline_config:
            module_name = pipeline_config['module']
            print(f"✅ Using module from pipeline config: {module_name}")

        if 'config' in pipeline_config:
            config = pipeline_config['config']
            print(f"✅ Using Hamilton config from pipeline")

        if 'default_final_vars' in pipeline_config and not args.final_vars:
            final_vars = pipeline_config['default_final_vars']
            print(f"✅ Using default final vars from pipeline: {final_vars}")

    elif args.config:
        # Direct config file
        if Path(args.config).exists():
            config = load_config_file(args.config)
        else:
            print(f"ℹ️  Config file not found: {args.config}")
            print("   Using empty config - specify module with --module")
            config = {}
    elif args.config_json:
        # JSON string config
        try:
            config = json.loads(args.config_json)
            print("✅ Loaded configuration from JSON string")
        except Exception as e:
            print(f"❌ Error parsing JSON configuration: {e}")
            sys.exit(1)
    else:
        print("ℹ️  No configuration provided, using empty config")

    # Validate module name
    if not module_name:
        print("❌ Module name required. Use --module or --pipeline with config file")
        sys.exit(1)

    # Load pipeline module
    module = load_pipeline_module(module_name)
    
    # Create Hamilton driver
    dr = create_hamilton_driver(module, config)
    
    # Handle list functions request
    if args.list_functions:
        list_available_functions(dr)
        return

    # Handle Hamilton's recommended visualization methods
    if args.display_all:
        display_full_dataflow(dr, args.output)
        return

    if args.upstream:
        node_names = [node.strip() for node in args.upstream.split(",")]
        display_lineage_analysis(dr, node_names, "upstream", args.output)
        return

    if args.downstream:
        node_names = [node.strip() for node in args.downstream.split(",")]
        display_lineage_analysis(dr, node_names, "downstream", args.output)
        return

    if args.path_between:
        node_names = [node.strip() for node in args.path_between.split(",")]
        if len(node_names) != 2:
            print("❌ --path-between requires exactly 2 nodes (format: 'node1,node2')")
            sys.exit(1)
        display_lineage_analysis(dr, node_names, "between", args.output)
        return
    
    # Determine final variables
    if args.final_vars:
        final_vars = [var.strip() for var in args.final_vars.split(",")]
        print(f"🎯 Target variables (from CLI): {final_vars}")
    elif final_vars:
        # Use pipeline defaults (already set above)
        print(f"🎯 Target variables (from pipeline config): {final_vars}")
    else:
        # Use all available variables (Hamilton will determine dependencies)
        all_vars = dr.list_available_variables()
        # Try to find reasonable leaf nodes or use a subset
        final_vars = [var for var in all_vars if not var.name.startswith("_")][:5]  # Limit to first 5
        print(f"🎯 Auto-selected target variables: {final_vars}")
        print("   💡 Use --final-vars to specify custom targets or --pipeline for defaults")
    
    # Create visualization
    visualize_dag(
        dr=dr,
        final_vars=final_vars,
        output_file=args.output,
        format=args.format,
        size=args.size,
        orient=args.orient
    )


if __name__ == "__main__":
    main()
