# CCOM6994_PV_Solar_Project

Using PV datasets to identify spatial correlation between PV installations and their surrounding Geographic and Socio-demographic variables.

## Project Structure

This project follows a structured directory layout to organize code, data, and outputs:

```
.
├── dataflows/      # Data processing workflows and pipelines
├── db/             # Database files and schemas
├── figures/        # Generated visualizations and plots
├── ingest/         # Data ingestion scripts
├── notebooks/      # Jupyter notebooks for analysis
├── utils/          # Utility functions and helper modules
├── pyproject.toml  # Project configuration and dependencies (uv)
├── requirements.txt # Legacy pip requirements file
└── README.md       # This file
```

## Installation

This project uses [uv](https://github.com/astral-sh/uv) for fast, reliable Python package management. We also provide a `requirements.txt` file for legacy pip compatibility.

### Installing uv

First, install uv using one of the following methods:

**On macOS and Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**On Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Using pip:**
```bash
pip install uv
```

For more installation options, see the [official uv documentation](https://github.com/astral-sh/uv#installation).

### Method 1: Using uv project (Recommended)

The recommended way to work with this project is using uv's project management:

```bash
# Install dependencies from pyproject.toml
uv sync

# Activate the project virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Run Jupyter notebook
jupyter notebook
```

### Method 2: Using uv venv with pip

If you prefer to use pip within a uv-managed virtual environment:

```bash
# Create a virtual environment with uv
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# or
.venv\Scripts\activate     # On Windows

# Install dependencies using pip
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook
```

### Method 3: Traditional pip (Legacy)

For systems without uv, you can use traditional pip:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # On macOS/Linux
# or
venv\Scripts\activate     # On Windows

# Install dependencies
pip install -r requirements.txt

# Run Jupyter notebook
jupyter notebook
```

## Usage

Once dependencies are installed, you can:

1. **Run Jupyter notebooks**: Navigate to the `notebooks/` directory and open `.ipynb` files
2. **Process data**: Use scripts in the `ingest/` directory to load and preprocess data
3. **Run workflows**: Execute data processing pipelines from the `dataflows/` directory
4. **Generate visualizations**: Outputs will be saved to the `figures/` directory

## Development

To contribute to this project:

```bash
# Install development dependencies (using uv)
uv sync --all-extras

# Or install dev dependencies with pip
pip install -r requirements.txt pytest black ruff
```

## License

This project is part of CCOM6994 coursework.
