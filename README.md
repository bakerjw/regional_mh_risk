# regional_mh_risk
[![DOI](https://zenodo.org/badge/878637905.svg)](https://doi.org/10.5281/zenodo.13994340)

This project contains Python functions to run a regional multi-hazard risk assessment under climate change. This code recreates figures
from Mongold, E., and Baker, J. W. (2025). "Quantifying climate change risk through natural hazard losses to inform adaptation action."
*Climatic Change*, 178(4), 82. https://doi.org/10.1007/s10584-025-03927-2

The building inventory values are set to their average for privacy, so some results will not align exactly with those in the paper.

## Overview

This package implements a multi-hazard risk assessment framework that evaluates the impacts of climate change on regional building inventories. The methodology integrates:

1. **Earthquake hazard** (including liquefaction susceptibility)
2. **Coastal flooding** (with sea level rise scenarios)
3. **Tsunami** inundation modeling

The framework supports analysis of adaptation strategies including building elevation, retreat, and seismic retrofitting.

## Installation

### Environment Setup
To re-create the conda environment used for this project:
```bash
conda env create -f environment.yml
conda activate mh_env
```

### Package Installation
Navigate to the package directory and install:
```bash
cd regional_mh_risk/
python setup.py install
```

## Required Data Downloads

Before running the analysis, download the following datasets:

1. **Tsunami hazard data**: Alameda_tsunami_tifs from [CA Geological Survey](https://www.conservation.ca.gov/cgs/tsunami/reports)
2. **Coastal flood data**: coastal_flood_rasters from [Adapting to Rising Tides](https://explorer.adaptingtorisingtides.org/download)

## Workflow

The complete analysis follows this sequential workflow:

### 0. Building Inventory Setup
```bash
python example/building_inventory.py
```
Sets up building inventory by filling tax assessor data with NSI (National Structure Inventory) data.

### 1. Earthquake Ground Motion Simulation
```bash
jupyter notebook example/ground_motions/run_pypsha_clean.ipynb
```
Generates probabilistic seismic hazard analysis using pypsha.

### 2. Liquefaction Analysis*
```bash
python example/setup_run.py
python example/flex_liq_run.py  # Or submit via slurm_array.sh for HPC
python example/earthquake_postprocess.py
```
Computes liquefaction potential using Boulanger & Idriss or Moss methodologies.

### 3. Coastal and Tsunami Hazards
```bash
python example/cf_run.py    # Coastal flooding
python example/tsu_run.py   # Tsunami
```

### 4. Risk Assessment
```bash
python example/risk_run.py
```
Generates risk metric figures and loss calculations.

### 5. Adaptation Analysis
```bash
python example/adapt_raise.py     # Building elevation strategy
python example/adapt_retreat.py   # Managed retreat strategy  
python example/adapt_retrofit.py* # Seismic retrofit strategy
```
These can be run in parallel after hazard analysis completion.

### 6. Adaptation Comparison
```bash
python example/adapt_compare.py
```
Post-processes and compares adaptation strategies.

*\*Computationally intensive steps designed for HPC clusters - may require significant resources for full dataset analysis.*

## Computational Requirements

- **Local execution**: Suitable for testing and small-scale analysis
- **HPC execution**: Recommended for full dataset analysis, especially liquefaction modeling
- **SLURM integration**: `slurm_array.sh` provided for cluster job submission
- **Memory requirements**: 4GB+ per core for liquefaction calculations

## Citation

If you use this code, please cite:

Mongold, E., and Baker, J. W. (2025). "Quantifying climate change risk through natural hazard losses to inform adaptation action." *Climatic Change*, 178(4), 82. https://doi.org/10.1007/s10584-025-03927-2
