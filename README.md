# dssat-py-batchrunner
A Python wrapper / batch-runner for executing DSSAT (CSM) simulations at scale, designed for gridded, multi-scenario workflows (climate ensembles, RCPs, rotations), with automated weather preparation and output post-processing.
# Cite 
Attia, A., Woli, P., Long, C.R., Rouquette, F.M. Jr, Smith, G.R., Datta, A., Feike., T., Rajan, N. 2025. Unlocking climate resilience by exploring the mitigation potential of improved rotation with cover cropping. J. of Environ. Manage. 391-126352. doi.org/10.1016/j.jenvman.2025.126352 
Attia, A., Woli, P., Long, C.R., Rouquette, F.M. Jr, Smith, G.R., Ibrahim, A.M.H. 2025. Mapping spatial zones of climate vulnerability and adaptive potential for major crops in the Texas High Plains. In Review- Model. Earth Syst. Env. doi.org/10.21203/rs.3.rs-6864209/v1
# dssat-py-batchrunner

A Python wrapper / batch-runner for executing DSSAT (CSM) simulations at scale, designed for gridded, multi-scenario workflows (climate ensembles, RCPs, rotations), with automated weather preparation and output post-processing.

> **Portfolio / showcase repository**
> This repository documents the pipeline and demonstrates the implementation approach used in a peer-reviewed study.  
> Full runnable assets (DSSAT binaries, proprietary inputs, large climate datasets, and institution-specific file paths) are not distributed here due to licensing and data-size constraints.

## What this project does

- **Reads gridded climate NetCDF (e.g., MACA)**, extracts point time series, converts units, and assembles DSSAT-ready daily weather inputs.
- **Writes DSSAT `.WTH` files** including **year-specific CO₂** concentrations for scenario-aware CO₂ fertilization.
- **Generates DSSAT batch files (`DSSBatch.v48`)** and orchestrates DSSAT runs via the DSSAT command-line executable.
- **Parses DSSAT outputs** (e.g., `Summary.OUT` and other `.OUT` files) into analysis-ready tables.
- Includes utilities for **GHG accounting** (e.g., N₂O emissions factors) and rotation-level aggregation.

## Scientific context (example application)

This workflow was used to run DSSAT v4.8 in a gridded spatio-temporal application (12-km grid) from 1990–2099 across multiple crop rotations and future scenarios, using a Python script to fetch/format MACA NetCDF weather at each grid point and post-process outputs.

- DSSAT v4.8 model application: see Methods in the associated paper.
- Grid execution and Python automation: see rotation analysis section in the paper.

## Key components (modules / scripts)

> Names below reflect the core building blocks in the current implementation.

### Weather pipeline
- **NetCDF → point time series → unit conversion → RH calculation**
  - Example logic: `Process_netCDF.py` (xarray/pandas; unit conversions; time slicing; nearest-point selection).
- **Write DSSAT `.WTH`**
  - `dssat_wthfunction.py` writes `.WTH` and maps CO₂ by year; includes basic TMAX/TMIN consistency checks.

### DSSAT run orchestration
- **Batch file generation + DSSAT CLI execution**
  - `Run_dssat_T_modified.py` and scenario variants (e.g., RCP45) generate `DSSBatch.v48` and run:
    `DSCSM048.EXE Q DSSBatch.v48`

### Output parsing
- **Summary.OUT parsing**
  - `readDssatSummary.py` reads `Summary.OUT` into a tidy DataFrame, including DSSAT date conversion.
- **Generic `.OUT` parsing for selected fields**
  - `read_dssat_Arno_workingV.py` parses DSSAT output sections by run/treatment and extracts requested columns.

### Emissions utility
- `N2Ofunction.py` implements direct + indirect N₂O estimation from N inputs and leaching, returning CO₂e if requested.

## How to use (high-level)

Because DSSAT binaries and large datasets are not included, this repo provides a **reference implementation** and recommended interfaces:

1. Prepare climate inputs:
   - Select GCM / scenario / years
   - Extract point series from NetCDF
   - Convert units and compute derived variables (e.g., RH)
2. Write `.WTH` files for each station/grid point (include CO₂ series)
3. For each experiment/treatment:
   - Patch DSSAT experiment templates (e.g., rotation sequences)
   - Write `DSSBatch.v48`
   - Execute DSSAT via CLI
4. Parse outputs:
   - Summary and detailed OUT files
   - Aggregate by period (historical / mid-century / end-century) and by scenario

## What’s intentionally not included
- DSSAT executables and model binaries (install separately).
- Large climate datasets (NetCDF) and institution-specific directories.
- Full experiment templates and calibration assets tied to specific projects.

## Citation
If you use the concepts or structure, please cite the associated publication and link back to this repository.

## Skills demonstrated (for hiring / technical review)
- Scientific workflow engineering (reproducible pipelines, batch orchestration)
- NetCDF/xarray processing, unit conversion, derived-variable computation
- DSSAT automation via CLI (batch files, input templating, output parsing)
- Data engineering with pandas (cleaning, joins, aggregation across scenarios)
- Domain modeling: crop rotations, CO₂ scenario handling, N₂O accounting

## License
Choose a license that matches your intent:
- “All rights reserved” (if you truly do not want reuse)
- or a standard open-source license (MIT/BSD/Apache) if you do.
