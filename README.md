# MMDS Project

Deep learning and ensemble forecasting pipeline for Bengaluru ride-hailing demand, supply, and cab reallocation.

## Project Structure

- `data/raw/`
  - Original source datasets.
- `data/processed/`
  - Cleaned and feature-ready datasets.
- `notebooks/`
  - Main and supporting notebooks.
  - Main notebook: `MMDS_Project.ipynb`.
- `scripts/`
  - Python scripts for modeling and utilities.
- `outputs/forecasts/`
  - Forecast result artifacts.
- `docs/`
  - Report and presentation.
- `archive/`
  - Old/generated/experimental files kept for reference.

## Main File

Use `notebooks/MMDS_Project.ipynb` as the primary project notebook.

## Quick Start

1. Open the notebook `notebooks/MMDS_Project.ipynb`.
2. Ensure required Python packages are installed in your environment.
3. Use datasets from `data/raw/` and `data/processed/`.

## Notes

- Generated training logs and scratch artifacts are managed through `.gitignore`.
- Keep only stable outputs in `outputs/forecasts/` for clean version control.
