# Ride-Hailing Demand Forecasting and Driver Reallocation System

A scalable big-data and machine learning framework for forecasting ride demand, predicting supply, and optimizing driver allocation in ride-hailing platforms.

## Overview

This project builds an end-to-end intelligent mobility system that analyzes large-scale ride data and predicts:

- Demand (ride requests)
- Supply (available drivers)
- Demand-supply imbalance (shortage)

The system combines Apache Spark (big data processing) with CatBoost and LightGBM (machine learning models) for accurate forecasting and decision-making.

## Key Features

- Large-scale data processing using Apache Spark
- Time-series forecasting for demand and supply
- Feature engineering (lag, rolling averages, temporal features)
- CatBoost and LightGBM model training
- Shortage prediction using demand minus supply
- Driver reallocation recommendation engine
- Visualization using heatmaps and trend graphs

## System Architecture

The system consists of three major layers:

### 1. Data Processing Layer (Apache Spark)

- Data cleaning and normalization
- Zone-based aggregation
- Feature engineering

### 2. Machine Learning Layer

- LightGBM (baseline model)
- CatBoost (final model)
- Time-series forecasting pipeline

### 3. Decision and Visualization Layer

- Demand trend analysis
- Zone-wise heatmaps
- Driver reallocation recommendations

## Dataset

- Large-scale ride-hailing dataset for Bengaluru
- Millions of trip records
- Key fields:
  - Timestamp
  - Pickup and drop location
  - Ride status
  - Fare
  - Distance

## Methodology

### Feature Engineering

- Time features (hour, day, month)
- Lag features (lag-1, lag-24)
- Rolling averages

### Forecasting

- Models: CatBoost, LightGBM
- Metrics: RMSE, MAE, R2
- CatBoost selected as best-performing model

### Shortage Calculation

```text
shortage = demand - supply
```

### Reallocation Engine

- Identifies surplus and shortage zones
- Matches zones based on proximity
- Recommends optimal driver movements

## Results

- CatBoost achieved the best performance (lowest RMSE)
- Accurate hourly demand and supply forecasts
- Better driver utilization and reduced wait times

The system demonstrates strong real-world applicability for ride-hailing platforms.

## Project Structure

- data/raw: source datasets
- data/processed: cleaned and feature-ready data
- notebooks: analysis and modeling notebooks
- scripts: Python scripts for forecasting logic
- outputs/forecasts: generated forecast outputs
- docs: project report and presentation
- archive: experimental and older artifacts

Main notebook: notebooks/MMDS_Project.ipynb

## How to Run

1. Open notebooks/MMDS_Project.ipynb in Jupyter or VS Code.
2. Install required Python packages used in the notebook and scripts.
3. Use input datasets from data/raw and data/processed.
4. Run the preprocessing, feature engineering, model training, and forecasting cells.
5. Review outputs in outputs/forecasts.

## Future Work

- Real-time streaming with Kafka and Spark Streaming
- Deep learning extensions (LSTM, GNN)
- Multi-city deployment
- Reinforcement learning for allocation
- Cloud deployment (AWS/GCP)
