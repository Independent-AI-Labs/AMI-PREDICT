# AMI Predict

Prediction engines and modeling toolkit for time series and event-driven data. AMI Predict provides modular building blocks for data ingestion, feature engineering, model training, backtesting, simulation, and optional execution/dashboard integrations.

- Data providers for historical and live feeds
- Feature pipelines for classical and microstructure signals
- Model trainers for Torch, LightGBM, CatBoost, and scikit-learn
- Backtesting, metrics, and regime detection
- Simulation engine for what-if and stress scenarios
- Optional web dashboard (Next.js) for monitoring and control

## Why AMI Predict

- Modular by design: swap providers, features, models, and metrics
- Scales from local CPU to accelerated hardware (Intel XPU, etc.)
- Repeatable experiments and clean configuration management
- Works for finance and beyond: any time series or event streams

## Core Architecture

- Core: configuration, logging, database utilities (`src/core`)
- Data providers: base + concrete providers (`src/data_providers`)
- Features and models: pipelines and trainers (`src/ml`)
- Backtesting and metrics: evaluation utilities (`src/ml`)
- Simulation: market/data simulation engines (`src/simulation`)
- Trading interfaces: orders, positions, engine hooks (`src/trading`)

See the Quickstart to run an end-to-end pipeline.

