# Feature Engineering

Feature pipelines transform raw input into model-ready features.

- Modules: `src/ml/feature_engineering.py`, `src/ml/microstructure_features.py`
- Common steps: resampling, lag features, rolling stats, microstructure signals

Patterns:
- Pure functions that accept/return dataframes
- Composable pipelines to tailor for each dataset

Tips:
- Start simple (lags, rolling mean/std) then add domain-specific features
- Validate leakage and alignment rigorously

