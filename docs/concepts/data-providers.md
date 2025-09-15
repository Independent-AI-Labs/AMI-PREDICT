# Data Providers

Data providers load historical and/or live data into normalized dataframes used across the toolkit.

- Base classes: `src/data_providers/base_provider.py`
- Built-ins: `src/data_providers/binance_provider.py`
- Sync utilities: `src/data_providers/data_sync.py`

Responsibilities:
- Define schema (timestamps, symbols, OHLCV, microstructure fields)
- Handle pagination, retries, and rate limits for remote APIs
- Cache locally (e.g., Parquet) for reproducibility

Extending:
- Subclass the base provider and implement `fetch()` or `stream()`
- Register the provider in configuration for easy selection

