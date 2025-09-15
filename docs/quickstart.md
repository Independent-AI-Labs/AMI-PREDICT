# Quickstart

This minimal example loads local data, builds features, trains a simple model, and evaluates with a backtest.

Prerequisites:
- Python 3.11+
- Install requirements: `pip install -r requirements.txt`

Example script:

```python
# quickstart_example.py
from pathlib import Path

from src.data_providers.data_sync import DataSync
from src.ml.feature_engineering import build_basic_features
from src.ml.models import train_lightgbm_classifier
from src.ml.backtester import Backtester

base = Path(__file__).resolve().parent
example = base / "data" / "parquet" / "BTC_USDT" / "2020" / "BTC_USDT_2020_08.parquet"

if not example.exists():
    raise SystemExit(f"Example dataset not found: {example}\n"
                     "Fetch or point to a local Parquet file.")

# 1) Load data
df = DataSync.read_parquet_file(example)

# 2) Build features/labels
X, y = build_basic_features(df)

# 3) Train model (LightGBM)
model, metrics = train_lightgbm_classifier(X, y)
print("Train metrics:", metrics)

# 4) Backtest simple strategy
bt = Backtester(model=model)
results = bt.run(df)
print("Backtest summary:", results.summary())
```

Run it:

```bash
python quickstart_example.py
```

Next steps:
- Read Concepts for data providers, features, and model choices
- Try the Guides to add a custom provider or use CatBoost
- Explore acceleration with Intel XPU (optional)

