#!/usr/bin/env python
"""
Test script to verify ML framework is working
"""

import numpy as np
import pandas as pd

# Test database
print("Testing Database...")
from database import ExperimentDB

db = ExperimentDB("test_experiments.db")
print("[OK] Database created")

# Test metrics
print("\nTesting Metrics Calculator...")
from metrics import MetricsCalculator

calc = MetricsCalculator()
y_true = np.array([1, 0, 1, 1, 0])
y_pred = np.array([1, 0, 1, 0, 0])
metrics = calc.classification_metrics(y_true, y_pred)
print(f"[OK] Metrics calculated: Accuracy={metrics['accuracy']:.2f}")

# Test feature engineering
print("\nTesting Feature Engineering...")
from feature_engineering import PriceFeatures

dates = pd.date_range(start="2024-01-01", periods=100, freq="1H")
df = pd.DataFrame(
    {
        "timestamp": dates,
        "open": np.random.uniform(100, 110, 100),
        "high": np.random.uniform(110, 120, 100),
        "low": np.random.uniform(90, 100, 100),
        "close": np.random.uniform(95, 115, 100),
        "volume": np.random.uniform(1000, 10000, 100),
    }
)

pf = PriceFeatures()
df = pf.compute(df)
print(f"[OK] Price features computed: {len([c for c in df.columns if 'returns' in c])} features added")

# Test experiment creation
print("\nTesting Experiment Manager...")
from experiment_manager import ExperimentRunner

runner = ExperimentRunner("test_experiments.db")
exp_id = runner.create_experiment(
    name="test_experiment",
    model_type="lightgbm",
    symbol="BTCUSDT",
    date_config={"train_start": "2024-01-01", "train_end": "2024-01-20", "val_end": "2024-01-25", "test_end": "2024-01-30"},
)
print(f"[OK] Experiment created with ID: {exp_id}")

# Test backtester
print("\nTesting Backtester...")
from backtester import Backtester

backtester = Backtester(initial_capital=10000)
data = pd.DataFrame({"close": np.random.uniform(100, 110, 50), "timestamp": pd.date_range(start="2024-01-01", periods=50, freq="1H")})
signals = pd.Series(np.random.choice([0, 1, -1], size=50))

results = backtester.run(data, signals)
print(f"[OK] Backtest completed: Final equity=${results['final_equity']:.2f}")

print("\n" + "=" * 50)
print("ALL TESTS PASSED! Framework is ready to use.")
print("=" * 50)

print("\nTo run experiments, use:")
print("  python run_experiments.py train --name my_exp --model lightgbm --symbol BTCUSDT")
print("  python run_experiments.py benchmark --symbol BTCUSDT")
print("  python run_experiments.py compare --model catboost --symbol ETHUSDT")
