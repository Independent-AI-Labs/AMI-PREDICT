# Backtesting & Metrics

Evaluate models and strategies with robust metrics and simulation.

- Backtester: `src/ml/backtester.py`
- Metrics: `src/ml/metrics.py`
- Regime detection: `src/ml/regime_detector.py`

Guidelines:
- Split by time; avoid leakage across train/validation/test
- Use walk-forward or rolling windows
- Report multiple metrics (accuracy, precision/recall, drawdown, Sharpe)

