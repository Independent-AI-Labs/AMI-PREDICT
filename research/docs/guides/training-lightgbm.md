# Train with LightGBM/CatBoost

Lightweight gradient boosting models are strong baselines for many datasets.

Steps:
- Prepare features/labels via feature pipeline
- Use trainers in `src/ml/models/{lightgbm_trainer.py, catboost_trainer.py}`
- Evaluate with backtesting utilities

Example (LightGBM):

```python
from src.ml.feature_engineering import build_basic_features
from src.ml.models.lightgbm_trainer import train_lgbm

X, y = build_basic_features(df)
model, metrics = train_lgbm(X, y)
```

