# Models

AMI Predict supports multiple model families:

- Deep Learning (PyTorch): temporal CNNs/Transformers (`src/ml/advanced_models.py`)
- Gradient Boosting: LightGBM, CatBoost (`src/ml/models/*.py`)
- Classical baselines: sklean-compatible models via common interfaces

Acceleration:
- Optional Intel XPU acceleration for PyTorch via `src/ml/xpu_optimizer.py`
- CPU remains the default; acceleration is additive

Training helpers:
- Utilities in `src/ml/models.py` and trainers under `src/ml/models/`
- Experiment tracking in `src/ml/experiment_manager.py`

