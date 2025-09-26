"""
LightGBM model trainer for CPU-only execution
"""
import logging
import time
from typing import Any, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class LightGBMTrainer:
    """LightGBM trainer optimized for CPU"""

    def __init__(self, task_type: str = "classification"):
        self.task_type = task_type
        self.model = None
        self.training_history = []
        self.best_iteration = None

    def get_default_params(self) -> dict[str, Any]:
        """Get default CPU-optimized parameters"""
        base_params = {
            "device_type": "cpu",
            "num_threads": -1,  # Use all CPU cores
            "verbosity": -1,
            "seed": 42,
            "boosting_type": "gbdt",
            "num_leaves": 31,
            "learning_rate": 0.05,
            "feature_fraction": 0.9,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "min_child_samples": 20,
            "n_estimators": 1000,
        }

        if self.task_type == "classification":
            base_params.update({"objective": "binary", "metric": "binary_logloss", "is_unbalance": True})
        else:
            base_params.update({"objective": "regression", "metric": "rmse"})

        return base_params

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        params: Optional[dict[str, Any]] = None,
        early_stopping_rounds: int = 50,
    ) -> dict[str, Any]:
        """Train LightGBM model"""

        start_time = time.time()

        # Use provided params or defaults
        train_params = params or self.get_default_params()

        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")

        # Callbacks for tracking
        callbacks = [lgb.log_evaluation(period=50), lgb.record_evaluation(self.training_history)]

        if X_val is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds))

        # Train model
        self.model = lgb.train(train_params, train_data, valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks)

        training_time = time.time() - start_time

        # Get best iteration
        if hasattr(self.model, "best_iteration"):
            self.best_iteration = self.model.best_iteration

        results = {
            "training_time": training_time,
            "best_iteration": self.best_iteration,
            "n_features": X_train.shape[1],
            "training_history": self.training_history,
        }

        logger.info(f"LightGBM training completed in {training_time:.2f}s")

        return results

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        predictions = self.model.predict(X, num_iteration=self.best_iteration or self.model.best_iteration)

        if self.task_type == "classification":
            return (predictions > 0.5).astype(int)
        return predictions

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")

        predictions = self.model.predict(X, num_iteration=self.best_iteration or self.model.best_iteration)

        if self.task_type == "classification":
            return np.column_stack([1 - predictions, predictions])
        return predictions.reshape(-1, 1)

    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.model is None:
            return pd.DataFrame()

        importance = self.model.feature_importance(importance_type="gain")
        feature_names = [f"feature_{i}" for i in range(len(importance))]

        return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values("importance", ascending=False)

    def save(self, path: str):
        """Save model to file"""
        if self.model:
            self.model.save_model(path)
            logger.info(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from file"""
        self.model = lgb.Booster(model_file=path)
        logger.info(f"Model loaded from {path}")
