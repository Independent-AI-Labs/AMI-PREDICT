"""
CatBoost model trainer for CPU-only execution
"""
import catboost as cb
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import time
import logging

logger = logging.getLogger(__name__)

class CatBoostTrainer:
    """CatBoost trainer optimized for CPU"""
    
    def __init__(self, task_type: str = 'classification'):
        self.task_type = task_type
        self.model = None
        self.training_history = []
        
    def get_default_params(self) -> Dict[str, Any]:
        """Get default CPU-optimized parameters"""
        base_params = {
            'task_type': 'CPU',
            'thread_count': -1,  # Use all CPU cores
            'verbose': False,
            'random_seed': 42,
            'iterations': 1000,
            'learning_rate': 0.03,
            'depth': 6,
            'l2_leaf_reg': 3,
            'border_count': 128,
            'bagging_temperature': 1
        }
        
        if self.task_type == 'classification':
            base_params.update({
                'loss_function': 'Logloss',
                'eval_metric': 'AUC',
                'auto_class_weights': 'Balanced'
            })
        else:
            base_params.update({
                'loss_function': 'RMSE',
                'eval_metric': 'RMSE'
            })
        
        return base_params
    
    def train(self,
             X_train: np.ndarray,
             y_train: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None,
             params: Optional[Dict[str, Any]] = None,
             cat_features: Optional[List[int]] = None,
             early_stopping_rounds: int = 50) -> Dict[str, Any]:
        """Train CatBoost model"""
        
        start_time = time.time()
        
        # Use provided params or defaults
        train_params = params or self.get_default_params()
        
        # Add early stopping if validation data provided
        if X_val is not None and y_val is not None:
            train_params['early_stopping_rounds'] = early_stopping_rounds
        
        # Create Pool objects
        train_pool = cb.Pool(
            data=X_train,
            label=y_train,
            cat_features=cat_features
        )
        
        eval_pool = None
        if X_val is not None and y_val is not None:
            eval_pool = cb.Pool(
                data=X_val,
                label=y_val,
                cat_features=cat_features
            )
        
        # Initialize model
        if self.task_type == 'classification':
            self.model = cb.CatBoostClassifier(**train_params)
        else:
            self.model = cb.CatBoostRegressor(**train_params)
        
        # Train model
        self.model.fit(
            train_pool,
            eval_set=eval_pool,
            plot=False,
            use_best_model=True if eval_pool else False
        )
        
        # Get training history
        if hasattr(self.model, 'evals_result_'):
            self.training_history = self.model.evals_result_
        
        training_time = time.time() - start_time
        
        results = {
            'training_time': training_time,
            'best_iteration': self.model.best_iteration_ if hasattr(self.model, 'best_iteration_') else None,
            'n_features': X_train.shape[1],
            'training_history': self.training_history
        }
        
        logger.info(f"CatBoost training completed in {training_time:.2f}s")
        
        return results
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        predictions = self.model.predict(X)
        
        if self.task_type == 'classification' and len(predictions.shape) == 1:
            return predictions.astype(int)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        if self.task_type == 'classification':
            return self.model.predict_proba(X)
        else:
            # For regression, return predictions as single column
            return self.model.predict(X).reshape(-1, 1)
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Get feature importance scores"""
        if self.model is None:
            return pd.DataFrame()
        
        importance = self.model.feature_importances_
        feature_names = [f'feature_{i}' for i in range(len(importance))]
        
        return pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
    
    def save(self, path: str):
        """Save model to file"""
        if self.model:
            self.model.save_model(path)
            logger.info(f"Model saved to {path}")
    
    def load(self, path: str):
        """Load model from file"""
        if self.task_type == 'classification':
            self.model = cb.CatBoostClassifier()
        else:
            self.model = cb.CatBoostRegressor()
        
        self.model.load_model(path)
        logger.info(f"Model loaded from {path}")