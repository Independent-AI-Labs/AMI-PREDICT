"""
Experiment manager for training and comparing models
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

from database import ExperimentDB
from metrics import MetricsCalculator
from feature_engineering import PriceFeatures, VolumeFeatures, MomentumFeatures, MovingAverages
from models.lightgbm_trainer import LightGBMTrainer
from models.catboost_trainer import CatBoostTrainer

logger = logging.getLogger(__name__)

class DataLoader:
    """Load and prepare data for experiments"""
    
    def __init__(self, data_path: str = "data/"):
        self.data_path = Path(data_path)
    
    def load_price_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load price data for a symbol and date range"""
        # For now, simulate loading - replace with actual data loading
        dates = pd.date_range(start=start_date, end=end_date, freq='1H')
        n = len(dates)
        
        # Simulate OHLCV data
        np.random.seed(42)
        close = 50000 * (1 + np.random.randn(n).cumsum() * 0.001)
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': close * (1 + np.random.randn(n) * 0.001),
            'high': close * (1 + np.abs(np.random.randn(n)) * 0.002),
            'low': close * (1 - np.abs(np.random.randn(n)) * 0.002),
            'close': close,
            'volume': np.random.uniform(1000, 10000, n)
        })
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering"""
        # Apply each feature extractor
        df = PriceFeatures().compute(df)
        df = VolumeFeatures().compute(df)
        df = MomentumFeatures().compute(df)
        df = MovingAverages().compute(df)
        
        # Create target (next period return)
        df['target'] = df['returns'].shift(-1)
        
        # Drop NaN values
        df = df.dropna()
        
        return df
    
    def split_data(self, df: pd.DataFrame, 
                  train_end: str, 
                  val_end: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train, validation, and test sets"""
        train = df[df['timestamp'] <= train_end]
        val = df[(df['timestamp'] > train_end) & (df['timestamp'] <= val_end)]
        test = df[df['timestamp'] > val_end]
        
        return train, val, test

class ExperimentRunner:
    """Run experiments and track results"""
    
    def __init__(self, db_path: str = "experiments.db"):
        self.db = ExperimentDB(db_path)
        self.data_loader = DataLoader()
        self.metrics_calc = MetricsCalculator()
    
    def create_experiment(self, 
                         name: str,
                         model_type: str,
                         symbol: str,
                         date_config: Dict[str, str],
                         hyperparameters: Optional[Dict] = None) -> int:
        """Create a new experiment"""
        
        config = {
            'training_start': date_config['train_start'],
            'training_end': date_config['train_end'],
            'validation_start': date_config['train_end'],
            'validation_end': date_config['val_end'],
            'test_start': date_config['val_end'],
            'test_end': date_config['test_end'],
            'symbol': symbol,
            'timeframe': '1H',
            'hyperparameters': hyperparameters or {}
        }
        
        experiment_id = self.db.create_experiment(name, model_type, config)
        logger.info(f"Created experiment {experiment_id}: {name}")
        
        return experiment_id
    
    def run_experiment(self, experiment_id: int) -> Dict[str, Any]:
        """Run a complete experiment"""
        
        # Get experiment config
        exp = self.db.get_experiment(experiment_id)
        if not exp:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        self.db.update_experiment_status(experiment_id, 'running')
        
        try:
            # Load and prepare data
            logger.info(f"Loading data for {exp['symbol']}")
            df = self.data_loader.load_price_data(
                exp['symbol'],
                exp['training_start'],
                exp['test_end']
            )
            
            # Engineer features
            df = self.data_loader.prepare_features(df)
            
            # Split data
            train_df, val_df, test_df = self.data_loader.split_data(
                df,
                exp['training_end'],
                exp['validation_end']
            )
            
            # Prepare feature matrices
            feature_cols = [col for col in df.columns 
                          if col not in ['timestamp', 'target', 'open', 'high', 'low', 'close', 'volume']]
            
            X_train = train_df[feature_cols].values
            y_train = (train_df['target'] > 0).astype(int).values
            
            X_val = val_df[feature_cols].values
            y_val = (val_df['target'] > 0).astype(int).values
            
            X_test = test_df[feature_cols].values
            y_test = (test_df['target'] > 0).astype(int).values
            
            # Train model
            logger.info(f"Training {exp['model_type']} model")
            results = self._train_model(
                exp['model_type'],
                X_train, y_train,
                X_val, y_val,
                exp.get('hyperparameters')
            )
            
            # Evaluate on all sets
            trainer = results['trainer']
            
            # Training set metrics
            train_pred = trainer.predict(X_train)
            train_metrics = self.metrics_calc.classification_metrics(y_train, train_pred)
            self.db.save_results(experiment_id, train_metrics, 'train')
            
            # Validation set metrics
            val_pred = trainer.predict(X_val)
            val_metrics = self.metrics_calc.classification_metrics(y_val, val_pred)
            self.db.save_results(experiment_id, val_metrics, 'validation')
            
            # Test set metrics
            test_pred = trainer.predict(X_test)
            test_proba = trainer.predict_proba(X_test)
            test_metrics = self.metrics_calc.classification_metrics(y_test, test_pred, test_proba)
            self.db.save_results(experiment_id, test_metrics, 'test')
            
            # Save predictions
            predictions_df = pd.DataFrame({
                'timestamp': test_df['timestamp'].values,
                'symbol': exp['symbol'],
                'prediction': test_pred,
                'actual': y_test
            })
            self.db.save_predictions(experiment_id, predictions_df)
            
            # Save model
            model_path = f"models/exp_{experiment_id}_{exp['model_type']}.pkl"
            Path("models").mkdir(exist_ok=True)
            trainer.save(model_path)
            self.db.save_model_artifact(experiment_id, 'model', model_path)
            
            self.db.update_experiment_status(experiment_id, 'completed')
            
            logger.info(f"Experiment {experiment_id} completed successfully")
            
            return {
                'experiment_id': experiment_id,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'training_time': results.get('training_time', 0)
            }
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {str(e)}")
            self.db.update_experiment_status(experiment_id, 'failed')
            raise
    
    def _train_model(self, model_type: str, 
                    X_train: np.ndarray, y_train: np.ndarray,
                    X_val: np.ndarray, y_val: np.ndarray,
                    hyperparameters: Optional[Dict] = None) -> Dict[str, Any]:
        """Train a specific model type"""
        
        if model_type == 'lightgbm':
            trainer = LightGBMTrainer(task_type='classification')
            results = trainer.train(X_train, y_train, X_val, y_val, hyperparameters)
        elif model_type == 'catboost':
            trainer = CatBoostTrainer(task_type='classification')
            results = trainer.train(X_train, y_train, X_val, y_val, hyperparameters)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        results['trainer'] = trainer
        return results
    
    def compare_experiments(self, experiment_ids: List[int]) -> pd.DataFrame:
        """Compare multiple experiments"""
        comparison = self.db.compare_experiments(experiment_ids)
        
        # Pivot for easier comparison
        pivot = comparison.pivot_table(
            index=['id', 'name', 'model_type'],
            columns=['dataset', 'metric_name'],
            values='metric_value'
        )
        
        return pivot
    
    def run_time_range_experiment(self, 
                                 model_type: str,
                                 symbol: str,
                                 time_ranges: List[int]) -> List[int]:
        """Run experiments for different training time ranges"""
        
        experiment_ids = []
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        for months in time_ranges:
            start_date = (datetime.now() - timedelta(days=months*30)).strftime('%Y-%m-%d')
            val_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            
            date_config = {
                'train_start': start_date,
                'train_end': val_date,
                'val_end': end_date,
                'test_end': end_date
            }
            
            name = f"{model_type}_{symbol}_{months}m"
            exp_id = self.create_experiment(name, model_type, symbol, date_config)
            
            self.run_experiment(exp_id)
            experiment_ids.append(exp_id)
        
        return experiment_ids