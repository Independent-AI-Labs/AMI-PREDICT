#!/usr/bin/env python
"""
Comprehensive scalping experiment runner.
Tests different architectures, data volumes, capital levels, and hyperparameters.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
import sqlite3
import redis
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from src.ml.scalping_models import (
    ScalpingModelFactory, TCN, TransformerScalper, 
    OnlineLSTM, EnsembleMetaLearner, DEVICE
)
from src.ml.microstructure_features import MicrostructureFeatureEngine

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scalping_experiment.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Redis connection for caching
REDIS_HOST = '172.72.72.2'
REDIS_PORT = 6379

class ScalpingExperiment:
    """Run comprehensive scalping experiments."""
    
    def __init__(self, data_dir: str = 'data', db_name: str = 'crypto_3months.db'):
        self.data_dir = Path(data_dir)
        self.db_path = self.data_dir / db_name  # Allow different databases
        self.results_dir = Path('experiment_results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Initialize Redis connection
        try:
            self.redis_client = redis.Redis(
                host=REDIS_HOST, 
                port=REDIS_PORT, 
                decode_responses=True,
                socket_connect_timeout=5
            )
            self.redis_client.ping()
            logger.info(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        except:
            logger.warning(f"Could not connect to Redis at {REDIS_HOST}:{REDIS_PORT}")
            self.redis_client = None
        
        # Feature engine
        self.feature_engine = MicrostructureFeatureEngine()
        
        # Results storage
        self.results = []
        
    def load_data(self, symbols: List[str], start_date: datetime = None, 
                  end_date: datetime = None, days: int = None) -> Dict[str, pd.DataFrame]:
        """
        Load data from database for multiple symbols.
        
        Args:
            symbols: List of trading pairs or single pair
            start_date: Start date for data
            end_date: End date for data
            days: Alternative - load last N days
            
        Returns:
            Dict of symbol -> DataFrame
        """
        if isinstance(symbols, str):
            symbols = [symbols]
            
        conn = sqlite3.connect(self.db_path)
        
        # Calculate date range
        if days:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
        elif not start_date or not end_date:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=90)  # Default 3 months
        
        data_dict = {}
        
        for symbol in symbols:
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM market_data_1m
            WHERE symbol = ? AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            """
            
            df = pd.read_sql_query(
                query, conn, 
                params=[symbol, start_date.isoformat(), end_date.isoformat()]
            )
        
            
            if len(df) > 0:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
                data_dict[symbol] = df
                logger.info(f"Loaded {len(df)} records for {symbol}")
            else:
                logger.warning(f"No data found for {symbol}")
        
        conn.close()
        return data_dict
    
    def prepare_features(self, ohlcv_data: pd.DataFrame, 
                        order_book_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Prepare features for model training."""
        if len(ohlcv_data) == 0:
            return pd.DataFrame()
        
        # Calculate microstructure features
        features = self.feature_engine.calculate_microstructure_features(
            ohlcv_data, order_book_data
        )
        
        # Calculate regime features
        regime_features = self.feature_engine.calculate_regime_features(ohlcv_data)
        
        # Combine all features
        all_features = pd.concat([features, regime_features], axis=1)
        
        # Cache features if Redis available
        if self.redis_client:
            cache_key = f"features_{len(ohlcv_data)}_{hash(str(ohlcv_data.index[0]))}"
            try:
                self.redis_client.setex(
                    cache_key, 
                    3600,  # 1 hour TTL
                    all_features.to_json()
                )
            except:
                pass
        
        return all_features
    
    def create_labels(self, ohlcv_data: pd.DataFrame, 
                     lookahead: int = 5) -> pd.Series:
        """
        Create labels for scalping prediction.
        
        Args:
            ohlcv_data: OHLCV data
            lookahead: Minutes to look ahead for price movement
            
        Returns:
            Binary labels (1 for profitable trade, 0 otherwise)
        """
        # Calculate future returns
        future_returns = ohlcv_data['close'].pct_change(lookahead).shift(-lookahead)
        
        # Label as profitable if return > 0.2% (after fees)
        labels = (future_returns > 0.002).astype(float)
        
        return labels
    
    def create_datasets(self, features: pd.DataFrame, labels: pd.Series,
                       sequence_length: int = 100, 
                       train_ratio: float = 0.8) -> Tuple[DataLoader, DataLoader]:
        """Create train and validation datasets."""
        logger.info(f"Initial features shape: {features.shape}, labels shape: {labels.shape}")
        
        # Remove NaN values
        valid_idx = ~(features.isna().any(axis=1) | labels.isna())
        features = features[valid_idx]
        labels = labels[valid_idx]
        
        logger.info(f"After NaN removal: features shape: {features.shape}, labels shape: {labels.shape}")
        
        if len(features) < sequence_length:
            logger.error(f"Not enough data: {len(features)} < {sequence_length}")
            return None, None
        
        # Create sequences
        X, y = [], []
        for i in range(len(features) - sequence_length):
            X.append(features.iloc[i:i+sequence_length].values)
            y.append(labels.iloc[i+sequence_length])
        
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.float32).reshape(-1, 1)
        
        # Normalize features
        X_mean = X.mean(axis=(0, 1), keepdims=True)
        X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
        X = (X - X_mean) / X_std
        
        # Split data
        split_idx = int(len(X) * train_ratio)
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Create DataLoaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train).to(DEVICE),
            torch.FloatTensor(y_train).to(DEVICE)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(DEVICE),
            torch.FloatTensor(y_val).to(DEVICE)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        
        logger.info(f"Created datasets - Train: {len(X_train)}, Val: {len(X_val)}")
        
        return train_loader, val_loader
    
    def train_model(self, model: nn.Module, train_loader: DataLoader, 
                   val_loader: DataLoader, epochs: int = 10,
                   learning_rate: float = 1e-3) -> Dict:
        """Train a model and return metrics."""
        if train_loader is None:
            return {'error': 'No data available'}
        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        
        train_losses = []
        val_accuracies = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            # Training
            model.train()
            epoch_loss = 0
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                
                if isinstance(model, OnlineLSTM):
                    output, _ = model(batch_x)
                else:
                    output = model(batch_x)
                
                loss = criterion(output, batch_y)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_x, batch_y in val_loader:
                    if isinstance(model, OnlineLSTM):
                        output, _ = model(batch_x)
                    else:
                        output = model(batch_x)
                    
                    predictions = (output > 0.5).float()
                    correct += (predictions == batch_y).sum().item()
                    total += batch_y.size(0)
            
            accuracy = correct / total if total > 0 else 0
            val_accuracies.append(accuracy)
            
            if (epoch + 1) % 5 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} - Loss: {train_losses[-1]:.4f}, "
                          f"Val Acc: {accuracy:.4f}")
        
        training_time = time.time() - start_time
        
        # Calculate final metrics
        metrics = {
            'final_loss': train_losses[-1],
            'best_accuracy': max(val_accuracies),
            'final_accuracy': val_accuracies[-1],
            'training_time': training_time,
            'epochs': epochs,
            'parameters': sum(p.numel() for p in model.parameters())
        }
        
        return metrics
    
    def backtest_model(self, model: nn.Module, features: pd.DataFrame,
                      ohlcv_data: pd.DataFrame, initial_capital: float = 10000,
                      position_size: float = 0.2, take_profit: float = 0.002,
                      stop_loss: float = 0.0015) -> Dict:
        """
        Backtest model with realistic trading simulation.
        
        Args:
            model: Trained model
            features: Feature data
            ohlcv_data: OHLCV data for prices
            initial_capital: Starting capital
            position_size: Fraction of capital per trade
            take_profit: Take profit threshold (0.002 = 0.2%)
            stop_loss: Stop loss threshold (0.0015 = 0.15%)
            
        Returns:
            Backtest metrics
        """
        model.eval()
        
        capital = initial_capital
        trades = []
        positions = []
        
        sequence_length = 100
        
        # Skip first sequence_length for proper sequencing
        for i in range(sequence_length, len(features) - 1):
            # Prepare input
            X = features.iloc[i-sequence_length:i].values
            X = torch.FloatTensor(X).unsqueeze(0).to(DEVICE)
            
            # Get prediction
            with torch.no_grad():
                if isinstance(model, OnlineLSTM):
                    pred, _ = model(X)
                else:
                    pred = model(X)
                
                confidence = pred.item()
            
            # Trading logic
            if confidence > 0.6 and len(positions) < 5:  # Max 5 concurrent positions
                # Enter position
                entry_price = ohlcv_data['close'].iloc[i]
                trade_size = capital * position_size
                
                position = {
                    'entry_time': ohlcv_data.index[i],
                    'entry_price': entry_price,
                    'size': trade_size,
                    'tp_price': entry_price * (1 + take_profit),
                    'sl_price': entry_price * (1 - stop_loss),
                    'confidence': confidence
                }
                positions.append(position)
            
            # Check existing positions
            positions_to_close = []
            for pos_idx, pos in enumerate(positions):
                current_price = ohlcv_data['close'].iloc[i]
                
                # Check exit conditions
                if current_price >= pos['tp_price']:
                    # Take profit hit
                    pnl = pos['size'] * take_profit
                    capital += pnl
                    trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': ohlcv_data.index[i],
                        'pnl': pnl,
                        'return': take_profit,
                        'result': 'TP'
                    })
                    positions_to_close.append(pos_idx)
                    
                elif current_price <= pos['sl_price']:
                    # Stop loss hit
                    pnl = -pos['size'] * stop_loss
                    capital += pnl
                    trades.append({
                        'entry_time': pos['entry_time'],
                        'exit_time': ohlcv_data.index[i],
                        'pnl': pnl,
                        'return': -stop_loss,
                        'result': 'SL'
                    })
                    positions_to_close.append(pos_idx)
            
            # Remove closed positions
            for idx in reversed(positions_to_close):
                positions.pop(idx)
        
        # Calculate metrics
        if len(trades) == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'sharpe_ratio': 0,
                'max_drawdown': 0,
                'total_return': 0,
                'annual_return': 0
            }
        
        trades_df = pd.DataFrame(trades)
        
        # Win rate
        win_rate = (trades_df['pnl'] > 0).mean()
        
        # Returns
        total_return = (capital - initial_capital) / initial_capital
        
        # Sharpe ratio (minute-level)
        returns = trades_df['return'].values
        if len(returns) > 1:
            sharpe = returns.mean() / (returns.std() + 1e-8) * np.sqrt(1440 * 252)
        else:
            sharpe = 0
        
        # Max drawdown
        cumulative_returns = (1 + pd.Series(returns)).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(drawdown.min())
        
        # Annual return (assuming 252 trading days)
        days_traded = (trades_df['exit_time'].max() - trades_df['entry_time'].min()).days
        if days_traded > 0:
            annual_return = (1 + total_return) ** (365 / days_traded) - 1
        else:
            annual_return = 0
        
        metrics = {
            'total_trades': len(trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'total_return': total_return,
            'annual_return': annual_return,
            'avg_trade_duration': trades_df.apply(
                lambda x: (x['exit_time'] - x['entry_time']).total_seconds() / 60, 
                axis=1
            ).mean() if len(trades_df) > 0 else 0
        }
        
        return metrics
    
    def run_experiment(self, model_type: str, symbol: str, 
                      years: int, capital: float,
                      hyperparams: Dict) -> Dict:
        """Run single experiment configuration."""
        logger.info(f"\nRunning experiment: {model_type}, {symbol}, "
                   f"{years}yr, ${capital}, params: {hyperparams}")
        
        # Load data - updated to handle new load_data signature
        if years:
            data_dict = self.load_data([symbol], days=years * 365)
        else:
            # Use all available data
            data_dict = self.load_data([symbol])
        
        if not data_dict or symbol not in data_dict:
            return {'error': 'No data available'}
        
        ohlcv_data = data_dict[symbol]
        order_book_data = pd.DataFrame()  # No order book data yet
        
        if len(ohlcv_data) == 0:
            return {'error': 'No data available'}
        
        # Prepare features
        features = self.prepare_features(ohlcv_data, order_book_data)
        
        # Create labels
        labels = self.create_labels(ohlcv_data)
        
        # Create datasets
        train_loader, val_loader = self.create_datasets(
            features, labels, 
            sequence_length=hyperparams.get('sequence_length', 100)
        )
        
        if train_loader is None:
            return {'error': 'Insufficient data for training'}
        
        # Create model
        factory = ScalpingModelFactory()
        input_size = features.shape[1]
        
        if model_type == 'tcn':
            model = factory.create_tcn(input_size, **hyperparams)
        elif model_type == 'transformer':
            model = factory.create_transformer(input_size, **hyperparams)
        elif model_type == 'lstm':
            model = factory.create_lstm(input_size, **hyperparams)
        elif model_type == 'ensemble':
            model = factory.create_ensemble(input_size, **hyperparams)
        else:
            return {'error': f'Unknown model type: {model_type}'}
        
        # Train model
        train_metrics = self.train_model(
            model, train_loader, val_loader,
            epochs=hyperparams.get('epochs', 10),
            learning_rate=hyperparams.get('learning_rate', 1e-3)
        )
        
        # Backtest model
        backtest_metrics = self.backtest_model(
            model, features, ohlcv_data,
            initial_capital=capital,
            position_size=hyperparams.get('position_size', 0.2),
            take_profit=hyperparams.get('take_profit', 0.002),
            stop_loss=hyperparams.get('stop_loss', 0.0015)
        )
        
        # Combine results
        result = {
            'model_type': model_type,
            'symbol': symbol,
            'years': years,
            'capital': capital,
            'hyperparams': hyperparams,
            **train_metrics,
            **backtest_metrics,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def run_comprehensive_experiments(self):
        """Run all experiment configurations."""
        # Experiment configurations
        models = ['tcn', 'transformer', 'lstm']
        symbols = ['BTC/USDT']  # Start with BTC, expand later
        data_years = [1, 2, 3, 5]
        capitals = [100, 1000, 10000, 100000]
        
        # Hyperparameter grid
        hyperparam_grid = {
            'learning_rate': [1e-4, 1e-3],
            'sequence_length': [50, 100, 200],
            'take_profit': [0.001, 0.002, 0.003],
            'stop_loss': [0.001, 0.0015, 0.002],
            'position_size': [0.1, 0.2, 0.3],
            'epochs': [10]
        }
        
        # Generate all combinations
        hyperparam_combinations = list(product(
            hyperparam_grid['learning_rate'],
            hyperparam_grid['sequence_length'],
            hyperparam_grid['take_profit'],
            hyperparam_grid['stop_loss'],
            hyperparam_grid['position_size'],
            hyperparam_grid['epochs']
        ))
        
        total_experiments = (len(models) * len(symbols) * len(data_years) * 
                           len(capitals) * len(hyperparam_combinations))
        
        logger.info(f"Starting {total_experiments} experiments...")
        
        experiment_count = 0
        
        for model in models:
            for symbol in symbols:
                for years in data_years:
                    for capital in capitals:
                        # Run subset of hyperparameter combinations for efficiency
                        for hp_combo in hyperparam_combinations[:3]:  # Limit to 3 for now
                            experiment_count += 1
                            
                            hyperparams = {
                                'learning_rate': hp_combo[0],
                                'sequence_length': hp_combo[1],
                                'take_profit': hp_combo[2],
                                'stop_loss': hp_combo[3],
                                'position_size': hp_combo[4],
                                'epochs': hp_combo[5]
                            }
                            
                            logger.info(f"\nExperiment {experiment_count}/{total_experiments}")
                            
                            try:
                                result = self.run_experiment(
                                    model, symbol, years, capital, hyperparams
                                )
                                self.results.append(result)
                                
                                # Save intermediate results
                                if experiment_count % 10 == 0:
                                    self.save_results()
                                    
                            except Exception as e:
                                logger.error(f"Experiment failed: {e}")
                                self.results.append({
                                    'model_type': model,
                                    'symbol': symbol,
                                    'years': years,
                                    'capital': capital,
                                    'error': str(e)
                                })
        
        # Save final results
        self.save_results()
        
        # Generate report
        self.generate_report()
    
    def save_results(self):
        """Save experiment results to file."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = self.results_dir / f'scalping_results_{timestamp}.json'
        
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"Saved {len(self.results)} results to {results_file}")
    
    def generate_report(self):
        """Generate comprehensive experiment report."""
        if len(self.results) == 0:
            logger.warning("No results to report")
            return
        
        # Convert to DataFrame for analysis
        df = pd.DataFrame(self.results)
        
        # Remove error results
        df_valid = df[~df['error'].notna()]
        
        if len(df_valid) == 0:
            logger.warning("No valid results to report")
            return
        
        report = []
        report.append("# Scalping Strategy Experiment Report")
        report.append(f"\n## Executive Summary")
        report.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
        report.append(f"**Total Experiments**: {len(self.results)}")
        report.append(f"**Successful**: {len(df_valid)}")
        report.append(f"**Failed**: {len(df) - len(df_valid)}")
        
        # Best configuration
        best_idx = df_valid['sharpe_ratio'].idxmax()
        best = df_valid.loc[best_idx]
        
        report.append(f"\n**Best Configuration**:")
        report.append(f"- Model: {best['model_type']}")
        report.append(f"- Data: {best['years']} years")
        report.append(f"- Capital: ${best['capital']:,.0f}")
        report.append(f"- Sharpe Ratio: {best['sharpe_ratio']:.2f}")
        report.append(f"- Win Rate: {best['win_rate']*100:.1f}%")
        report.append(f"- Annual Return: {best['annual_return']*100:.1f}%")
        
        # Performance by model
        report.append(f"\n## Performance by Model Architecture")
        for model in df_valid['model_type'].unique():
            model_df = df_valid[df_valid['model_type'] == model]
            report.append(f"\n### {model.upper()}")
            report.append(f"- Avg Sharpe: {model_df['sharpe_ratio'].mean():.2f}")
            report.append(f"- Avg Win Rate: {model_df['win_rate'].mean()*100:.1f}%")
            report.append(f"- Avg Annual Return: {model_df['annual_return'].mean()*100:.1f}%")
            report.append(f"- Avg Training Time: {model_df['training_time'].mean():.1f}s")
        
        # Performance by data volume
        report.append(f"\n## Performance by Training Data Volume")
        for years in sorted(df_valid['years'].unique()):
            years_df = df_valid[df_valid['years'] == years]
            report.append(f"\n### {years} Year(s)")
            report.append(f"- Avg Sharpe: {years_df['sharpe_ratio'].mean():.2f}")
            report.append(f"- Avg Accuracy: {years_df['final_accuracy'].mean()*100:.1f}%")
            report.append(f"- Total Trades: {years_df['total_trades'].mean():.0f}")
        
        # Performance by capital
        report.append(f"\n## Performance by Capital Size")
        for capital in sorted(df_valid['capital'].unique()):
            cap_df = df_valid[df_valid['capital'] == capital]
            report.append(f"\n### ${capital:,.0f}")
            report.append(f"- Avg Sharpe: {cap_df['sharpe_ratio'].mean():.2f}")
            report.append(f"- Avg Max Drawdown: {cap_df['max_drawdown'].mean()*100:.1f}%")
            report.append(f"- Avg Annual Return: {cap_df['annual_return'].mean()*100:.1f}%")
        
        # Save report
        report_file = self.results_dir / f'experiment_report_{datetime.now().strftime("%Y%m%d")}.md'
        with open(report_file, 'w') as f:
            f.write('\n'.join(report))
        
        logger.info(f"Report saved to {report_file}")
        
        # Print summary
        print("\n" + "="*60)
        print('\n'.join(report[:20]))  # Print first part of report
        print("="*60)


def run_consolidated_benchmark(
    db_name: str = 'crypto_3months.db',
    symbols: List[str] = None,
    model_types: List[str] = None,
    train_days: int = 60,
    test_days: int = 30,
    multi_symbol_training: bool = False
):
    """
    Run consolidated training and benchmarking.
    
    Args:
        db_name: Database to use
        symbols: List of symbols to train on
        model_types: List of model types to test
        train_days: Days for training
        test_days: Days for testing
        multi_symbol_training: Train single model on multiple symbols
    """
    logger.info("="*60)
    logger.info("CONSOLIDATED TRAINING & BENCHMARKING")
    logger.info("="*60)
    
    # Default symbols
    if symbols is None:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    # Default models
    if model_types is None:
        model_types = ['tcn', 'lstm']
    
    experiment = ScalpingExperiment(db_name=db_name)
    
    # Calculate date ranges
    end_date = datetime.now()
    test_start = end_date - timedelta(days=test_days)
    train_start = test_start - timedelta(days=train_days)
    
    logger.info(f"Database: {db_name}")
    logger.info(f"Training period: {train_start.date()} to {test_start.date()} ({train_days} days)")
    logger.info(f"Testing period: {test_start.date()} to {end_date.date()} ({test_days} days)")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Models: {', '.join(model_types)}")
    logger.info(f"Multi-symbol training: {multi_symbol_training}")
    logger.info("="*60)
    
    results = []
    
    if multi_symbol_training:
        # Train single model on all symbols
        logger.info("\n[MULTI-SYMBOL TRAINING]")
        
        # Load training data for all symbols
        train_data = experiment.load_data(symbols, train_start, test_start)
        test_data = experiment.load_data(symbols, test_start, end_date)
        
        if not train_data or not test_data:
            logger.error("Failed to load data")
            return
        
        # Combine all symbol data
        combined_train = pd.concat(list(train_data.values()))
        combined_test = pd.concat(list(test_data.values()))
        
        logger.info(f"Combined training data: {len(combined_train)} records")
        logger.info(f"Combined test data: {len(combined_test)} records")
        
        for model_type in model_types:
            logger.info(f"\nTraining {model_type} on all symbols...")
            
            # Prepare features
            train_features = experiment.prepare_features(combined_train)
            test_features = experiment.prepare_features(combined_test)
            
            # Create labels
            train_labels = experiment.create_labels(combined_train)
            test_labels = experiment.create_labels(combined_test)
            
            # Create datasets
            train_loader, val_loader = experiment.create_datasets(
                train_features, train_labels, sequence_length=100
            )
            
            if train_loader is None:
                logger.error(f"Failed to create datasets for {model_type}")
                continue
            
            # Train model
            factory = ScalpingModelFactory()
            if model_type == 'tcn':
                model = factory.create_tcn(train_features.shape[1])
            elif model_type == 'lstm':
                model = factory.create_lstm(train_features.shape[1])
            else:
                continue
            
            # Train
            train_metrics = experiment.train_model(
                model, train_loader, val_loader, epochs=20
            )
            
            # Test on each symbol separately
            for symbol in symbols:
                if symbol in test_data:
                    logger.info(f"Testing on {symbol}...")
                    backtest_metrics = experiment.backtest_model(
                        model, test_features, test_data[symbol],
                        initial_capital=10000
                    )
                    
                    results.append({
                        'model_type': model_type,
                        'training': 'multi-symbol',
                        'test_symbol': symbol,
                        'train_days': train_days,
                        'test_days': test_days,
                        **train_metrics,
                        **backtest_metrics
                    })
    
    else:
        # Train separate model for each symbol
        logger.info("\n[SINGLE-SYMBOL TRAINING]")
        
        for symbol in symbols:
            logger.info(f"\nProcessing {symbol}...")
            
            # Load data
            train_data = experiment.load_data([symbol], train_start, test_start)
            test_data = experiment.load_data([symbol], test_start, end_date)
            
            if not train_data or symbol not in train_data:
                logger.warning(f"No data for {symbol}")
                continue
            
            symbol_train = train_data[symbol]
            symbol_test = test_data[symbol]
            
            logger.info(f"Training data: {len(symbol_train)} records")
            logger.info(f"Test data: {len(symbol_test)} records")
            
            for model_type in model_types:
                logger.info(f"Training {model_type}...")
                
                # Run experiment
                result = experiment.run_experiment(
                    model_type=model_type,
                    symbol=symbol,
                    years=None,  # We're using date ranges instead
                    capital=10000,
                    hyperparams={
                        'learning_rate': 1e-3,
                        'sequence_length': 100,
                        'epochs': 20,
                        'take_profit': 0.002,
                        'stop_loss': 0.0015,
                        'position_size': 0.2
                    }
                )
                
                results.append({
                    'training': 'single-symbol',
                    'train_symbol': symbol,
                    'test_symbol': symbol,
                    'train_days': train_days,
                    'test_days': test_days,
                    **result
                })
    
    # Save and display results
    experiment.results = results
    experiment.save_results()
    experiment.generate_report()
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("BENCHMARK RESULTS SUMMARY")
    logger.info("="*60)
    
    for result in results:
        if 'error' not in result:
            logger.info(f"\n{result.get('model_type', 'Unknown')} - {result.get('test_symbol', 'Unknown')}:")
            logger.info(f"  Training: {result.get('training', 'Unknown')}")
            logger.info(f"  Accuracy: {result.get('final_accuracy', 0)*100:.1f}%")
            logger.info(f"  Win Rate: {result.get('win_rate', 0)*100:.1f}%")
            logger.info(f"  Sharpe: {result.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Total Return: {result.get('total_return', 0)*100:.1f}%")
    
    return results


def main():
    """Main entry point with argument parsing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run scalping experiments')
    parser.add_argument('--db', default='crypto_3months.db', help='Database name')
    parser.add_argument('--symbols', nargs='+', help='Symbols to trade')
    parser.add_argument('--models', nargs='+', help='Model types to test')
    parser.add_argument('--train-days', type=int, default=60, help='Training days')
    parser.add_argument('--test-days', type=int, default=30, help='Testing days')
    parser.add_argument('--multi-symbol', action='store_true', help='Train on multiple symbols')
    
    args = parser.parse_args()
    
    return run_consolidated_benchmark(
        db_name=args.db,
        symbols=args.symbols,
        model_types=args.models,
        train_days=args.train_days,
        test_days=args.test_days,
        multi_symbol_training=args.multi_symbol
    )


if __name__ == "__main__":
    main()