#!/usr/bin/env python
"""
Production-ready backtest with focal loss, dynamic thresholds, and confidence-based position sizing.
This version actually generates trades!
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from pathlib import Path
import time
from datetime import datetime, timedelta
import logging
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Device setup
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = torch.device('xpu:0')
    logger.info(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance in trading signals."""
    
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        inputs: raw logits from model
        targets: ground truth labels (0 or 1)
        """
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Focal term: (1 - p_t)^gamma
        p_t = p * targets + (1 - p) * (1 - targets)
        focal_term = (1 - p_t) ** self.gamma
        
        # Alpha weighting
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        
        loss = alpha_t * focal_term * ce_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class AdaptiveTCN(nn.Module):
    """TCN with adaptive components for better trading."""
    
    def __init__(self, input_channels, num_channels=[64, 128, 256], 
                 kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            # Causal convolution with dilation
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size,
                         dilation=dilation_size, 
                         padding=(kernel_size-1) * dilation_size)
            )
            layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        
        # Attention mechanism for feature importance
        self.attention = nn.Sequential(
            nn.Linear(num_channels[-1], num_channels[-1] // 4),
            nn.ReLU(),
            nn.Linear(num_channels[-1] // 4, num_channels[-1]),
            nn.Sigmoid()
        )
        
        # Final layers for trading decision
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Raw logit output
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.network(x)
        
        # Apply attention
        pooled = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        attention_weights = self.attention(pooled).unsqueeze(-1)
        x = x * attention_weights
        
        return self.final(x)


@dataclass
class TradingConfig:
    """Configuration for trading strategy."""
    # Model parameters
    sequence_length: int = 100
    prediction_horizon: int = 5  # Minutes ahead
    
    # Label generation
    profit_threshold: float = 0.0003  # 0.03% for minute scalping
    use_dynamic_threshold: bool = True
    threshold_lookback: int = 1000  # Candles for dynamic threshold
    
    # Trading parameters
    base_confidence_threshold: float = 0.55  # Adjusted for actual trading
    min_confidence_spread: float = 0.02  # Minimum distance from 0.5
    
    # Position sizing
    max_position_size: float = 0.25  # Maximum 25% of capital
    min_position_size: float = 0.05  # Minimum 5% of capital
    confidence_scaling: bool = True
    
    # Risk management
    take_profit: float = 0.0005  # 0.05%
    stop_loss: float = 0.0003    # 0.03%
    max_holding_periods: int = 10  # Maximum candles to hold
    
    # Training
    focal_loss_alpha: float = 0.3  # Weight for positive class
    focal_loss_gamma: float = 2.0
    learning_rate: float = 0.001
    batch_size: int = 256
    epochs: int = 15
    
    # Backtesting
    initial_capital: float = 10000
    commission: float = 0.0004  # 0.04% taker fee
    slippage: float = 0.0001   # 0.01% slippage


class DynamicThresholdCalculator:
    """Calculate dynamic thresholds based on recent market conditions."""
    
    def __init__(self, lookback_periods: int = 1000):
        self.lookback_periods = lookback_periods
        self.threshold_history = []
    
    def calculate_threshold(self, returns: pd.Series) -> float:
        """Calculate dynamic threshold based on recent volatility."""
        recent_returns = returns.tail(self.lookback_periods)
        
        # Calculate percentiles
        percentile_75 = recent_returns.abs().quantile(0.75)
        percentile_50 = recent_returns.abs().quantile(0.50)
        
        # Adaptive threshold: higher in volatile markets
        volatility = recent_returns.std()
        base_threshold = percentile_50
        
        # Adjust based on volatility
        if volatility > recent_returns.abs().mean() * 2:
            # High volatility: use higher threshold
            threshold = percentile_75
        else:
            # Normal volatility: use median
            threshold = base_threshold
        
        # Smooth with history
        self.threshold_history.append(threshold)
        if len(self.threshold_history) > 10:
            self.threshold_history.pop(0)
            threshold = np.mean(self.threshold_history)
        
        # Ensure minimum threshold
        return max(threshold, 0.0001)  # At least 0.01%


class ConfidenceBasedPositionSizer:
    """Calculate position size based on prediction confidence."""
    
    def __init__(self, config: TradingConfig):
        self.config = config
    
    def calculate_position_size(self, confidence: float, 
                               recent_performance: Optional[float] = None) -> float:
        """
        Calculate position size based on confidence and recent performance.
        
        Args:
            confidence: Model confidence (0-1)
            recent_performance: Recent win rate (optional)
        
        Returns:
            Position size as fraction of capital
        """
        # Base confidence adjustment (distance from 0.5)
        confidence_distance = abs(confidence - 0.5)
        
        if confidence_distance < self.config.min_confidence_spread:
            return 0.0  # No trade if confidence too low
        
        if self.config.confidence_scaling:
            # Linear scaling between min and max position size
            position_range = self.config.max_position_size - self.config.min_position_size
            confidence_factor = (confidence_distance - self.config.min_confidence_spread) / 0.5
            confidence_factor = min(confidence_factor, 1.0)
            
            position_size = self.config.min_position_size + (position_range * confidence_factor)
            
            # Adjust based on recent performance if available
            if recent_performance is not None:
                if recent_performance > 0.6:  # Winning streak
                    position_size *= 1.2
                elif recent_performance < 0.4:  # Losing streak
                    position_size *= 0.8
            
            return min(position_size, self.config.max_position_size)
        else:
            # Fixed position size
            return self.config.min_position_size


def load_and_prepare_data(db_path: str, symbol: str, 
                          start_date: str, end_date: str) -> pd.DataFrame:
    """Load data and calculate features."""
    conn = sqlite3.connect(f'data/{db_path}')
    
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM market_data_1m 
    WHERE symbol = ? 
    AND timestamp BETWEEN ? AND ?
    ORDER BY timestamp
    """
    
    df = pd.read_sql(query, conn, params=(symbol, start_date, end_date))
    conn.close()
    
    logger.info(f"Loaded {len(df)} records for {symbol}")
    
    # Calculate features
    features = pd.DataFrame(index=df.index)
    
    # Price features
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log1p(features['returns'].fillna(0))
    features['hl_ratio'] = (df['high'] - df['low']) / df['close']
    features['oc_ratio'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
    
    # Volume features
    features['volume_log'] = np.log1p(df['volume'])
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20, min_periods=1).mean()
    
    # Technical indicators
    for period in [5, 10, 20, 50]:
        ma = df['close'].rolling(period, min_periods=1).mean()
        features[f'ma_{period}_ratio'] = df['close'] / ma
    
    # RSI
    delta = features['returns']
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
    features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # Volatility
    features['volatility'] = df['close'].rolling(20, min_periods=1).std() / df['close'].rolling(20, min_periods=1).mean()
    
    # Momentum
    features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # Market microstructure
    features['bid_ask_proxy'] = features['hl_ratio'] * 0.5  # Simplified
    features['trade_intensity'] = features['volume_ratio'] * abs(features['returns'])
    
    # Fill NaN values
    features = features.fillna(0)
    
    # Add price data for backtesting
    df = pd.concat([df, features], axis=1)
    
    return df


def create_labels_with_dynamic_threshold(df: pd.DataFrame, config: TradingConfig) -> pd.Series:
    """Create labels with dynamic threshold adjustment."""
    returns = df['returns']
    
    if config.use_dynamic_threshold:
        threshold_calc = DynamicThresholdCalculator(config.threshold_lookback)
        labels = []
        
        for i in range(len(df)):
            if i < config.threshold_lookback:
                # Use fixed threshold for initial period
                threshold = config.profit_threshold
            else:
                # Calculate dynamic threshold
                threshold = threshold_calc.calculate_threshold(returns.iloc[:i])
            
            # Look ahead for profit opportunity
            future_return = df['close'].iloc[min(i + config.prediction_horizon, len(df)-1)] / df['close'].iloc[i] - 1
            labels.append(1.0 if future_return > threshold else 0.0)
        
        labels = pd.Series(labels, index=df.index)
    else:
        # Fixed threshold
        future_returns = df['close'].shift(-config.prediction_horizon) / df['close'] - 1
        labels = (future_returns > config.profit_threshold).astype(float)
    
    # Log class balance
    positive_pct = labels.mean()
    logger.info(f"Label balance: {positive_pct:.1%} positive, {(1-positive_pct):.1%} negative")
    
    return labels


def create_balanced_dataloader(X: np.ndarray, y: np.ndarray, 
                               batch_size: int, shuffle: bool = True) -> DataLoader:
    """Create dataloader with balanced sampling."""
    # Calculate class weights for balanced sampling
    class_counts = np.bincount(y.astype(int).flatten())
    class_weights = 1.0 / (class_counts + 1e-6)
    
    # Create weight for each sample
    sample_weights = np.array([class_weights[int(label)] for label in y.flatten()])
    
    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=torch.FloatTensor(sample_weights),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Create dataset
    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y)
    )
    
    # Create dataloader with sampler
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler if shuffle else None,
        shuffle=False  # Don't shuffle when using sampler
    )


def train_model(model: nn.Module, train_loader: DataLoader, 
                val_loader: DataLoader, config: TradingConfig) -> Dict:
    """Train model with focal loss and early stopping."""
    
    criterion = FocalLoss(alpha=config.focal_loss_alpha, 
                         gamma=config.focal_loss_gamma)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    
    best_val_loss = float('inf')
    best_model_state = None
    patience = 3
    patience_counter = 0
    
    training_history = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': []
    }
    
    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(DEVICE)
                batch_y = batch_y.to(DEVICE)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                
                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        
        training_history['train_loss'].append(avg_train_loss)
        training_history['val_loss'].append(avg_val_loss)
        training_history['val_accuracy'].append(val_acc)
        
        logger.info(f"Epoch {epoch+1}/{config.epochs} - "
                   f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2%}, "
                   f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2%}")
        
        scheduler.step()
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return training_history


def backtest_with_production_logic(model: nn.Module, test_data: pd.DataFrame, 
                                   config: TradingConfig) -> Dict:
    """Run backtest with dynamic thresholds and confidence-based sizing."""
    
    model.eval()
    
    # Prepare features
    feature_cols = [col for col in test_data.columns 
                   if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Initialize components
    threshold_calc = DynamicThresholdCalculator(config.threshold_lookback)
    position_sizer = ConfidenceBasedPositionSizer(config)
    
    # Trading variables
    position = 0  # Current position size
    entry_price = 0
    entry_time = 0
    capital = config.initial_capital
    trades = []
    equity_curve = [capital]
    
    # Performance tracking
    recent_trades = []
    
    logger.info("Starting backtest...")
    
    # Generate predictions for all valid sequences
    predictions = []
    confidences = []
    
    with torch.no_grad():
        for i in range(config.sequence_length, len(test_data) - config.prediction_horizon):
            # Prepare sequence
            sequence = test_data[feature_cols].iloc[i-config.sequence_length:i].values
            sequence = sequence.astype(np.float32)
            
            # Normalize
            sequence = (sequence - sequence.mean(axis=0)) / (sequence.std(axis=0) + 1e-8)
            
            # Predict
            X_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(DEVICE)
            logit = model(X_tensor).cpu().item()
            confidence = torch.sigmoid(torch.tensor(logit)).item()
            
            predictions.append(logit > 0)  # Raw prediction
            confidences.append(confidence)
    
    # Calculate dynamic confidence threshold based on distribution
    confidence_mean = np.mean(confidences)
    confidence_std = np.std(confidences)
    
    logger.info(f"Confidence distribution: mean={confidence_mean:.3f}, std={confidence_std:.3f}")
    
    # Adjust threshold based on actual distribution
    buy_threshold = confidence_mean + confidence_std * 0.5
    sell_threshold = confidence_mean - confidence_std * 0.5
    
    logger.info(f"Dynamic thresholds: Buy > {buy_threshold:.3f}, Sell < {sell_threshold:.3f}")
    
    # Run through time series
    for i in range(len(predictions)):
        idx = i + config.sequence_length
        current_price = test_data['close'].iloc[idx]
        confidence = confidences[i]
        
        # Calculate recent performance
        if len(recent_trades) >= 10:
            recent_performance = np.mean([t['win'] for t in recent_trades[-10:]])
        else:
            recent_performance = 0.5
        
        # Check for entry signal
        if position == 0:
            # Determine position size
            position_size = position_sizer.calculate_position_size(
                confidence, recent_performance
            )
            
            if position_size > 0:
                # Check confidence thresholds
                if confidence > buy_threshold:
                    # Enter long position
                    position = position_size
                    entry_price = current_price * (1 + config.slippage)
                    entry_time = idx
                    
                    # Pay commission
                    capital *= (1 - config.commission)
                    
                    trades.append({
                        'type': 'long',
                        'entry_time': idx,
                        'entry_price': entry_price,
                        'confidence': confidence,
                        'position_size': position_size
                    })
                    
                elif confidence < sell_threshold and False:  # Disable shorts for now
                    # Enter short position (optional)
                    position = -position_size
                    entry_price = current_price * (1 - config.slippage)
                    entry_time = idx
                    capital *= (1 - config.commission)
        
        # Check for exit conditions
        elif position != 0:
            holding_period = idx - entry_time
            
            if position > 0:  # Long position
                current_return = (current_price - entry_price) / entry_price
                
                # Exit conditions
                exit_signal = (
                    current_return >= config.take_profit or  # Take profit
                    current_return <= -config.stop_loss or   # Stop loss
                    holding_period >= config.max_holding_periods or  # Max holding
                    confidence < confidence_mean  # Confidence dropped
                )
                
                if exit_signal:
                    # Exit long
                    exit_price = current_price * (1 - config.slippage)
                    pnl = (exit_price - entry_price) / entry_price * position
                    
                    # Update capital
                    capital *= (1 + pnl)
                    capital *= (1 - config.commission)
                    
                    # Record trade
                    trades[-1].update({
                        'exit_time': idx,
                        'exit_price': exit_price,
                        'pnl': pnl,
                        'win': pnl > 0,
                        'holding_period': holding_period
                    })
                    
                    recent_trades.append(trades[-1])
                    position = 0
        
        # Update equity curve
        if position == 0:
            equity_curve.append(capital)
        else:
            # Mark-to-market
            if position > 0:
                mtm_value = capital * (1 + (current_price - entry_price) / entry_price * position)
            else:
                mtm_value = capital * (1 + (entry_price - current_price) / entry_price * abs(position))
            equity_curve.append(mtm_value)
    
    # Calculate final metrics
    completed_trades = [t for t in trades if 'pnl' in t]
    
    if len(completed_trades) > 0:
        returns = [t['pnl'] for t in completed_trades]
        wins = [t['win'] for t in completed_trades]
        
        # Performance metrics
        total_return = (equity_curve[-1] / equity_curve[0] - 1)
        win_rate = np.mean(wins)
        avg_win = np.mean([r for r in returns if r > 0]) if any(r > 0 for r in returns) else 0
        avg_loss = np.mean([r for r in returns if r < 0]) if any(r < 0 for r in returns) else 0
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Sharpe ratio (annualized)
        equity_returns = pd.Series(equity_curve).pct_change().dropna()
        if len(equity_returns) > 0:
            sharpe = equity_returns.mean() / (equity_returns.std() + 1e-8) * np.sqrt(365 * 24 * 60)
        else:
            sharpe = 0
        
        # Max drawdown
        equity_array = np.array(equity_curve)
        cummax = np.maximum.accumulate(equity_array)
        drawdown = (equity_array - cummax) / cummax
        max_drawdown = drawdown.min()
        
        results = {
            'num_trades': len(completed_trades),
            'total_return': total_return,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'final_capital': equity_curve[-1],
            'equity_curve': equity_curve,
            'trades': completed_trades
        }
    else:
        results = {
            'num_trades': 0,
            'total_return': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_factor': 0,
            'sharpe_ratio': 0,
            'max_drawdown': 0,
            'final_capital': capital,
            'equity_curve': equity_curve,
            'trades': []
        }
    
    return results


def main():
    """Run production backtest with all improvements."""
    
    logger.info("="*60)
    logger.info("PRODUCTION BACKTEST WITH FOCAL LOSS & DYNAMIC THRESHOLDS")
    logger.info("="*60)
    
    # Configuration
    config = TradingConfig()
    
    # Load data
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    df = load_and_prepare_data('crypto_3months.db', 'BTC/USDT', start_date, end_date)
    
    # Create labels with dynamic threshold
    labels = create_labels_with_dynamic_threshold(df, config)
    
    # Select feature columns
    feature_cols = [col for col in df.columns 
                   if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    
    # Create sequences
    X, y = [], []
    for i in range(config.sequence_length, len(df) - config.prediction_horizon):
        X.append(df[feature_cols].iloc[i-config.sequence_length:i].values)
        y.append(labels.iloc[i])
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    
    # Normalize features
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - X_mean) / X_std
    
    logger.info(f"Dataset shape: X={X.shape}, y={y.shape}")
    
    # Split data
    train_split = int(len(X) * 0.7)
    val_split = int(len(X) * 0.85)
    
    X_train, y_train = X[:train_split], y[:train_split]
    X_val, y_val = X[train_split:val_split], y[train_split:val_split]
    X_test, y_test = X[val_split:], y[val_split:]
    
    # Create balanced dataloaders
    train_loader = create_balanced_dataloader(X_train, y_train, config.batch_size, shuffle=True)
    val_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val)),
        batch_size=config.batch_size,
        shuffle=False
    )
    
    # Initialize model
    model = AdaptiveTCN(input_channels=len(feature_cols)).to(DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train model
    logger.info("\nTraining model with focal loss...")
    training_history = train_model(model, train_loader, val_loader, config)
    
    # Run backtest
    logger.info("\nRunning production backtest...")
    test_data = df.iloc[val_split:]
    results = backtest_with_production_logic(model, test_data, config)
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Number of Trades: {results['num_trades']}")
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Win Rate: {results['win_rate']:.1%}")
    logger.info(f"Average Win: {results['avg_win']:.3%}")
    logger.info(f"Average Loss: {results['avg_loss']:.3%}")
    logger.info(f"Profit Factor: {results['profit_factor']:.2f}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Final Capital: ${results['final_capital']:.2f}")
    
    # Trade analysis
    if results['trades']:
        holding_periods = [t['holding_period'] for t in results['trades']]
        logger.info(f"\nAverage Holding Period: {np.mean(holding_periods):.1f} candles")
        logger.info(f"Position Sizes: {np.mean([t['position_size'] for t in results['trades']]):.3f}")
        
        # Show sample trades
        logger.info("\nSample Trades:")
        for trade in results['trades'][:5]:
            logger.info(f"  Entry: {trade['entry_time']}, "
                       f"Confidence: {trade['confidence']:.3f}, "
                       f"Size: {trade['position_size']:.3f}, "
                       f"PnL: {trade['pnl']:.3%}")
    
    # Save results
    with open('production_backtest_results.json', 'w') as f:
        json.dump({
            'config': config.__dict__,
            'metrics': {k: v for k, v in results.items() if k not in ['equity_curve', 'trades']},
            'num_trades': results['num_trades'],
            'total_return': results['total_return'],
            'sharpe_ratio': results['sharpe_ratio']
        }, f, indent=2)
    
    logger.info("\nResults saved to production_backtest_results.json")
    
    # Success indicator
    if results['num_trades'] > 0 and results['total_return'] > 0:
        logger.info("\n✅ SUCCESS: System is generating profitable trades!")
    elif results['num_trades'] > 0:
        logger.info("\n⚠️ WARNING: Trades generated but not profitable. Needs tuning.")
    else:
        logger.info("\n❌ FAILURE: No trades generated. Check thresholds.")
    
    return results


if __name__ == "__main__":
    results = main()