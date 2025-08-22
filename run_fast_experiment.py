#!/usr/bin/env python
"""
FAST experiment runner that actually uses the hardware!
- Uses XPU for training
- Parallel feature calculation
- Batch processing
"""

import sys
import os
import sqlite3
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from datetime import datetime, timedelta
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Force XPU usage
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = torch.device('xpu:0')  # Use first XPU
    logger.info(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
    logger.info(f"XPU Memory: {torch.xpu.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Set XPU to use maximum performance
    torch.xpu.set_device(0)
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {DEVICE}")

# Use all CPU cores
NUM_WORKERS = mp.cpu_count()
logger.info(f"Using {NUM_WORKERS} CPU cores for data loading")

class FastTCN(nn.Module):
    """Fast TCN optimized for XPU."""
    def __init__(self, input_channels, num_channels=[64, 128, 256], kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(nn.Conv1d(in_channels, out_channels, kernel_size,
                                   dilation=dilation_size, padding=(kernel_size-1) * dilation_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        self.network = nn.Sequential(*layers)
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels[-1], 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.network(x)
        return self.final(x)

def load_data_fast(db_path, symbol, start_date, end_date):
    """Load data FAST using optimal query."""
    conn = sqlite3.connect(f'data/{db_path}')
    
    query = f"""
    SELECT timestamp, open, high, low, close, volume
    FROM market_data_1m 
    WHERE symbol = ? 
    AND timestamp BETWEEN ? AND ?
    ORDER BY timestamp
    """
    
    df = pd.read_sql(query, conn, params=(symbol, start_date, end_date))
    conn.close()
    
    logger.info(f"Loaded {len(df)} records for {symbol}")
    return df

def calculate_features_fast(df):
    """Calculate features FAST using vectorized operations."""
    features = pd.DataFrame(index=df.index)
    
    # Price features (vectorized)
    features['returns'] = df['close'].pct_change()
    features['log_returns'] = np.log1p(features['returns'])
    features['hl_ratio'] = (df['high'] - df['low']) / df['close']
    features['oc_ratio'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
    
    # Volume features
    features['volume_log'] = np.log1p(df['volume'])
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20, min_periods=1).mean()
    
    # Fast moving averages using pandas rolling
    for period in [5, 10, 20, 50]:
        ma = df['close'].rolling(period, min_periods=1).mean()
        features[f'ma_{period}_ratio'] = df['close'] / ma
    
    # Volatility
    features['volatility'] = df['close'].rolling(20, min_periods=1).std() / df['close'].rolling(20, min_periods=1).mean()
    
    # Simple RSI (vectorized)
    delta = features['returns']
    gain = delta.where(delta > 0, 0).rolling(14, min_periods=1).mean()
    loss = -delta.where(delta < 0, 0).rolling(14, min_periods=1).mean()
    features['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # Momentum
    features['momentum_5'] = df['close'] / df['close'].shift(5) - 1
    features['momentum_10'] = df['close'] / df['close'].shift(10) - 1
    
    # Fill NaN with 0 (fast)
    features = features.fillna(0)
    
    return features

def create_sequences_batch(features, labels, seq_length=100, batch_size=1024):
    """Create sequences in batches for efficiency."""
    n_samples = len(features) - seq_length
    
    # Pre-allocate arrays
    X = np.zeros((n_samples, seq_length, features.shape[1]), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    
    # Vectorized sequence creation
    for i in range(n_samples):
        X[i] = features.iloc[i:i+seq_length].values
        y[i] = labels.iloc[i+seq_length]
    
    # Normalize in batch
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - X_mean) / X_std
    
    return X, y

def train_fast(model, train_loader, val_loader, epochs=10, device=DEVICE):
    """Train model FAST using XPU."""
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=0.01,
        epochs=epochs,
        steps_per_epoch=len(train_loader)
    )
    
    model.train()
    best_val_acc = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        # Training 
        for batch_X, batch_y in train_loader:
            # Move batch to device
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            train_correct += (predicted == batch_y).sum().item()
            train_total += batch_y.size(0)
        
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_X)
                predicted = (outputs > 0.5).float()
                val_correct += (predicted == batch_y).sum().item()
                val_total += batch_y.size(0)
        
        model.train()
        
        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        epoch_time = time.time() - start_time
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        logger.info(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
                   f"Loss: {train_loss/len(train_loader):.4f}, "
                   f"Train: {train_acc:.2%}, Val: {val_acc:.2%}, "
                   f"LR: {scheduler.get_last_lr()[0]:.6f}")
    
    return best_val_acc

def backtest_fast(model, test_data, features, seq_length=100):
    """Fast backtesting with vectorized operations."""
    model.eval()
    
    with torch.no_grad():
        # Prepare test sequences
        X_test = []
        for i in range(len(test_data) - seq_length):
            X_test.append(features.iloc[i:i+seq_length].values)
        
        X_test = np.array(X_test, dtype=np.float32)
        
        # Normalize
        X_mean = X_test.mean(axis=(0, 1), keepdims=True)
        X_std = X_test.std(axis=(0, 1), keepdims=True) + 1e-8
        X_test = (X_test - X_mean) / X_std
        
        # Batch prediction for speed
        X_test_tensor = torch.FloatTensor(X_test).to(DEVICE)
        predictions = model(X_test_tensor).cpu().numpy().flatten()
    
    # Vectorized backtest calculation
    signals = (predictions > 0.5).astype(int)
    returns = test_data['close'].pct_change().iloc[seq_length:].values
    
    # Strategy returns
    strategy_returns = returns * signals
    
    # Calculate metrics (vectorized)
    total_return = (1 + strategy_returns).prod() - 1
    sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(1440 * 365)
    win_rate = (strategy_returns[signals == 1] > 0).mean() if signals.sum() > 0 else 0
    max_drawdown = (np.maximum.accumulate(1 + strategy_returns.cumsum()) - (1 + strategy_returns.cumsum())).max()
    
    return {
        'total_return': total_return,
        'sharpe_ratio': sharpe,
        'win_rate': win_rate,
        'max_drawdown': max_drawdown,
        'num_trades': signals.sum()
    }

def main():
    logger.info("="*60)
    logger.info("FAST HARDWARE-ACCELERATED EXPERIMENT")
    logger.info("="*60)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"CPU Cores: {NUM_WORKERS}")
    
    # Load data FAST
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')
    
    df = load_data_fast('crypto_3months.db', 'BTC/USDT', start_date, end_date)
    
    # Calculate features FAST
    start = time.time()
    features = calculate_features_fast(df)
    logger.info(f"Features calculated in {time.time() - start:.1f}s")
    
    # Create labels
    future_returns = df['close'].shift(-5) / df['close'] - 1
    labels = (future_returns > 0.002).astype(float)
    
    # Create sequences
    start = time.time()
    X, y = create_sequences_batch(features, labels, seq_length=100)
    logger.info(f"Sequences created in {time.time() - start:.1f}s - Shape: {X.shape}")
    
    # Split data
    split_idx = int(len(X) * 0.7)
    val_idx = int(len(X) * 0.85)
    
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:val_idx], y[split_idx:val_idx]
    X_test, y_test = X[val_idx:], y[val_idx:]
    
    # Create DataLoaders - keep data on CPU, move to XPU in batches
    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=256,  # Large batch for XPU efficiency
        shuffle=True,
        pin_memory=True,
        num_workers=0  # Workers not needed since data is on GPU
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=256,
        shuffle=False,
        pin_memory=True,
        num_workers=0
    )
    
    # Create and train model
    model = FastTCN(input_channels=features.shape[1]).to(DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train FAST
    start = time.time()
    best_val_acc = train_fast(model, train_loader, val_loader, epochs=10)
    training_time = time.time() - start
    logger.info(f"Training completed in {training_time:.1f}s")
    logger.info(f"Best validation accuracy: {best_val_acc:.2%}")
    
    # Backtest
    test_df = df.iloc[val_idx+100:]
    test_features = features.iloc[val_idx:]
    
    start = time.time()
    results = backtest_fast(model, test_df, test_features)
    logger.info(f"Backtesting completed in {time.time() - start:.1f}s")
    
    # Results
    logger.info("="*60)
    logger.info("RESULTS:")
    logger.info("="*60)
    logger.info(f"Total Return: {results['total_return']:.2%}")
    logger.info(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    logger.info(f"Win Rate: {results['win_rate']:.2%}")
    logger.info(f"Max Drawdown: {results['max_drawdown']:.2%}")
    logger.info(f"Number of Trades: {results['num_trades']}")
    logger.info(f"Training Speed: {len(X_train) / training_time:.0f} samples/sec")
    
    # Check XPU utilization
    if hasattr(torch.xpu, 'memory_allocated'):
        mem_used = torch.xpu.memory_allocated() / 1e9
        mem_total = torch.xpu.get_device_properties(0).total_memory / 1e9
        logger.info(f"XPU Memory: {mem_used:.1f}/{mem_total:.1f} GB ({mem_used/mem_total*100:.1f}%)")

if __name__ == "__main__":
    main()