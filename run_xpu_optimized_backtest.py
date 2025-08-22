#!/usr/bin/env python
"""
XPU-OPTIMIZED backtest - Actually uses the GPU for everything!
Batched predictions, vectorized operations, minimal CPU usage.
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
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
import time
from datetime import datetime, timedelta
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

sys.path.append(str(Path(__file__).parent))

# FORCE XPU USAGE
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = torch.device('xpu:0')
    logger.info(f"✅ Using Intel XPU: {torch.xpu.get_device_name(0)}")
    logger.info(f"   Memory: {torch.xpu.get_device_properties(0).total_memory / 1e9:.1f} GB")
    # Set for maximum performance
    torch.xpu.set_device(0)
    torch.backends.cudnn.benchmark = True
else:
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"⚠️ Using: {DEVICE}")


class FocalLossFast(nn.Module):
    """Fast focal loss implementation for XPU."""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        # All operations on GPU
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = self.alpha * ((1 - p_t) ** self.gamma) * ce_loss
        return loss.mean()


class FastScalpingNet(nn.Module):
    """Optimized network for XPU - minimal memory transfers."""
    
    def __init__(self, input_dim=16, hidden_dim=128):
        super().__init__()
        # Simplified architecture for speed
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=2, 
                         batch_first=True, dropout=0.2)
        
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        # x: (batch, seq_len, features)
        gru_out, _ = self.gru(x)
        
        # Attention weights
        attn_weights = F.softmax(self.attention(gru_out), dim=1)
        
        # Weighted average
        context = torch.sum(gru_out * attn_weights, dim=1)
        
        # Classification
        return self.classifier(context)


def load_data_vectorized(db_path: str, symbol: str) -> pd.DataFrame:
    """Load and process data with vectorized operations."""
    logger.info(f"Loading {symbol} data...")
    
    conn = sqlite3.connect(f'data/{db_path}')
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM market_data_1m 
    WHERE symbol = ?
    ORDER BY timestamp
    LIMIT 100000
    """
    
    df = pd.read_sql(query, conn, params=(symbol,))
    conn.close()
    
    logger.info(f"Loaded {len(df)} records")
    
    # VECTORIZED feature calculation - all at once
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log1p(df['returns'].fillna(0))
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    df['oc_ratio'] = (df['close'] - df['open']) / (df['open'].replace(0, 1))
    df['volume_log'] = np.log1p(df['volume'])
    
    # Fast rolling calculations
    for period in [5, 10, 20]:
        df[f'sma_{period}'] = df['close'].rolling(period, min_periods=1).mean()
        df[f'sma_ratio_{period}'] = df['close'] / df[f'sma_{period}']
    
    # RSI vectorized
    delta = df['returns']
    gain = delta.clip(lower=0).rolling(14, min_periods=1).mean()
    loss = -delta.clip(upper=0).rolling(14, min_periods=1).mean()
    df['rsi'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # Volatility
    df['volatility'] = df['returns'].rolling(20, min_periods=1).std()
    
    # Momentum
    df['momentum'] = df['close'] / df['close'].shift(10) - 1
    
    # Fill NaN
    df = df.fillna(0)
    
    return df


def prepare_data_gpu_batch(df: pd.DataFrame, seq_len: int = 50, 
                           batch_size: int = 1024) -> tuple:
    """Prepare all data in batches on GPU."""
    
    feature_cols = ['returns', 'log_returns', 'hl_ratio', 'oc_ratio', 
                   'volume_log', 'sma_ratio_5', 'sma_ratio_10', 'sma_ratio_20',
                   'rsi', 'volatility', 'momentum']
    
    # Convert to numpy for speed
    features = df[feature_cols].values.astype(np.float32)
    
    # Simple labels - just predict if price goes up
    labels = (df['close'].shift(-5) > df['close']).astype(np.float32).values
    
    # Create sequences - VECTORIZED
    n_samples = len(features) - seq_len - 5
    
    # Pre-allocate on CPU
    X = np.zeros((n_samples, seq_len, len(feature_cols)), dtype=np.float32)
    y = np.zeros((n_samples, 1), dtype=np.float32)
    
    # Vectorized sequence creation
    for i in range(n_samples):
        X[i] = features[i:i+seq_len]
        y[i] = labels[i+seq_len]
    
    # Normalize ONCE
    X_mean = X.mean(axis=(0, 1), keepdims=True)
    X_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
    X = (X - X_mean) / X_std
    
    logger.info(f"Created {n_samples} sequences on CPU")
    
    # Move to GPU in one shot
    X_tensor = torch.FloatTensor(X).to(DEVICE)
    y_tensor = torch.FloatTensor(y).to(DEVICE)
    
    logger.info(f"Moved data to XPU - Shape: {X_tensor.shape}")
    
    return X_tensor, y_tensor, X_mean, X_std


def train_on_xpu(model: nn.Module, X_train: torch.Tensor, y_train: torch.Tensor,
                 X_val: torch.Tensor, y_val: torch.Tensor, 
                 epochs: int = 10, batch_size: int = 512) -> dict:
    """Train entirely on XPU - no CPU transfers."""
    
    criterion = FocalLossFast(alpha=0.3, gamma=2.0).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=0.002)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=0.01, epochs=epochs, 
        steps_per_epoch=len(X_train) // batch_size + 1
    )
    
    # Create datasets - already on GPU
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
    
    logger.info(f"Training on XPU with batch size {batch_size}...")
    
    best_val_acc = 0
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        
        for batch_X, batch_y in train_loader:
            # Data already on GPU!
            optimizer.zero_grad(set_to_none=True)
            
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            with torch.no_grad():
                preds = (torch.sigmoid(outputs) > 0.5).float()
                train_correct += (preds == batch_y).sum().item()
        
        # Validation - all on GPU
        model.eval()
        val_correct = 0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                outputs = model(batch_X)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (preds == batch_y).sum().item()
        
        train_acc = train_correct / len(y_train)
        val_acc = val_correct / len(y_val)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
        
        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
                   f"Loss: {train_loss/len(train_loader):.4f}, "
                   f"Train: {train_acc:.2%}, Val: {val_acc:.2%}")
    
    return {'best_val_acc': best_val_acc}


def backtest_gpu_optimized(model: nn.Module, df: pd.DataFrame, 
                          X_test: torch.Tensor, seq_len: int = 50) -> dict:
    """Run backtest with GPU-accelerated predictions."""
    
    model.eval()
    
    # Predict ALL AT ONCE on GPU
    logger.info("Generating predictions on XPU...")
    start_pred = time.time()
    
    with torch.no_grad():
        # Process in large batches for efficiency
        batch_size = 2048
        all_predictions = []
        
        for i in range(0, len(X_test), batch_size):
            batch = X_test[i:i+batch_size]
            logits = model(batch)
            probs = torch.sigmoid(logits)
            all_predictions.append(probs)
        
        # Concatenate all predictions
        predictions = torch.cat(all_predictions, dim=0).cpu().numpy().flatten()
    
    pred_time = time.time() - start_pred
    logger.info(f"Generated {len(predictions)} predictions in {pred_time:.2f}s "
               f"({len(predictions)/pred_time:.0f} pred/sec)")
    
    # Dynamic thresholds based on prediction distribution
    pred_mean = predictions.mean()
    pred_std = predictions.std()
    
    buy_threshold = pred_mean + pred_std * 0.5
    sell_threshold = pred_mean - pred_std * 0.5
    
    logger.info(f"Prediction stats: mean={pred_mean:.3f}, std={pred_std:.3f}")
    logger.info(f"Thresholds: Buy>{buy_threshold:.3f}, Sell<{sell_threshold:.3f}")
    
    # Vectorized backtest
    test_start = len(df) - len(predictions) - seq_len
    test_prices = df['close'].iloc[test_start:test_start+len(predictions)].values
    test_returns = np.diff(test_prices) / test_prices[:-1]
    
    # Generate signals
    signals = np.zeros(len(predictions))
    signals[predictions > buy_threshold] = 1
    signals[predictions < sell_threshold] = -1
    
    # Calculate position sizes based on confidence
    confidence_distance = np.abs(predictions - pred_mean)
    position_sizes = np.clip(confidence_distance * 2, 0.05, 0.25)
    
    # Simulate trading
    capital = 10000
    position = 0
    trades = []
    equity = [capital]
    
    for i in range(1, len(signals)):
        if position == 0 and signals[i] != 0:
            # Enter position
            position = signals[i] * position_sizes[i]
            entry_price = test_prices[i]
            entry_idx = i
            
        elif position != 0:
            # Check exit conditions
            current_return = (test_prices[i] - entry_price) / entry_price * position
            
            # Exit conditions
            if (current_return > 0.001 or  # 0.1% profit
                current_return < -0.0005 or  # 0.05% loss
                i - entry_idx > 10 or  # Max 10 candles
                signals[i] * position < 0):  # Signal reversal
                
                # Exit position
                pnl_pct = current_return
                capital *= (1 + pnl_pct)
                
                trades.append({
                    'entry': entry_idx,
                    'exit': i,
                    'pnl': pnl_pct,
                    'confidence': predictions[entry_idx],
                    'size': abs(position)
                })
                
                position = 0
                equity.append(capital)
    
    # Calculate metrics
    if len(trades) > 0:
        returns = [t['pnl'] for t in trades]
        total_return = (capital / 10000 - 1)
        win_rate = np.mean([r > 0 for r in returns])
        
        # Sharpe ratio
        if len(returns) > 1:
            sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252 * 24 * 60)
        else:
            sharpe = 0
        
        results = {
            'num_trades': len(trades),
            'total_return': total_return,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'final_capital': capital,
            'avg_confidence': np.mean([t['confidence'] for t in trades]),
            'avg_position_size': np.mean([t['size'] for t in trades])
        }
    else:
        results = {
            'num_trades': 0,
            'total_return': 0,
            'win_rate': 0,
            'sharpe_ratio': 0,
            'final_capital': capital,
            'avg_confidence': 0,
            'avg_position_size': 0
        }
    
    return results


def main():
    """Main function - XPU optimized."""
    
    logger.info("="*60)
    logger.info("🚀 XPU-OPTIMIZED PRODUCTION BACKTEST")
    logger.info("="*60)
    
    # Load data
    df = load_data_vectorized('crypto_3months.db', 'BTC/USDT')
    
    # Prepare data on GPU
    seq_len = 50
    X, y, X_mean, X_std = prepare_data_gpu_batch(df, seq_len=seq_len, batch_size=1024)
    
    # Split data - all on GPU
    train_split = int(len(X) * 0.7)
    val_split = int(len(X) * 0.85)
    
    X_train = X[:train_split]
    y_train = y[:train_split]
    X_val = X[train_split:val_split]
    y_val = y[train_split:val_split]
    X_test = X[val_split:]
    y_test = y[val_split:]
    
    logger.info(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Create model on XPU
    model = FastScalpingNet(input_dim=11).to(DEVICE)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Check XPU memory usage
    if hasattr(torch.xpu, 'memory_allocated'):
        mem_used = torch.xpu.memory_allocated() / 1e9
        logger.info(f"XPU Memory Used: {mem_used:.2f} GB")
    
    # Train on XPU
    logger.info("\n🔥 Training on XPU...")
    train_start = time.time()
    
    train_results = train_on_xpu(
        model, X_train, y_train, X_val, y_val,
        epochs=10, batch_size=512
    )
    
    train_time = time.time() - train_start
    logger.info(f"Training completed in {train_time:.1f}s")
    logger.info(f"Throughput: {len(X_train) * 10 / train_time:.0f} samples/sec")
    
    # Run backtest
    logger.info("\n📊 Running GPU-accelerated backtest...")
    backtest_results = backtest_gpu_optimized(model, df, X_test, seq_len)
    
    # Display results
    logger.info("\n" + "="*60)
    logger.info("💰 BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Trades: {backtest_results['num_trades']}")
    logger.info(f"Return: {backtest_results['total_return']:.2%}")
    logger.info(f"Win Rate: {backtest_results['win_rate']:.1%}")
    logger.info(f"Sharpe: {backtest_results['sharpe_ratio']:.2f}")
    logger.info(f"Final Capital: ${backtest_results['final_capital']:.2f}")
    logger.info(f"Avg Confidence: {backtest_results['avg_confidence']:.3f}")
    logger.info(f"Avg Position Size: {backtest_results['avg_position_size']:.3f}")
    
    # Check XPU utilization
    if hasattr(torch.xpu, 'memory_allocated'):
        mem_used = torch.xpu.memory_allocated() / 1e9
        mem_total = torch.xpu.get_device_properties(0).total_memory / 1e9
        logger.info(f"\n🎮 XPU Memory: {mem_used:.1f}/{mem_total:.1f} GB "
                   f"({mem_used/mem_total*100:.1f}% utilized)")
    
    # Success check
    if backtest_results['num_trades'] > 0:
        if backtest_results['total_return'] > 0:
            logger.info("\n✅ SUCCESS: Profitable trades generated!")
        else:
            logger.info("\n⚠️ WARNING: Trades generated but not profitable")
    else:
        logger.info("\n❌ No trades generated")
    
    # Save results
    with open('xpu_backtest_results.json', 'w') as f:
        json.dump(backtest_results, f, indent=2)
    
    return backtest_results


if __name__ == "__main__":
    results = main()