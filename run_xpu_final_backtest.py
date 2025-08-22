#!/usr/bin/env python
"""
Final XPU backtest - FORCE trades by using classification instead of regression.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime
import time

# Check XPU availability
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = torch.device('xpu')
    print(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
    print(f"XPU Memory: {torch.xpu.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device('cpu')
    print("WARNING: Using CPU - XPU not available!")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


class TradingNet(nn.Module):
    """3-class classification: BUY, SELL, HOLD"""
    
    def __init__(self, input_dim=16, hidden_dim=256):
        super().__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                           batch_first=True, dropout=0.2)
        
        # 3 outputs for 3 classes
        self.fc = nn.Linear(hidden_dim, 3)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        logits = self.fc(last_out)
        return logits  # Return raw logits for CrossEntropyLoss


def create_sequences_gpu(data, seq_len=20):
    """Create sequences with 3-class labels."""
    sequences = []
    labels = []
    
    features = ['open', 'high', 'low', 'close', 'volume', 
                'returns', 'volatility', 'rsi', 'macd', 'bb_signal',
                'vwap_ratio', 'volume_ratio', 'price_momentum',
                'micro_trend', 'spread', 'tick_imbalance']
    
    X = data[features].values
    
    # Create 3-class labels based on future returns
    future_returns = data['returns'].shift(-1).fillna(0).values
    volatility = data['volatility'].rolling(20).mean().fillna(0.001).values
    
    # Dynamic thresholds
    buy_threshold = volatility * 1.0
    sell_threshold = -volatility * 1.0
    
    # 0 = HOLD, 1 = BUY, 2 = SELL
    y = np.where(future_returns > buy_threshold, 1,
                 np.where(future_returns < sell_threshold, 2, 0))
    
    for i in range(seq_len, len(X) - 1):  # -1 because we look ahead
        sequences.append(X[i-seq_len:i])
        labels.append(y[i])
    
    # Log class distribution
    unique, counts = np.unique(labels, return_counts=True)
    class_dist = dict(zip(['HOLD', 'BUY', 'SELL'], 
                          [counts[unique == i][0] if i in unique else 0 
                           for i in range(3)]))
    logger.info(f"Class distribution: {class_dist}")
    
    return (torch.FloatTensor(np.array(sequences)).to(DEVICE),
            torch.LongTensor(labels).to(DEVICE))


def run_backtest_gpu(model, data, seq_len=20, initial_capital=10000):
    """Backtest with 3-class predictions."""
    model.eval()
    
    features = ['open', 'high', 'low', 'close', 'volume',
                'returns', 'volatility', 'rsi', 'macd', 'bb_signal',
                'vwap_ratio', 'volume_ratio', 'price_momentum',
                'micro_trend', 'spread', 'tick_imbalance']
    
    X = torch.FloatTensor(data[features].values).to(DEVICE)
    prices = data['close'].values
    
    # Generate predictions
    sequences = []
    for i in range(seq_len, len(X)):
        sequences.append(X[i-seq_len:i].unsqueeze(0))
    
    if not sequences:
        return {}
    
    # Batch prediction
    batch_size = 512
    all_preds = []
    
    logger.info("Generating predictions on XPU...")
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.cat(sequences[i:i+batch_size], dim=0)
            logits = model(batch)
            # Get class predictions (0=HOLD, 1=BUY, 2=SELL)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds)
    
    predictions = torch.cat(all_preds).cpu().numpy()
    
    pred_time = time.time() - start_time
    logger.info(f"Generated {len(predictions)} predictions in {pred_time:.2f}s")
    
    # Count predictions
    unique, counts = np.unique(predictions, return_counts=True)
    pred_dist = dict(zip(['HOLD', 'BUY', 'SELL'], 
                        [counts[unique == i][0] if i in unique else 0 
                         for i in range(3)]))
    logger.info(f"Prediction distribution: {pred_dist}")
    
    # Simulate trading
    capital = initial_capital
    position = 0
    trades = []
    
    for i in range(len(predictions)):
        pred = predictions[i]
        price = prices[i + seq_len]
        
        if pred == 1 and position == 0:  # BUY
            shares = capital * 0.95 / price  # Use 95% of capital
            position = shares
            trades.append({
                'type': 'buy',
                'price': price,
                'shares': shares,
                'capital': capital
            })
            capital -= shares * price
            
        elif pred == 2 and position > 0:  # SELL
            capital += position * price
            trades.append({
                'type': 'sell',
                'price': price,
                'shares': position,
                'capital': capital
            })
            position = 0
    
    # Close final position
    if position > 0:
        final_price = prices[-1]
        capital += position * final_price
        trades.append({
            'type': 'sell',
            'price': final_price,
            'shares': position,
            'capital': capital
        })
    
    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital * 100
    num_trades = len(trades)
    
    if num_trades > 0:
        # Calculate win rate from paired trades
        wins = 0
        paired_trades = 0
        for i in range(0, len(trades)-1, 2):
            if i+1 < len(trades) and trades[i]['type'] == 'buy' and trades[i+1]['type'] == 'sell':
                paired_trades += 1
                if trades[i+1]['price'] > trades[i]['price']:
                    wins += 1
        
        win_rate = (wins / paired_trades * 100) if paired_trades > 0 else 0
    else:
        win_rate = 0
    
    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'final_capital': capital,
        'predictions': pred_dist
    }


def main():
    logger.info("="*60)
    logger.info("FINAL XPU BACKTEST - 3-CLASS CLASSIFICATION")
    logger.info("="*60)
    
    # Load data - use new 30-day data
    data_path = Path("data/btc_30d/btc_usdt_30d.parquet")
    if not data_path.exists():
        # Fallback to old data
        data_files = list(Path("data/parquet/BTC_USDT").glob("**/*.parquet"))
        if not data_files:
            logger.error("No BTC data files found")
            return
        dfs = []
        for f in data_files:
            dfs.append(pd.read_parquet(f))
        data = pd.concat(dfs, ignore_index=True)
        logger.info(f"Loaded {len(data):,} records from {len(data_files)} old files")
    else:
        data = pd.read_parquet(data_path)
        logger.info(f"Loaded {len(data):,} records from {data_path}")
    
    # Add required features
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(20).std()
    data['rsi'] = 50
    data['macd'] = 0
    data['bb_signal'] = 0
    data['vwap_ratio'] = data['close'] / data['close'].rolling(20).mean()
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    data['price_momentum'] = data['close'].pct_change(5)
    data['micro_trend'] = data['close'].pct_change()
    data['spread'] = (data['high'] - data['low']) / data['close']
    data['tick_imbalance'] = data['volume'].diff()
    data = data.fillna(0)
    
    # Create sequences
    seq_len = 20
    X, y = create_sequences_gpu(data, seq_len)
    logger.info(f"Created {len(X):,} sequences on {DEVICE}")
    
    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Model and training
    model = TradingNet(input_dim=16, hidden_dim=256).to(DEVICE)
    
    # Weighted loss to handle class imbalance
    class_counts = torch.bincount(y_train)
    class_weights = 1.0 / (class_counts.float() + 1)
    class_weights = class_weights / class_weights.sum()
    
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(DEVICE))
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # Training
    batch_size = 512
    num_epochs = 10
    
    logger.info(f"\nTraining on {DEVICE}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        correct = 0
        
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_X = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == batch_y).sum().item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            _, val_pred = torch.max(val_outputs, 1)
            val_correct = (val_pred == y_test).sum().item()
            val_acc = val_correct / len(y_test) * 100
        
        train_acc = correct / len(X_train) * 100
        avg_loss = total_loss / (len(X_train) // batch_size)
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
                   f"Loss: {avg_loss:.4f}, Train: {train_acc:.2f}%, Val: {val_acc:.2f}%")
    
    # Run backtest
    logger.info("\nRunning backtest...")
    test_data = data.iloc[split_idx+seq_len:]
    results = run_backtest_gpu(model, test_data, seq_len=seq_len)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Trades: {results['num_trades']}")
    logger.info(f"Return: {results['total_return']:.2f}%")
    logger.info(f"Win Rate: {results['win_rate']:.1f}%")
    logger.info(f"Final Capital: ${results['final_capital']:.2f}")
    logger.info(f"Predictions: {results['predictions']}")
    
    if DEVICE.type == 'xpu':
        mem_used = torch.xpu.memory_allocated() / 1e9
        mem_total = torch.xpu.get_device_properties(0).total_memory / 1e9
        logger.info(f"\nXPU Memory: {mem_used:.1f}/{mem_total:.1f} GB ({mem_used/mem_total*100:.1f}%)")
    
    if results['num_trades'] == 0:
        logger.warning("\nNo trades generated!")
    else:
        logger.info(f"\nSUCCESS! Generated {results['num_trades']} trades")


if __name__ == "__main__":
    main()