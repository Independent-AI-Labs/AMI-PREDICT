#!/usr/bin/env python
"""
XPU-optimized backtest with aggressive focal loss and proper weight initialization.
Uses Intel Arc A770 for maximum performance with trade generation focus.
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


class AggressiveFocalLoss(nn.Module):
    """Aggressive focal loss to force trade generation."""
    
    def __init__(self, alpha=0.1, gamma=3.0):  # More aggressive parameters
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Aggressive alpha weighting - heavily penalize not trading
        alpha_t = torch.where(targets == 1, 1 - self.alpha, self.alpha)
        
        loss = alpha_t * focal_weight * bce_loss
        return loss.mean()


class AggressiveScalpingNet(nn.Module):
    """Network with aggressive initialization for trade generation."""
    
    def __init__(self, input_dim=16, hidden_dim=256):
        super().__init__()
        
        # Larger network for more capacity
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers=3, 
                         batch_first=True, dropout=0.3)
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1)
        )
        
        # Multiple heads for decision making
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        
        # Aggressive weight initialization
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights to encourage trading."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Xavier with gain > 1 for more aggressive outputs
                nn.init.xavier_uniform_(m.weight, gain=2.0)
                if m.bias is not None:
                    # Positive bias to encourage trading
                    nn.init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param, gain=1.5)
                    elif 'bias' in name:
                        nn.init.constant_(param, 0.1)
    
    def forward(self, x):
        # GRU encoding
        gru_out, _ = self.gru(x)
        
        # Attention weights
        attn_weights = torch.softmax(self.attention(gru_out), dim=1)
        
        # Weighted average
        context = torch.sum(gru_out * attn_weights, dim=1)
        
        # Output layers with ReLU to encourage positive signals
        out = torch.relu(self.fc1(context))
        out = self.dropout(out)
        out = self.fc2(out)
        
        # Sigmoid with temperature scaling for more extreme predictions
        temperature = 0.5  # Lower temperature = more extreme predictions
        return torch.sigmoid(out / temperature)


def create_sequences_gpu(data, seq_len=20):
    """Create sequences directly on GPU."""
    sequences = []
    labels = []
    
    features = ['open', 'high', 'low', 'close', 'volume', 
                'returns', 'volatility', 'rsi', 'macd', 'bb_signal',
                'vwap_ratio', 'volume_ratio', 'price_momentum',
                'micro_trend', 'spread', 'tick_imbalance']
    
    X = data[features].values
    
    # Create buy/sell signals with more aggressive thresholds
    returns = data['returns'].values
    volatility = data['volatility'].rolling(20).mean().fillna(0.001).values
    
    # More aggressive signal generation
    buy_threshold = volatility * 0.5  # Lower threshold
    sell_threshold = -volatility * 0.5
    
    y = np.where(returns > buy_threshold, 1,
                 np.where(returns < sell_threshold, -1, 0))
    
    # Convert 0/-1/1 to binary (1 = trade, 0 = no trade)
    # But weight towards trading
    y_binary = np.where(y != 0, 1, 0).astype(np.float32)
    
    # Add some random trades to prevent model from learning to never trade
    noise_mask = np.random.random(len(y_binary)) < 0.1  # 10% random trades
    y_binary[noise_mask] = 1
    
    for i in range(seq_len, len(X)):
        sequences.append(X[i-seq_len:i])
        labels.append(y_binary[i])
    
    # Log class distribution
    trade_pct = np.mean(labels) * 100
    logger.info(f"Signal distribution: {trade_pct:.1f}% trades, {100-trade_pct:.1f}% no-trades")
    
    return (torch.FloatTensor(np.array(sequences)).to(DEVICE),
            torch.FloatTensor(labels).reshape(-1, 1).to(DEVICE))


def run_backtest_gpu(model, data, seq_len=20, initial_capital=10000):
    """GPU-accelerated backtest with aggressive trading."""
    model.eval()
    
    features = ['open', 'high', 'low', 'close', 'volume',
                'returns', 'volatility', 'rsi', 'macd', 'bb_signal',
                'vwap_ratio', 'volume_ratio', 'price_momentum',
                'micro_trend', 'spread', 'tick_imbalance']
    
    X = torch.FloatTensor(data[features].values).to(DEVICE)
    prices = data['close'].values
    
    # Generate all predictions at once on GPU
    sequences = []
    for i in range(seq_len, len(X)):
        sequences.append(X[i-seq_len:i].unsqueeze(0))
    
    if not sequences:
        return {}
    
    # Batch prediction
    batch_size = 512
    all_preds = []
    all_confidence = []
    
    logger.info("Generating predictions on XPU...")
    start_time = time.time()
    
    with torch.no_grad():
        for i in range(0, len(sequences), batch_size):
            batch = torch.cat(sequences[i:i+batch_size], dim=0)
            preds = model(batch)
            all_preds.append(preds)
            
            # Calculate confidence (distance from 0.5)
            confidence = torch.abs(preds - 0.5) * 2
            all_confidence.append(confidence)
    
    predictions = torch.cat(all_preds).cpu().numpy().flatten()
    confidences = torch.cat(all_confidence).cpu().numpy().flatten()
    
    pred_time = time.time() - start_time
    logger.info(f"Generated {len(predictions)} predictions in {pred_time:.2f}s ({len(predictions)/pred_time:.0f} pred/sec)")
    
    # Aggressive dynamic thresholds
    pred_mean = np.mean(predictions)
    pred_std = np.std(predictions)
    
    # Use percentiles for more aggressive trading
    buy_threshold = np.percentile(predictions, 60)  # Top 40% are buys
    sell_threshold = np.percentile(predictions, 40)  # Bottom 40% are sells
    
    logger.info(f"Prediction stats: mean={pred_mean:.3f}, std={pred_std:.3f}")
    logger.info(f"Thresholds: Buy>{buy_threshold:.3f}, Sell<{sell_threshold:.3f}")
    
    # Simulate trading
    capital = initial_capital
    position = 0
    trades = []
    
    for i in range(len(predictions)):
        pred = predictions[i]
        confidence = confidences[i]
        price = prices[i + seq_len]
        
        # Aggressive position sizing based on confidence
        position_size = min(0.5, 0.1 + confidence * 0.4)  # 10-50% of capital
        
        if pred > buy_threshold and position <= 0:
            # Buy signal
            shares = (capital * position_size) / price
            position = shares
            trades.append({
                'type': 'buy',
                'price': price,
                'shares': shares,
                'confidence': confidence,
                'capital': capital
            })
            capital -= shares * price
            
        elif pred < sell_threshold and position > 0:
            # Sell signal
            capital += position * price
            trades.append({
                'type': 'sell',
                'price': price,
                'shares': position,
                'confidence': confidence,
                'capital': capital
            })
            position = 0
    
    # Close final position
    if position > 0:
        final_price = prices[-1]
        capital += position * final_price
    
    # Calculate metrics
    total_return = (capital - initial_capital) / initial_capital * 100
    num_trades = len(trades)
    
    if num_trades > 0:
        winning_trades = sum(1 for i in range(0, len(trades)-1, 2) 
                           if i+1 < len(trades) and 
                           trades[i+1]['price'] > trades[i]['price'])
        win_rate = winning_trades / (num_trades // 2) * 100 if num_trades >= 2 else 0
        
        avg_confidence = np.mean([t['confidence'] for t in trades])
        avg_position = np.mean([t['shares'] * t['price'] / t['capital'] 
                               for t in trades if t['type'] == 'buy'])
    else:
        win_rate = 0
        avg_confidence = 0
        avg_position = 0
    
    # Calculate Sharpe ratio
    if num_trades > 2:
        trade_returns = []
        for i in range(0, len(trades)-1, 2):
            if i+1 < len(trades):
                ret = (trades[i+1]['price'] - trades[i]['price']) / trades[i]['price']
                trade_returns.append(ret)
        
        if trade_returns:
            sharpe = np.mean(trade_returns) / (np.std(trade_returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe = 0
    else:
        sharpe = 0
    
    return {
        'total_return': total_return,
        'num_trades': num_trades,
        'win_rate': win_rate,
        'sharpe_ratio': sharpe,
        'final_capital': capital,
        'avg_confidence': avg_confidence,
        'avg_position_size': avg_position
    }


def main():
    logger.info("="*60)
    logger.info("XPU-OPTIMIZED AGGRESSIVE BACKTEST")
    logger.info("="*60)
    
    # Load data - use existing parquet files
    data_files = list(Path("data/parquet/BTC_USDT").glob("**/*.parquet"))
    if not data_files:
        logger.error("No BTC data files found")
        return
    
    # Load all BTC data
    dfs = []
    for f in data_files:
        dfs.append(pd.read_parquet(f))
    data = pd.concat(dfs, ignore_index=True)
    
    logger.info(f"Loaded {len(data):,} records from {len(data_files)} files")
    
    # Add required features
    data['returns'] = data['close'].pct_change()
    data['volatility'] = data['returns'].rolling(20).std()
    data['rsi'] = 50  # Placeholder
    data['macd'] = 0
    data['bb_signal'] = 0
    data['vwap_ratio'] = data['close'] / data['close'].rolling(20).mean()
    data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
    data['price_momentum'] = data['close'].pct_change(5)
    data['micro_trend'] = data['close'].pct_change()
    data['spread'] = (data['high'] - data['low']) / data['close']
    data['tick_imbalance'] = data['volume'].diff()
    data = data.fillna(0)
    
    # Use recent data for better patterns
    data = data.tail(150000)
    logger.info(f"Using {len(data):,} records")
    
    # Create sequences on GPU
    seq_len = 20
    X, y = create_sequences_gpu(data, seq_len)
    logger.info(f"Created {len(X):,} sequences on {DEVICE}")
    
    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_test, y_test = X[split_idx:], y[split_idx:]
    
    logger.info(f"Train: {len(X_train):,}, Test: {len(X_test):,}")
    
    # Check XPU memory
    if DEVICE.type == 'xpu':
        mem_used = torch.xpu.memory_allocated() / 1e9
        mem_total = torch.xpu.get_device_properties(0).total_memory / 1e9
        logger.info(f"XPU Memory: {mem_used:.2f}/{mem_total:.1f} GB ({mem_used/mem_total*100:.1f}% utilized)")
    
    # Create model with aggressive initialization
    model = AggressiveScalpingNet(input_dim=16, hidden_dim=256).to(DEVICE)
    criterion = AggressiveFocalLoss(alpha=0.1, gamma=3.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    
    # Training with larger batches for GPU efficiency
    batch_size = 1024
    num_epochs = 20
    
    logger.info(f"\nTraining on {DEVICE} with batch size {batch_size}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        correct = 0
        
        # Shuffle indices
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_X = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pred_binary = (outputs > 0.5).float()
            correct += (pred_binary == batch_y).sum().item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_pred = (val_outputs > 0.5).float()
            val_acc = (val_pred == y_test).float().mean().item() * 100
        
        scheduler.step()
        
        train_acc = correct / len(X_train) * 100
        avg_loss = total_loss / (len(X_train) // batch_size)
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} ({epoch_time:.1f}s) - "
                   f"Loss: {avg_loss:.4f}, Train: {train_acc:.2f}%, Val: {val_acc:.2f}%")
    
    logger.info(f"Training completed in {time.time() - epoch_start:.1f}s")
    
    # Calculate throughput
    total_samples = num_epochs * len(X_train)
    total_time = num_epochs * epoch_time
    logger.info(f"Throughput: {total_samples/total_time:.0f} samples/sec")
    
    # Run backtest on test data
    logger.info("\nRunning GPU-accelerated backtest...")
    test_data = data.iloc[split_idx+seq_len:]
    results = run_backtest_gpu(model, test_data, seq_len=seq_len)
    
    # Print results
    logger.info("\n" + "="*60)
    logger.info("BACKTEST RESULTS")
    logger.info("="*60)
    logger.info(f"Trades: {results['num_trades']}")
    logger.info(f"Return: {results['total_return']:.2f}%")
    logger.info(f"Win Rate: {results['win_rate']:.1f}%")
    logger.info(f"Sharpe: {results['sharpe_ratio']:.2f}")
    logger.info(f"Final Capital: ${results['final_capital']:.2f}")
    logger.info(f"Avg Confidence: {results['avg_confidence']:.3f}")
    logger.info(f"Avg Position Size: {results['avg_position_size']:.3f}")
    
    # Final memory check
    if DEVICE.type == 'xpu':
        mem_used = torch.xpu.memory_allocated() / 1e9
        mem_total = torch.xpu.get_device_properties(0).total_memory / 1e9
        logger.info(f"\nXPU Memory: {mem_used:.1f}/{mem_total:.1f} GB ({mem_used/mem_total*100:.1f}% utilized)")
    
    if results['num_trades'] == 0:
        logger.warning("\nStill no trades - need even more aggressive parameters")
    else:
        logger.info(f"\nGenerated {results['num_trades']} trades!")


if __name__ == "__main__":
    main()