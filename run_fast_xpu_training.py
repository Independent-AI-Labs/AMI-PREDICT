#!/usr/bin/env python
"""
Fast XPU training on 5-year data WITHOUT complex feature engineering.
Direct SQLite to XPU pipeline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import sqlite3
from pathlib import Path
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Check XPU
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = torch.device('xpu')
    logger.info(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
    logger.info(f"XPU Memory: {torch.xpu.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device('cpu')
    logger.info("WARNING: Using CPU")


class FastTCN(nn.Module):
    """Simple TCN for fast training."""
    
    def __init__(self, input_size=10, hidden_size=128):
        super().__init__()
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, dilation=2)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1, dilation=4)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        # x: (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = torch.relu(self.conv1(x))
        x = self.dropout(x)
        x = torch.relu(self.conv2(x))
        x = self.dropout(x)
        x = torch.relu(self.conv3(x))
        x = self.pool(x).squeeze(-1)
        return torch.sigmoid(self.fc(x))


def load_data_fast(db_path, symbol, limit=None):
    """Load data with SIMPLE features only."""
    conn = sqlite3.connect(db_path)
    
    query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM market_data_1m
        WHERE symbol = '{symbol}'
        ORDER BY timestamp
    """
    
    if limit:
        query += f" LIMIT {limit}"
    
    logger.info(f"Loading {symbol} data...")
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df):,} records")
    
    # SIMPLE features only - no complex calculations
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['hl_ratio'] = (df['high'] - df['low']) / df['close']
    df['volume_norm'] = df['volume'] / df['volume'].rolling(20, min_periods=1).mean()
    df['price_ma_ratio'] = df['close'] / df['close'].rolling(20, min_periods=1).mean()
    
    # Simple momentum
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_10'] = df['close'].pct_change(10)
    
    # Simple volatility
    df['volatility'] = df['returns'].rolling(20, min_periods=1).std()
    
    # Price position in range
    df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-10)
    
    df = df.fillna(0)
    
    return df


def create_sequences_fast(df, seq_len=50):
    """Create sequences for training."""
    features = ['returns', 'log_returns', 'hl_ratio', 'volume_norm', 
                'price_ma_ratio', 'momentum_5', 'momentum_10', 
                'volatility', 'price_position', 'volume']
    
    X = df[features].values
    y = (df['returns'].shift(-1) > 0).astype(float).values
    
    sequences = []
    labels = []
    
    for i in range(seq_len, len(X) - 1):
        sequences.append(X[i-seq_len:i])
        labels.append(y[i])
    
    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)


def train_on_xpu(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=2048):
    """Train model on XPU with large batches."""
    model = model.to(DEVICE)
    
    # Move data to XPU
    X_train = torch.FloatTensor(X_train).to(DEVICE)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(DEVICE)
    X_val = torch.FloatTensor(X_val).to(DEVICE)
    y_val = torch.FloatTensor(y_val).reshape(-1, 1).to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    logger.info(f"Training on {DEVICE} with batch size {batch_size}")
    logger.info(f"Train samples: {len(X_train):,}, Val samples: {len(X_val):,}")
    
    train_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        correct = 0
        
        # Shuffle
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
            
            total_loss += loss.item() * len(batch_X)
            predictions = (outputs > 0.5).float()
            correct += (predictions == batch_y).sum().item()
        
        # Validation
        model.eval()
        val_correct = 0
        val_loss = 0
        
        with torch.no_grad():
            for i in range(0, len(X_val), batch_size):
                batch_X = X_val[i:i+batch_size]
                batch_y = y_val[i:i+batch_size]
                
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item() * len(batch_X)
                predictions = (outputs > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
        
        scheduler.step()
        
        train_acc = correct / len(X_train) * 100
        val_acc = val_correct / len(X_val) * 100
        avg_loss = total_loss / len(X_train)
        avg_val_loss = val_loss / len(X_val)
        epoch_time = time.time() - epoch_start
        
        logger.info(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
                   f"Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
                   f"Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
    
    train_time = time.time() - train_start
    throughput = len(X_train) * epochs / train_time
    
    logger.info(f"\nTraining complete in {train_time:.1f}s")
    logger.info(f"Throughput: {throughput:.0f} samples/sec")
    
    if DEVICE.type == 'xpu':
        mem_used = torch.xpu.memory_allocated() / 1e9
        mem_total = torch.xpu.get_device_properties(0).total_memory / 1e9
        logger.info(f"XPU Memory: {mem_used:.2f}/{mem_total:.1f} GB ({mem_used/mem_total*100:.1f}%)")
    
    return model


def main():
    logger.info("="*60)
    logger.info("FAST XPU TRAINING ON 5-YEAR DATA")
    logger.info("="*60)
    
    # Load data
    db_path = 'data/crypto_5years.db'
    
    # Train on all symbols
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
    
    all_sequences = []
    all_labels = []
    
    for symbol in symbols:
        df = load_data_fast(db_path, symbol, limit=500000)  # 500K records per symbol
        
        # Create sequences
        X, y = create_sequences_fast(df, seq_len=50)
        logger.info(f"{symbol}: {len(X):,} sequences created")
        
        all_sequences.append(X)
        all_labels.append(y)
    
    # Combine all data
    X_combined = np.concatenate(all_sequences)
    y_combined = np.concatenate(all_labels)
    
    logger.info(f"\nTotal sequences: {len(X_combined):,}")
    
    # Shuffle combined data
    indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[indices]
    y_combined = y_combined[indices]
    
    # Split
    split_idx = int(len(X_combined) * 0.8)
    X_train = X_combined[:split_idx]
    y_train = y_combined[:split_idx]
    X_val = X_combined[split_idx:]
    y_val = y_combined[split_idx:]
    
    # Create and train model
    model = FastTCN(input_size=10, hidden_size=128)
    
    # Train with large batch size for XPU efficiency
    batch_size = 4096 if DEVICE.type == 'xpu' else 512
    model = train_on_xpu(model, X_train, y_train, X_val, y_val, 
                        epochs=10, batch_size=batch_size)
    
    # Save model
    model_path = Path('models/fast_tcn_5year.pth')
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    logger.info(f"\nModel saved to {model_path}")
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)


if __name__ == "__main__":
    main()