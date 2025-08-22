#!/usr/bin/env python
"""
Parallel training using PostgreSQL for data loading.
Optimized for Intel XPU with parallel data fetching.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import logging
from pathlib import Path
import json

# Import our models
from src.ml.scalping_models import ScalpingModelFactory

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# PostgreSQL configuration
PG_HOST = '172.72.72.2'
PG_PORT = 5432
PG_DB = 'postgres'
PG_USER = 'postgres'
PG_PASS = 'postgres'

# Check XPU
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    DEVICE = torch.device('xpu')
    logger.info(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
    logger.info(f"XPU Memory: {torch.xpu.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    DEVICE = torch.device('cpu')
    logger.info("Using CPU")


class ParallelDataLoader:
    """Parallel data loader using PostgreSQL."""
    
    def __init__(self, symbols, start_date, end_date, batch_size=100000):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.batch_size = batch_size
        
        # Create connection pool
        self.pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=len(symbols) * 2,
            host=PG_HOST,
            port=PG_PORT,
            database=PG_DB,
            user=PG_USER,
            password=PG_PASS
        )
        
    def load_symbol_data(self, symbol):
        """Load data for a single symbol from PostgreSQL."""
        conn = self.pool.getconn()
        try:
            query = """
                SELECT timestamp, open, high, low, close, volume
                FROM market_data_1m
                WHERE symbol = %s 
                AND timestamp BETWEEN %s AND %s
                ORDER BY timestamp
            """
            
            df = pd.read_sql_query(
                query, conn,
                params=(symbol, self.start_date, self.end_date)
            )
            
            if len(df) > 0:
                # Add features
                df['returns'] = df['close'].pct_change()
                df['volatility'] = df['returns'].rolling(20).std()
                df['rsi'] = self.calculate_rsi(df['close'])
                df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
                df['price_momentum'] = df['close'].pct_change(5)
                df = df.fillna(0)
                
            return symbol, df
            
        finally:
            self.pool.putconn(conn)
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def load_all_parallel(self):
        """Load all symbols in parallel."""
        logger.info(f"Loading data for {len(self.symbols)} symbols in parallel...")
        
        all_data = {}
        with ThreadPoolExecutor(max_workers=len(self.symbols)) as executor:
            futures = [executor.submit(self.load_symbol_data, symbol) 
                      for symbol in self.symbols]
            
            for future in as_completed(futures):
                symbol, df = future.result()
                all_data[symbol] = df
                logger.info(f"Loaded {len(df):,} records for {symbol}")
        
        return all_data


def create_sequences(data, seq_len=100):
    """Create sequences for training."""
    features = ['open', 'high', 'low', 'close', 'volume', 
                'returns', 'volatility', 'rsi', 'volume_ratio', 'price_momentum']
    
    sequences = []
    labels = []
    
    for symbol, df in data.items():
        if len(df) < seq_len + 1:
            continue
            
        X = df[features].values
        y = (df['returns'].shift(-1) > 0).astype(float).values
        
        for i in range(seq_len, len(X) - 1):
            sequences.append(X[i-seq_len:i])
            labels.append(y[i])
    
    return np.array(sequences), np.array(labels)


def train_model_xpu(model, X_train, y_train, X_val, y_val, epochs=10):
    """Train model on XPU."""
    model = model.to(DEVICE)
    
    # Convert to tensors
    X_train = torch.FloatTensor(X_train).to(DEVICE)
    y_train = torch.FloatTensor(y_train).reshape(-1, 1).to(DEVICE)
    X_val = torch.FloatTensor(X_val).to(DEVICE)
    y_val = torch.FloatTensor(y_val).reshape(-1, 1).to(DEVICE)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    
    batch_size = 1024 if DEVICE.type == 'xpu' else 32
    
    logger.info(f"Training on {DEVICE} with batch size {batch_size}")
    
    best_val_loss = float('inf')
    train_start = time.time()
    
    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        
        # Shuffle data
        indices = torch.randperm(len(X_train))
        
        for i in range(0, len(X_train), batch_size):
            batch_idx = indices[i:i+batch_size]
            batch_X = X_train[batch_idx]
            batch_y = y_train[batch_idx]
            
            optimizer.zero_grad()
            
            # Handle different model types
            if hasattr(model, 'forward'):
                outputs = model(batch_X)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
            else:
                outputs = model(batch_X)
            
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            if isinstance(val_outputs, tuple):
                val_outputs = val_outputs[0]
            val_loss = criterion(val_outputs, y_val).item()
            
            # Accuracy
            val_pred = (val_outputs > 0.5).float()
            val_acc = (val_pred == y_val).float().mean().item() * 100
        
        epoch_time = time.time() - epoch_start
        avg_loss = total_loss / (len(X_train) // batch_size)
        
        logger.info(f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
                   f"Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}, "
                   f"Val Acc: {val_acc:.2f}%")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    train_time = time.time() - train_start
    throughput = len(X_train) * epochs / train_time
    
    logger.info(f"Training complete in {train_time:.1f}s")
    logger.info(f"Throughput: {throughput:.0f} samples/sec")
    
    if DEVICE.type == 'xpu':
        mem_used = torch.xpu.memory_allocated() / 1e9
        mem_total = torch.xpu.get_device_properties(0).total_memory / 1e9
        logger.info(f"XPU Memory: {mem_used:.2f}/{mem_total:.1f} GB ({mem_used/mem_total*100:.1f}%)")
    
    return model


def main():
    logger.info("="*60)
    logger.info("PARALLEL POSTGRESQL + XPU TRAINING")
    logger.info("="*60)
    
    # Configuration
    symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT']
    start_date = '2024-01-01'
    end_date = '2025-01-01'
    
    # Load data in parallel from PostgreSQL
    loader = ParallelDataLoader(symbols, start_date, end_date)
    data_start = time.time()
    data = loader.load_all_parallel()
    data_time = time.time() - data_start
    
    total_records = sum(len(df) for df in data.values())
    logger.info(f"Loaded {total_records:,} total records in {data_time:.1f}s")
    logger.info(f"Loading speed: {total_records/data_time:.0f} records/sec")
    
    # Create sequences
    logger.info("\nCreating training sequences...")
    X, y = create_sequences(data, seq_len=100)
    logger.info(f"Created {len(X):,} sequences")
    
    # Split data
    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]
    
    logger.info(f"Train: {len(X_train):,}, Validation: {len(X_val):,}")
    
    # Train models
    models_to_train = ['tcn', 'transformer', 'lstm']
    factory = ScalpingModelFactory()
    results = {}
    
    for model_name in models_to_train:
        logger.info(f"\n{'='*40}")
        logger.info(f"Training {model_name.upper()}")
        logger.info(f"{'='*40}")
        
        if model_name == 'tcn':
            model = factory.create_tcn(input_size=10)
        elif model_name == 'transformer':
            model = factory.create_transformer(input_size=10)
        elif model_name == 'lstm':
            model = factory.create_lstm(input_size=10)
        
        model = train_model_xpu(model, X_train, y_train, X_val, y_val, epochs=5)
        
        # Save model
        model_path = Path(f'models/{model_name}_pg_parallel.pth')
        model_path.parent.mkdir(exist_ok=True)
        torch.save(model.state_dict(), model_path)
        logger.info(f"Saved model to {model_path}")
        
        results[model_name] = {
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'device': str(DEVICE)
        }
    
    # Save results
    with open('parallel_training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    
    loader.pool.closeall()


if __name__ == "__main__":
    main()