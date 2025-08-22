#!/usr/bin/env python
"""
Deep Learning Experiment with GPU Acceleration
Training sophisticated models on 2-year data
"""
import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.core import ConfigManager, Database

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    # Check for XPU first, then CUDA, then CPU
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        DEVICE = torch.device('xpu')
        device_name = torch.xpu.get_device_name(0)
        print(f"PyTorch available. Using Intel XPU: {device_name}")
    elif torch.cuda.is_available():
        DEVICE = torch.device('cuda')
        print(f"PyTorch available. Using CUDA GPU")
    else:
        DEVICE = torch.device('cpu')
        print(f"PyTorch available. Using CPU")
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available. Please install: pip install torch")

COMMISSION_RATE = 0.00075  # Binance with BNB

class DeepLearningExperiment:
    """Deep learning models for crypto trading"""
    
    def __init__(self):
        self.config = ConfigManager()
        self.database = Database(self.config.get('database'))
        
    def load_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Load data from database"""
        session = self.database.get_session()
        try:
            from src.core.database import MarketData
            
            data = []
            for row in session.query(MarketData).filter(
                MarketData.symbol == symbol,
                MarketData.timeframe == timeframe
            ).order_by(MarketData.timestamp):
                data.append({
                    'timestamp': row.timestamp,
                    'open': float(row.open),
                    'high': float(row.high),
                    'low': float(row.low),
                    'close': float(row.close),
                    'volume': float(row.volume)
                })
            
            return pd.DataFrame(data)
            
        finally:
            session.close()
    
    def prepare_sequences(self, df: pd.DataFrame, seq_length: int = 60):
        """Prepare sequences for deep learning"""
        
        # Calculate features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['volume_change'] = df['volume'].pct_change()
        
        # Normalize OHLCV
        df['open_norm'] = df['open'] / df['close']
        df['high_norm'] = df['high'] / df['close']
        df['low_norm'] = df['low'] / df['close']
        df['volume_norm'] = df['volume'] / df['volume'].rolling(20).mean()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = (100 - (100 / (1 + rs))) / 100  # Normalize to 0-1
        
        # Target
        df['target'] = (df['close'].shift(-5) > df['close']).astype(int)
        
        # Drop NaN
        df = df.dropna()
        
        # Select features for sequences
        feature_cols = ['returns', 'log_returns', 'volume_change', 
                       'open_norm', 'high_norm', 'low_norm', 'volume_norm', 'rsi']
        
        # Create sequences
        X, y = [], []
        for i in range(seq_length, len(df) - 5):
            X.append(df[feature_cols].iloc[i-seq_length:i].values)
            y.append(df['target'].iloc[i])
        
        return np.array(X), np.array(y), df.iloc[seq_length:-5]
    
    def create_lstm_model(self, input_dim: int, seq_length: int):
        """Create LSTM model"""
        class LSTMModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm1 = nn.LSTM(input_dim, 128, batch_first=True, dropout=0.2)
                self.lstm2 = nn.LSTM(128, 64, batch_first=True, dropout=0.2)
                self.fc1 = nn.Linear(64, 32)
                self.fc2 = nn.Linear(32, 2)
                self.dropout = nn.Dropout(0.2)
                self.relu = nn.ReLU()
                
            def forward(self, x):
                x, _ = self.lstm1(x)
                x, _ = self.lstm2(x)
                x = x[:, -1, :]  # Take last timestep
                x = self.relu(self.fc1(x))
                x = self.dropout(x)
                x = self.fc2(x)
                return x
        
        return LSTMModel()
    
    def create_transformer_model(self, input_dim: int, seq_length: int):
        """Create Transformer model"""
        class TransformerModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.input_projection = nn.Linear(input_dim, 128)
                self.positional_encoding = nn.Parameter(torch.randn(1, seq_length, 128))
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=128, nhead=8, dim_feedforward=512, dropout=0.1
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
                self.fc1 = nn.Linear(128, 64)
                self.fc2 = nn.Linear(64, 2)
                self.dropout = nn.Dropout(0.1)
                
            def forward(self, x):
                x = self.input_projection(x)
                x = x + self.positional_encoding
                x = x.transpose(0, 1)  # Transformer expects seq_len first
                x = self.transformer(x)
                x = x.mean(dim=0)  # Global average pooling
                x = self.dropout(x)
                x = torch.relu(self.fc1(x))
                x = self.fc2(x)
                return x
        
        return TransformerModel()
    
    def train_model(self, model, X_train, y_train, X_val, y_val, epochs=20):
        """Train deep learning model"""
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(DEVICE)
        y_train_t = torch.LongTensor(y_train).to(DEVICE)
        X_val_t = torch.FloatTensor(X_val).to(DEVICE)
        y_val_t = torch.LongTensor(y_val).to(DEVICE)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        
        # Optimizer and loss
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        model.to(DEVICE)
        model.train()
        
        print(f"Training on {DEVICE}...")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            # Validation
            model.eval()
            with torch.no_grad():
                val_outputs = model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
                val_preds = torch.argmax(val_outputs, dim=1)
                val_acc = (val_preds == y_val_t).float().mean()
            model.train()
            
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Val Acc: {val_acc:.2%}")
        
        return model
    
    def predict(self, model, X_test):
        """Make predictions"""
        model.eval()
        X_test_t = torch.FloatTensor(X_test).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(X_test_t)
            predictions = torch.argmax(outputs, dim=1)
        
        return predictions.cpu().numpy()
    
    def backtest(self, predictions, test_df, initial_capital):
        """Simple backtest"""
        capital = initial_capital
        position = 0
        
        for i in range(len(predictions) - 1):
            price = test_df.iloc[i]['close']
            
            if predictions[i] == 1 and position == 0:
                # Buy
                position = capital * 0.95 / price
                capital = capital * 0.05
                
            elif predictions[i] == 0 and position > 0:
                # Sell
                sell_value = position * price
                capital = capital + sell_value * (1 - COMMISSION_RATE)
                position = 0
        
        # Close final position
        if position > 0:
            final_price = test_df.iloc[-1]['close']
            capital = capital + position * final_price * (1 - COMMISSION_RATE)
        
        return (capital - initial_capital) / initial_capital
    
    def run_experiment(self, symbol: str, timeframe: str):
        """Run complete deep learning experiment"""
        print(f"\n{'='*60}")
        print(f"Deep Learning Experiment: {symbol} {timeframe}")
        print(f"{'='*60}")
        
        if not TORCH_AVAILABLE:
            print("PyTorch not available. Skipping.")
            return None
        
        # Load data
        print("Loading data...")
        df = self.load_data(symbol, timeframe)
        print(f"Loaded {len(df):,} records")
        
        # Prepare sequences
        print("Preparing sequences...")
        seq_length = 60
        X, y, seq_df = self.prepare_sequences(df, seq_length)
        
        if len(X) < 1000:
            print(f"Insufficient sequences: {len(X)}")
            return None
        
        # Split data
        n = len(X)
        train_end = int(n * 0.6)
        val_end = int(n * 0.8)
        
        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]
        test_df = seq_df.iloc[val_end:]
        
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        results = {}
        
        # Train LSTM
        print("\nTraining LSTM...")
        lstm_model = self.create_lstm_model(X.shape[2], seq_length)
        lstm_model = self.train_model(lstm_model, X_train, y_train, X_val, y_val, epochs=10)
        
        lstm_preds = self.predict(lstm_model, X_test)
        lstm_acc = np.mean(lstm_preds == y_test)
        lstm_return = self.backtest(lstm_preds, test_df, 10000)
        
        print(f"LSTM - Accuracy: {lstm_acc:.2%}, Return: {lstm_return:.2%}")
        results['LSTM'] = {'accuracy': lstm_acc, 'return': lstm_return}
        
        # Train Transformer
        print("\nTraining Transformer...")
        transformer_model = self.create_transformer_model(X.shape[2], seq_length)
        transformer_model = self.train_model(transformer_model, X_train, y_train, X_val, y_val, epochs=10)
        
        trans_preds = self.predict(transformer_model, X_test)
        trans_acc = np.mean(trans_preds == y_test)
        trans_return = self.backtest(trans_preds, test_df, 10000)
        
        print(f"Transformer - Accuracy: {trans_acc:.2%}, Return: {trans_return:.2%}")
        results['Transformer'] = {'accuracy': trans_acc, 'return': trans_return}
        
        # Save results
        with open(f'dl_results_{symbol.replace("/", "_")}_{timeframe}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        return results

def main():
    """Run deep learning experiments"""
    experiment = DeepLearningExperiment()
    
    print("DEEP LEARNING CRYPTO TRADING EXPERIMENT")
    print("="*60)
    print("Training LSTM and Transformer models")
    print("Using GPU acceleration if available")
    print("="*60)
    
    # Test on main pairs
    configs = [
        ('BTC/USDT', '1h'),
        ('ETH/USDT', '1h'),
    ]
    
    all_results = {}
    
    for symbol, timeframe in configs:
        results = experiment.run_experiment(symbol, timeframe)
        if results:
            all_results[f"{symbol}_{timeframe}"] = results
    
    # Summary
    print("\n" + "="*60)
    print("DEEP LEARNING SUMMARY")
    print("="*60)
    
    for config, results in all_results.items():
        print(f"\n{config}:")
        for model_name, metrics in results.items():
            print(f"  {model_name}:")
            print(f"    Accuracy: {metrics['accuracy']:.2%}")
            print(f"    Return: {metrics['return']:.2%}")
    
    print("\n[COMPLETE] Deep learning experiments finished!")
    print(f"Results saved to dl_results_*.json files")

if __name__ == "__main__":
    main()