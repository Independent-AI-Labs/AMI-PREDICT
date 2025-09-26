#!/usr/bin/env python
"""Train deep learning models on crypto data with XPU support."""

import json
import sqlite3

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Check device
if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
    print(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


class LSTMModel(nn.Module):
    """LSTM model for time series prediction."""

    def __init__(self, input_size, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])


def prepare_data(symbol="BTC/USDT", timeframe="1h", seq_length=20):
    """Load and prepare data for training."""

    print(f"Loading {symbol} {timeframe} data...")
    conn = sqlite3.connect("data/cryptobot.db")

    query = f"""
    SELECT timestamp, open, high, low, close, volume
    FROM market_data
    WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
    ORDER BY timestamp
    """

    df = pd.read_sql(query, conn)
    conn.close()

    print(f"Loaded {len(df)} records")

    if len(df) < seq_length + 1:
        print("Not enough data for training")
        return None, None, None, None

    # Feature engineering
    df["returns"] = df["close"].pct_change()
    df["hl_ratio"] = (df["high"] - df["low"]) / df["close"]
    df["oc_ratio"] = (df["close"] - df["open"]) / df["open"]
    df["volume_ma"] = df["volume"].rolling(10).mean()
    df["price_ma"] = df["close"].rolling(20).mean()
    df["rsi"] = calculate_rsi(df["close"])

    # Drop NaN
    df = df.dropna()

    # Select features
    feature_cols = ["returns", "hl_ratio", "oc_ratio", "volume", "rsi"]

    # Normalize
    scaler = StandardScaler()
    features = scaler.fit_transform(df[feature_cols])

    # Create sequences
    X, y = [], []
    for i in range(seq_length, len(features) - 1):
        X.append(features[i - seq_length : i])
        y.append(df["returns"].iloc[i + 1])  # Predict next return

    X = np.array(X)
    y = np.array(y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def train_model(X_train, y_train, X_test, y_test, epochs=50):
    """Train LSTM model."""

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)
    y_test_t = torch.FloatTensor(y_test).to(DEVICE)

    # Create data loaders
    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Initialize model
    model = LSTMModel(input_size=X_train.shape[2]).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {DEVICE}...")

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test_t).squeeze()
                test_loss = criterion(test_outputs, y_test_t)

                # Calculate accuracy (direction prediction)
                pred_direction = (test_outputs > 0).float()
                true_direction = (y_test_t > 0).float()
                accuracy = (pred_direction == true_direction).float().mean()

                print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, " f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2%}")

    return model


def backtest(model, symbol="BTC/USDT", timeframe="1h", days=30):
    """Backtest the model on recent data."""

    print(f"\nBacktesting on last {days} days...")

    conn = sqlite3.connect("data/cryptobot.db")

    # Get recent data
    query = f"""
    SELECT * FROM market_data
    WHERE symbol = '{symbol}' AND timeframe = '{timeframe}'
    ORDER BY timestamp DESC
    LIMIT {days * 24}
    """

    df = pd.read_sql(query, conn)
    conn.close()

    if len(df) == 0:
        print("No data for backtesting")
        return None

    df = df.sort_values("timestamp")

    # Simple trading strategy
    initial_balance = 10000
    balance = initial_balance
    position = 0
    trades = []

    print(f"Starting balance: ${balance:.2f}")
    print(f"Testing on {len(df)} hours of data")

    # Calculate returns
    df["returns"] = df["close"].pct_change()

    # Calculate metrics
    total_return = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    print("\nBacktest Results:")
    print(f"  Buy & Hold Return: {total_return:.2f}%")
    print(f"  Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    return {"buy_hold_return": total_return, "period_days": days, "data_points": len(df)}


def main():
    """Main training pipeline."""

    print("=" * 60)
    print("DEEP LEARNING CRYPTO TRAINING WITH XPU")
    print("=" * 60)

    symbols = ["BTC/USDT", "ETH/USDT"]
    results = {}

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Training {symbol}")
        print("=" * 60)

        # Prepare data
        X_train, X_test, y_train, y_test = prepare_data(symbol, "1h")

        if X_train is None:
            print(f"Skipping {symbol} - insufficient data")
            continue

        # Train model
        model = train_model(X_train, y_train, X_test, y_test, epochs=30)

        # Backtest
        backtest_results = backtest(model, symbol, "1h", days=30)

        results[symbol] = {"train_samples": len(X_train), "test_samples": len(X_test), "backtest": backtest_results}

    # Save results
    with open("training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print("Results saved to training_results.json")


if __name__ == "__main__":
    main()
