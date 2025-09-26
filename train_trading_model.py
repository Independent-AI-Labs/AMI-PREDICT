"""
Train a model specifically for trading with proper labels and risk management
"""
import json
import logging
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

warnings.filterwarnings("ignore")

from advanced_models import AttentionTCN
from experiment_tracker import ExperimentTracker

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TradingDataset(torch.utils.data.Dataset):
    """Dataset for trading-specific training"""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_trading_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced trading features - exactly 15 features"""
    features = pd.DataFrame(index=df.index)

    # Price features (4)
    features["returns"] = df["close"].pct_change()
    features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    features["high_low_ratio"] = df["high"] / df["low"]
    features["close_open_ratio"] = df["close"] / df["open"]

    # Volume features (2)
    volume_sma = df["volume"].rolling(20).mean()
    features["volume_ratio"] = df["volume"] / volume_sma
    features["volume_std"] = df["volume"].rolling(20).std() / volume_sma

    # Market microstructure (3)
    features["bid_ask_proxy"] = (df["high"] - df["low"]) / df["close"]  # Volatility proxy
    features["price_efficiency"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)
    features["volume_price_trend"] = features["returns"] * features["volume_ratio"]

    # Advanced momentum (3)
    for period in [5, 10, 20]:
        features[f"momentum_{period}"] = df["close"] / df["close"].shift(period) - 1

    # Risk indicators (3)
    features["rsi"] = calculate_rsi(df["close"], 14)
    features["volatility_20"] = features["returns"].rolling(20).std()
    features["volatility_ratio"] = features["returns"].rolling(5).std() / features["volatility_20"]

    return features


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def create_trading_labels(df: pd.DataFrame, horizon: int = 30, stop_loss: float = 0.02, take_profit: float = 0.03) -> pd.Series:
    """
    Create trading-specific labels with risk management

    Labels:
    - 1: Profitable trade (hits take profit before stop loss)
    - 0: Losing trade or no clear signal
    """
    labels = []

    for i in range(len(df) - horizon):
        entry_price = df.iloc[i]["close"]
        future_prices = df.iloc[i + 1 : i + horizon + 1]["close"].values

        # Calculate returns for each future point
        returns = (future_prices - entry_price) / entry_price

        # Check if take profit or stop loss is hit first
        take_profit_hit = np.where(returns >= take_profit)[0]
        stop_loss_hit = np.where(returns <= -stop_loss)[0]

        if len(take_profit_hit) > 0 and len(stop_loss_hit) > 0:
            # Both hit - which comes first?
            if take_profit_hit[0] < stop_loss_hit[0]:
                labels.append(1)  # Profitable
            else:
                labels.append(0)  # Loss
        elif len(take_profit_hit) > 0:
            labels.append(1)  # Profitable
        else:
            labels.append(0)  # Loss or no clear signal

    # Pad the end
    labels.extend([0] * horizon)

    return pd.Series(labels, index=df.index)


def prepare_sequences(features: pd.DataFrame, labels: pd.Series, sequence_length: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Prepare sequences for training"""
    X, y = [], []

    for i in range(sequence_length, len(features)):
        X.append(features.iloc[i - sequence_length : i].values)
        y.append(labels.iloc[i])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_trading_model():
    """Train a model specifically for trading"""

    logger.info("=" * 60)
    logger.info("TRAINING TRADING-SPECIFIC MODEL")
    logger.info("=" * 60)

    # Check device
    if torch.xpu.is_available():
        device = "xpu"
        logger.info(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
    else:
        device = "cpu"
        logger.info("Using CPU")

    # Load data
    import sqlite3

    conn = sqlite3.connect("data/crypto_5years.db")

    # Use 2022-2023 data for training
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM market_data_1m
    WHERE symbol = 'BTC/USDT'
    AND timestamp >= '2022-01-01'
    AND timestamp <= '2023-12-31'
    ORDER BY timestamp
    """

    df = pd.read_sql_query(query, conn, parse_dates=["timestamp"])
    conn.close()

    df = df.set_index("timestamp")
    logger.info(f"Loaded {len(df)} records")

    # Create features and labels
    logger.info("Creating trading features and labels...")
    features_df = create_trading_features(df)
    labels = create_trading_labels(df, horizon=30, stop_loss=0.02, take_profit=0.03)

    # Drop NaN values
    features_df = features_df.dropna()
    labels = labels[features_df.index]

    # Check label distribution
    label_dist = labels.value_counts(normalize=True)
    logger.info(f"Label distribution: {label_dist.to_dict()}")

    # Prepare sequences
    X, y = prepare_sequences(features_df, labels, sequence_length=50)
    logger.info(f"Prepared {len(X)} sequences")

    # Split data (80/20)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Create datasets and dataloaders
    train_dataset = TradingDataset(X_train, y_train)
    val_dataset = TradingDataset(X_val, y_val)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=0)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1024, shuffle=False, num_workers=0)

    # Initialize model
    model = AttentionTCN(input_size=15, hidden_size=128, num_layers=4)
    model = model.to(device)

    # Loss function - Binary Cross Entropy for trading signals
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Training loop
    best_val_acc = 0
    best_val_loss = float("inf")
    epochs = 20

    tracker = ExperimentTracker()
    experiment_id = tracker.start_experiment(
        name="TradingModel",
        model_type="AttentionTCN",
        hyperparams={
            "epochs": epochs,
            "batch_size": 1024,
            "learning_rate": 0.001,
            "optimizer": "AdamW",
            "loss": "BCEWithLogitsLoss",
            "stop_loss": 0.02,
            "take_profit": 0.03,
        },
        description="Trading-specific model with risk management labels",
    )

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            train_correct += (predictions == batch_y).sum().item()
            train_total += batch_y.size(0)

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)

                val_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                val_correct += (predictions == batch_y).sum().item()
                val_total += batch_y.size(0)

        # Calculate metrics
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        train_acc = 100 * train_correct / train_total
        val_acc = 100 * val_correct / val_total

        # Log epoch results
        tracker.log_epoch(experiment_id, epoch + 1, train_loss, val_loss, train_acc, val_acc, scheduler.get_last_lr()[0])

        logger.info(
            f"Epoch {epoch+1}/{epochs} - " f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, " f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%"
        )

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                "models/TradingModel_best.pth",
            )

            logger.info(f"Saved best model with validation accuracy: {val_acc:.2f}%")

        scheduler.step()

    # Complete experiment
    tracker.end_experiment(experiment_id, success=True)

    # Save final results
    results = {
        "model": "AttentionTCN_Trading",
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
        "training_date": datetime.now().isoformat(),
        "device": device,
        "label_distribution": label_dist.to_dict(),
        "training_samples": len(X_train),
        "validation_samples": len(X_val),
    }

    with open("experiments/trading_model_results.json", "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nTraining completed! Best validation accuracy: {best_val_acc:.2f}%")

    return model, results


if __name__ == "__main__":
    model, results = train_trading_model()
