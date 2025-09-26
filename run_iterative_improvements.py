#!/usr/bin/env python
"""
Iterative improvement training script.
Tests multiple advanced architectures and tracks experiments.
"""

import logging
import sqlite3
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn, optim

# Import our modules
from advanced_models import AttentionTCN, GRUAttention, HybridCNNLSTM, TransformerTrader, WaveNetModel
from experiment_tracker import ExperimentTracker

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Check XPU
if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
    logger.info(f"Using Intel XPU: {torch.xpu.get_device_name(0)}")
else:
    DEVICE = torch.device("cpu")
    logger.info("Using CPU")


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction="none")
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        loss = alpha_t * focal_weight * bce_loss
        return loss.mean()


def load_data(db_path: str, symbol: str, limit: int = 200000) -> pd.DataFrame:
    """Load data with advanced features."""
    conn = sqlite3.connect(db_path)

    query = f"""
        SELECT timestamp, open, high, low, close, volume
        FROM market_data_1m
        WHERE symbol = '{symbol}'
        ORDER BY timestamp
        LIMIT {limit}
    """

    logger.info(f"Loading {symbol} data...")
    df = pd.read_sql_query(query, conn)
    conn.close()

    # Advanced features
    df["returns"] = df["close"].pct_change()
    df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

    # Volatility features
    df["volatility_5"] = df["returns"].rolling(5).std()
    df["volatility_20"] = df["returns"].rolling(20).std()

    # Price features
    df["hl_ratio"] = (df["high"] - df["low"]) / df["close"]
    df["co_ratio"] = (df["close"] - df["open"]) / df["open"]

    # Volume features
    df["volume_ratio"] = df["volume"] / df["volume"].rolling(20).mean()
    df["volume_std"] = df["volume"].rolling(20).std() / df["volume"].rolling(20).mean()

    # Momentum
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_10"] = df["close"].pct_change(10)
    df["momentum_20"] = df["close"].pct_change(20)

    # RSI
    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12).mean()
    ema_26 = df["close"].ewm(span=26).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # Bollinger Bands
    sma_20 = df["close"].rolling(20).mean()
    std_20 = df["close"].rolling(20).std()
    df["bb_upper"] = sma_20 + 2 * std_20
    df["bb_lower"] = sma_20 - 2 * std_20
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df = df.fillna(0)

    return df


def create_sequences(df: pd.DataFrame, seq_len: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Create sequences with all features."""
    features = [
        "returns",
        "log_returns",
        "volatility_5",
        "volatility_20",
        "hl_ratio",
        "co_ratio",
        "volume_ratio",
        "volume_std",
        "momentum_5",
        "momentum_10",
        "momentum_20",
        "rsi",
        "macd",
        "macd_signal",
        "bb_position",
    ]

    X = df[features].values

    # Create labels with class weighting consideration
    future_returns = df["returns"].shift(-1).fillna(0).values
    threshold = np.percentile(np.abs(future_returns[future_returns != 0]), 50)
    y = (future_returns > threshold).astype(float)

    sequences = []
    labels = []

    for i in range(seq_len, len(X) - 1):
        sequences.append(X[i - seq_len : i])
        labels.append(y[i])

    return np.array(sequences, dtype=np.float32), np.array(labels, dtype=np.float32)


def train_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    tracker: ExperimentTracker,
    epochs: int = 20,
    batch_size: int = 512,
) -> dict:
    """Train a model and track the experiment."""

    # Start experiment
    hyperparams = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": 0.001,
        "optimizer": "AdamW",
        "loss": "FocalLoss",
        "model_params": sum(p.numel() for p in model.parameters()),
    }

    exp_id = tracker.start_experiment(
        name=f"{model_name}_improvement", model_type=model_name, hyperparams=hyperparams, description=f"Testing {model_name} with advanced features"
    )

    logger.info(f"\nTraining {model_name} (Experiment: {exp_id})")

    # Move to device
    model = model.to(DEVICE)
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).reshape(-1, 1).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).reshape(-1, 1).to(DEVICE)

    # Setup training
    criterion = FocalLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0
    train_start = time.time()

    for epoch in range(epochs):
        model.train()
        epoch_start = time.time()
        total_loss = 0
        correct = 0

        # Shuffle
        indices = torch.randperm(len(X_train_t))

        for i in range(0, len(X_train_t), batch_size):
            batch_idx = indices[i : i + batch_size]
            batch_X = X_train_t[batch_idx]
            batch_y = y_train_t[batch_idx]

            optimizer.zero_grad()

            try:
                outputs = model(batch_X)
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(1)

                loss = criterion(outputs, batch_y)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()

                total_loss += loss.item() * len(batch_X)
                predictions = (outputs > 0.5).float()
                correct += (predictions == batch_y).sum().item()

            except Exception as e:
                logger.error(f"Error in training batch: {e}")
                continue

        # Validation
        model.eval()
        val_correct = 0
        val_loss = 0

        with torch.no_grad():
            for i in range(0, len(X_val_t), batch_size):
                batch_X = X_val_t[i : i + batch_size]
                batch_y = y_val_t[i : i + batch_size]

                try:
                    outputs = model(batch_X)
                    if outputs.dim() == 1:
                        outputs = outputs.unsqueeze(1)

                    val_loss += criterion(outputs, batch_y).item() * len(batch_X)
                    predictions = (outputs > 0.5).float()
                    val_correct += (predictions == batch_y).sum().item()

                except Exception as e:
                    logger.error(f"Error in validation: {e}")
                    continue

        scheduler.step()

        # Calculate metrics
        train_acc = correct / len(X_train_t) * 100
        val_acc = val_correct / len(X_val_t) * 100
        avg_loss = total_loss / len(X_train_t)
        avg_val_loss = val_loss / len(X_val_t)
        epoch_time = time.time() - epoch_start

        # Log to tracker
        tracker.log_epoch(
            epoch=epoch + 1,
            train_loss=avg_loss,
            val_loss=avg_val_loss,
            train_acc=train_acc,
            val_acc=val_acc,
            epoch_time=epoch_time,
            lr=scheduler.get_last_lr()[0],
        )

        logger.info(
            f"Epoch {epoch+1}/{epochs} ({epoch_time:.1f}s) - "
            f"Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
            f"Train: {train_acc:.2f}%, Val: {val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save best model
            model_path = Path("models") / f"{model_name}_best.pth"
            model_path.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), model_path)

    train_time = time.time() - train_start

    # Final metrics
    final_metrics = {"best_val_acc": best_val_acc, "train_time": train_time, "throughput": len(X_train_t) * epochs / train_time}

    tracker.log_metrics(final_metrics)
    tracker.end_experiment(success=True)

    logger.info(f"Completed {model_name}: Best Val Acc = {best_val_acc:.2f}%")

    return final_metrics


def main():
    logger.info("=" * 60)
    logger.info("ITERATIVE MODEL IMPROVEMENTS")
    logger.info("=" * 60)

    # Initialize tracker
    tracker = ExperimentTracker()

    # Load data
    db_path = "data/crypto_5years.db"
    symbols = ["BTC/USDT", "ETH/USDT"]

    all_sequences = []
    all_labels = []

    for symbol in symbols:
        df = load_data(db_path, symbol, limit=200000)
        X, y = create_sequences(df, seq_len=50)
        logger.info(f"{symbol}: {len(X):,} sequences")
        all_sequences.append(X)
        all_labels.append(y)

    # Combine data
    X = np.concatenate(all_sequences)
    y = np.concatenate(all_labels)

    logger.info(f"\nTotal sequences: {len(X):,}")
    logger.info(f"Positive class: {np.mean(y)*100:.1f}%")

    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]

    # Split
    split_idx = int(len(X) * 0.8)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_val = X[split_idx:]
    y_val = y[split_idx:]

    logger.info(f"Train: {len(X_train):,}, Val: {len(X_val):,}")

    # Test different models
    models_to_test = [
        ("AttentionTCN", AttentionTCN(input_size=15, hidden_size=128)),
        ("WaveNetModel", WaveNetModel(input_size=15, residual_channels=32)),
        ("HybridCNNLSTM", HybridCNNLSTM(input_size=15, cnn_channels=64)),
        ("TransformerTrader", TransformerTrader(input_size=15, d_model=128, num_layers=4)),
        ("GRUAttention", GRUAttention(input_size=15, hidden_size=128)),
    ]

    results = {}

    # Adjust batch size based on device
    batch_size = 1024 if DEVICE.type == "xpu" else 64

    for model_name, model in models_to_test:
        try:
            metrics = train_model(model, X_train, y_train, X_val, y_val, model_name, tracker, epochs=10, batch_size=batch_size)
            results[model_name] = metrics

        except Exception as e:
            logger.error(f"Failed to train {model_name}: {e}")
            tracker.end_experiment(success=False, error_msg=str(e))

    # Generate report
    logger.info("\n" + "=" * 60)
    logger.info("EXPERIMENT SUMMARY")
    logger.info("=" * 60)

    comparison_df = tracker.compare_experiments()
    if not comparison_df.empty:
        print(comparison_df.to_string())

    # Save report
    report = tracker.generate_report()
    logger.info("\nReport saved to experiments/EXPERIMENT_REPORT.md")

    # Show best model
    best_model = tracker.get_best_model()
    if best_model:
        logger.info(f"\n🏆 Best Model: {best_model['model_type']}")
        logger.info(f"   Validation Accuracy: {best_model['summary']['best_val_acc']:.2f}%")

    logger.info("\n✅ Iterative improvements complete!")


if __name__ == "__main__":
    main()
