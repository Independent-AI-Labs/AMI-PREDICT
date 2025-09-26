"""
Improved backtesting with proper risk management and position sizing
"""
import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

from advanced_models import AttentionTCN
from src.ml.backtester import Backtester

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class ImprovedTradingSystem:
    """Enhanced trading system with risk management"""

    def __init__(
        self,
        initial_capital: float = 100000,
        max_position_size: float = 0.1,  # Max 10% per trade
        stop_loss: float = 0.02,  # 2% stop loss
        take_profit: float = 0.03,  # 3% take profit
        min_holding_period: int = 60,  # 60 minutes minimum
        confidence_threshold: float = 0.7,  # 70% confidence required
        max_daily_trades: int = 5,  # Maximum trades per day
        max_drawdown_limit: float = 0.15,
    ):  # 15% max drawdown
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_position_size = max_position_size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.min_holding_period = min_holding_period
        self.confidence_threshold = confidence_threshold
        self.max_daily_trades = max_daily_trades
        self.max_drawdown_limit = max_drawdown_limit

        self.positions = {}
        self.trades = []
        self.equity_curve = [initial_capital]
        self.daily_trade_count = {}
        self.current_drawdown = 0
        self.peak_equity = initial_capital

    def calculate_position_size(self, confidence: float, volatility: float) -> float:
        """Calculate position size based on Kelly Criterion and volatility"""

        # Kelly fraction (simplified)
        win_rate = 0.4  # Conservative estimate
        avg_win_loss_ratio = 1.5  # Take profit / stop loss
        kelly_fraction = (win_rate * avg_win_loss_ratio - (1 - win_rate)) / avg_win_loss_ratio

        # Adjust for confidence
        confidence_adjustment = (confidence - 0.5) * 2  # Scale 0.5-1.0 to 0-1

        # Volatility adjustment (inverse relationship)
        volatility_adjustment = 1 / (1 + volatility * 10)

        # Final position size
        position_size = kelly_fraction * confidence_adjustment * volatility_adjustment
        position_size = min(position_size, self.max_position_size)
        position_size = max(position_size, 0.01)  # Minimum 1%

        return position_size

    def check_risk_limits(self, date: pd.Timestamp) -> bool:
        """Check if we're within risk limits"""

        # Check daily trade limit
        date_str = date.strftime("%Y-%m-%d")
        if self.daily_trade_count.get(date_str, 0) >= self.max_daily_trades:
            return False

        # Check drawdown limit
        if self.current_drawdown > self.max_drawdown_limit:
            return False

        return True

    def update_equity(self, current_value: float):
        """Update equity and drawdown"""
        self.equity_curve.append(current_value)

        if current_value > self.peak_equity:
            self.peak_equity = current_value

        self.current_drawdown = (self.peak_equity - current_value) / self.peak_equity


class MarketRegimeDetector:
    """Detect market regime for better trading decisions"""

    @staticmethod
    def detect_regime(df: pd.DataFrame, lookback: int = 100) -> str:
        """
        Detect current market regime
        Returns: 'trending_up', 'trending_down', 'ranging', 'volatile'
        """
        if len(df) < lookback:
            return "unknown"

        recent_data = df.tail(lookback)
        returns = recent_data["close"].pct_change().dropna()

        # Calculate metrics
        trend = (recent_data["close"].iloc[-1] - recent_data["close"].iloc[0]) / recent_data["close"].iloc[0]
        volatility = returns.std()

        # Determine regime
        if abs(trend) < 0.02:  # Less than 2% movement
            if volatility > 0.02:
                return "volatile"
            else:
                return "ranging"
        elif trend > 0.02:
            return "trending_up"
        else:
            return "trending_down"


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create advanced features including market microstructure"""
    features = pd.DataFrame(index=df.index)

    # Price features (4)
    features["returns"] = df["close"].pct_change()
    features["log_returns"] = np.log(df["close"] / df["close"].shift(1))
    features["high_low_ratio"] = df["high"] / df["low"]
    features["close_open_ratio"] = df["close"] / df["open"]

    # Market microstructure (4)
    features["spread_proxy"] = (df["high"] - df["low"]) / df["close"]
    features["price_efficiency"] = abs(df["close"] - df["open"]) / (df["high"] - df["low"] + 1e-10)
    features["volume_imbalance"] = (df["volume"] - df["volume"].rolling(20).mean()) / df["volume"].rolling(20).std()
    features["price_pressure"] = features["returns"] * np.log(df["volume"] + 1)

    # Technical indicators (4)
    features["rsi"] = calculate_rsi(df["close"], 14)
    features["bbands_position"] = (df["close"] - df["close"].rolling(20).mean()) / (df["close"].rolling(20).std() * 2)
    macd = df["close"].ewm(span=12).mean() - df["close"].ewm(span=26).mean()
    features["macd"] = macd / df["close"]
    features["volatility_ratio"] = df["close"].rolling(5).std() / df["close"].rolling(20).std()

    # Momentum (3)
    features["momentum_5"] = df["close"] / df["close"].shift(5) - 1
    features["momentum_10"] = df["close"] / df["close"].shift(10) - 1
    features["momentum_20"] = df["close"] / df["close"].shift(20) - 1

    return features


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def generate_smart_signals(model: torch.nn.Module, df: pd.DataFrame, trading_system: ImprovedTradingSystem, device: str = "cpu") -> pd.Series:
    """Generate intelligent trading signals with risk management"""

    # Create features
    features_df = create_advanced_features(df)
    features_df = features_df.dropna()

    # Prepare sequences
    sequence_length = 50
    X = []
    indices = []

    for i in range(sequence_length, len(features_df)):
        X.append(features_df.iloc[i - sequence_length : i].values)
        indices.append(features_df.index[i])

    if not X:
        return pd.Series(dtype=float)

    X = np.array(X, dtype=np.float32)

    # Generate predictions
    batch_size = 1024
    predictions = []

    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            batch = torch.FloatTensor(X[i : i + batch_size]).to(device)
            outputs = model(batch)
            probs = torch.sigmoid(outputs)
            predictions.extend(probs.squeeze().cpu().numpy())

    predictions = np.array(predictions)

    # Create signals with risk management
    signals = []
    current_position = None
    position_entry_idx = None

    regime_detector = MarketRegimeDetector()

    for i, (idx, pred) in enumerate(zip(indices, predictions, strict=False)):
        # Get current market data
        current_idx = df.index.get_loc(idx)
        if current_idx < 100:
            signals.append(0)
            continue

        recent_df = df.iloc[max(0, current_idx - 100) : current_idx + 1]
        volatility = recent_df["close"].pct_change().std()
        regime = regime_detector.detect_regime(recent_df)

        # Check if we should close position
        if current_position is not None:
            bars_held = i - position_entry_idx
            entry_price = df.loc[indices[position_entry_idx], "close"]
            current_price = df.loc[idx, "close"]
            pnl_pct = (current_price - entry_price) / entry_price

            # Check exit conditions
            if (
                pnl_pct >= trading_system.take_profit  # Take profit
                or pnl_pct <= -trading_system.stop_loss  # Stop loss
                or bars_held >= trading_system.min_holding_period * 2
            ):  # Max holding
                signals.append(-1)  # Close position
                current_position = None
                position_entry_idx = None
            else:
                signals.append(0)  # Hold

        # Check if we should open position
        elif pred > trading_system.confidence_threshold:
            # Additional filters
            if (
                regime in ["trending_up", "ranging"]  # Favorable regime
                and volatility < 0.03  # Not too volatile
                and trading_system.check_risk_limits(idx)
            ):  # Within risk limits
                signals.append(1)  # Open position
                current_position = "long"
                position_entry_idx = i

                # Update daily trade count
                date_str = idx.strftime("%Y-%m-%d")
                trading_system.daily_trade_count[date_str] = trading_system.daily_trade_count.get(date_str, 0) + 1
            else:
                signals.append(0)
        else:
            signals.append(0)

    return pd.Series(signals, index=indices)


def run_improved_backtest(start_date: str = "2023-01-01", end_date: str = "2023-12-31") -> dict[str, Any]:
    """Run improved backtest with risk management"""

    logger.info("=" * 60)
    logger.info("IMPROVED BACKTESTING SYSTEM")
    logger.info("=" * 60)

    # Load model
    device = "xpu" if torch.xpu.is_available() else "cpu"
    model = AttentionTCN(input_size=15, hidden_size=128, num_layers=4)
    checkpoint = torch.load("models/AttentionTCN_best.pth", map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Load data
    import sqlite3

    conn = sqlite3.connect("data/crypto_5years.db")
    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM market_data_1m
    WHERE symbol = 'BTC/USDT'
    AND timestamp >= ?
    AND timestamp <= ?
    ORDER BY timestamp
    """
    df = pd.read_sql_query(query, conn, params=(start_date, end_date), parse_dates=["timestamp"])
    conn.close()
    df = df.set_index("timestamp")

    logger.info(f"Loaded {len(df)} records from {start_date} to {end_date}")

    # Initialize trading system
    trading_system = ImprovedTradingSystem(
        initial_capital=100000,
        max_position_size=0.1,
        stop_loss=0.02,
        take_profit=0.03,
        min_holding_period=60,
        confidence_threshold=0.7,
        max_daily_trades=5,
        max_drawdown_limit=0.15,
    )

    # Generate signals
    signals = generate_smart_signals(model, df, trading_system, device)

    # Log signal statistics
    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")

    # Align data
    aligned_df = df.loc[signals.index]

    # Run backtest
    backtester = Backtester(initial_capital=trading_system.initial_capital, commission=0.001, slippage=0.0005)

    results = backtester.run(data=aligned_df, signals=signals, position_size=trading_system.max_position_size)

    # Print results
    print_improved_results(results)

    # Save results
    save_improved_results(results, start_date, end_date)

    return results


def print_improved_results(results: dict[str, Any]):
    """Print improved backtest results"""

    print("\n" + "=" * 60)
    print("IMPROVED BACKTEST RESULTS")
    print("=" * 60)

    stats = results["portfolio_stats"]

    print("\nPERFORMANCE METRICS")
    print("-" * 40)
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")

    print("\nTRADING STATISTICS")
    print("-" * 40)
    print(f"Total Trades: {stats['total_trades']}")
    print(f"Win Rate: {stats['win_rate']:.2f}%")
    print(f"Profit Factor: {stats['profit_factor']:.2f}")
    print(f"Average Win: ${stats['avg_win']:.2f}")
    print(f"Average Loss: ${stats['avg_loss']:.2f}")

    # Risk-adjusted metrics
    if results["total_return"] > 0 and results["max_drawdown"] < 0:
        calmar = results["total_return"] / abs(results["max_drawdown"])
        print(f"Calmar Ratio: {calmar:.2f}")


def save_improved_results(results: dict[str, Any], start_date: str, end_date: str):
    """Save improved results"""

    save_data = {
        "timestamp": datetime.now().isoformat(),
        "system": "ImprovedTradingSystem",
        "period": f"{start_date} to {end_date}",
        "metrics": {
            "total_return": results["total_return"],
            "final_equity": results["final_equity"],
            "max_drawdown": results["max_drawdown"],
            "sharpe_ratio": results["sharpe_ratio"],
            "win_rate": results["portfolio_stats"]["win_rate"],
            "profit_factor": results["portfolio_stats"]["profit_factor"],
        },
    }

    output_file = Path(f"experiments/improved_backtest_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2)

    logger.info(f"Results saved to {output_file}")


if __name__ == "__main__":
    # Test on 2023 data
    results = run_improved_backtest("2023-01-01", "2023-12-31")
