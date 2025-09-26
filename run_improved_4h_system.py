"""
Improved trading system using 4-hour timeframes based on research findings
"""
import json
import logging
import sqlite3
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class ImprovedTradingSystem:
    """Trading system optimized for 4-hour timeframes"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = [initial_capital]

    def backtest(self, df: pd.DataFrame, signals: pd.Series) -> dict[str, Any]:
        """Run backtest with proper position sizing"""

        position = 0
        entry_price = 0
        entry_time = None
        trades = []
        equity = self.initial_capital
        equity_curve = [equity]
        max_position_size = 0.5  # Use max 50% of capital per trade

        for i in range(len(signals)):
            if i >= len(df):
                break

            signal = signals.iloc[i]
            price = df.iloc[i]["close"]
            timestamp = df.index[i]

            if signal == 1 and position == 0:
                # Buy signal - use Kelly Criterion for position sizing
                position_value = equity * max_position_size
                position = position_value / price
                entry_price = price
                entry_time = timestamp

                # Deduct commission
                equity -= position * price * 1.001

            elif signal == -1 and position > 0:
                # Sell signal
                exit_price = price
                trade_return = (exit_price - entry_price) / entry_price

                # Calculate P&L including commission
                exit_value = position * exit_price * 0.999
                entry_value = position * entry_price * 1.001
                pnl = exit_value - entry_value

                equity += exit_value

                trades.append(
                    {
                        "entry_time": entry_time,
                        "exit_time": timestamp,
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "return": trade_return,
                        "holding_hours": (timestamp - entry_time).total_seconds() / 3600,
                    }
                )

                position = 0
                entry_price = 0
                entry_time = None

            # Update equity
            if position > 0:
                current_value = equity + position * price
            else:
                current_value = equity
            equity_curve.append(current_value)

        # Calculate metrics
        return self.calculate_metrics(equity_curve, trades)

    def calculate_metrics(self, equity_curve: list, trades: list) -> dict[str, Any]:
        """Calculate comprehensive trading metrics"""

        equity_arr = np.array(equity_curve)
        returns = np.diff(equity_arr) / equity_arr[:-1]

        # Calculate Sharpe ratio (annualized for 4-hour bars)
        # 6 bars per day, 252 trading days per year
        periods_per_year = 6 * 252
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year)
        else:
            sharpe = 0

        # Max drawdown
        running_max = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100

        # Trade statistics
        if trades:
            wins = [t for t in trades if t["pnl"] > 0]
            losses = [t for t in trades if t["pnl"] <= 0]

            win_rate = len(wins) / len(trades) * 100
            avg_win = np.mean([t["pnl"] for t in wins]) if wins else 0
            avg_loss = np.mean([t["pnl"] for t in losses]) if losses else 0

            if losses and avg_loss != 0:
                profit_factor = abs(sum(w["pnl"] for w in wins) / sum(l["pnl"] for l in losses))
            else:
                profit_factor = float("inf") if wins else 0

            avg_holding = np.mean([t["holding_hours"] for t in trades])
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_holding = 0

        return {
            "final_equity": equity_arr[-1],
            "total_return": (equity_arr[-1] - self.initial_capital) / self.initial_capital * 100,
            "num_trades": len(trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "avg_holding_hours": avg_holding,
            "equity_curve": equity_curve,
        }


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1-minute data to 4-hour bars"""

    # Resample OHLCV data
    df_4h = pd.DataFrame()
    df_4h["open"] = df["open"].resample("4H").first()
    df_4h["high"] = df["high"].resample("4H").max()
    df_4h["low"] = df["low"].resample("4H").min()
    df_4h["close"] = df["close"].resample("4H").last()
    df_4h["volume"] = df["volume"].resample("4H").sum()

    # Remove any NaN rows
    df_4h = df_4h.dropna()

    return df_4h


def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def generate_4h_signals(df: pd.DataFrame) -> pd.Series:
    """Generate trading signals optimized for 4-hour timeframe"""

    # Calculate indicators
    df["rsi"] = calculate_rsi(df["close"], period=14)
    df["sma_10"] = df["close"].rolling(10).mean()  # 10 periods = 40 hours
    df["sma_20"] = df["close"].rolling(20).mean()  # 20 periods = 80 hours
    df["sma_50"] = df["close"].rolling(50).mean()  # 50 periods = 200 hours

    # Volume analysis
    df["volume_sma"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma"]

    # Volatility
    df["returns"] = df["close"].pct_change()
    df["volatility"] = df["returns"].rolling(20).std()

    # MACD
    df["ema_12"] = df["close"].ewm(span=12).mean()
    df["ema_26"] = df["close"].ewm(span=26).mean()
    df["macd"] = df["ema_12"] - df["ema_26"]
    df["macd_signal"] = df["macd"].ewm(span=9).mean()

    # Initialize signals
    signals = pd.Series(0, index=df.index)

    # Multi-condition entry signals
    # Buy when:
    # 1. RSI recovering from oversold (crossed above 35)
    # 2. Price above SMA 20
    # 3. MACD bullish crossover
    # 4. Volume confirmation

    df["rsi_prev"] = df["rsi"].shift(1)
    df["macd_prev"] = df["macd"].shift(1)

    buy_conditions = (
        (df["rsi"] > 35)
        & (df["rsi_prev"] <= 35)  # RSI recovery
        & (df["close"] > df["sma_20"])  # Above medium-term average
        & (df["macd"] > df["macd_signal"])
        & (df["macd_prev"] <= df["macd_signal"])  # MACD crossover
        & (df["volume_ratio"] > 1.1)  # Volume confirmation
    )

    # Sell when:
    # 1. RSI overbought (above 70)
    # 2. MACD bearish crossover
    # 3. Price below SMA 10 (faster exit)

    sell_conditions = (
        (df["rsi"] > 70)  # Overbought
        | ((df["macd"] < df["macd_signal"]) & (df["macd_prev"] >= df["macd_signal"]))  # MACD bearish
        | (df["close"] < df["sma_10"])  # Below short-term average
    )

    # Generate signals with state management
    position = False
    entry_idx = None
    min_holding_periods = 6  # 24 hours minimum (6 * 4 hours)

    for i in range(len(df)):
        if position:
            # Check exit conditions
            if entry_idx is not None:
                periods_held = i - entry_idx
                if sell_conditions.iloc[i] and periods_held >= min_holding_periods:
                    signals.iloc[i] = -1
                    position = False
                    entry_idx = None
        else:
            # Check entry conditions
            if buy_conditions.iloc[i]:
                signals.iloc[i] = 1
                position = True
                entry_idx = i

    return signals


def run_improved_4h_backtest():
    """Run backtest on 4-hour timeframe"""

    logger.info("=" * 60)
    logger.info("IMPROVED 4-HOUR TIMEFRAME TRADING SYSTEM")
    logger.info("=" * 60)

    # Load data
    conn = sqlite3.connect("data/crypto_5years.db")

    query = """
    SELECT timestamp, open, high, low, close, volume
    FROM market_data_1m
    WHERE symbol = 'BTC/USDT'
    AND timestamp >= '2023-01-01'
    AND timestamp <= '2023-12-31'
    ORDER BY timestamp
    """

    df = pd.read_sql_query(query, conn, parse_dates=["timestamp"])
    conn.close()

    df = df.set_index("timestamp")
    logger.info(f"Loaded {len(df)} 1-minute records")

    # Resample to 4-hour bars
    df_4h = resample_to_4h(df)
    logger.info(f"Resampled to {len(df_4h)} 4-hour bars")

    # Generate signals
    signals = generate_4h_signals(df_4h)

    buy_signals = (signals == 1).sum()
    sell_signals = (signals == -1).sum()
    logger.info(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")

    # Run backtest
    system = ImprovedTradingSystem(initial_capital=100000)
    results = system.backtest(df_4h, signals)

    # Print results
    print("\n" + "=" * 60)
    print("4-HOUR TIMEFRAME RESULTS")
    print("=" * 60)
    print("Initial Capital: $100,000")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Avg Holding: {results['avg_holding_hours']:.1f} hours")

    # Compare with buy and hold
    first_price = df_4h.iloc[0]["close"]
    last_price = df_4h.iloc[-1]["close"]
    buy_hold_return = (last_price - first_price) / first_price * 100

    print(f"\nBuy & Hold Return: {buy_hold_return:.2f}%")
    print(f"Strategy Alpha: {results['total_return'] - buy_hold_return:.2f}%")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "timeframe": "4H",
        "period": "2023 Full Year",
        "results": {
            "total_return": results["total_return"],
            "final_equity": results["final_equity"],
            "num_trades": results["num_trades"],
            "win_rate": results["win_rate"],
            "profit_factor": results["profit_factor"],
            "sharpe_ratio": results["sharpe_ratio"],
            "max_drawdown": results["max_drawdown"],
            "buy_hold_return": buy_hold_return,
            "alpha": results["total_return"] - buy_hold_return,
        },
    }

    with open("experiments/improved_4h_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("\nResults saved to experiments/improved_4h_results.json")

    return results


if __name__ == "__main__":
    results = run_improved_4h_backtest()
