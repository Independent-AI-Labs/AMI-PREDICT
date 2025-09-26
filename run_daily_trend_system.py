"""
Simple daily timeframe trend following system based on proven strategies
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


class DailyTrendSystem:
    """Simple trend following system on daily timeframe"""

    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.trades = []

    def backtest(self, df: pd.DataFrame) -> dict[str, Any]:
        """Run trend following backtest"""

        # Calculate indicators
        df["sma_20"] = df["close"].rolling(20).mean()
        df["sma_50"] = df["close"].rolling(50).mean()
        df["sma_200"] = df["close"].rolling(200).mean()

        # RSI for filtering
        df["rsi"] = self.calculate_rsi(df["close"], 14)

        # ATR for position sizing and stops
        df["atr"] = self.calculate_atr(df, 14)

        # Volume filter
        df["volume_sma"] = df["volume"].rolling(20).mean()

        # Trading signals
        position = 0
        entry_price = 0
        entry_time = None
        stop_loss = 0

        equity = self.initial_capital
        equity_curve = [equity]
        trades = []

        for i in range(200, len(df)):  # Start after we have 200-day SMA
            current_price = df.iloc[i]["close"]
            current_time = df.index[i]

            if position == 0:
                # Entry conditions - Golden Cross with filters
                if (
                    df.iloc[i]["sma_20"] > df.iloc[i]["sma_50"]  # Short-term uptrend
                    and df.iloc[i - 1]["sma_20"] <= df.iloc[i - 1]["sma_50"]  # Just crossed
                    and df.iloc[i]["close"] > df.iloc[i]["sma_200"]  # Long-term uptrend
                    and df.iloc[i]["rsi"] < 70  # Not overbought
                    and df.iloc[i]["volume"] > df.iloc[i]["volume_sma"] * 1.2
                ):  # Volume confirmation
                    # Calculate position size using ATR-based risk
                    risk_per_trade = 0.02  # Risk 2% per trade
                    atr = df.iloc[i]["atr"]
                    stop_distance = 2 * atr  # Stop at 2 ATR below entry

                    position_value = (equity * risk_per_trade) / (stop_distance / current_price)
                    position_value = min(position_value, equity * 0.25)  # Max 25% position

                    position = position_value / current_price
                    entry_price = current_price
                    entry_time = current_time
                    stop_loss = entry_price - stop_distance

                    # Deduct commission
                    equity -= position * current_price * 1.001

            else:
                # Exit conditions
                exit_signal = False
                exit_reason = ""

                # Stop loss
                if current_price <= stop_loss:
                    exit_signal = True
                    exit_reason = "stop_loss"

                # Death cross exit
                elif df.iloc[i]["sma_20"] < df.iloc[i]["sma_50"] and df.iloc[i - 1]["sma_20"] >= df.iloc[i - 1]["sma_50"]:
                    exit_signal = True
                    exit_reason = "death_cross"

                # Trailing stop - move stop up as price rises
                elif current_price > entry_price * 1.1:  # If up 10%+
                    new_stop = current_price - 2 * df.iloc[i]["atr"]
                    stop_loss = max(stop_loss, new_stop)

                if exit_signal:
                    # Exit position
                    exit_value = position * current_price * 0.999  # Commission
                    equity += exit_value

                    trade_return = (current_price - entry_price) / entry_price
                    pnl = exit_value - (position * entry_price * 1.001)

                    trades.append(
                        {
                            "entry_time": entry_time,
                            "exit_time": current_time,
                            "entry_price": entry_price,
                            "exit_price": current_price,
                            "return": trade_return,
                            "pnl": pnl,
                            "exit_reason": exit_reason,
                            "holding_days": (current_time - entry_time).days,
                        }
                    )

                    position = 0
                    entry_price = 0
                    stop_loss = 0

            # Update equity
            if position > 0:
                current_value = equity + position * current_price
            else:
                current_value = equity
            equity_curve.append(current_value)

        return self.calculate_metrics(equity_curve, trades, df)

    def calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))

    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df["high"] - df["low"]
        high_close = abs(df["high"] - df["close"].shift())
        low_close = abs(df["low"] - df["close"].shift())

        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)

        return true_range.rolling(period).mean()

    def calculate_metrics(self, equity_curve: list, trades: list, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate comprehensive metrics"""

        equity_arr = np.array(equity_curve)
        returns = np.diff(equity_arr) / equity_arr[:-1]

        # Sharpe ratio (daily)
        if len(returns) > 0 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
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
            avg_win = np.mean([t["return"] for t in wins]) * 100 if wins else 0
            avg_loss = np.mean([t["return"] for t in losses]) * 100 if losses else 0

            if losses:
                profit_factor = abs(sum(w["pnl"] for w in wins) / sum(l["pnl"] for l in losses))
            else:
                profit_factor = float("inf") if wins else 0

            avg_holding = np.mean([t["holding_days"] for t in trades])

            # Exit reason analysis
            exit_reasons = {}
            for t in trades:
                reason = t.get("exit_reason", "unknown")
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            avg_holding = 0
            exit_reasons = {}

        # Buy and hold comparison
        buy_hold_return = (df.iloc[-1]["close"] - df.iloc[0]["close"]) / df.iloc[0]["close"] * 100
        strategy_return = (equity_arr[-1] - self.initial_capital) / self.initial_capital * 100

        return {
            "final_equity": equity_arr[-1],
            "total_return": strategy_return,
            "buy_hold_return": buy_hold_return,
            "alpha": strategy_return - buy_hold_return,
            "num_trades": len(trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "sharpe_ratio": sharpe,
            "max_drawdown": max_drawdown,
            "avg_holding_days": avg_holding,
            "exit_reasons": exit_reasons,
            "trades": trades[-10:] if trades else [],  # Last 10 trades
        }


def resample_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Resample minute data to daily bars"""

    df_daily = pd.DataFrame()
    df_daily["open"] = df["open"].resample("1D").first()
    df_daily["high"] = df["high"].resample("1D").max()
    df_daily["low"] = df["low"].resample("1D").min()
    df_daily["close"] = df["close"].resample("1D").last()
    df_daily["volume"] = df["volume"].resample("1D").sum()

    return df_daily.dropna()


def run_daily_trend_backtest():
    """Run daily trend following backtest"""

    logger.info("=" * 60)
    logger.info("DAILY TREND FOLLOWING SYSTEM")
    logger.info("=" * 60)

    # Load data
    conn = sqlite3.connect("data/crypto_5years.db")

    # Use 2022-2023 for more data
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
    logger.info(f"Loaded {len(df)} minute records")

    # Resample to daily
    df_daily = resample_to_daily(df)
    logger.info(f"Resampled to {len(df_daily)} daily bars")

    # Run backtest
    system = DailyTrendSystem(initial_capital=100000)
    results = system.backtest(df_daily)

    # Print results
    print("\n" + "=" * 60)
    print("DAILY TREND FOLLOWING RESULTS (2022-2023)")
    print("=" * 60)
    print("Initial Capital: $100,000")
    print(f"Final Equity: ${results['final_equity']:,.2f}")
    print(f"Total Return: {results['total_return']:.2f}%")
    print(f"Buy & Hold Return: {results['buy_hold_return']:.2f}%")
    print(f"Alpha: {results['alpha']:.2f}%")
    print("\nTRADING STATISTICS")
    print("-" * 40)
    print(f"Number of Trades: {results['num_trades']}")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Average Win: {results['avg_win']:.2f}%")
    print(f"Average Loss: {results['avg_loss']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2f}%")
    print(f"Avg Holding: {results['avg_holding_days']:.0f} days")

    if results["exit_reasons"]:
        print("\nEXIT REASONS")
        print("-" * 40)
        for reason, count in results["exit_reasons"].items():
            print(f"{reason}: {count}")

    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "Daily Trend Following",
        "period": "2022-2023",
        "results": {k: v for k, v in results.items() if k != "trades"},
    }

    with open("experiments/daily_trend_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    logger.info("\nResults saved to experiments/daily_trend_results.json")

    # Show recent trades
    if results["trades"]:
        print("\nRECENT TRADES")
        print("-" * 40)
        for i, trade in enumerate(results["trades"][-5:], 1):
            print(
                f"Trade {i}: {trade['entry_time'].strftime('%Y-%m-%d')} -> "
                f"{trade['exit_time'].strftime('%Y-%m-%d')} "
                f"({trade['holding_days']}d) "
                f"Return: {trade['return']*100:.2f}% "
                f"Exit: {trade['exit_reason']}"
            )

    return results


if __name__ == "__main__":
    results = run_daily_trend_backtest()
