"""
Backtesting engine for evaluating trading strategies
"""
import logging
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Represents a single trade"""

    entry_time: pd.Timestamp
    exit_time: Optional[pd.Timestamp]
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    exit_price: Optional[float]
    size: float
    pnl: Optional[float]
    status: str  # 'open' or 'closed'


class Portfolio:
    """Manage portfolio state during backtesting"""

    def __init__(self, initial_capital: float, commission: float = 0.001):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.positions = {}
        self.trades = []
        self.equity_curve = [initial_capital]

    def open_position(self, symbol: str, side: str, price: float, size: float, timestamp: pd.Timestamp):
        """Open a new position"""
        # Calculate commission
        trade_value = size * price
        commission_cost = trade_value * self.commission

        # Check if we have enough cash
        if trade_value + commission_cost > self.cash:
            return None

        # Deduct from cash
        self.cash -= trade_value + commission_cost

        # Create trade
        trade = Trade(entry_time=timestamp, exit_time=None, symbol=symbol, side=side, entry_price=price, exit_price=None, size=size, pnl=None, status="open")

        self.positions[symbol] = trade
        self.trades.append(trade)

        return trade

    def close_position(self, symbol: str, price: float, timestamp: pd.Timestamp):
        """Close an existing position"""
        if symbol not in self.positions:
            return None

        trade = self.positions[symbol]
        trade.exit_time = timestamp
        trade.exit_price = price
        trade.status = "closed"

        # Calculate PnL
        if trade.side == "long":
            gross_pnl = (price - trade.entry_price) * trade.size
        else:
            gross_pnl = (trade.entry_price - price) * trade.size

        # Deduct commission
        exit_commission = price * trade.size * self.commission
        trade.pnl = gross_pnl - exit_commission

        # Add to cash
        self.cash += price * trade.size - exit_commission

        # Remove from positions
        del self.positions[symbol]

        return trade

    def update_equity(self, current_prices: dict[str, float]):
        """Update equity based on current prices"""
        total_value = self.cash

        for symbol, trade in self.positions.items():
            if symbol in current_prices:
                current_price = current_prices[symbol]
                if trade.side == "long":
                    position_value = current_price * trade.size
                else:
                    position_value = (2 * trade.entry_price - current_price) * trade.size
                total_value += position_value

        self.equity_curve.append(total_value)
        return total_value

    def get_stats(self) -> dict[str, float]:
        """Calculate portfolio statistics"""
        closed_trades = [t for t in self.trades if t.status == "closed"]

        if not closed_trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "avg_win": 0,
                "avg_loss": 0,
                "profit_factor": 0,
                "total_pnl": 0,
                "total_return": 0,
            }

        pnls = [t.pnl for t in closed_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]

        return {
            "total_trades": len(closed_trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": len(wins) / len(closed_trades) * 100 if closed_trades else 0,
            "avg_win": np.mean(wins) if wins else 0,
            "avg_loss": np.mean(losses) if losses else 0,
            "profit_factor": sum(wins) / abs(sum(losses)) if losses else 0,
            "total_pnl": sum(pnls),
            "total_return": (self.equity_curve[-1] - self.initial_capital) / self.initial_capital * 100,
        }


class Backtester:
    """Main backtesting engine"""

    def __init__(self, initial_capital: float = 10000, commission: float = 0.001, slippage: float = 0.001):
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage

    def run(self, data: pd.DataFrame, signals: pd.Series, position_size: float = 0.1) -> dict[str, Any]:
        """
        Run backtest on data with signals

        Args:
            data: DataFrame with OHLCV data
            signals: Series with trading signals (1=buy, -1=sell, 0=hold)
            position_size: Fraction of capital to use per trade
        """

        portfolio = Portfolio(self.initial_capital, self.commission)

        for i in range(len(data)):
            timestamp = data.index[i] if isinstance(data.index, pd.DatetimeIndex) else data.iloc[i]["timestamp"]
            current_price = data.iloc[i]["close"]
            signal = signals.iloc[i] if i < len(signals) else 0

            # Apply slippage
            buy_price = current_price * (1 + self.slippage)
            sell_price = current_price * (1 - self.slippage)

            # Check for position changes
            symbol = "asset"  # Generic symbol for single asset backtest

            if signal == 1 and symbol not in portfolio.positions:
                # Buy signal - open long position
                trade_size = (portfolio.cash * position_size) / buy_price
                portfolio.open_position(symbol, "long", buy_price, trade_size, timestamp)

            elif signal == -1 and symbol in portfolio.positions:
                # Sell signal - close position
                portfolio.close_position(symbol, sell_price, timestamp)

            # Update equity
            portfolio.update_equity({symbol: current_price})

        # Close any remaining positions
        for symbol in list(portfolio.positions.keys()):
            last_price = data.iloc[-1]["close"]
            portfolio.close_position(symbol, last_price, data.index[-1])

        # Calculate performance metrics
        equity_curve = np.array(portfolio.equity_curve)
        returns = np.diff(equity_curve) / equity_curve[:-1]

        results = {
            "portfolio_stats": portfolio.get_stats(),
            "equity_curve": equity_curve,
            "returns": returns,
            "trades": len(portfolio.trades),
            "final_equity": equity_curve[-1],
            "total_return": (equity_curve[-1] - self.initial_capital) / self.initial_capital * 100,
            "max_drawdown": self._calculate_max_drawdown(equity_curve),
            "sharpe_ratio": self._calculate_sharpe_ratio(returns),
            "trades_list": portfolio.trades,
        }

        return results

    def _calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        """Calculate maximum drawdown"""
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return np.min(drawdown) * 100

    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio (annualized)"""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
