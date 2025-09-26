"""
Position management module
"""

from datetime import datetime
from typing import Any, Optional

from loguru import logger

from ..core import ConfigManager, Database


class Position:
    """Represents a trading position"""

    def __init__(self, data: dict[str, Any]):
        """Initialize position

        Args:
            data: Position data
        """
        self.id = data.get("id")
        self.symbol = data["symbol"]
        self.side = data["side"]  # LONG or SHORT
        self.entry_price = data["entry_price"]
        self.size = data["size"]
        self.current_price = data.get("current_price", self.entry_price)
        self.exit_price = data.get("exit_price")
        self.stop_loss = data.get("stop_loss")
        self.take_profit = data.get("take_profit")
        self.opened_at = data.get("opened_at", datetime.now())
        self.closed_at = data.get("closed_at")
        self.status = data.get("status", "open")  # open, closed
        self.pnl = data.get("pnl", 0)
        self.pnl_percent = data.get("pnl_percent", 0)
        self.strategy = data.get("strategy")
        self.metadata = data.get("metadata", {})

    def calculate_pnl(self, current_price: float) -> float:
        """Calculate P&L

        Args:
            current_price: Current market price

        Returns:
            P&L amount
        """
        if self.side == "LONG":
            return (current_price - self.entry_price) * self.size
        else:  # SHORT
            return (self.entry_price - current_price) * self.size

    def calculate_pnl_percent(self, current_price: float) -> float:
        """Calculate P&L percentage

        Args:
            current_price: Current market price

        Returns:
            P&L percentage
        """
        pnl = self.calculate_pnl(current_price)
        return (pnl / (self.entry_price * self.size)) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert position to dictionary

        Returns:
            Position data
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "entry_price": self.entry_price,
            "size": self.size,
            "current_price": self.current_price,
            "exit_price": self.exit_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "status": self.status,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "strategy": self.strategy,
            "metadata": self.metadata,
        }


class PositionManager:
    """Manages trading positions"""

    def __init__(self, config: ConfigManager, database: Database):
        """Initialize position manager

        Args:
            config: Configuration manager
            database: Database instance
        """
        self.config = config
        self.database = database
        self.log = logger.bind(name=__name__)

        self.positions = {}  # Position ID -> Position
        self.position_counter = 0
        self.initial_balance = config.get("trading.initial_balance", 10000)
        self.current_balance = self.initial_balance
        self.peak_balance = self.initial_balance

    async def open_position(self, **kwargs) -> Position:
        """Open a new position

        Returns:
            Created position
        """
        self.position_counter += 1
        position_data = {"id": f"POS_{self.position_counter}", "opened_at": datetime.now(), "status": "open", **kwargs}

        position = Position(position_data)
        self.positions[position.id] = position

        self.log.info(f"Opened position {position.id}: {position.symbol} {position.side}")

        return position

    async def close_position(self, position_id: str, exit_price: float = None, reason: str = None, emergency: bool = False) -> Position:
        """Close a position

        Args:
            position_id: Position ID
            exit_price: Exit price
            reason: Close reason
            emergency: Emergency close flag

        Returns:
            Closed position
        """
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")

        position = self.positions[position_id]

        if position.status == "closed":
            return position

        # Set exit price
        if exit_price:
            position.exit_price = exit_price
        else:
            position.exit_price = position.current_price

        # Calculate final P&L
        position.pnl = position.calculate_pnl(position.exit_price)
        position.pnl_percent = position.calculate_pnl_percent(position.exit_price)

        # Update balance
        self.current_balance += position.pnl

        # Update peak balance for drawdown calculation
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance

        # Mark as closed
        position.status = "closed"
        position.closed_at = datetime.now()
        position.metadata["close_reason"] = reason
        position.metadata["emergency"] = emergency

        # Save to database
        self.database.save_trade(
            {
                "timestamp": position.opened_at,
                "symbol": position.symbol,
                "side": position.side,
                "entry_price": position.entry_price,
                "exit_price": position.exit_price,
                "size": position.size,
                "pnl": position.pnl,
                "pnl_percent": position.pnl_percent,
                "strategy": position.strategy,
                "mode": self.config.mode,
                "metadata": position.metadata,
            }
        )

        self.log.info(f"Closed position {position.id}: P&L ${position.pnl:.2f} ({position.pnl_percent:.2f}%)")

        return position

    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID

        Args:
            position_id: Position ID

        Returns:
            Position or None
        """
        return self.positions.get(position_id)

    def get_open_positions(self) -> list[dict[str, Any]]:
        """Get all open positions

        Returns:
            List of open positions
        """
        return [pos.to_dict() for pos in self.positions.values() if pos.status == "open"]

    def get_all_positions(self) -> list[dict[str, Any]]:
        """Get all positions

        Returns:
            List of all positions
        """
        return [pos.to_dict() for pos in self.positions.values()]

    def get_available_balance(self) -> float:
        """Get available balance for trading

        Returns:
            Available balance
        """
        # Calculate balance locked in open positions
        locked_balance = sum(pos.entry_price * pos.size for pos in self.positions.values() if pos.status == "open")

        return self.current_balance - locked_balance

    def calculate_drawdown(self) -> float:
        """Calculate current drawdown

        Returns:
            Drawdown percentage
        """
        if self.peak_balance == 0:
            return 0

        drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
        return max(0, drawdown)

    def get_daily_pnl(self) -> float:
        """Get today's P&L

        Returns:
            Daily P&L
        """
        today = datetime.now().date()
        daily_pnl = 0

        for pos in self.positions.values():
            if pos.status == "closed" and pos.closed_at.date() == today:
                daily_pnl += pos.pnl
            elif pos.status == "open":
                # Include unrealized P&L
                daily_pnl += pos.calculate_pnl(pos.current_price)

        return daily_pnl

    def calculate_pnl_percent(self, position: dict[str, Any], current_price: float) -> float:
        """Calculate P&L percentage for a position

        Args:
            position: Position data
            current_price: Current price

        Returns:
            P&L percentage
        """
        if position["side"] == "LONG":
            pnl = (current_price - position["entry_price"]) * position["size"]
        else:
            pnl = (position["entry_price"] - current_price) * position["size"]

        return (pnl / (position["entry_price"] * position["size"])) * 100

    def get_performance_metrics(self) -> dict[str, Any]:
        """Get performance metrics

        Returns:
            Performance metrics
        """
        closed_positions = [pos for pos in self.positions.values() if pos.status == "closed"]

        if not closed_positions:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "average_win": 0,
                "average_loss": 0,
                "profit_factor": 0,
                "max_drawdown": 0,
            }

        winning_trades = [pos for pos in closed_positions if pos.pnl > 0]
        losing_trades = [pos for pos in closed_positions if pos.pnl <= 0]

        total_wins = sum(pos.pnl for pos in winning_trades)
        total_losses = abs(sum(pos.pnl for pos in losing_trades))

        return {
            "total_trades": len(closed_positions),
            "winning_trades": len(winning_trades),
            "losing_trades": len(losing_trades),
            "win_rate": (len(winning_trades) / len(closed_positions) * 100) if closed_positions else 0,
            "total_pnl": self.current_balance - self.initial_balance,
            "average_win": (total_wins / len(winning_trades)) if winning_trades else 0,
            "average_loss": (total_losses / len(losing_trades)) if losing_trades else 0,
            "profit_factor": (total_wins / total_losses) if total_losses > 0 else 0,
            "max_drawdown": self.calculate_drawdown() * 100,
        }
