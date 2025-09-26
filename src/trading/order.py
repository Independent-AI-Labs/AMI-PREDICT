"""
Order management module
"""

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from loguru import logger

from ..core import ConfigManager, Database


class OrderType(Enum):
    """Order types"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"


class OrderStatus(Enum):
    """Order status"""

    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class Order:
    """Represents a trading order"""

    def __init__(self, data: dict[str, Any]):
        """Initialize order

        Args:
            data: Order data
        """
        self.id = data.get("id")
        self.symbol = data["symbol"]
        self.side = data["side"]  # BUY or SELL
        self.order_type = data.get("order_type", OrderType.MARKET.value)
        self.size = data["size"]
        self.price = data.get("price")
        self.limit_price = data.get("limit_price")
        self.stop_price = data.get("stop_price")
        self.status = data.get("status", OrderStatus.PENDING.value)
        self.filled_size = data.get("filled_size", 0)
        self.average_price = data.get("average_price")
        self.created_at = data.get("created_at", datetime.now())
        self.filled_at = data.get("filled_at")
        self.metadata = data.get("metadata", {})

    def is_filled(self) -> bool:
        """Check if order is filled

        Returns:
            True if filled
        """
        return self.status == OrderStatus.FILLED.value

    def is_pending(self) -> bool:
        """Check if order is pending

        Returns:
            True if pending
        """
        return self.status == OrderStatus.PENDING.value

    def to_dict(self) -> dict[str, Any]:
        """Convert order to dictionary

        Returns:
            Order data
        """
        return {
            "id": self.id,
            "symbol": self.symbol,
            "side": self.side,
            "order_type": self.order_type,
            "size": self.size,
            "price": self.price,
            "limit_price": self.limit_price,
            "stop_price": self.stop_price,
            "status": self.status,
            "filled_size": self.filled_size,
            "average_price": self.average_price,
            "created_at": self.created_at,
            "filled_at": self.filled_at,
            "metadata": self.metadata,
        }


class OrderManager:
    """Manages trading orders"""

    def __init__(self, config: ConfigManager, database: Database):
        """Initialize order manager

        Args:
            config: Configuration manager
            database: Database instance
        """
        self.config = config
        self.database = database
        self.log = logger.bind(name=__name__)

        self.orders = {}  # Order ID -> Order
        self.order_counter = 0

        # Simulation parameters
        self.latency_ms = config.get("simulation.latency_ms", 50)
        self.slippage_percent = config.get("simulation.slippage_percent", 0.1) / 100

    async def create_order(
        self, symbol: str, side: str, size: float, order_type: str = "MARKET", limit_price: Optional[float] = None, stop_price: Optional[float] = None
    ) -> dict[str, Any]:
        """Create a new order

        Args:
            symbol: Trading pair
            side: BUY or SELL
            size: Order size
            order_type: Order type
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders

        Returns:
            Created order data
        """
        self.order_counter += 1

        order_data = {
            "id": f"ORD_{self.order_counter}",
            "symbol": symbol,
            "side": side,
            "size": size,
            "order_type": order_type,
            "limit_price": limit_price,
            "stop_price": stop_price,
            "created_at": datetime.now(),
            "status": OrderStatus.PENDING.value,
        }

        order = Order(order_data)
        self.orders[order.id] = order

        self.log.info(f"Created order {order.id}: {symbol} {side} {size}")

        # Simulate order execution based on mode
        if self.config.is_simulation or self.config.is_paper:
            await self._simulate_order_execution(order)
        elif self.config.is_live:
            await self._execute_live_order(order)

        return order.to_dict()

    async def _simulate_order_execution(self, order: Order):
        """Simulate order execution

        Args:
            order: Order to execute
        """
        import asyncio
        import random

        # Simulate latency
        await asyncio.sleep(self.latency_ms / 1000)

        # Get current price (placeholder)
        base_prices = {"BTC/USDT": 43000, "ETH/USDT": 2300, "BNB/USDT": 315, "ADA/USDT": 0.58, "SOL/USDT": 98}

        if order.symbol in base_prices:
            current_price = base_prices[order.symbol] * (1 + random.uniform(-0.01, 0.01))

            # Apply slippage
            if order.side == "BUY":
                execution_price = current_price * (1 + self.slippage_percent)
            else:
                execution_price = current_price * (1 - self.slippage_percent)

            # Fill order
            order.status = OrderStatus.FILLED.value
            order.filled_size = order.size
            order.average_price = execution_price
            order.price = execution_price
            order.filled_at = datetime.now()

            self.log.info(f"Order {order.id} filled at {execution_price:.2f}")
        else:
            order.status = OrderStatus.REJECTED.value
            self.log.warning(f"Order {order.id} rejected: Unknown symbol")

    async def _execute_live_order(self, order: Order):
        """Execute live order on exchange

        Args:
            order: Order to execute
        """
        # Placeholder for live trading implementation
        self.log.warning("Live trading not yet implemented")
        order.status = OrderStatus.REJECTED.value

    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order

        Args:
            order_id: Order ID

        Returns:
            True if cancelled
        """
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]

        if order.is_pending():
            order.status = OrderStatus.CANCELLED.value
            self.log.info(f"Order {order_id} cancelled")
            return True

        return False

    async def cancel_all_orders(self):
        """Cancel all pending orders"""
        pending_orders = [order for order in self.orders.values() if order.is_pending()]

        for order in pending_orders:
            await self.cancel_order(order.id)

        self.log.info(f"Cancelled {len(pending_orders)} orders")

    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID

        Args:
            order_id: Order ID

        Returns:
            Order or None
        """
        return self.orders.get(order_id)

    def get_pending_orders(self) -> list[dict[str, Any]]:
        """Get all pending orders

        Returns:
            List of pending orders
        """
        return [order.to_dict() for order in self.orders.values() if order.is_pending()]

    def get_all_orders(self) -> list[dict[str, Any]]:
        """Get all orders

        Returns:
            List of all orders
        """
        return [order.to_dict() for order in self.orders.values()]
