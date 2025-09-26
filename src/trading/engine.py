"""
Main trading engine
"""

import asyncio
from datetime import datetime
from typing import Any, Optional

from loguru import logger

from ..core import ConfigManager, Database
from ..ml import EnsembleModel, FeatureEngineer, RegimeDetector
from .order import OrderManager
from .position import PositionManager


class TradingEngine:
    """Main trading engine"""

    def __init__(self, config: ConfigManager, database: Database):
        """Initialize trading engine

        Args:
            config: Configuration manager
            database: Database instance
        """
        self.config = config
        self.database = database
        self.log = logger.bind(name=__name__)

        self.position_manager = PositionManager(config, database)
        self.order_manager = OrderManager(config, database)

        # Initialize ML components
        self.ensemble_model = EnsembleModel(config.get("ml", {}))
        self.regime_detector = RegimeDetector(config.get("ml", {}))
        self.feature_engineer = FeatureEngineer()

        self.is_running = False
        self.tasks = []
        self.predictions_made = 0

        # Trading parameters
        self.pairs = config.get("trading.pairs", [])
        self.timeframes = config.get("trading.timeframes", ["5m"])
        self.initial_balance = config.get("trading.initial_balance", 10000)
        self.stake_per_trade = config.get("trading.stake_per_trade", 0.02)
        self.max_open_trades = config.get("trading.max_open_trades", 5)

        # Risk parameters
        self.max_drawdown = config.get("risk.max_drawdown", 0.10)
        self.daily_loss_limit = config.get("risk.daily_loss_limit", 0.03)
        self.stop_loss = config.get("risk.stop_loss", 0.02)
        self.take_profit = config.get("risk.take_profit", 0.05)

        self.log.info(f"Trading engine initialized for {len(self.pairs)} pairs")

    async def start(self):
        """Start trading engine"""
        if self.is_running:
            self.log.warning("Trading engine already running")
            return

        self.log.info("Starting trading engine...")
        self.is_running = True

        # Start main trading loop
        self.tasks.append(asyncio.create_task(self._trading_loop()))

        # Start position monitoring
        self.tasks.append(asyncio.create_task(self._monitor_positions()))

        # Start risk monitoring
        self.tasks.append(asyncio.create_task(self._monitor_risk()))

        self.log.info("Trading engine started")

    async def stop(self):
        """Stop trading engine"""
        if not self.is_running:
            return

        self.log.info("Stopping trading engine...")
        self.is_running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        # Wait for tasks to complete
        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

        self.log.info("Trading engine stopped")

    async def emergency_stop(self):
        """Emergency stop - close all positions immediately"""
        self.log.warning("EMERGENCY STOP initiated!")

        # Stop trading
        self.is_running = False

        # Close all positions
        positions = self.position_manager.get_open_positions()
        for position in positions:
            self.log.info(f"Emergency closing position: {position['symbol']}")
            await self.position_manager.close_position(position["id"], emergency=True)

        # Cancel all pending orders
        await self.order_manager.cancel_all_orders()

        self.log.warning("Emergency stop completed")

    async def _trading_loop(self):
        """Main trading loop"""
        while self.is_running:
            try:
                # Check for trading opportunities
                for pair in self.pairs:
                    # Check if we can open new positions
                    if len(self.position_manager.get_open_positions()) >= self.max_open_trades:
                        continue

                    # Generate trading signal (placeholder - will be replaced with ML predictions)
                    signal = await self._generate_signal(pair)

                    if signal and signal["confidence"] > 0.6:
                        # Execute trade
                        await self._execute_trade(pair, signal)

                # Sleep before next iteration
                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.log.error(f"Error in trading loop: {e}")
                await asyncio.sleep(10)

    async def _generate_signal(self, pair: str) -> Optional[dict[str, Any]]:
        """Generate trading signal for a pair using ML models

        Args:
            pair: Trading pair

        Returns:
            Trading signal or None
        """
        try:
            # Get market data
            import pandas as pd

            market_data = self.database.get_market_data(pair, "1m", limit=100)

            if market_data is None or len(market_data) < 50:
                # Fallback to random signals if not enough data
                import random

                if random.random() > 0.95:
                    return {
                        "symbol": pair,
                        "side": random.choice(["BUY", "SELL"]),
                        "confidence": random.uniform(0.6, 0.9),
                        "predicted_return": random.uniform(0.01, 0.05),
                        "strategy": "random",
                    }
                return None

            # Convert to DataFrame
            df = pd.DataFrame(market_data)

            # Detect market regime
            regime, regime_confidence = self.regime_detector.detect_regime(df)

            # Engineer features
            features = self.feature_engineer.create_features(df)

            if len(features) == 0:
                return None

            # Normalize features
            normalized_features = self.feature_engineer.normalize_features(features)

            # Get predictions from ensemble
            predictions, model_predictions = self.ensemble_model.predict(normalized_features.tail(1))
            predicted_return = float(predictions[0])

            self.predictions_made += 1

            # Get regime parameters
            regime_params = self.regime_detector.get_regime_params(regime)

            # Generate signal based on prediction and regime
            confidence = abs(predicted_return) * 10 * regime_confidence  # Scale to 0-1
            confidence = min(confidence, 0.95)

            # Check if signal meets threshold
            if confidence > regime_params["entry_threshold"]:
                side = "BUY" if predicted_return > 0 else "SELL"

                # Save prediction to database
                self.database.save_prediction(
                    {
                        "timestamp": datetime.now(),
                        "symbol": pair,
                        "predicted_return": predicted_return,
                        "confidence": confidence,
                        "regime": regime.value,
                        "models": {k: float(v[0]) for k, v in model_predictions.items()},
                        "features": dict(list(features.items())[:10]),  # Save top 10 features
                    }
                )

                return {
                    "symbol": pair,
                    "side": side,
                    "confidence": confidence,
                    "predicted_return": abs(predicted_return),
                    "strategy": "ensemble",
                    "regime": regime.value,
                    "position_size_multiplier": regime_params["position_size_multiplier"],
                }

            return None

        except Exception as e:
            self.log.error(f"Error generating ML signal for {pair}: {e}")
            # Fallback to simple random signal
            import random

            if random.random() > 0.98:
                return {"symbol": pair, "side": random.choice(["BUY", "SELL"]), "confidence": 0.6, "predicted_return": 0.02, "strategy": "fallback"}
            return None

    async def _execute_trade(self, pair: str, signal: dict[str, Any]):
        """Execute trade based on signal

        Args:
            pair: Trading pair
            signal: Trading signal
        """
        try:
            # Calculate position size
            balance = self.position_manager.get_available_balance()
            position_size = balance * self.stake_per_trade

            # Create order
            order = await self.order_manager.create_order(symbol=pair, side=signal["side"], size=position_size, order_type="MARKET")

            if order:
                # Create position
                position = await self.position_manager.open_position(
                    symbol=pair,
                    side=signal["side"],
                    entry_price=order["price"],
                    size=order["size"],
                    stop_loss=self.stop_loss,
                    take_profit=self.take_profit,
                    strategy=signal["strategy"],
                )

                self.log.info(f"Opened position: {pair} {signal['side']} @ {order['price']}")

                # Save trade to database
                self.database.save_trade(
                    {
                        "timestamp": datetime.now(),
                        "symbol": pair,
                        "side": signal["side"],
                        "entry_price": order["price"],
                        "size": order["size"],
                        "strategy": signal["strategy"],
                        "mode": self.config.mode,
                        "metadata": {"confidence": signal["confidence"], "predicted_return": signal["predicted_return"]},
                    }
                )

        except Exception as e:
            self.log.error(f"Failed to execute trade: {e}")

    async def _monitor_positions(self):
        """Monitor open positions"""
        while self.is_running:
            try:
                positions = self.position_manager.get_open_positions()

                for position in positions:
                    # Check stop loss and take profit
                    current_price = await self._get_current_price(position["symbol"])

                    if current_price:
                        pnl_percent = self.position_manager.calculate_pnl_percent(position, current_price)

                        # Check stop loss
                        if pnl_percent <= -self.stop_loss:
                            self.log.info(f"Stop loss triggered for {position['symbol']}")
                            await self.position_manager.close_position(position["id"], exit_price=current_price, reason="stop_loss")

                        # Check take profit
                        elif pnl_percent >= self.take_profit:
                            self.log.info(f"Take profit triggered for {position['symbol']}")
                            await self.position_manager.close_position(position["id"], exit_price=current_price, reason="take_profit")

                await asyncio.sleep(2)  # Check every 2 seconds

            except Exception as e:
                self.log.error(f"Error monitoring positions: {e}")
                await asyncio.sleep(5)

    async def _monitor_risk(self):
        """Monitor risk metrics"""
        while self.is_running:
            try:
                # Calculate current drawdown
                drawdown = self.position_manager.calculate_drawdown()

                if drawdown > self.max_drawdown:
                    self.log.warning(f"Max drawdown exceeded: {drawdown:.2%}")
                    await self.emergency_stop()
                    break

                # Check daily loss limit
                daily_pnl = self.position_manager.get_daily_pnl()
                if daily_pnl < -self.daily_loss_limit * self.initial_balance:
                    self.log.warning(f"Daily loss limit exceeded: ${daily_pnl:.2f}")
                    await self.emergency_stop()
                    break

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.log.error(f"Error monitoring risk: {e}")
                await asyncio.sleep(10)

    async def _get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for symbol

        Args:
            symbol: Trading pair

        Returns:
            Current price or None
        """
        # Placeholder - will be replaced with real price feed
        import random

        base_prices = {"BTC/USDT": 43000, "ETH/USDT": 2300, "BNB/USDT": 315, "ADA/USDT": 0.58, "SOL/USDT": 98}

        if symbol in base_prices:
            return base_prices[symbol] * (1 + random.uniform(-0.01, 0.01))

        return None

    def get_positions(self) -> list[dict[str, Any]]:
        """Get all positions

        Returns:
            List of positions
        """
        return self.position_manager.get_all_positions()

    async def get_ticker(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get ticker data for symbol

        Args:
            symbol: Trading pair

        Returns:
            Ticker data
        """
        price = await self._get_current_price(symbol)
        if price:
            return {
                "price": price,
                "change24h": 2.5,  # Placeholder
                "volume24h": 1000000,  # Placeholder
                "high24h": price * 1.02,
                "low24h": price * 0.98,
            }
        return None
