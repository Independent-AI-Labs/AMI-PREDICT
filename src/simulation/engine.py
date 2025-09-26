"""
Real-time simulation engine for 24-hour benchmark
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any

from loguru import logger

from ..core import ConfigManager, Database
from .data_feed import SimulatedDataFeed
from .market_simulator import MarketSimulator


class SimulationEngine:
    """Real-time simulation engine"""

    def __init__(self, config: ConfigManager, database: Database):
        """Initialize simulation engine

        Args:
            config: Configuration manager
            database: Database instance
        """
        self.config = config
        self.database = database
        self.log = logger.bind(name=__name__)

        self.market_simulator = MarketSimulator(config)
        self.data_feed = SimulatedDataFeed(config)

        self.is_running = False
        self.start_time = None
        self.simulation_time = 0
        self.tasks = []

        # Simulation parameters
        self.duration = config.get("simulation.duration", "24h")
        self.speed_multiplier = config.get("simulation.speed_multiplier", 1)

        # Performance tracking
        self.metrics = {
            "trades_executed": 0,
            "signals_generated": 0,
            "predictions_made": 0,
            "total_pnl": 0,
            "max_drawdown": 0,
            "win_rate": 0,
            "sharpe_ratio": 0,
        }

        self.log.info(f"Simulation engine initialized for {self.duration}")

    async def start(self):
        """Start simulation engine"""
        if self.is_running:
            self.log.warning("Simulation already running")
            return

        self.log.info("Starting simulation engine...")
        self.is_running = True
        self.start_time = datetime.now()

        # Start market simulator
        self.tasks.append(asyncio.create_task(self._run_market_simulation()))

        # Start data feed
        self.tasks.append(asyncio.create_task(self._run_data_feed()))

        # Start metrics collector
        self.tasks.append(asyncio.create_task(self._collect_metrics()))

        # Start performance reporter
        self.tasks.append(asyncio.create_task(self._report_performance()))

        self.log.info("Simulation engine started")

    async def stop(self):
        """Stop simulation engine"""
        if not self.is_running:
            return

        self.log.info("Stopping simulation engine...")
        self.is_running = False

        # Cancel all tasks
        for task in self.tasks:
            task.cancel()

        await asyncio.gather(*self.tasks, return_exceptions=True)
        self.tasks.clear()

        # Save final metrics
        await self._save_final_metrics()

        self.log.info("Simulation engine stopped")

    async def _run_market_simulation(self):
        """Run market simulation"""
        while self.is_running:
            try:
                # Update simulation time
                self.simulation_time += 1

                # Generate market data
                market_data = await self.market_simulator.generate_tick()

                # Save to database
                for symbol, data in market_data.items():
                    self.database.save_market_data(data, symbol, "1m")

                # Check if simulation duration reached
                if self._is_simulation_complete():
                    self.log.info("Simulation duration reached")
                    await self.stop()
                    break

                # Sleep based on speed multiplier
                await asyncio.sleep(1 / self.speed_multiplier)

            except Exception as e:
                self.log.error(f"Error in market simulation: {e}")
                await asyncio.sleep(1)

    async def _run_data_feed(self):
        """Run simulated data feed"""
        while self.is_running:
            try:
                # Generate data feed events
                events = await self.data_feed.generate_events()

                for event in events:
                    if event["type"] == "signal":
                        self.metrics["signals_generated"] += 1
                    elif event["type"] == "prediction":
                        self.metrics["predictions_made"] += 1

                await asyncio.sleep(0.5 / self.speed_multiplier)

            except Exception as e:
                self.log.error(f"Error in data feed: {e}")
                await asyncio.sleep(1)

    async def _collect_metrics(self):
        """Collect performance metrics"""
        while self.is_running:
            try:
                # Update metrics from trading engine
                trades = self.database.get_trades(mode="simulation", start=self.start_time)

                # Get predictions count from database
                predictions = self.database.get_predictions(start=self.start_time)
                if predictions:
                    self.metrics["predictions_made"] = len(predictions)

                if trades:
                    self.metrics["trades_executed"] = len(trades)

                    # Calculate P&L
                    total_pnl = sum(t["pnl"] for t in trades if t["pnl"])
                    self.metrics["total_pnl"] = total_pnl

                    # Calculate win rate
                    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
                    self.metrics["win_rate"] = (len(winning_trades) / len(trades) * 100) if trades else 0

                    # Calculate drawdown
                    running_pnl = 0
                    peak_pnl = 0
                    max_drawdown = 0

                    for trade in sorted(trades, key=lambda x: x["timestamp"]):
                        running_pnl += trade.get("pnl", 0)
                        if running_pnl > peak_pnl:
                            peak_pnl = running_pnl
                        drawdown = (peak_pnl - running_pnl) / peak_pnl if peak_pnl > 0 else 0
                        max_drawdown = max(max_drawdown, drawdown)

                    self.metrics["max_drawdown"] = max_drawdown * 100

                    # Calculate Sharpe ratio (simplified)
                    if len(trades) > 1:
                        returns = [t.get("pnl_percent", 0) for t in trades]
                        avg_return = sum(returns) / len(returns)
                        std_return = (sum((r - avg_return) ** 2 for r in returns) / len(returns)) ** 0.5
                        self.metrics["sharpe_ratio"] = (avg_return / std_return * (252**0.5)) if std_return > 0 else 0

                await asyncio.sleep(10)  # Update every 10 seconds

            except Exception as e:
                self.log.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(10)

    async def _report_performance(self):
        """Report performance metrics periodically"""
        while self.is_running:
            try:
                elapsed = datetime.now() - self.start_time

                self.log.info(
                    f"Simulation Progress: {elapsed} | "
                    f"Trades: {self.metrics['trades_executed']} | "
                    f"P&L: ${self.metrics['total_pnl']:.2f} | "
                    f"Win Rate: {self.metrics['win_rate']:.1f}% | "
                    f"Drawdown: {self.metrics['max_drawdown']:.1f}%"
                )

                # Save performance to database
                self.database.save_performance(
                    {
                        "timestamp": datetime.now(),
                        "mode": "simulation",
                        "total_pnl": self.metrics["total_pnl"],
                        "win_rate": self.metrics["win_rate"],
                        "sharpe_ratio": self.metrics["sharpe_ratio"],
                        "max_drawdown": self.metrics["max_drawdown"],
                        "total_trades": self.metrics["trades_executed"],
                        "winning_trades": 0,  # Will be calculated
                        "losing_trades": 0,  # Will be calculated
                        "metadata": {
                            "signals_generated": self.metrics["signals_generated"],
                            "predictions_made": self.metrics["predictions_made"],
                            "elapsed_time": str(elapsed),
                        },
                    }
                )

                await asyncio.sleep(60)  # Report every minute

            except Exception as e:
                self.log.error(f"Error reporting performance: {e}")
                await asyncio.sleep(60)

    async def _save_final_metrics(self):
        """Save final simulation metrics"""
        elapsed = datetime.now() - self.start_time

        self.log.info("=" * 50)
        self.log.info("SIMULATION COMPLETE")
        self.log.info("=" * 50)
        self.log.info(f"Duration: {elapsed}")
        self.log.info(f"Trades Executed: {self.metrics['trades_executed']}")
        self.log.info(f"Signals Generated: {self.metrics['signals_generated']}")
        self.log.info(f"Predictions Made: {self.metrics['predictions_made']}")
        self.log.info(f"Total P&L: ${self.metrics['total_pnl']:.2f}")
        self.log.info(f"Win Rate: {self.metrics['win_rate']:.1f}%")
        self.log.info(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.2f}")
        self.log.info(f"Max Drawdown: {self.metrics['max_drawdown']:.1f}%")
        self.log.info("=" * 50)

        # Save final performance
        self.database.save_performance(
            {
                "timestamp": datetime.now(),
                "mode": "simulation_final",
                "total_pnl": self.metrics["total_pnl"],
                "win_rate": self.metrics["win_rate"],
                "sharpe_ratio": self.metrics["sharpe_ratio"],
                "max_drawdown": self.metrics["max_drawdown"],
                "total_trades": self.metrics["trades_executed"],
                "winning_trades": 0,
                "losing_trades": 0,
                "metadata": {
                    "signals_generated": self.metrics["signals_generated"],
                    "predictions_made": self.metrics["predictions_made"],
                    "total_duration": str(elapsed),
                    "final_report": True,
                },
            }
        )

    def _is_simulation_complete(self) -> bool:
        """Check if simulation duration is complete

        Returns:
            True if complete
        """
        if not self.start_time:
            return False

        # Parse duration
        if self.duration.endswith("h"):
            hours = int(self.duration[:-1])
            target_duration = timedelta(hours=hours)
        elif self.duration.endswith("d"):
            days = int(self.duration[:-1])
            target_duration = timedelta(days=days)
        else:
            # Default to 24 hours
            target_duration = timedelta(hours=24)

        # Adjust for speed multiplier
        target_duration = target_duration / self.speed_multiplier

        elapsed = datetime.now() - self.start_time
        return elapsed >= target_duration

    def get_metrics(self) -> dict[str, Any]:
        """Get current simulation metrics

        Returns:
            Current metrics
        """
        return self.metrics.copy()
