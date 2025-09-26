"""
Main entry point for CryptoBot Pro
"""

import asyncio
import signal
import sys
from pathlib import Path
from typing import Optional

import click
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api import create_api_app
from src.core import ConfigManager, Database, Logger
from src.simulation import SimulationEngine
from src.trading import TradingEngine


class CryptoBotPro:
    """Main application class"""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize CryptoBot Pro

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigManager(config_path)
        self.logger = Logger(self.config.get("logging", {}))
        self.log = self.logger.get_logger(__name__)
        self.database = Database(self.config.get("database", {}))

        self.trading_engine = None
        self.simulation_engine = None
        self.api_app = None

        self.log.info(f"CryptoBot Pro initialized in {self.config.mode} mode")

    async def start(self):
        """Start the application"""
        self.log.info("Starting CryptoBot Pro...")

        # Initialize engines based on mode
        if self.config.is_simulation:
            self.log.info("Initializing simulation engine...")
            self.simulation_engine = SimulationEngine(self.config, self.database)
            await self.simulation_engine.start()

        # Initialize trading engine
        self.log.info("Initializing trading engine...")
        self.trading_engine = TradingEngine(self.config, self.database)
        await self.trading_engine.start()

        # Start API server
        self.log.info("Starting API server...")
        await self.start_api_server()

    async def start_api_server(self):
        """Start FastAPI server"""
        self.api_app = create_api_app(self.config, self.database, self.trading_engine)

        config = uvicorn.Config(
            app=self.api_app,
            host=self.config.get("api.host", "0.0.0.0"),
            port=self.config.get("api.port", 8000),
            log_level=self.config.get("logging.level", "info").lower(),
        )
        server = uvicorn.Server(config)
        await server.serve()

    async def stop(self):
        """Stop the application"""
        self.log.info("Stopping CryptoBot Pro...")

        if self.trading_engine:
            await self.trading_engine.stop()

        if self.simulation_engine:
            await self.simulation_engine.stop()

        self.log.info("CryptoBot Pro stopped")

    def run(self):
        """Run the application"""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Handle signals
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda s, f: loop.create_task(self.stop()))

        try:
            loop.run_until_complete(self.start())
        except KeyboardInterrupt:
            self.log.info("Received interrupt signal")
        finally:
            loop.run_until_complete(self.stop())
            loop.close()


@click.group()
def cli():
    """CryptoBot Pro CLI"""


@cli.command()
@click.option("--mode", type=click.Choice(["simulation", "paper", "live"]), default="simulation", help="Trading mode")
@click.option("--config", type=click.Path(exists=True), help="Configuration file path")
@click.option("--duration", type=str, default="24h", help="Run duration (e.g., 24h, 7d)")
@click.option("--pairs", type=str, help="Trading pairs (comma-separated)")
def run(mode: str, config: Optional[str], duration: str, pairs: Optional[str]):
    """Run CryptoBot Pro"""
    click.echo(f"Starting CryptoBot Pro in {mode} mode...")

    # Create app instance
    app = CryptoBotPro(config)

    # Override mode if specified
    if mode:
        app.config.set("system.mode", mode)

    # Override pairs if specified
    if pairs:
        app.config.set("trading.pairs", pairs.split(","))

    # Run the application
    app.run()


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Configuration file path")
def backtest(config: Optional[str]):
    """Run historical backtest"""
    click.echo("Running historical backtest...")

    from src.backtesting import BacktestEngine

    app = CryptoBotPro(config)
    engine = BacktestEngine(app.config, app.database)

    # Run backtest
    asyncio.run(engine.run())

    # Display results
    results = engine.get_results()
    click.echo("Backtest Results:")
    click.echo(f"  Total P&L: ${results['total_pnl']:.2f}")
    click.echo(f"  Win Rate: {results['win_rate']:.1f}%")
    click.echo(f"  Sharpe Ratio: {results['sharpe_ratio']:.2f}")
    click.echo(f"  Max Drawdown: {results['max_drawdown']:.1f}%")


@cli.command()
@click.option("--config", type=click.Path(exists=True), help="Configuration file path")
def train(config: Optional[str]):
    """Train ML models"""
    click.echo("Training ML models...")

    from src.ml import ModelTrainer

    app = CryptoBotPro(config)
    trainer = ModelTrainer(app.config, app.database)

    # Train models
    asyncio.run(trainer.train_all())

    click.echo("Model training complete!")


@cli.command()
def benchmark():
    """Run 24-hour benchmark test"""
    click.echo("Starting 24-hour benchmark test...")
    click.echo("This will run a parallel simulation against real market data")

    # Create app in simulation mode
    app = CryptoBotPro()
    app.config.set("system.mode", "simulation")
    app.config.set("simulation.duration", "24h")

    # Run benchmark
    app.run()


if __name__ == "__main__":
    cli()
