#!/usr/bin/env python
"""
Run 24-hour benchmark test
"""

import asyncio
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.api import create_api_app
from src.core import ConfigManager, Database, Logger
from src.simulation import SimulationEngine
from src.trading import TradingEngine


async def run_benchmark():
    """Run 24-hour benchmark test"""
    print("=" * 60)
    print("CRYPTOBOT PRO - 24-HOUR BENCHMARK TEST")
    print("=" * 60)
    print(f"Start Time: {datetime.now()}")
    print("-" * 60)

    # Initialize components
    config = ConfigManager()
    config.set("system.mode", "simulation")
    config.set("simulation.duration", "24h")
    config.set("simulation.speed_multiplier", 1440)  # 1440x speed = 24h in 1 minute for testing

    logger = Logger(config.get("logging"))
    log = logger.get_logger(__name__)

    database = Database(config.get("database"))

    # Initialize engines
    log.info("Initializing simulation engine...")
    simulation_engine = SimulationEngine(config, database)

    log.info("Initializing trading engine...")
    trading_engine = TradingEngine(config, database)

    # Start API server in background
    log.info("Starting API server...")
    api_app = create_api_app(config, database, trading_engine)

    # Run server in background task
    from uvicorn import Config, Server

    server_config = Config(app=api_app, host="0.0.0.0", port=8000, log_level="warning")
    server = Server(server_config)
    server_task = asyncio.create_task(server.serve())

    # Wait for server to start
    await asyncio.sleep(2)

    print("\n" + "=" * 60)
    print("BENCHMARK STARTED")
    print("=" * 60)
    print("Dashboard: http://localhost:3000")
    print("API: http://localhost:8000")
    print("-" * 60)
    print("Running simulation at 1440x speed (24 hours in 1 minute)")
    print("-" * 60)

    # Start engines
    await simulation_engine.start()
    await trading_engine.start()

    # Wait for simulation to complete
    while simulation_engine.is_running:
        await asyncio.sleep(1)

        # Print progress
        metrics = simulation_engine.get_metrics()
        print(f"\rProgress: Trades: {metrics['trades_executed']} | " f"P&L: ${metrics['total_pnl']:.2f} | " f"Win Rate: {metrics['win_rate']:.1f}%", end="")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    # Get final metrics
    final_metrics = simulation_engine.get_metrics()

    print("\nFINAL RESULTS:")
    print("-" * 60)
    print(f"Total Trades: {final_metrics['trades_executed']}")
    print(f"Total P&L: ${final_metrics['total_pnl']:.2f}")
    print(f"Win Rate: {final_metrics['win_rate']:.1f}%")
    print(f"Sharpe Ratio: {final_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {final_metrics['max_drawdown']:.1f}%")
    print(f"Signals Generated: {final_metrics['signals_generated']}")
    print(f"Predictions Made: {final_metrics['predictions_made']}")
    print("-" * 60)

    # Stop everything
    await trading_engine.stop()
    server.should_exit = True
    await server_task

    print("\nBenchmark test completed successfully!")
    print("Check the dashboard for detailed results.")

    return final_metrics


def main():
    """Main entry point"""
    try:
        # Run the benchmark
        metrics = asyncio.run(run_benchmark())

        # Determine success
        if metrics["total_pnl"] > 0 and metrics["win_rate"] > 50:
            print("\n[SUCCESS] Benchmark passed! System is profitable.")
            return 0
        else:
            print("\n[WARNING] Benchmark complete but needs optimization.")
            return 1

    except KeyboardInterrupt:
        print("\n\nBenchmark interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n[ERROR] Benchmark failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
