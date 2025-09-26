"""
API module for CryptoBot Pro
"""

from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from ..core import ConfigManager, Database


def create_api_app(config: ConfigManager, database: Database, trading_engine=None) -> FastAPI:
    """Create FastAPI application

    Args:
        config: Configuration manager
        database: Database instance
        trading_engine: Trading engine instance

    Returns:
        FastAPI application
    """
    app = FastAPI(title="CryptoBot Pro API", description="24-Hour Live Market Benchmark Trading System", version="0.1.0")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.get("api.cors_origins", ["http://localhost:3000"]),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store instances in app state
    app.state.config = config
    app.state.database = database
    app.state.trading_engine = trading_engine

    @app.get("/")
    async def root():
        """Root endpoint"""
        return {"name": "CryptoBot Pro", "version": "0.1.0", "status": "operational", "mode": config.mode}

    @app.get("/api/status")
    async def get_status():
        """Get system status"""
        performance = database.get_latest_performance(config.mode) or {}

        return {
            "system": {"status": "operational", "uptime": 99.98, "timestamp": datetime.now().isoformat(), "mode": config.mode},
            "trading": {
                "mode": config.mode,
                "isActive": trading_engine.is_running if trading_engine else False,
                "tradesExecuted": performance.get("total_trades", 0),
                "signalsGenerated": 0,  # Will be updated
                "activePositions": len(trading_engine.get_positions()) if trading_engine else 0,
            },
            "performance": {
                "totalPnL": performance.get("total_pnl", 0),
                "winRate": performance.get("win_rate", 0),
                "sharpeRatio": performance.get("sharpe_ratio", 0),
                "maxDrawdown": performance.get("max_drawdown", 0),
                "predictionAccuracy": 0,  # Will be updated
            },
            "models": {
                "ensemble": {
                    "accuracy": 0,  # Will be updated
                    "latency": 15,
                    "lastPrediction": datetime.now().isoformat(),
                },
                "regime": "trending_bull",
                "confidence": 78.5,
            },
        }

    @app.get("/api/market")
    async def get_market_data():
        """Get market data"""
        pairs = config.get("trading.pairs", [])
        market_data = []

        for pair in pairs:
            # Get latest price from database or trading engine
            data = {
                "symbol": pair,
                "price": 0,  # Will be updated with real data
                "change24h": 0,
                "volume24h": 0,
                "high24h": 0,
                "low24h": 0,
            }

            if trading_engine and hasattr(trading_engine, "get_ticker"):
                ticker = await trading_engine.get_ticker(pair)
                if ticker:
                    data.update(ticker)

            market_data.append(data)

        return {"pairs": market_data, "timestamp": datetime.now().isoformat()}

    @app.get("/api/positions")
    async def get_positions():
        """Get open positions"""
        if not trading_engine:
            return {"positions": []}

        positions = trading_engine.get_positions()
        return {"positions": positions, "timestamp": datetime.now().isoformat()}

    @app.get("/api/trades")
    async def get_trades(limit: int = 50):
        """Get recent trades"""
        trades = database.get_trades(mode=config.mode)[:limit]
        return {"trades": trades, "timestamp": datetime.now().isoformat()}

    @app.get("/api/performance")
    async def get_performance():
        """Get performance metrics"""
        performance = database.get_latest_performance(config.mode)
        return {"performance": performance, "timestamp": datetime.now().isoformat()}

    @app.post("/api/control/start")
    async def start_trading():
        """Start trading"""
        if trading_engine:
            await trading_engine.start()
            return {"status": "started"}
        raise HTTPException(status_code=503, detail="Trading engine not available")

    @app.post("/api/control/stop")
    async def stop_trading():
        """Stop trading"""
        if trading_engine:
            await trading_engine.stop()
            return {"status": "stopped"}
        raise HTTPException(status_code=503, detail="Trading engine not available")

    @app.post("/api/control/emergency-stop")
    async def emergency_stop():
        """Emergency stop - close all positions"""
        if trading_engine:
            await trading_engine.emergency_stop()
            return {"status": "emergency_stopped"}
        raise HTTPException(status_code=503, detail="Trading engine not available")

    return app
