"""
Database management system
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd
from sqlalchemy import JSON, Column, DateTime, Float, Index, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

Base = declarative_base()


class MarketData(Base):
    """Market data table"""

    __tablename__ = "market_data"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=False)
    timeframe = Column(String(10), nullable=False)

    __table_args__ = (Index("ix_market_data_symbol_timestamp", "symbol", "timestamp"),)


class Trade(Base):
    """Trade history table"""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    side = Column(String(10), nullable=False)  # BUY/SELL
    entry_price = Column(Float, nullable=False)
    exit_price = Column(Float)
    size = Column(Float, nullable=False)
    pnl = Column(Float)
    pnl_percent = Column(Float)
    strategy = Column(String(50))
    mode = Column(String(20))  # simulation/paper/live
    meta_data = Column(JSON)


class Prediction(Base):
    """Model predictions table"""

    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(20), nullable=False)
    model = Column(String(50), nullable=False)
    prediction = Column(Float, nullable=False)
    confidence = Column(Float)
    actual = Column(Float)
    features = Column(JSON)

    __table_args__ = (Index("ix_predictions_symbol_timestamp", "symbol", "timestamp"),)


class Performance(Base):
    """Performance metrics table"""

    __tablename__ = "performance"

    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    mode = Column(String(20), nullable=False)
    total_pnl = Column(Float)
    win_rate = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    total_trades = Column(Integer)
    winning_trades = Column(Integer)
    losing_trades = Column(Integer)
    meta_data = Column(JSON)


class Database:
    """Database management class"""

    def __init__(self, config: dict[str, Any]):
        """Initialize database

        Args:
            config: Database configuration
        """
        self.config = config
        self.engine = None
        self.session_factory = None
        self._init_database()

    def _init_database(self):
        """Initialize database connection and tables"""
        db_type = self.config.get("type", "sqlite")

        if db_type == "sqlite":
            db_path = Path(self.config.get("path", "data/cryptobot.db"))
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Create engine with proper settings for concurrent access
            self.engine = create_engine(
                f"sqlite:///{db_path}", connect_args={"check_same_thread": False}, poolclass=StaticPool, echo=self.config.get("echo", False)
            )
        else:
            raise ValueError(f"Unsupported database type: {db_type}")

        # Create tables
        Base.metadata.create_all(self.engine)

        # Create session factory
        self.session_factory = sessionmaker(bind=self.engine)

    def get_session(self) -> Session:
        """Get database session

        Returns:
            Database session
        """
        return self.session_factory()

    def save_market_data(self, data: pd.DataFrame, symbol: str, timeframe: str):
        """Save market data to database

        Args:
            data: Market data DataFrame
            symbol: Trading pair symbol
            timeframe: Timeframe
        """
        session = self.get_session()
        try:
            for _, row in data.iterrows():
                market_data = MarketData(
                    timestamp=row["timestamp"],
                    symbol=symbol,
                    open=row["open"],
                    high=row["high"],
                    low=row["low"],
                    close=row["close"],
                    volume=row["volume"],
                    timeframe=timeframe,
                )
                session.add(market_data)
            session.commit()
        finally:
            session.close()

    def get_market_data(
        self, symbol: str, timeframe: str, start: Optional[datetime] = None, end: Optional[datetime] = None, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Get market data from database

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            start: Start timestamp
            end: End timestamp

        Returns:
            Market data DataFrame
        """
        session = self.get_session()
        try:
            query = session.query(MarketData).filter(MarketData.symbol == symbol, MarketData.timeframe == timeframe)

            if start:
                query = query.filter(MarketData.timestamp >= start)
            if end:
                query = query.filter(MarketData.timestamp <= end)

            query = query.order_by(MarketData.timestamp.desc())

            if limit:
                query = query.limit(limit)

            data = query.all()
            # Reverse to get chronological order
            data = list(reversed(data))

            if not data:
                return pd.DataFrame()

            return pd.DataFrame([{"timestamp": d.timestamp, "open": d.open, "high": d.high, "low": d.low, "close": d.close, "volume": d.volume} for d in data])
        finally:
            session.close()

    def save_trade(self, trade_data: dict[str, Any]):
        """Save trade to database

        Args:
            trade_data: Trade information
        """
        session = self.get_session()
        try:
            # Convert metadata to meta_data if present
            if "metadata" in trade_data:
                trade_data["meta_data"] = trade_data.pop("metadata")
            trade = Trade(**trade_data)
            session.add(trade)
            session.commit()
        finally:
            session.close()

    def get_trades(self, mode: Optional[str] = None, start: Optional[datetime] = None) -> list[dict[str, Any]]:
        """Get trades from database

        Args:
            mode: Trading mode filter
            start: Start timestamp

        Returns:
            List of trades
        """
        session = self.get_session()
        try:
            query = session.query(Trade)

            if mode:
                query = query.filter(Trade.mode == mode)
            if start:
                query = query.filter(Trade.timestamp >= start)

            trades = query.order_by(Trade.timestamp.desc()).all()

            return [
                {
                    "id": t.id,
                    "timestamp": t.timestamp,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "size": t.size,
                    "pnl": t.pnl,
                    "pnl_percent": t.pnl_percent,
                    "strategy": t.strategy,
                    "mode": t.mode,
                    "metadata": t.meta_data,
                }
                for t in trades
            ]
        finally:
            session.close()

    def save_prediction(self, prediction_data: dict[str, Any]):
        """Save model prediction to database

        Args:
            prediction_data: Prediction information
        """
        session = self.get_session()
        try:
            prediction = Prediction(**prediction_data)
            session.add(prediction)
            session.commit()
        finally:
            session.close()

    def save_performance(self, performance_data: dict[str, Any]):
        """Save performance metrics to database

        Args:
            performance_data: Performance metrics
        """
        session = self.get_session()
        try:
            # Convert metadata to meta_data if present
            if "metadata" in performance_data:
                performance_data["meta_data"] = performance_data.pop("metadata")
            performance = Performance(**performance_data)
            session.add(performance)
            session.commit()
        finally:
            session.close()

    def get_latest_performance(self, mode: str) -> Optional[dict[str, Any]]:
        """Get latest performance metrics

        Args:
            mode: Trading mode

        Returns:
            Performance metrics
        """
        session = self.get_session()
        try:
            perf = session.query(Performance).filter(Performance.mode == mode).order_by(Performance.timestamp.desc()).first()

            if not perf:
                return None

            return {
                "timestamp": perf.timestamp,
                "total_pnl": perf.total_pnl,
                "win_rate": perf.win_rate,
                "sharpe_ratio": perf.sharpe_ratio,
                "max_drawdown": perf.max_drawdown,
                "total_trades": perf.total_trades,
                "winning_trades": perf.winning_trades,
                "losing_trades": perf.losing_trades,
                "metadata": perf.meta_data,
            }
        finally:
            session.close()

    def get_predictions(self, start: Optional[datetime] = None) -> list[dict[str, Any]]:
        """Get predictions from database

        Args:
            start: Start timestamp

        Returns:
            List of predictions
        """
        session = self.get_session()
        try:
            query = session.query(Prediction)

            if start:
                query = query.filter(Prediction.timestamp >= start)

            predictions = query.all()

            return [
                {
                    "id": p.id,
                    "timestamp": p.timestamp,
                    "symbol": p.symbol,
                    "predicted_return": p.predicted_return,
                    "confidence": p.confidence,
                    "regime": p.regime,
                    "models": p.models,
                    "features": p.features,
                    "actual_return": p.actual_return,
                    "metadata": p.meta_data,
                }
                for p in predictions
            ]
        finally:
            session.close()
