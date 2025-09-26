"""Base class for data providers"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional

import pandas as pd


class BaseDataProvider(ABC):
    """Abstract base class for all data providers"""

    def __init__(self, config: dict[str, Any]):
        """Initialize data provider

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.name = self.__class__.__name__

    @abstractmethod
    async def fetch_ohlcv(
        self, symbol: str, timeframe: str, start: Optional[datetime] = None, end: Optional[datetime] = None, limit: Optional[int] = None
    ) -> pd.DataFrame:
        """Fetch OHLCV data

        Args:
            symbol: Trading pair symbol (e.g., 'BTC/USDT')
            timeframe: Timeframe (e.g., '1m', '5m', '1h', '1d')
            start: Start datetime
            end: End datetime
            limit: Maximum number of candles

        Returns:
            DataFrame with OHLCV data
        """

    @abstractmethod
    async def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Fetch current ticker data

        Args:
            symbol: Trading pair symbol

        Returns:
            Ticker data dictionary
        """

    @abstractmethod
    async def fetch_order_book(self, symbol: str, limit: int = 100) -> dict[str, Any]:
        """Fetch order book

        Args:
            symbol: Trading pair symbol
            limit: Depth of order book

        Returns:
            Order book data
        """

    @abstractmethod
    async def stream_trades(self, symbol: str, callback: callable):
        """Stream real-time trades

        Args:
            symbol: Trading pair symbol
            callback: Function to call with trade data
        """

    @abstractmethod
    async def stream_ohlcv(self, symbol: str, timeframe: str, callback: callable):
        """Stream real-time OHLCV data

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe
            callback: Function to call with OHLCV data
        """

    @abstractmethod
    def get_supported_symbols(self) -> list[str]:
        """Get list of supported trading pairs

        Returns:
            List of symbol strings
        """

    @abstractmethod
    def get_supported_timeframes(self) -> list[str]:
        """Get list of supported timeframes

        Returns:
            List of timeframe strings
        """

    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol is supported

        Args:
            symbol: Trading pair symbol

        Returns:
            True if supported
        """
        return symbol in self.get_supported_symbols()

    def validate_timeframe(self, timeframe: str) -> bool:
        """Validate if timeframe is supported

        Args:
            timeframe: Timeframe string

        Returns:
            True if supported
        """
        return timeframe in self.get_supported_timeframes()
