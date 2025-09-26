"""
Simulated data feed for testing
"""

import asyncio
import random
from datetime import datetime
from typing import Any

from loguru import logger

from ..core import ConfigManager


class SimulatedDataFeed:
    """Simulates real-time data feed events"""

    def __init__(self, config: ConfigManager):
        """Initialize data feed

        Args:
            config: Configuration manager
        """
        self.config = config
        self.log = logger.bind(name=__name__)

        self.pairs = config.get("trading.pairs", [])

        # Event generation probabilities
        self.signal_probability = 0.05
        self.prediction_probability = 0.1
        self.news_probability = 0.01

        self.event_counter = 0

    async def generate_events(self) -> list[dict[str, Any]]:
        """Generate random events

        Returns:
            List of events
        """
        events = []

        for pair in self.pairs:
            # Generate trading signal
            if random.random() < self.signal_probability:
                events.append(self._generate_signal(pair))

            # Generate prediction
            if random.random() < self.prediction_probability:
                events.append(self._generate_prediction(pair))

            # Generate news event
            if random.random() < self.news_probability:
                events.append(self._generate_news(pair))

        return events

    def _generate_signal(self, pair: str) -> dict[str, Any]:
        """Generate trading signal event

        Args:
            pair: Trading pair

        Returns:
            Signal event
        """
        self.event_counter += 1

        signal_types = ["BUY", "SELL", "HOLD"]
        strengths = ["STRONG", "MEDIUM", "WEAK"]
        strategies = ["trend_following", "mean_reversion", "momentum", "arbitrage"]

        return {
            "id": f"SIG_{self.event_counter}",
            "type": "signal",
            "timestamp": datetime.now(),
            "pair": pair,
            "signal": random.choice(signal_types),
            "strength": random.choice(strengths),
            "confidence": random.uniform(0.5, 0.95),
            "strategy": random.choice(strategies),
            "metadata": {"rsi": random.uniform(20, 80), "macd": random.uniform(-0.5, 0.5), "volume_ratio": random.uniform(0.5, 2.0)},
        }

    def _generate_prediction(self, pair: str) -> dict[str, Any]:
        """Generate price prediction event

        Args:
            pair: Trading pair

        Returns:
            Prediction event
        """
        self.event_counter += 1

        models = ["lightgbm", "catboost", "lstm", "ensemble"]

        return {
            "id": f"PRED_{self.event_counter}",
            "type": "prediction",
            "timestamp": datetime.now(),
            "pair": pair,
            "model": random.choice(models),
            "prediction": random.uniform(-0.05, 0.05),  # Predicted return
            "confidence": random.uniform(0.4, 0.9),
            "timeframe": "5m",
            "features_used": random.randint(50, 200),
            "metadata": {"model_version": "1.0.0", "training_date": "2024-01-01", "accuracy_score": random.uniform(0.5, 0.7)},
        }

    def _generate_news(self, pair: str) -> dict[str, Any]:
        """Generate news event

        Args:
            pair: Trading pair

        Returns:
            News event
        """
        self.event_counter += 1

        sentiments = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
        impacts = ["HIGH", "MEDIUM", "LOW"]

        headlines = [
            f"Major institution announces {pair.split('/')[0]} investment",
            f"Regulatory update affects {pair.split('/')[0]} trading",
            f"Technical breakthrough in {pair.split('/')[0]} network",
            f"Market analysis: {pair.split('/')[0]} shows strong momentum",
            f"Breaking: {pair.split('/')[0]} partnership announcement",
        ]

        return {
            "id": f"NEWS_{self.event_counter}",
            "type": "news",
            "timestamp": datetime.now(),
            "pair": pair,
            "headline": random.choice(headlines),
            "sentiment": random.choice(sentiments),
            "impact": random.choice(impacts),
            "confidence": random.uniform(0.6, 0.95),
            "source": random.choice(["Reuters", "Bloomberg", "CoinDesk", "Twitter"]),
            "metadata": {"mentions": random.randint(1, 100), "reach": random.randint(1000, 100000)},
        }

    async def subscribe(self, callback):
        """Subscribe to data feed events

        Args:
            callback: Callback function for events
        """
        self.log.info("Subscribed to data feed")

        while True:
            events = await self.generate_events()
            for event in events:
                await callback(event)

            await asyncio.sleep(1)  # Generate events every second
