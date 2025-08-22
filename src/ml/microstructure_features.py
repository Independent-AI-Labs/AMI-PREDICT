#!/usr/bin/env python
"""
Microstructure feature engineering for high-frequency scalping.
Calculates order flow, VWAP, tick momentum, and other HFT indicators.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from collections import deque
import talib
from numba import jit
import torch

class MicrostructureFeatureEngine:
    """
    Extract microstructure features for scalping strategies.
    Optimized for real-time calculation with minimal latency.
    """
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        """
        Initialize feature engine.
        
        Args:
            lookback_periods: Dict of feature -> lookback period
        """
        self.lookback_periods = lookback_periods or {
            'tick_momentum': 10,
            'order_flow': 20,
            'vwap': 100,
            'volume_profile': 500,
            'microstructure_noise': 50
        }
        
        # Buffers for streaming calculation
        self.price_buffer = deque(maxlen=max(self.lookback_periods.values()))
        self.volume_buffer = deque(maxlen=max(self.lookback_periods.values()))
        self.bid_buffer = deque(maxlen=max(self.lookback_periods.values()))
        self.ask_buffer = deque(maxlen=max(self.lookback_periods.values()))
        
    @staticmethod
    @jit(nopython=True)
    def _calculate_order_flow_imbalance(bid_volumes: np.ndarray, 
                                       ask_volumes: np.ndarray) -> float:
        """
        Calculate order flow imbalance (OFI).
        
        OFI = (Bid Volume - Ask Volume) / (Bid Volume + Ask Volume)
        Range: [-1, 1] where positive indicates buying pressure
        """
        total_volume = bid_volumes.sum() + ask_volumes.sum()
        if total_volume == 0:
            return 0.0
        return (bid_volumes.sum() - ask_volumes.sum()) / total_volume
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_vwap(prices: np.ndarray, volumes: np.ndarray) -> float:
        """Calculate Volume Weighted Average Price."""
        if volumes.sum() == 0:
            return prices.mean()
        return np.sum(prices * volumes) / volumes.sum()
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_tick_momentum(prices: np.ndarray, window: int = 10) -> float:
        """
        Calculate tick-by-tick momentum.
        
        Measures the ratio of up-ticks to down-ticks in the window.
        """
        if len(prices) < 2:
            return 0.0
        
        price_changes = np.diff(prices[-window:])
        up_ticks = np.sum(price_changes > 0)
        down_ticks = np.sum(price_changes < 0)
        
        if up_ticks + down_ticks == 0:
            return 0.0
        
        return (up_ticks - down_ticks) / (up_ticks + down_ticks)
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_spread_ratio(bids: np.ndarray, asks: np.ndarray) -> float:
        """
        Calculate bid-ask spread ratio.
        
        Spread Ratio = (Ask - Bid) / Mid-price
        Lower values indicate tighter spreads (better liquidity)
        """
        mid_prices = (bids + asks) / 2
        spreads = asks - bids
        
        if mid_prices.mean() == 0:
            return 0.0
        
        return spreads.mean() / mid_prices.mean()
    
    @staticmethod
    @jit(nopython=True)
    def _calculate_trade_size_clustering(volumes: np.ndarray, 
                                        n_clusters: int = 3) -> np.ndarray:
        """
        Calculate trade size clustering.
        
        Groups trades into small/medium/large and returns distribution.
        """
        if len(volumes) == 0:
            return np.zeros(n_clusters)
        
        # Simple percentile-based clustering
        percentiles = np.percentile(volumes, [33, 67])
        
        small_trades = np.sum(volumes < percentiles[0])
        medium_trades = np.sum((volumes >= percentiles[0]) & (volumes < percentiles[1]))
        large_trades = np.sum(volumes >= percentiles[1])
        
        total = len(volumes)
        if total == 0:
            return np.zeros(n_clusters)
        
        return np.array([small_trades/total, medium_trades/total, large_trades/total])
    
    def calculate_microstructure_features(self, 
                                         ohlcv_data: pd.DataFrame,
                                         order_book_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Calculate all microstructure features.
        
        Args:
            ohlcv_data: DataFrame with columns [timestamp, open, high, low, close, volume]
            order_book_data: Optional DataFrame with [timestamp, bid, ask, bid_vol, ask_vol]
            
        Returns:
            DataFrame with microstructure features
        """
        features = pd.DataFrame(index=ohlcv_data.index)
        
        # Price and volume arrays
        prices = ohlcv_data['close'].values
        volumes = ohlcv_data['volume'].values
        
        # 1. VWAP and deviation
        features['vwap'] = pd.Series(prices).rolling(
            self.lookback_periods['vwap']).apply(
            lambda x: self._calculate_vwap(x.values, 
                                          volumes[-len(x):])
        )
        features['vwap_deviation'] = (prices - features['vwap']) / features['vwap']
        
        # 2. Tick momentum
        features['tick_momentum'] = pd.Series(prices).rolling(
            self.lookback_periods['tick_momentum']).apply(
            lambda x: self._calculate_tick_momentum(x.values)
        )
        
        # 3. Volume profile features
        features['volume_mean'] = pd.Series(volumes).rolling(
            self.lookback_periods['volume_profile']).mean()
        features['volume_std'] = pd.Series(volumes).rolling(
            self.lookback_periods['volume_profile']).std()
        features['volume_zscore'] = (volumes - features['volume_mean']) / features['volume_std']
        
        # 4. Price momentum at different scales
        for period in [5, 10, 20, 50]:
            features[f'return_{period}'] = pd.Series(prices).pct_change(period)
            features[f'volatility_{period}'] = pd.Series(prices).pct_change().rolling(period).std()
        
        # 5. Microstructure noise ratio
        # Ratio of high-frequency to low-frequency volatility
        features['noise_ratio'] = features['volatility_5'] / features['volatility_50']
        
        # 6. Order book features (if available)
        if order_book_data is not None and len(order_book_data) > 0 and 'bid' in order_book_data.columns:
            bids = order_book_data['bid'].values
            asks = order_book_data['ask'].values
            bid_vols = order_book_data.get('bid_vol', pd.Series(np.ones_like(bids))).values
            ask_vols = order_book_data.get('ask_vol', pd.Series(np.ones_like(asks))).values
            
            # Order flow imbalance
            features['order_flow_imbalance'] = pd.Series(bid_vols).rolling(
                self.lookback_periods['order_flow']).apply(
                lambda x: self._calculate_order_flow_imbalance(
                    x.values, ask_vols[-len(x):]
                )
            )
            
            # Spread features
            features['spread'] = asks - bids
            features['spread_ratio'] = features['spread'] / ((asks + bids) / 2)
            features['spread_ma'] = features['spread'].rolling(20).mean()
            
            # Mid-price
            features['mid_price'] = (bids + asks) / 2
            features['mid_price_return'] = features['mid_price'].pct_change()
            
            # Book pressure
            features['book_pressure'] = bid_vols / (bid_vols + ask_vols)
            
        # 7. Technical indicators for scalping
        features['rsi_14'] = talib.RSI(prices, timeperiod=14)
        features['rsi_5'] = talib.RSI(prices, timeperiod=5)
        
        # MACD for momentum
        macd, signal, hist = talib.MACD(prices, fastperiod=12, slowperiod=26, signalperiod=9)
        features['macd'] = macd
        features['macd_signal'] = signal
        features['macd_hist'] = hist
        
        # Bollinger Bands for volatility
        upper, middle, lower = talib.BBANDS(prices, timeperiod=20)
        features['bb_upper'] = upper
        features['bb_lower'] = lower
        features['bb_width'] = (upper - lower) / middle
        features['bb_position'] = (prices - lower) / (upper - lower)
        
        # ATR for volatility
        features['atr_14'] = talib.ATR(ohlcv_data['high'].values, 
                                       ohlcv_data['low'].values, 
                                       prices, timeperiod=14)
        
        # 8. Support/Resistance levels
        features['resistance_1'] = pd.Series(ohlcv_data['high']).rolling(20).max()
        features['support_1'] = pd.Series(ohlcv_data['low']).rolling(20).min()
        features['price_to_resistance'] = (prices - features['resistance_1']) / prices
        features['price_to_support'] = (prices - features['support_1']) / prices
        
        # 9. Volume indicators
        features['obv'] = talib.OBV(prices, volumes)
        features['ad'] = talib.AD(ohlcv_data['high'].values,
                                 ohlcv_data['low'].values,
                                 prices, volumes)
        
        # 10. Efficiency ratio (trending vs ranging)
        def efficiency_ratio(prices, period=10):
            price_change = abs(prices.iloc[-1] - prices.iloc[0])
            path_sum = (prices.diff().abs()).sum()
            if path_sum == 0:
                return 0
            return price_change / path_sum
        
        features['efficiency_ratio'] = pd.Series(prices).rolling(10).apply(
            efficiency_ratio, raw=False
        )
        
        # Fill NaN values
        features = features.ffill().fillna(0)
        
        return features
    
    def calculate_regime_features(self, ohlcv_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate market regime features.
        
        Args:
            ohlcv_data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with regime features
        """
        features = pd.DataFrame(index=ohlcv_data.index)
        prices = ohlcv_data['close'].values
        returns = pd.Series(prices).pct_change()
        
        # 1. Volatility regime (GARCH-like)
        features['realized_vol'] = returns.rolling(20).std() * np.sqrt(1440)  # Annualized
        features['vol_of_vol'] = features['realized_vol'].rolling(50).std()
        
        # 2. Trend strength
        # ADX for trend strength
        features['adx'] = talib.ADX(ohlcv_data['high'].values,
                                    ohlcv_data['low'].values,
                                    prices, timeperiod=14)
        
        # Linear regression slope
        def trend_strength(prices, period=20):
            if len(prices) < period:
                return 0
            x = np.arange(period)
            y = prices[-period:]
            slope = np.polyfit(x, y, 1)[0]
            return slope / prices[-1]  # Normalize by current price
        
        features['trend_strength'] = pd.Series(prices).rolling(20).apply(
            lambda x: trend_strength(x.values), raw=False
        )
        
        # 3. Mean reversion indicator
        features['hurst_exponent'] = self._calculate_hurst_exponent(returns)
        
        # 4. Regime classification
        # Simple regime: trending bull/bear, ranging
        features['regime_trend'] = np.where(features['trend_strength'] > 0.001, 1,
                                           np.where(features['trend_strength'] < -0.001, -1, 0))
        features['regime_volatility'] = np.where(features['realized_vol'] > features['realized_vol'].rolling(100).mean(), 1, 0)
        
        return features
    
    def _calculate_hurst_exponent(self, returns: pd.Series, 
                                 lags: List[int] = None) -> pd.Series:
        """
        Calculate rolling Hurst exponent.
        
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        if lags is None:
            lags = [2, 4, 8, 16, 32]
        
        def hurst(ts):
            if len(ts) < max(lags):
                return 0.5
            
            tau = []
            lagvec = []
            
            for lag in lags:
                pp = np.subtract(ts[lag:], ts[:-lag])
                lagvec.append(lag)
                tau.append(np.std(pp))
            
            # Linear fit to log-log plot
            with np.errstate(all='ignore'):
                slope = np.polyfit(np.log(lagvec), np.log(tau), 1)[0]
            
            return slope
        
        return returns.rolling(100).apply(hurst, raw=False).fillna(0.5)
    
    def create_feature_tensor(self, features: pd.DataFrame, 
                            sequence_length: int = 100) -> torch.Tensor:
        """
        Convert features to tensor for model input.
        
        Args:
            features: DataFrame with features
            sequence_length: Length of sequences for model
            
        Returns:
            Tensor of shape (n_samples, sequence_length, n_features)
        """
        # Normalize features
        normalized = (features - features.mean()) / (features.std() + 1e-8)
        
        # Create sequences
        n_features = normalized.shape[1]
        n_samples = len(normalized) - sequence_length + 1
        
        sequences = np.zeros((n_samples, sequence_length, n_features))
        
        for i in range(n_samples):
            sequences[i] = normalized.iloc[i:i+sequence_length].values
        
        return torch.FloatTensor(sequences)
    
    def update_buffers(self, price: float, volume: float, 
                      bid: Optional[float] = None, ask: Optional[float] = None):
        """
        Update internal buffers for streaming calculation.
        
        Args:
            price: Current price
            volume: Current volume
            bid: Current bid price
            ask: Current ask price
        """
        self.price_buffer.append(price)
        self.volume_buffer.append(volume)
        
        if bid is not None:
            self.bid_buffer.append(bid)
        if ask is not None:
            self.ask_buffer.append(ask)
    
    def get_realtime_features(self) -> Dict[str, float]:
        """
        Calculate features from current buffers for real-time prediction.
        
        Returns:
            Dict of feature_name -> value
        """
        if len(self.price_buffer) < 2:
            return {}
        
        features = {}
        
        prices = np.array(self.price_buffer)
        volumes = np.array(self.volume_buffer)
        
        # Core features
        features['vwap'] = self._calculate_vwap(prices, volumes)
        features['vwap_deviation'] = (prices[-1] - features['vwap']) / features['vwap']
        features['tick_momentum'] = self._calculate_tick_momentum(prices)
        features['volume_zscore'] = (volumes[-1] - volumes.mean()) / (volumes.std() + 1e-8)
        
        # Returns
        if len(prices) >= 5:
            features['return_5'] = (prices[-1] - prices[-5]) / prices[-5]
        if len(prices) >= 10:
            features['return_10'] = (prices[-1] - prices[-10]) / prices[-10]
        
        # Order book features
        if len(self.bid_buffer) > 0 and len(self.ask_buffer) > 0:
            bids = np.array(self.bid_buffer)
            asks = np.array(self.ask_buffer)
            
            features['spread'] = asks[-1] - bids[-1]
            features['mid_price'] = (asks[-1] + bids[-1]) / 2
            features['spread_ratio'] = features['spread'] / features['mid_price']
        
        return features


def test_feature_engine():
    """Test microstructure feature calculation."""
    print("Testing Microstructure Feature Engine...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic OHLCV data
    prices = 50000 + np.cumsum(np.random.randn(n_samples) * 10)
    
    ohlcv_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='1min'),
        'open': prices + np.random.randn(n_samples) * 5,
        'high': prices + abs(np.random.randn(n_samples)) * 10,
        'low': prices - abs(np.random.randn(n_samples)) * 10,
        'close': prices,
        'volume': abs(np.random.randn(n_samples)) * 1000000
    })
    
    # Generate synthetic order book data
    spread = abs(np.random.randn(n_samples)) * 5
    order_book_data = pd.DataFrame({
        'timestamp': ohlcv_data['timestamp'],
        'bid': prices - spread/2,
        'ask': prices + spread/2,
        'bid_vol': abs(np.random.randn(n_samples)) * 100000,
        'ask_vol': abs(np.random.randn(n_samples)) * 100000
    })
    
    # Initialize engine
    engine = MicrostructureFeatureEngine()
    
    # Calculate features
    print("\nCalculating microstructure features...")
    features = engine.calculate_microstructure_features(ohlcv_data, order_book_data)
    print(f"Generated {features.shape[1]} features for {features.shape[0]} samples")
    
    # Calculate regime features
    print("\nCalculating regime features...")
    regime_features = engine.calculate_regime_features(ohlcv_data)
    print(f"Generated {regime_features.shape[1]} regime features")
    
    # Combine all features
    all_features = pd.concat([features, regime_features], axis=1)
    
    # Display sample features
    print("\nSample features (last 5 rows):")
    print(all_features.tail())
    
    # Feature statistics
    print("\nFeature statistics:")
    print(all_features.describe().T[['mean', 'std', 'min', 'max']])
    
    # Test real-time feature calculation
    print("\nTesting real-time feature calculation...")
    for i in range(100):
        engine.update_buffers(
            price=prices[i],
            volume=ohlcv_data['volume'].iloc[i],
            bid=order_book_data['bid'].iloc[i],
            ask=order_book_data['ask'].iloc[i]
        )
    
    realtime_features = engine.get_realtime_features()
    print(f"Real-time features: {realtime_features}")
    
    # Test tensor creation
    print("\nCreating feature tensor for model input...")
    tensor = engine.create_feature_tensor(all_features.iloc[:500], sequence_length=50)
    print(f"Tensor shape: {tensor.shape}")
    print(f"Tensor dtype: {tensor.dtype}")
    
    print("\n✅ Microstructure feature engine working correctly!")


if __name__ == "__main__":
    test_feature_engine()