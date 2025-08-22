"""Market Regime Detection using Hidden Markov Models"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum


class MarketRegime(Enum):
    """Market regime states"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"


class RegimeDetector:
    """Detect market regimes using statistical methods"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_regime = MarketRegime.SIDEWAYS
        self.regime_history = []
        self.transition_matrix = self._initialize_transition_matrix()
        self.regime_features = {}
        self.confidence = 0.5
        
    def _initialize_transition_matrix(self) -> Dict[str, Dict[str, float]]:
        """Initialize regime transition probabilities"""
        return {
            MarketRegime.TRENDING_BULL.value: {
                MarketRegime.TRENDING_BULL.value: 0.7,
                MarketRegime.SIDEWAYS.value: 0.15,
                MarketRegime.TRENDING_BEAR.value: 0.1,
                MarketRegime.HIGH_VOLATILITY.value: 0.05
            },
            MarketRegime.TRENDING_BEAR.value: {
                MarketRegime.TRENDING_BEAR.value: 0.7,
                MarketRegime.SIDEWAYS.value: 0.15,
                MarketRegime.TRENDING_BULL.value: 0.1,
                MarketRegime.HIGH_VOLATILITY.value: 0.05
            },
            MarketRegime.SIDEWAYS.value: {
                MarketRegime.SIDEWAYS.value: 0.6,
                MarketRegime.TRENDING_BULL.value: 0.15,
                MarketRegime.TRENDING_BEAR.value: 0.15,
                MarketRegime.HIGH_VOLATILITY.value: 0.1
            },
            MarketRegime.HIGH_VOLATILITY.value: {
                MarketRegime.HIGH_VOLATILITY.value: 0.5,
                MarketRegime.SIDEWAYS.value: 0.3,
                MarketRegime.TRENDING_BULL.value: 0.1,
                MarketRegime.TRENDING_BEAR.value: 0.1
            }
        }
    
    def detect_regime(self, market_data: pd.DataFrame) -> Tuple[MarketRegime, float]:
        """Detect current market regime"""
        if len(market_data) < 50:
            return MarketRegime.SIDEWAYS, 0.5
        
        # Calculate regime features
        features = self._calculate_regime_features(market_data)
        self.regime_features = features
        
        # Determine regime based on features
        regime = self._classify_regime(features)
        
        # Calculate confidence
        confidence = self._calculate_confidence(features, regime)
        
        # Update history
        self.regime_history.append({
            'timestamp': datetime.now(),
            'regime': regime,
            'confidence': confidence,
            'features': features
        })
        
        # Keep only recent history
        if len(self.regime_history) > 100:
            self.regime_history = self.regime_history[-100:]
        
        self.current_regime = regime
        self.confidence = confidence
        
        return regime, confidence
    
    def _calculate_regime_features(self, market_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate features for regime detection"""
        features = {}
        
        # Price trend features
        returns = market_data['close'].pct_change().dropna()
        
        # Short and long term trends
        sma_20 = market_data['close'].rolling(20).mean()
        sma_50 = market_data['close'].rolling(50).mean()
        
        features['trend_strength'] = float((market_data['close'].iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1])
        features['trend_consistency'] = float((sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1])
        
        # Volatility features
        features['volatility'] = float(returns.rolling(20).std().iloc[-1])
        features['volatility_ratio'] = float(
            returns.rolling(10).std().iloc[-1] / returns.rolling(50).std().iloc[-1]
        ) if returns.rolling(50).std().iloc[-1] > 0 else 1.0
        
        # Market structure
        features['higher_highs'] = self._count_higher_highs(market_data['high'].tail(20))
        features['lower_lows'] = self._count_lower_lows(market_data['low'].tail(20))
        
        # Volume analysis
        volume_sma = market_data['volume'].rolling(20).mean()
        features['volume_trend'] = float(
            (market_data['volume'].iloc[-1] - volume_sma.iloc[-1]) / volume_sma.iloc[-1]
        ) if volume_sma.iloc[-1] > 0 else 0
        
        # Momentum
        features['rsi'] = self._calculate_rsi(market_data['close'])
        features['momentum'] = float((market_data['close'].iloc[-1] - market_data['close'].iloc[-20]) / market_data['close'].iloc[-20])
        
        # Mean reversion
        features['mean_reversion'] = float(
            (market_data['close'].iloc[-1] - sma_20.iloc[-1]) / market_data['close'].iloc[-1]
        )
        
        return features
    
    def _classify_regime(self, features: Dict[str, float]) -> MarketRegime:
        """Classify market regime based on features"""
        
        # High volatility check
        if features['volatility_ratio'] > 1.5 or features['volatility'] > 0.03:
            return MarketRegime.HIGH_VOLATILITY
        
        # Trending bull
        if (features['trend_strength'] > 0.02 and 
            features['trend_consistency'] > 0.01 and
            features['higher_highs'] > features['lower_lows'] and
            features['momentum'] > 0.02):
            return MarketRegime.TRENDING_BULL
        
        # Trending bear
        if (features['trend_strength'] < -0.02 and
            features['trend_consistency'] < -0.01 and
            features['lower_lows'] > features['higher_highs'] and
            features['momentum'] < -0.02):
            return MarketRegime.TRENDING_BEAR
        
        # Default to sideways
        return MarketRegime.SIDEWAYS
    
    def _calculate_confidence(self, features: Dict[str, float], regime: MarketRegime) -> float:
        """Calculate confidence in regime detection"""
        confidence = 0.5
        
        if regime == MarketRegime.TRENDING_BULL:
            # Strong trend indicators increase confidence
            if features['trend_strength'] > 0.05:
                confidence += 0.2
            if features['momentum'] > 0.05:
                confidence += 0.15
            if features['higher_highs'] > 3:
                confidence += 0.1
            if features['rsi'] > 60 and features['rsi'] < 80:
                confidence += 0.05
                
        elif regime == MarketRegime.TRENDING_BEAR:
            # Strong bearish indicators
            if features['trend_strength'] < -0.05:
                confidence += 0.2
            if features['momentum'] < -0.05:
                confidence += 0.15
            if features['lower_lows'] > 3:
                confidence += 0.1
            if features['rsi'] < 40 and features['rsi'] > 20:
                confidence += 0.05
                
        elif regime == MarketRegime.HIGH_VOLATILITY:
            # High volatility confirmation
            if features['volatility_ratio'] > 2.0:
                confidence += 0.3
            if features['volatility'] > 0.05:
                confidence += 0.2
                
        elif regime == MarketRegime.SIDEWAYS:
            # Range-bound indicators
            if abs(features['trend_strength']) < 0.01:
                confidence += 0.2
            if abs(features['mean_reversion']) < 0.02:
                confidence += 0.15
            if features['rsi'] > 40 and features['rsi'] < 60:
                confidence += 0.15
        
        return min(confidence, 1.0)
    
    def _count_higher_highs(self, highs: pd.Series) -> int:
        """Count number of higher highs"""
        count = 0
        for i in range(1, len(highs)):
            if highs.iloc[i] > highs.iloc[i-1]:
                count += 1
        return count
    
    def _count_lower_lows(self, lows: pd.Series) -> int:
        """Count number of lower lows"""
        count = 0
        for i in range(1, len(lows)):
            if lows.iloc[i] < lows.iloc[i-1]:
                count += 1
        return count
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        if loss.iloc[-1] == 0:
            return 100.0
            
        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))
        return float(rsi)
    
    def get_regime_params(self, regime: MarketRegime) -> Dict[str, Any]:
        """Get recommended parameters for given regime"""
        params = {
            MarketRegime.TRENDING_BULL: {
                'position_size_multiplier': 1.2,
                'stop_loss': 0.02,
                'take_profit': 0.05,
                'max_positions': 5,
                'entry_threshold': 0.6,
                'preferred_strategies': ['trend_following', 'momentum']
            },
            MarketRegime.TRENDING_BEAR: {
                'position_size_multiplier': 0.8,
                'stop_loss': 0.015,
                'take_profit': 0.03,
                'max_positions': 3,
                'entry_threshold': 0.7,
                'preferred_strategies': ['mean_reversion', 'short']
            },
            MarketRegime.SIDEWAYS: {
                'position_size_multiplier': 1.0,
                'stop_loss': 0.015,
                'take_profit': 0.025,
                'max_positions': 4,
                'entry_threshold': 0.65,
                'preferred_strategies': ['mean_reversion', 'range_trading']
            },
            MarketRegime.HIGH_VOLATILITY: {
                'position_size_multiplier': 0.5,
                'stop_loss': 0.025,
                'take_profit': 0.04,
                'max_positions': 2,
                'entry_threshold': 0.75,
                'preferred_strategies': ['volatility_breakout', 'options']
            }
        }
        
        return params.get(regime, params[MarketRegime.SIDEWAYS])
    
    def predict_next_regime(self) -> Tuple[MarketRegime, float]:
        """Predict next regime based on transition matrix"""
        if not self.regime_history:
            return MarketRegime.SIDEWAYS, 0.25
        
        current = self.current_regime.value
        transitions = self.transition_matrix[current]
        
        # Find most likely next regime
        max_prob = 0
        next_regime = MarketRegime.SIDEWAYS
        
        for regime_str, prob in transitions.items():
            if prob > max_prob:
                max_prob = prob
                next_regime = MarketRegime(regime_str)
        
        return next_regime, max_prob
    
    def update_transition_matrix(self, actual_transitions: List[Tuple[MarketRegime, MarketRegime]]):
        """Update transition matrix based on observed transitions"""
        # Count transitions
        transition_counts = {}
        for from_regime, to_regime in actual_transitions:
            from_str = from_regime.value
            to_str = to_regime.value
            
            if from_str not in transition_counts:
                transition_counts[from_str] = {}
            if to_str not in transition_counts[from_str]:
                transition_counts[from_str][to_str] = 0
                
            transition_counts[from_str][to_str] += 1
        
        # Update probabilities
        for from_regime in transition_counts:
            total = sum(transition_counts[from_regime].values())
            if total > 0:
                for to_regime in transition_counts[from_regime]:
                    # Blend old and new probabilities
                    old_prob = self.transition_matrix[from_regime].get(to_regime, 0)
                    new_prob = transition_counts[from_regime][to_regime] / total
                    self.transition_matrix[from_regime][to_regime] = 0.7 * old_prob + 0.3 * new_prob
    
    def get_summary(self) -> Dict[str, Any]:
        """Get regime detection summary"""
        return {
            'current_regime': self.current_regime.value,
            'confidence': self.confidence,
            'features': self.regime_features,
            'history_length': len(self.regime_history),
            'regime_params': self.get_regime_params(self.current_regime)
        }