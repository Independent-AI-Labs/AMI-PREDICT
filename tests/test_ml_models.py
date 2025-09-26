"""Tests for ML models"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.ml import CatBoostModel, EnsembleModel, FeatureEngineer, LightGBMModel, LSTMModel, RandomForestModel, RegimeDetector
from src.ml.regime_detector import MarketRegime


class TestFeatureEngineer:
    """Test feature engineering"""

    def test_create_features(self):
        """Test feature creation"""
        # Create sample market data
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")
        market_data = pd.DataFrame(
            {
                "open": np.random.randn(100).cumsum() + 100,
                "high": np.random.randn(100).cumsum() + 101,
                "low": np.random.randn(100).cumsum() + 99,
                "close": np.random.randn(100).cumsum() + 100,
                "volume": np.random.rand(100) * 1000000,
            },
            index=dates,
        )

        engineer = FeatureEngineer()
        features = engineer.create_features(market_data)

        assert len(features) > 0
        assert "returns" in features.columns
        assert "rsi" in features.columns
        assert "macd" in features.columns
        assert "volatility_20" in features.columns
        print("[OK] Feature creation test passed")

    def test_normalize_features(self):
        """Test feature normalization"""
        # Create sample features
        dates = pd.date_range(start="2024-01-01", periods=200, freq="1h")
        features = pd.DataFrame({"feature1": np.random.randn(200) * 10, "feature2": np.random.randn(200) * 100, "hour": np.arange(200) % 24}, index=dates)

        engineer = FeatureEngineer()
        normalized = engineer.normalize_features(features)

        assert len(normalized) == len(features)
        # Check that non-categorical features are normalized
        assert normalized["feature1"].std() < features["feature1"].std()
        # Check that categorical features are preserved
        assert (normalized["hour"] == features["hour"]).all()
        print("[OK] Feature normalization test passed")


class TestMLModels:
    """Test individual ML models"""

    def create_sample_data(self, n_samples=100):
        """Create sample training data"""
        X = pd.DataFrame({f"feature_{i}": np.random.randn(n_samples) for i in range(10)})
        y = pd.Series(np.random.randn(n_samples) * 0.01)
        return X, y

    def test_lightgbm_model(self):
        """Test LightGBM model"""
        X, y = self.create_sample_data()

        model = LightGBMModel({})
        metrics = model.train(X, y)

        assert model.is_trained
        assert "mae" in metrics
        assert "r2" in metrics

        predictions = model.predict(X.head(10))
        assert len(predictions) == 10
        print("[OK] LightGBM model test passed")

    def test_catboost_model(self):
        """Test CatBoost model"""
        X, y = self.create_sample_data()

        model = CatBoostModel({})
        metrics = model.train(X, y)

        assert model.is_trained
        assert "mae" in metrics
        assert "r2" in metrics

        predictions = model.predict(X.head(10))
        assert len(predictions) == 10
        print("[OK] CatBoost model test passed")

    def test_lstm_model(self):
        """Test LSTM model"""
        X, y = self.create_sample_data()

        model = LSTMModel({"sequence_length": 20})
        metrics = model.train(X, y)

        assert model.is_trained
        assert "mae" in metrics

        predictions = model.predict(X.head(10))
        assert len(predictions) == 10
        print("[OK] LSTM model test passed")

    def test_random_forest_model(self):
        """Test Random Forest model"""
        X, y = self.create_sample_data()

        model = RandomForestModel({})
        metrics = model.train(X, y)

        assert model.is_trained
        assert "mae" in metrics
        assert "r2" in metrics

        predictions = model.predict(X.head(10))
        assert len(predictions) == 10
        print("[OK] Random Forest model test passed")


class TestEnsembleModel:
    """Test ensemble model"""

    def test_ensemble_training(self):
        """Test ensemble model training"""
        X = pd.DataFrame({f"feature_{i}": np.random.randn(100) for i in range(10)})
        y = pd.Series(np.random.randn(100) * 0.01)

        ensemble = EnsembleModel({})
        metrics = ensemble.train(X, y)

        assert ensemble.is_trained
        assert "models" in metrics
        assert "weights" in metrics
        assert "ensemble_mae" in metrics
        print("[OK] Ensemble training test passed")

    def test_ensemble_prediction(self):
        """Test ensemble prediction"""
        X = pd.DataFrame({f"feature_{i}": np.random.randn(100) for i in range(10)})
        y = pd.Series(np.random.randn(100) * 0.01)

        ensemble = EnsembleModel({})
        ensemble.train(X, y)

        predictions, model_predictions = ensemble.predict(X.head(10))

        assert len(predictions) == 10
        assert len(model_predictions) == 4  # 4 models in ensemble
        assert "lightgbm" in model_predictions
        assert "catboost" in model_predictions
        print("[OK] Ensemble prediction test passed")

    def test_feature_importance(self):
        """Test feature importance aggregation"""
        X = pd.DataFrame({f"feature_{i}": np.random.randn(100) for i in range(10)})
        y = pd.Series(np.random.randn(100) * 0.01)

        ensemble = EnsembleModel({})
        ensemble.train(X, y)

        importance = ensemble.get_feature_importance()

        assert len(importance) > 0
        assert sum(importance.values()) > 0
        print("[OK] Feature importance test passed")


class TestRegimeDetector:
    """Test market regime detection"""

    def create_trending_data(self, trend="bull"):
        """Create trending market data"""
        dates = pd.date_range(start="2024-01-01", periods=100, freq="1h")

        if trend == "bull":
            base_price = 100 + np.arange(100) * 0.5
        else:
            base_price = 100 - np.arange(100) * 0.5

        noise = np.random.randn(100) * 0.5

        market_data = pd.DataFrame(
            {
                "open": base_price + noise,
                "high": base_price + abs(noise) + 1,
                "low": base_price - abs(noise) - 1,
                "close": base_price + noise * 0.5,
                "volume": np.random.rand(100) * 1000000,
            },
            index=dates,
        )

        return market_data

    def test_regime_detection(self):
        """Test regime detection"""
        detector = RegimeDetector({})

        # Test bull market detection
        bull_data = self.create_trending_data("bull")
        regime, confidence = detector.detect_regime(bull_data)

        assert regime in [MarketRegime.TRENDING_BULL, MarketRegime.SIDEWAYS]
        assert 0 <= confidence <= 1
        print("[OK] Regime detection test passed")

    def test_regime_parameters(self):
        """Test regime parameters"""
        detector = RegimeDetector({})

        params = detector.get_regime_params(MarketRegime.TRENDING_BULL)

        assert "position_size_multiplier" in params
        assert "stop_loss" in params
        assert "take_profit" in params
        assert "preferred_strategies" in params
        print("[OK] Regime parameters test passed")

    def test_regime_transition(self):
        """Test regime transition prediction"""
        detector = RegimeDetector({})

        # Simulate some regime history
        bull_data = self.create_trending_data("bull")
        detector.detect_regime(bull_data)

        next_regime, probability = detector.predict_next_regime()

        assert next_regime in [MarketRegime.TRENDING_BULL, MarketRegime.TRENDING_BEAR, MarketRegime.SIDEWAYS, MarketRegime.HIGH_VOLATILITY]
        assert 0 <= probability <= 1
        print("[OK] Regime transition test passed")


def run_tests():
    """Run all ML tests"""
    print("=" * 60)
    print("RUNNING ML MODEL TESTS")
    print("=" * 60)

    # Test feature engineering
    print("\nTesting Feature Engineering...")
    feature_tests = TestFeatureEngineer()
    feature_tests.test_create_features()
    feature_tests.test_normalize_features()

    # Test individual models
    print("\nTesting Individual ML Models...")
    model_tests = TestMLModels()
    model_tests.test_lightgbm_model()
    model_tests.test_catboost_model()
    model_tests.test_lstm_model()
    model_tests.test_random_forest_model()

    # Test ensemble
    print("\nTesting Ensemble Model...")
    ensemble_tests = TestEnsembleModel()
    ensemble_tests.test_ensemble_training()
    ensemble_tests.test_ensemble_prediction()
    ensemble_tests.test_feature_importance()

    # Test regime detector
    print("\nTesting Regime Detector...")
    regime_tests = TestRegimeDetector()
    regime_tests.test_regime_detection()
    regime_tests.test_regime_parameters()
    regime_tests.test_regime_transition()

    print("\n" + "=" * 60)
    print("[SUCCESS] ALL ML MODEL TESTS PASSED!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
