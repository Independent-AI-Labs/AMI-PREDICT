"""
Performance metrics calculator for trading models
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score


class MetricsCalculator:
    """Calculate performance metrics for models"""

    @staticmethod
    def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray = None) -> dict[str, float]:
        """Calculate classification metrics"""
        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1_score": f1_score(y_true, y_pred, zero_division=0),
        }

        if y_proba is not None and len(y_proba.shape) > 1:
            try:
                metrics["roc_auc"] = roc_auc_score(y_true, y_proba[:, 1])
            except:
                metrics["roc_auc"] = 0.5

        return metrics

    @staticmethod
    def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
        """Calculate regression metrics"""
        return {
            "mae": mean_absolute_error(y_true, y_pred),
            "mse": mean_squared_error(y_true, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
            "r2": r2_score(y_true, y_pred),
        }

    @staticmethod
    def trading_metrics(returns: np.ndarray) -> dict[str, float]:
        """Calculate trading-specific metrics"""
        if len(returns) == 0:
            return {"total_return": 0, "sharpe_ratio": 0, "sortino_ratio": 0, "max_drawdown": 0, "win_rate": 0, "profit_factor": 0}

        # Total return
        total_return = np.sum(returns)

        # Sharpe ratio (assuming 0 risk-free rate)
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0

        # Sortino ratio (only downside volatility)
        downside_returns = returns[returns < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
        sortino = np.mean(returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0

        # Maximum drawdown
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) if len(drawdown) > 0 else 0

        # Win rate
        wins = returns > 0
        win_rate = np.mean(wins) * 100

        # Profit factor
        gross_profits = np.sum(returns[returns > 0])
        gross_losses = abs(np.sum(returns[returns < 0]))
        profit_factor = gross_profits / gross_losses if gross_losses > 0 else 0

        return {
            "total_return": total_return,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
        }

    @staticmethod
    def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate directional accuracy for price predictions"""
        y_true_dir = (y_true > 0).astype(int)
        y_pred_dir = (y_pred > 0).astype(int)
        return accuracy_score(y_true_dir, y_pred_dir)
