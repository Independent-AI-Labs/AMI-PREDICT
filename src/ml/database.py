"""
Experiment tracking database for ML model training and benchmarking
"""
import json
import sqlite3
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class ExperimentDB:
    def __init__(self, db_path: str = "experiments.db"):
        self.db_path = Path(db_path)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.create_tables()

    def create_tables(self):
        """Create database tables for experiment tracking"""
        cursor = self.conn.cursor()

        # Experiments table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                model_type TEXT NOT NULL,
                training_start DATE,
                training_end DATE,
                validation_start DATE,
                validation_end DATE,
                test_start DATE,
                test_end DATE,
                symbol TEXT,
                timeframe TEXT,
                features TEXT,
                hyperparameters TEXT,
                status TEXT DEFAULT 'created',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP
            )
        """
        )

        # Results table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_value REAL,
                dataset TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """
        )

        # Predictions table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                timestamp TIMESTAMP,
                symbol TEXT,
                prediction REAL,
                actual REAL,
                prediction_proba TEXT,
                features TEXT,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """
        )

        # Training history table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS training_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                epoch INTEGER,
                train_loss REAL,
                val_loss REAL,
                train_metric REAL,
                val_metric REAL,
                learning_rate REAL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """
        )

        # Model artifacts table
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS model_artifacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_id INTEGER NOT NULL,
                artifact_type TEXT NOT NULL,
                file_path TEXT,
                file_size INTEGER,
                checksum TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(id)
            )
        """
        )

        self.conn.commit()

    def create_experiment(self, name: str, model_type: str, config: dict[str, Any]) -> int:
        """Create a new experiment"""
        cursor = self.conn.cursor()

        cursor.execute(
            """
            INSERT INTO experiments (
                name, model_type, training_start, training_end,
                validation_start, validation_end, test_start, test_end,
                symbol, timeframe, features, hyperparameters, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                name,
                model_type,
                config.get("training_start"),
                config.get("training_end"),
                config.get("validation_start"),
                config.get("validation_end"),
                config.get("test_start"),
                config.get("test_end"),
                config.get("symbol"),
                config.get("timeframe"),
                json.dumps(config.get("features", [])),
                json.dumps(config.get("hyperparameters", {})),
                "created",
            ),
        )

        self.conn.commit()
        return cursor.lastrowid

    def update_experiment_status(self, experiment_id: int, status: str):
        """Update experiment status"""
        cursor = self.conn.cursor()

        if status == "completed":
            cursor.execute(
                """
                UPDATE experiments
                SET status = ?, completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """,
                (status, experiment_id),
            )
        else:
            cursor.execute(
                """
                UPDATE experiments
                SET status = ?
                WHERE id = ?
            """,
                (status, experiment_id),
            )

        self.conn.commit()

    def save_results(self, experiment_id: int, metrics: dict[str, float], dataset: str):
        """Save experiment results"""
        cursor = self.conn.cursor()

        for metric_name, metric_value in metrics.items():
            cursor.execute(
                """
                INSERT INTO results (experiment_id, metric_name, metric_value, dataset)
                VALUES (?, ?, ?, ?)
            """,
                (experiment_id, metric_name, metric_value, dataset),
            )

        self.conn.commit()

    def save_predictions(self, experiment_id: int, predictions_df: pd.DataFrame):
        """Save model predictions"""
        cursor = self.conn.cursor()

        for _, row in predictions_df.iterrows():
            cursor.execute(
                """
                INSERT INTO predictions (
                    experiment_id, timestamp, symbol, prediction, actual,
                    prediction_proba, features
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment_id,
                    row.get("timestamp"),
                    row.get("symbol"),
                    row.get("prediction"),
                    row.get("actual"),
                    json.dumps(row.get("prediction_proba", [])),
                    json.dumps(row.get("features", {})),
                ),
            )

        self.conn.commit()

    def save_training_history(self, experiment_id: int, history: list[dict[str, Any]]):
        """Save training history for neural networks"""
        cursor = self.conn.cursor()

        for epoch_data in history:
            cursor.execute(
                """
                INSERT INTO training_history (
                    experiment_id, epoch, train_loss, val_loss,
                    train_metric, val_metric, learning_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    experiment_id,
                    epoch_data.get("epoch"),
                    epoch_data.get("train_loss"),
                    epoch_data.get("val_loss"),
                    epoch_data.get("train_metric"),
                    epoch_data.get("val_metric"),
                    epoch_data.get("learning_rate"),
                ),
            )

        self.conn.commit()

    def save_model_artifact(self, experiment_id: int, artifact_type: str, file_path: str, metadata: dict[str, Any] = None):
        """Save model artifact information"""
        cursor = self.conn.cursor()

        file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0

        cursor.execute(
            """
            INSERT INTO model_artifacts (
                experiment_id, artifact_type, file_path, file_size, metadata
            ) VALUES (?, ?, ?, ?, ?)
        """,
            (experiment_id, artifact_type, file_path, file_size, json.dumps(metadata or {})),
        )

        self.conn.commit()

    def get_experiment(self, experiment_id: int) -> Optional[dict[str, Any]]:
        """Get experiment details"""
        cursor = self.conn.cursor()

        cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def get_experiment_results(self, experiment_id: int) -> pd.DataFrame:
        """Get all results for an experiment"""
        query = """
            SELECT * FROM results
            WHERE experiment_id = ?
            ORDER BY dataset, metric_name
        """
        return pd.read_sql_query(query, self.conn, params=(experiment_id,))

    def get_experiment_predictions(self, experiment_id: int) -> pd.DataFrame:
        """Get predictions for an experiment"""
        query = """
            SELECT * FROM predictions
            WHERE experiment_id = ?
            ORDER BY timestamp
        """
        return pd.read_sql_query(query, self.conn, params=(experiment_id,))

    def compare_experiments(self, experiment_ids: list[int]) -> pd.DataFrame:
        """Compare multiple experiments"""
        placeholders = ",".join("?" * len(experiment_ids))
        query = f"""
            SELECT
                e.id,
                e.name,
                e.model_type,
                r.metric_name,
                r.metric_value,
                r.dataset
            FROM experiments e
            JOIN results r ON e.id = r.experiment_id
            WHERE e.id IN ({placeholders})
            ORDER BY e.id, r.dataset, r.metric_name
        """
        return pd.read_sql_query(query, self.conn, params=experiment_ids)

    def get_best_experiments(self, metric: str = "sharpe_ratio", dataset: str = "test", limit: int = 10) -> pd.DataFrame:
        """Get best performing experiments by metric"""
        query = """
            SELECT
                e.*,
                r.metric_value
            FROM experiments e
            JOIN results r ON e.id = r.experiment_id
            WHERE r.metric_name = ? AND r.dataset = ?
            ORDER BY r.metric_value DESC
            LIMIT ?
        """
        return pd.read_sql_query(query, self.conn, params=(metric, dataset, limit))

    def cleanup_failed_experiments(self):
        """Clean up failed or incomplete experiments"""
        cursor = self.conn.cursor()

        # Delete experiments older than 7 days that never completed
        cursor.execute(
            """
            DELETE FROM experiments
            WHERE status != 'completed'
            AND created_at < datetime('now', '-7 days')
        """
        )

        self.conn.commit()

    def close(self):
        """Close database connection"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
