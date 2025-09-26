#!/usr/bin/env python
"""
Experiment tracking system for model improvements.
Tracks all experiments, hyperparameters, and results.
"""

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd


class ExperimentTracker:
    """Track and log all experiments with results."""

    def __init__(self, experiment_dir: str = "experiments"):
        self.experiment_dir = Path(experiment_dir)
        self.experiment_dir.mkdir(exist_ok=True)
        self.current_experiment = None
        self.results_file = self.experiment_dir / "all_results.json"
        self.best_model_file = self.experiment_dir / "best_model.json"

        # Load existing results
        self.all_results = self._load_results()

    def _load_results(self) -> list:
        """Load existing experiment results."""
        if self.results_file.exists():
            with open(self.results_file) as f:
                return json.load(f)
        return []

    def _save_results(self):
        """Save all results to file."""
        with open(self.results_file, "w") as f:
            json.dump(self.all_results, f, indent=2, default=str)

    def start_experiment(self, name: str, model_type: str, hyperparams: dict[str, Any], description: str = "") -> str:
        """Start a new experiment."""
        experiment_id = hashlib.md5(f"{name}_{model_type}_{time.time()}".encode()).hexdigest()[:8]

        self.current_experiment = {
            "id": experiment_id,
            "name": name,
            "model_type": model_type,
            "hyperparams": hyperparams,
            "description": description,
            "start_time": datetime.now().isoformat(),
            "status": "running",
            "metrics": {},
            "training_history": [],
            "device": "xpu" if self._check_xpu() else "cpu",
        }

        # Create experiment directory
        exp_dir = self.experiment_dir / experiment_id
        exp_dir.mkdir(exist_ok=True)

        return experiment_id

    def _check_xpu(self) -> bool:
        """Check if XPU is available."""
        try:
            import torch

            return hasattr(torch, "xpu") and torch.xpu.is_available()
        except:
            return False

    def log_epoch(self, epoch: int, train_loss: float, val_loss: float, train_acc: float, val_acc: float, epoch_time: float, lr: Optional[float] = None):
        """Log epoch results."""
        if not self.current_experiment:
            return

        epoch_data = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "epoch_time": epoch_time,
            "lr": lr,
        }

        self.current_experiment["training_history"].append(epoch_data)

    def log_metrics(self, metrics: dict[str, Any]):
        """Log final metrics."""
        if not self.current_experiment:
            return

        self.current_experiment["metrics"].update(metrics)

    def end_experiment(self, success: bool = True, error_msg: str = ""):
        """End current experiment and save results."""
        if not self.current_experiment:
            return

        self.current_experiment["end_time"] = datetime.now().isoformat()
        self.current_experiment["status"] = "success" if success else "failed"
        self.current_experiment["error"] = error_msg

        # Calculate summary statistics
        if self.current_experiment["training_history"]:
            history = self.current_experiment["training_history"]
            self.current_experiment["summary"] = {
                "best_val_acc": max(h["val_acc"] for h in history),
                "best_val_loss": min(h["val_loss"] for h in history),
                "final_val_acc": history[-1]["val_acc"],
                "final_val_loss": history[-1]["val_loss"],
                "total_epochs": len(history),
                "avg_epoch_time": np.mean([h["epoch_time"] for h in history]),
            }

        # Save experiment
        exp_file = self.experiment_dir / self.current_experiment["id"] / "experiment.json"
        with open(exp_file, "w") as f:
            json.dump(self.current_experiment, f, indent=2, default=str)

        # Add to all results
        self.all_results.append(self.current_experiment)
        self._save_results()

        # Check if this is the best model
        self._update_best_model()

        self.current_experiment = None

    def _update_best_model(self):
        """Update best model if current is better."""
        if not self.all_results:
            return

        # Find best model by validation accuracy
        best = max(self.all_results, key=lambda x: x.get("summary", {}).get("best_val_acc", 0))

        with open(self.best_model_file, "w") as f:
            json.dump(best, f, indent=2, default=str)

    def get_best_model(self) -> Optional[dict]:
        """Get the best model so far."""
        if self.best_model_file.exists():
            with open(self.best_model_file) as f:
                return json.load(f)
        return None

    def compare_experiments(self, metric: str = "val_acc") -> pd.DataFrame:
        """Compare all experiments by a metric."""
        if not self.all_results:
            return pd.DataFrame()

        data = []
        for exp in self.all_results:
            if exp["status"] == "success" and "summary" in exp:
                data.append(
                    {
                        "id": exp["id"],
                        "name": exp["name"],
                        "model_type": exp["model_type"],
                        "val_acc": exp["summary"].get("best_val_acc", 0),
                        "val_loss": exp["summary"].get("best_val_loss", float("inf")),
                        "epochs": exp["summary"].get("total_epochs", 0),
                        "avg_time": exp["summary"].get("avg_epoch_time", 0),
                        "device": exp.get("device", "unknown"),
                    }
                )

        df = pd.DataFrame(data)
        if not df.empty:
            df = df.sort_values("val_acc", ascending=False)

        return df

    def plot_learning_curves(self, experiment_id: str):
        """Plot learning curves for an experiment."""
        exp = None
        for e in self.all_results:
            if e["id"] == experiment_id:
                exp = e
                break

        if not exp or not exp["training_history"]:
            return None

        import matplotlib.pyplot as plt

        history = exp["training_history"]
        epochs = [h["epoch"] for h in history]
        train_acc = [h["train_acc"] for h in history]
        val_acc = [h["val_acc"] for h in history]
        train_loss = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"] for h in history]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

        # Accuracy plot
        ax1.plot(epochs, train_acc, label="Train Accuracy")
        ax1.plot(epochs, val_acc, label="Val Accuracy")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title(f'Accuracy - {exp["name"]}')
        ax1.legend()
        ax1.grid(True)

        # Loss plot
        ax2.plot(epochs, train_loss, label="Train Loss")
        ax2.plot(epochs, val_loss, label="Val Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss")
        ax2.set_title(f'Loss - {exp["name"]}')
        ax2.legend()
        ax2.grid(True)

        plt.tight_layout()

        # Save plot
        plot_file = self.experiment_dir / experiment_id / "learning_curves.png"
        plt.savefig(plot_file)
        plt.close()

        return plot_file

    def generate_report(self) -> str:
        """Generate a markdown report of all experiments."""
        report = ["# Experiment Results Report", f"\nGenerated: {datetime.now().isoformat()}\n"]

        # Summary statistics
        successful = [e for e in self.all_results if e["status"] == "success"]
        report.append("## Summary")
        report.append(f"- Total Experiments: {len(self.all_results)}")
        report.append(f"- Successful: {len(successful)}")
        report.append(f"- Failed: {len(self.all_results) - len(successful)}")

        # Best model
        best = self.get_best_model()
        if best:
            report.append("\n## Best Model")
            report.append(f"- ID: {best['id']}")
            report.append(f"- Type: {best['model_type']}")
            report.append(f"- Best Val Accuracy: {best['summary']['best_val_acc']:.2f}%")
            report.append(f"- Best Val Loss: {best['summary']['best_val_loss']:.4f}")

        # Comparison table
        df = self.compare_experiments()
        if not df.empty:
            report.append("\n## All Experiments Comparison")
            report.append(df.to_markdown(index=False))

        # Detailed results
        report.append("\n## Detailed Results")
        for exp in successful:
            report.append(f"\n### {exp['name']} ({exp['id']})")
            report.append(f"- Model: {exp['model_type']}")
            report.append(f"- Device: {exp.get('device', 'unknown')}")
            report.append(f"- Hyperparameters: {json.dumps(exp['hyperparams'], indent=2)}")
            if "summary" in exp:
                report.append(f"- Best Val Acc: {exp['summary']['best_val_acc']:.2f}%")
                report.append(f"- Final Val Acc: {exp['summary']['final_val_acc']:.2f}%")
                report.append(f"- Epochs: {exp['summary']['total_epochs']}")
                report.append(f"- Avg Epoch Time: {exp['summary']['avg_epoch_time']:.1f}s")

        report_text = "\n".join(report)

        # Save report
        report_file = self.experiment_dir / "EXPERIMENT_REPORT.md"
        with open(report_file, "w") as f:
            f.write(report_text)

        return report_text


# Global tracker instance
tracker = ExperimentTracker()


if __name__ == "__main__":
    # Test the tracker
    tracker = ExperimentTracker()

    # Simulate an experiment
    exp_id = tracker.start_experiment(
        name="Test TCN", model_type="TCN", hyperparams={"hidden_size": 128, "layers": 3}, description="Testing experiment tracker"
    )

    # Log some epochs
    for epoch in range(5):
        tracker.log_epoch(
            epoch=epoch + 1,
            train_loss=0.7 - epoch * 0.05,
            val_loss=0.7 - epoch * 0.04,
            train_acc=50 + epoch * 2,
            val_acc=50 + epoch * 1.5,
            epoch_time=10.5,
            lr=0.001,
        )

    # Log metrics
    tracker.log_metrics({"sharpe_ratio": 1.2, "win_rate": 52.5, "max_drawdown": 8.5})

    # End experiment
    tracker.end_experiment(success=True)

    # Generate report
    report = tracker.generate_report()
    print(report)
