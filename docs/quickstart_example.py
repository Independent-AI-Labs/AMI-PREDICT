from pathlib import Path

from src.data_providers.data_sync import DataSync
from src.ml.backtester import Backtester
from src.ml.feature_engineering import build_basic_features
from src.ml.models import train_lightgbm_classifier


def main() -> None:
    base = Path(__file__).resolve().parent.parent
    example = base / "data" / "parquet" / "BTC_USDT" / "2020" / "BTC_USDT_2020_08.parquet"

    if not example.exists():
        raise SystemExit(f"Example dataset not found: {example}\n" "Fetch or point to a local Parquet file before running.")

    df = DataSync.read_parquet_file(example)
    X, y = build_basic_features(df)

    model, metrics = train_lightgbm_classifier(X, y)
    print("Train metrics:", metrics)

    bt = Backtester(model=model)
    results = bt.run(df)
    print("Backtest summary:", results.summary())


if __name__ == "__main__":
    main()
