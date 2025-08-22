# CryptoBot Pro

A sophisticated cryptocurrency trading system with ML/AI capabilities for automated trading and market analysis.

## Features

- **Machine Learning Models**: Ensemble of LightGBM, CatBoost, LSTM, and Random Forest models
- **Deep Learning Support**: Transformer, WaveNet, and TCN architectures with GPU acceleration
- **Real-time Trading**: Support for simulation, paper, and live trading modes
- **Risk Management**: Stop-loss, take-profit, trailing stops, and portfolio heat management
- **Web Dashboard**: Next.js-based monitoring and control interface
- **Data Infrastructure**: Binance integration with historical data support

## Project Structure

```
├── src/                    # Core Python application
│   ├── core/              # Core utilities (config, database, logging)
│   ├── ml/                # Machine learning models and training
│   ├── trading/           # Trading engine and order management
│   ├── simulation/        # Market simulation engine
│   ├── data_providers/    # Exchange data providers
│   └── api/               # FastAPI backend
├── web/                   # Next.js frontend dashboard
├── config/                # Configuration files
├── data/                  # Database and data storage
├── tests/                 # Test suite
└── logs/                  # Application logs
```

## Setup

### Prerequisites

- Python 3.12+
- Node.js 18+ (for web dashboard)
- Intel Arc GPU (optional, for XPU acceleration)

### Installation

1. Create virtual environment:
```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. For Intel XPU support:
```bash
pip install -r requirements-xpu.txt
```

4. Install web dependencies:
```bash
cd web
npm install
```

## Configuration

Edit `config/main_config.yml` to configure:
- Exchange API credentials
- Trading pairs and timeframes
- Risk management parameters
- ML model settings

## Usage

### Download Historical Data
```bash
python download_historical_data.py
```

### Train ML Models
```bash
python run_deep_learning.py
```

### Run 24-Hour Benchmark
```bash
python run_benchmark.py
```

### Start Trading System
```bash
python src/main.py --mode simulation --duration 24h
```

### Start Web Dashboard
```bash
python start_server.py
```
Then open http://localhost:3000

## Trading Modes

- **Simulation**: Test strategies with historical data
- **Paper**: Real-time trading with simulated funds
- **Live**: Real trading (use with caution)

## Performance Targets

- **Prediction Accuracy**: >55% directional accuracy
- **Sharpe Ratio**: 1.5-2.0
- **Annual Returns**: 10-20% after fees
- **Max Drawdown**: <10%

## Risk Warning

Cryptocurrency trading carries significant risk. This software is for educational purposes. Always test thoroughly in simulation mode before considering live trading.

## License

Proprietary - All rights reserved

## Support

For issues or questions, please check the documentation in `FINAL_GOAL.md` and `PROJECT_STATUS.md`.