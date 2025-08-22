# CryptoBot Pro Dashboard

## Overview
Real-time monitoring dashboard for the CryptoBot Pro 24-hour benchmark trading system.

## Features
- **Live Market Overview**: Real-time price updates for BTC, ETH, BNB, ADA, SOL
- **Performance Metrics**: P&L, Win Rate, Sharpe Ratio, Drawdown tracking
- **Model Performance**: Individual and ensemble model accuracy monitoring
- **Position Management**: Live tracking of open positions and P&L
- **Risk Monitor**: Real-time risk metrics and alerts
- **System Status**: CPU, memory, latency, and network monitoring

## Quick Start

### Install Dependencies
```bash
npm install
```

### Start Development Server
```bash
node start-server.js
```

The dashboard will be available at http://localhost:3000

### Stop Server
```bash
node stop-server.js
```

## Dashboard Components

### Market Overview
- Real-time price updates every 3 seconds
- 24-hour change percentages
- Trading volume indicators

### Trading Status
- Current operational mode (Simulation/Paper/Live)
- Trades executed counter
- Signals generated counter
- Emergency stop controls

### Performance Metrics
- Total P&L tracking
- Win rate percentage
- Sharpe ratio calculation
- Maximum drawdown monitoring
- Prediction accuracy

### Live Chart
- Real-time price visualization
- Volume indicators
- 24h high/low markers

### Model Performance
- Individual model accuracy (LightGBM, CatBoost, LSTM)
- Ensemble performance
- Latency monitoring
- Regime detection status

### Risk Monitor
- Current drawdown vs max allowed
- Portfolio heat map
- Correlation monitoring
- Margin usage tracking

### System Status
- CPU and memory usage
- Network latency
- Data processing rate
- API call metrics

## API Endpoints

- `GET /api/status` - System and trading status
- `GET /api/market` - Market data for tracked pairs
- `GET /api/positions` - Open positions (coming soon)
- `GET /api/performance` - Performance metrics (coming soon)

## Technology Stack
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Recharts**: Data visualization
- **Lucide Icons**: Icon library

## Development

### Project Structure
```
web/
├── app/              # Next.js app directory
│   ├── api/         # API routes
│   ├── globals.css  # Global styles
│   ├── layout.tsx   # Root layout
│   └── page.tsx     # Dashboard page
├── components/       # React components
│   ├── Header.tsx
│   ├── MarketOverview.tsx
│   ├── TradingStatus.tsx
│   ├── PerformanceMetrics.tsx
│   ├── LiveChart.tsx
│   ├── ModelPerformance.tsx
│   ├── PositionsTable.tsx
│   ├── RiskMonitor.tsx
│   └── SystemStatus.tsx
├── public/          # Static assets
└── package.json     # Dependencies
```

### Customization
- Edit `tailwind.config.js` for theme customization
- Modify components in `/components` directory
- Add new API routes in `/app/api`

## Integration with Python Backend

The dashboard is designed to integrate with the Python trading engine. Update the API routes in `/app/api` to connect to your Python backend endpoints.

Example integration:
```typescript
// app/api/market/route.ts
const response = await fetch('http://localhost:8000/api/market')
const data = await response.json()
return NextResponse.json(data)
```

## Monitoring

The dashboard provides real-time monitoring for the 24-hour benchmark test:
1. System performance metrics
2. Trading activity and signals
3. Model prediction accuracy
4. Risk exposure levels
5. P&L tracking

## License
MIT