# /predict vs. `spec.txt` Architecture Review

## Scope

Assessment covers the current code and configuration under `domains/predict` against the institutional trading requirements captured in `spec.txt`.

## Architectural Alignment

- **Runtime model** – Implementation runs a monolithic FastAPI + asyncio trading loop labeled “CryptoBot Pro” (`src/main.py:23-101`), whereas the spec mandates a microservices, containerised stack centred on a GBP/USD FIX executor (`spec.txt:11-12`, `spec.txt:182-188`).
- **Deployment targets** – Current config relies on local execution with optional simulation/paper/live modes; spec expects a production Docker deployment for a single IC Markets account (`spec.txt:6-12`).
- **Data stores** – SQLite with SQLAlchemy models for `market_data`, `trades`, `predictions`, and `performance` (`src/core/database.py:104-347`) replaces the required PostgreSQL schema and Redis caches (`spec.txt:189-220`).

## Trading & Risk Features

- **Instrument coverage** – Code focuses on multiple crypto pairs sourced from Binance/Coinbase (`config/main_config.yml:58-118`), conflicting with the spec’s single GBP/USD instrument, lot sizing, and pip-based risk constants (`spec.txt:18-40`).
- **Position sizing** – Dynamic percentage-of-equity sizing (`src/trading/engine.py:241-257`) does not meet the 0.16 fixed lot mandate with 5% daily loss cap and ATR tiers (`spec.txt:18-34`).
- **Session controls** – No enforcement of the four UTC trading windows or 22:00 force-close rule; `_trading_loop` opens positions continuously (`spec.txt:42-54`).
- **Signal pipeline** – Ensemble fallback to random trades and missing confluence scoring / pattern detectors (`src/trading/engine.py:124-231`) diverge from the required Fibonacci, triangle, VWAP, and ATR-weighted system (`spec.txt:55-177`).
- **Machine learning stack** – Ensemble lacks the Transformer and XGBoost components and treats LSTM as a stub (`src/ml/models.py:60-219`), short of the ensemble spec (`spec.txt:93-111`).

## Infrastructure & Monitoring

- **FIX connectivity** – Live order path is unimplemented (`src/trading/order.py:167-177`) with no FIX 4.4 messaging, port bindings, or host settings (`spec.txt:221-246`).
- **Logging & dashboards** – JSON logger writes to `logs/cryptobot.log` without the retention/location scheme in the spec, and Grafana/Prometheus settings are config placeholders only (`config/main_config.yml:164-193`, `spec.txt:207-239`).
- **Anomaly handling** – No implementations for the price/volume/spread anomaly controls or trading halts defined in Section 12 (`spec.txt:288-308`).

## Notable Implementation Gaps

- Imported `FeatureEngineer` class is undefined, causing runtime errors before any trading logic runs (`src/trading/engine.py:34-38`).
- Database accessor for predictions expects fields not created by the ORM model (uses `predicted_return`, `regime`, and `models` columns that do not exist), indicating unfinished telemetry wiring.
- Emergency stop only manipulates in-memory state; it does not coordinate with external brokers, caches, or downstream services described in the spec (`src/trading/engine.py:96-113`).

## Recommendations

1. Decide whether to align the current codebase with the GBP/USD FIX mandate or formally diverge and document a separate product scope.
2. If pursuing alignment, introduce the spec-defined services: FIX executor, Redis/PostgreSQL storage, Docker orchestration, and event-driven components.
3. Implement the mandated trading controls (session windows, confluence scoring, ATR stops, fixed lot sizing) and remove random signal fallbacks.
4. Build the monitoring surface—Grafana dashboards, anomaly handlers, logging retention—to satisfy Sections 10-12 of the spec.
5. Address schema/implementation mismatches (missing classes, incorrect ORM fields) before extending functionality.
