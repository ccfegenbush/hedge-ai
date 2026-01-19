# Hedge AI - Backend

This is the Python backend for the Hedge AI investment platform.

## Project Overview

A Python-based investment platform for generating, backtesting, and paper trading stock strategies. Provides both a CLI interface and a REST API.

## Tech Stack

- **Language:** Python 3.11+
- **Data Source:** yfinance (Yahoo Finance)
- **API Framework:** FastAPI
- **CLI:** Typer
- **Testing:** pytest

## Project Structure

```
src/
├── api/              # FastAPI REST API
│   ├── main.py       # API routes and app
│   └── schemas.py    # Pydantic request/response models
├── backtest/         # Backtesting engine
│   ├── engine.py     # Core backtest runner
│   └── metrics.py    # Performance metrics (Sharpe, drawdown, etc.)
├── data/             # Data providers
│   ├── provider.py   # Abstract data provider interface
│   └── yahoo.py      # Yahoo Finance implementation
├── models/           # Data models
│   └── schemas.py    # Core dataclasses (Trade, Position, OHLCV, etc.)
├── strategies/       # Trading strategies
│   ├── base.py       # Abstract strategy interface
│   └── momentum.py   # Momentum strategy implementation
└── cli.py            # CLI entry point
```

## Key Commands

```bash
# Run tests
python -m pytest tests/ -v

# Start API server
uvicorn src.api.main:app --reload --port 8000

# Run CLI backtest
python -m src.cli backtest --strategy momentum --tickers AAPL --start 2024-01-01
```

## API Endpoints

- `GET /` - Health check
- `GET /api/strategies` - List available strategies
- `POST /api/backtest` - Run a backtest

## Adding New Strategies

1. Create a new file in `src/strategies/`
2. Extend the `Strategy` base class from `src/strategies/base.py`
3. Implement `info` property and `generate_signals()` method
4. Register the strategy in `src/api/main.py` and `src/cli.py` STRATEGIES dict

## Frontend

The frontend is a separate Next.js project: https://github.com/ccfegenbush/hedge-ai-ui
