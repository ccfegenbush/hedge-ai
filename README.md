# Hedge AI

An investment platform for generating, backtesting, and paper trading stock strategies.

## Features

- **Strategy Framework**: Extensible strategy interface for implementing trading strategies
- **Momentum Strategy**: Built-in momentum strategy using moving average crossover signals
- **Backtesting Engine**: Run strategies against historical data with performance metrics
- **Performance Metrics**: Sharpe ratio, Sortino ratio, max drawdown, win rate, and more
- **CLI Interface**: Easy-to-use command line interface for running backtests
- **REST API**: FastAPI-based API for web frontend integration

## Installation

```bash
# Clone the repository
cd hedge-ai

# Install dependencies
pip install -r requirements.txt
```

## Usage

### List Available Strategies

```bash
python -m src.cli list-strategies
```

### Run a Backtest

```bash
python -m src.cli backtest --strategy momentum --tickers AAPL,MSFT --start 2023-01-01 --capital 10000
```

### CLI Options

```
Options:
  -s, --strategy TEXT     Strategy to backtest (default: momentum)
  -t, --tickers TEXT      Comma-separated list of ticker symbols (default: AAPL)
  --start TEXT            Start date (YYYY-MM-DD) [required]
  --end TEXT              End date (YYYY-MM-DD), defaults to today
  -c, --capital FLOAT     Initial capital (default: 10000)
  --commission FLOAT      Commission per trade (default: 0)
  -l, --lookback INTEGER  Lookback period for momentum strategy (default: 20)
```

### Start the API Server

```bash
uvicorn src.api.main:app --reload --port 8000
```

### API Endpoints

- `GET /` - Health check
- `GET /api/strategies` - List available strategies
- `POST /api/backtest` - Run a backtest

Example API request:
```bash
curl -X POST http://localhost:8000/api/backtest \
  -H "Content-Type: application/json" \
  -d '{"strategy": "momentum", "tickers": ["AAPL"], "start_date": "2024-01-01", "initial_capital": 10000}'
```

## Project Structure

```
hedge-ai/
├── src/
│   ├── api/                   # FastAPI REST API
│   │   ├── main.py            # API routes
│   │   └── schemas.py         # Pydantic models
│   ├── cli.py                 # CLI entry point
│   ├── strategies/
│   │   ├── base.py            # Abstract strategy interface
│   │   └── momentum.py        # Momentum strategy implementation
│   ├── data/
│   │   ├── provider.py        # Data fetching abstraction
│   │   └── yahoo.py           # yfinance implementation
│   ├── backtest/
│   │   ├── engine.py          # Backtesting engine
│   │   └── metrics.py         # Performance metrics
│   └── models/
│       └── schemas.py         # Data classes for trades, positions, etc.
├── tests/
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## License

MIT
