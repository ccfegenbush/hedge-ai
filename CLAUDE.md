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

---

## Code Standards

### Git Worktree Workflow (Default)

Use git worktrees to work on multiple branches simultaneously in different terminal tabs.

**Directory Structure:**
```
hedge-ai/                       # Main repo (stays on main)
hedge-ai-worktrees/             # Feature worktrees
  ├── add-new-strategy/         # feature/add-new-strategy branch
  └── fix-backtest-bug/         # feature/fix-backtest-bug branch
```

**Starting a New Feature:**
```bash
# From main repo, pull latest changes first
git checkout main
git pull origin main

# Create worktrees directory if needed
mkdir -p ../hedge-ai-worktrees

# Create new worktree with feature branch
git worktree add ../hedge-ai-worktrees/<feature-name> -b feature/<feature-name>

# Then open new terminal tab and:
cd ../hedge-ai-worktrees/<feature-name>
```

**Feature Development Process:**
1. Pull latest main and create worktree (see above)
2. Work in the worktree directory (new terminal tab)
3. Make changes and run tests: `python -m pytest tests/ -v`
4. Commit changes (wait for user approval)
5. Push branch (wait for user approval)
6. Create PR via `gh pr create` (wait for user approval)
7. Merge PR via `gh pr merge` (wait for user approval)
8. Cleanup worktree: `git worktree remove ../hedge-ai-worktrees/<feature-name>`

### Git Rules
- Never commit without explicitly asking for permission first
- Never push without explicitly asking for permission first
- Never merge PRs without explicitly asking for permission first
- Branch names must follow pattern: `feature/description` or `fix/description`
- Commit messages must be descriptive, not generic ("fix bug" is not acceptable)
