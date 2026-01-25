"""Tests for the Polymarket backtesting engine."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.backtest.polymarket_engine import (
    PolymarketBacktestEngine,
    PolymarketPortfolio,
    run_polymarket_backtest,
)
from src.models.polymarket_schemas import (
    Market,
    MarketCategory,
    MarketStatus,
    Outcome,
    PolymarketPosition,
    PriceHistory,
)
from src.strategies.polymarket_composite import CompositeStrategy


@pytest.fixture
def sample_market():
    """Create a sample market for testing."""
    return Market(
        condition_id="test_market",
        question="Test market?",
        description="A test market",
        category=MarketCategory.OTHER,
        status=MarketStatus.ACTIVE,
        outcomes=[
            Outcome(outcome_id="yes", name="Yes", price=0.6, token_id="token_yes"),
            Outcome(outcome_id="no", name="No", price=0.4, token_id="token_no"),
        ],
        volume=100000,
        liquidity=10000,
        created_at=datetime.now() - timedelta(days=30),
        end_date=datetime.now() + timedelta(days=30),
    )


@pytest.fixture
def sample_price_history():
    """Create sample price history."""
    dates = pd.date_range(end=datetime.now(), periods=100, freq="h")
    prices = [0.5 + 0.005 * i for i in range(100)]
    volumes = [1000] * 100

    df = pd.DataFrame({
        "timestamp": dates,
        "price": prices,
        "volume": volumes,
    })

    return PriceHistory(
        condition_id="test_market",
        outcome_id="yes",
        data=df,
    )


class TestPolymarketPortfolio:
    """Tests for PolymarketPortfolio."""

    def test_initial_state(self):
        """Test portfolio initial state."""
        portfolio = PolymarketPortfolio(cash=10000)

        assert portfolio.cash == 10000
        assert len(portfolio.positions) == 0
        assert portfolio.total_value == 10000

    def test_with_position(self):
        """Test portfolio with position."""
        portfolio = PolymarketPortfolio(
            cash=5000,
            positions={
                "market_1": PolymarketPosition(
                    market_id="market_1",
                    outcome_id="yes",
                    tokens=100,
                    avg_cost=0.5,
                    entry_time=datetime.now(),
                )
            },
        )

        # 5000 cash + 100 tokens * 0.5 = 5050
        assert portfolio.total_value == 5050

    def test_value_at_prices(self):
        """Test portfolio value at given prices."""
        portfolio = PolymarketPortfolio(
            cash=5000,
            positions={
                "market_1": PolymarketPosition(
                    market_id="market_1",
                    outcome_id="yes",
                    tokens=100,
                    avg_cost=0.5,
                    entry_time=datetime.now(),
                )
            },
        )

        # At price 0.7: 5000 + 100 * 0.7 = 5070
        value = portfolio.value_at_prices({"market_1": 0.7})
        assert value == 5070


class TestPolymarketBacktestEngine:
    """Tests for PolymarketBacktestEngine."""

    def test_engine_initialization(self):
        """Test engine initialization with defaults."""
        engine = PolymarketBacktestEngine()

        assert engine.initial_capital == 10000
        assert engine.max_position_pct == 0.1
        assert engine.min_signal_strength == 0.3

    def test_engine_custom_params(self):
        """Test engine with custom parameters."""
        engine = PolymarketBacktestEngine(
            initial_capital=50000,
            max_position_pct=0.2,
            min_signal_strength=0.5,
            slippage_bps=20,
        )

        assert engine.initial_capital == 50000
        assert engine.max_position_pct == 0.2
        assert engine.min_signal_strength == 0.5
        assert engine.slippage_bps == 20

    @pytest.mark.asyncio
    async def test_run_backtest_empty_markets(self):
        """Test backtest with no markets."""
        engine = PolymarketBacktestEngine()
        strategy = CompositeStrategy()

        result = await engine.run(
            strategy=strategy,
            markets=[],
            price_histories={},
            start_date=datetime.now() - timedelta(days=30),
            end_date=datetime.now(),
        )

        assert result.initial_capital == 10000
        assert result.final_value == 10000  # No trades
        assert len(result.trades) == 0
        assert result.markets_traded == 0

    @pytest.mark.asyncio
    async def test_run_backtest_single_market(
        self, sample_market, sample_price_history
    ):
        """Test backtest with a single market."""
        engine = PolymarketBacktestEngine(
            min_signal_strength=0.0,  # Allow all signals
        )
        strategy = CompositeStrategy()

        result = await engine.run(
            strategy=strategy,
            markets=[sample_market],
            price_histories={sample_market.condition_id: sample_price_history},
            start_date=datetime.now() - timedelta(days=5),
            end_date=datetime.now(),
            signal_interval_hours=24,
        )

        assert result.initial_capital == 10000
        assert len(result.portfolio_values) > 0

    @pytest.mark.asyncio
    async def test_filter_history(self, sample_price_history):
        """Test price history filtering."""
        engine = PolymarketBacktestEngine()

        end_time = datetime.now() - timedelta(hours=50)
        filtered = engine._filter_history(sample_price_history, end_time)

        assert len(filtered.data) < len(sample_price_history.data)

    @pytest.mark.asyncio
    async def test_backtest_result_properties(
        self, sample_market, sample_price_history
    ):
        """Test backtest result properties."""
        engine = PolymarketBacktestEngine()
        strategy = CompositeStrategy()

        result = await engine.run(
            strategy=strategy,
            markets=[sample_market],
            price_histories={sample_market.condition_id: sample_price_history},
            start_date=datetime.now() - timedelta(days=5),
            end_date=datetime.now(),
        )

        # Check computed properties
        assert hasattr(result, "total_return")
        assert hasattr(result, "total_return_pct")
        assert hasattr(result, "win_count")
        assert hasattr(result, "loss_count")
        assert hasattr(result, "resolution_accuracy")


class TestBacktestConvenienceFunction:
    """Tests for the run_polymarket_backtest convenience function."""

    @pytest.mark.asyncio
    async def test_run_polymarket_backtest(
        self, sample_market, sample_price_history
    ):
        """Test the convenience function."""
        strategy = CompositeStrategy()

        result = await run_polymarket_backtest(
            strategy=strategy,
            markets=[sample_market],
            price_histories={sample_market.condition_id: sample_price_history},
            start_date=datetime.now() - timedelta(days=5),
            end_date=datetime.now(),
            initial_capital=10000,
        )

        assert result is not None
        assert result.initial_capital == 10000
