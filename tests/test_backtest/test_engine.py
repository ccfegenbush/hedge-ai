from datetime import datetime

import pandas as pd
import pytest

from src.backtest.engine import BacktestEngine
from src.models.schemas import OHLCVSeries, Signal, TradeSignal, OrderSide
from src.strategies.base import Strategy, StrategyInfo


class MockStrategy(Strategy):
    """Mock strategy for testing that returns predefined signals."""

    def __init__(self, signals: list[TradeSignal]):
        self._signals = signals

    @property
    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name="mock", description="Mock strategy", parameters={}
        )

    def generate_signals(self, data: OHLCVSeries) -> list[TradeSignal]:
        return [s for s in self._signals if s.ticker == data.ticker]


@pytest.fixture
def sample_data():
    """Create sample OHLCV data."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    prices = [100, 102, 104, 103, 105, 107, 106, 108, 110, 112]
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 2 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume": [1000000] * 10,
        },
        index=dates,
    )
    return OHLCVSeries(ticker="TEST", data=df)


@pytest.fixture
def buy_sell_signals(sample_data):
    """Create buy and sell signals."""
    dates = sample_data.data.index
    return [
        TradeSignal(
            ticker="TEST",
            signal=Signal.BUY,
            timestamp=dates[2].to_pydatetime(),
            price=104,
        ),
        TradeSignal(
            ticker="TEST",
            signal=Signal.SELL,
            timestamp=dates[7].to_pydatetime(),
            price=108,
        ),
    ]


class TestBacktestEngine:
    def test_init_defaults(self):
        engine = BacktestEngine()
        assert engine.initial_capital == 10000.0
        assert engine.commission == 0.0
        assert engine.position_size_pct == 1.0

    def test_init_custom_params(self):
        engine = BacktestEngine(
            initial_capital=50000, commission=5.0, position_size_pct=0.5
        )
        assert engine.initial_capital == 50000
        assert engine.commission == 5.0
        assert engine.position_size_pct == 0.5

    def test_position_size_pct_clamped(self):
        engine = BacktestEngine(position_size_pct=1.5)
        assert engine.position_size_pct == 1.0

        engine = BacktestEngine(position_size_pct=-0.5)
        assert engine.position_size_pct == 0.0

    def test_run_with_no_signals(self, sample_data):
        """Test backtest with no signals returns initial capital."""
        strategy = MockStrategy([])
        engine = BacktestEngine(initial_capital=10000)

        result = engine.run(strategy, sample_data)

        assert result.initial_capital == 10000
        assert result.final_value == 10000
        assert result.num_trades == 0

    def test_run_with_buy_sell(self, sample_data, buy_sell_signals):
        """Test backtest with buy and sell signals."""
        strategy = MockStrategy(buy_sell_signals)
        engine = BacktestEngine(initial_capital=10000, commission=0)

        result = engine.run(strategy, sample_data)

        assert result.num_trades == 2
        assert result.final_value > result.initial_capital  # Made profit

        # Check trades
        assert result.trades[0].side == OrderSide.BUY
        assert result.trades[1].side == OrderSide.SELL

    def test_run_with_commission(self, sample_data, buy_sell_signals):
        """Test that commission is deducted from trades."""
        strategy = MockStrategy(buy_sell_signals)
        engine_no_comm = BacktestEngine(initial_capital=10000, commission=0)
        engine_with_comm = BacktestEngine(initial_capital=10000, commission=10)

        result_no_comm = engine_no_comm.run(strategy, sample_data)
        result_with_comm = engine_with_comm.run(strategy, sample_data)

        # Commission should reduce final value
        assert result_with_comm.final_value < result_no_comm.final_value

    def test_buy_only_holds_position(self, sample_data):
        """Test that buy without sell holds position to end."""
        dates = sample_data.data.index
        buy_only = [
            TradeSignal(
                ticker="TEST",
                signal=Signal.BUY,
                timestamp=dates[2].to_pydatetime(),
                price=104,
            ),
        ]
        strategy = MockStrategy(buy_only)
        engine = BacktestEngine(initial_capital=10000)

        result = engine.run(strategy, sample_data)

        # Should have 1 buy trade
        assert result.num_trades == 1
        assert result.trades[0].side == OrderSide.BUY

        # Final value should reflect position at final price (112)
        # Bought at 104, ended at 112 = profit
        assert result.final_value > result.initial_capital

    def test_sell_without_position_ignored(self, sample_data):
        """Test that sell signal without position is ignored."""
        dates = sample_data.data.index
        sell_only = [
            TradeSignal(
                ticker="TEST",
                signal=Signal.SELL,
                timestamp=dates[2].to_pydatetime(),
                price=104,
            ),
        ]
        strategy = MockStrategy(sell_only)
        engine = BacktestEngine(initial_capital=10000)

        result = engine.run(strategy, sample_data)

        # No trades should execute
        assert result.num_trades == 0
        assert result.final_value == 10000

    def test_portfolio_values_series(self, sample_data, buy_sell_signals):
        """Test that portfolio values are tracked correctly."""
        strategy = MockStrategy(buy_sell_signals)
        engine = BacktestEngine(initial_capital=10000)

        result = engine.run(strategy, sample_data)

        # Should have values for each day
        assert len(result.portfolio_values) == len(sample_data)
        assert isinstance(result.portfolio_values, pd.Series)

    def test_position_size_pct_limits_trade(self, sample_data, buy_sell_signals):
        """Test that position_size_pct limits trade size."""
        strategy = MockStrategy(buy_sell_signals)

        engine_full = BacktestEngine(initial_capital=10000, position_size_pct=1.0)
        engine_half = BacktestEngine(initial_capital=10000, position_size_pct=0.5)

        result_full = engine_full.run(strategy, sample_data)
        result_half = engine_half.run(strategy, sample_data)

        # Half position should buy fewer shares
        assert result_half.trades[0].quantity < result_full.trades[0].quantity
