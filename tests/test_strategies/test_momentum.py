import pandas as pd
import pytest

from src.models.schemas import OHLCVSeries, Signal
from src.strategies.momentum import MomentumStrategy


@pytest.fixture
def strategy():
    return MomentumStrategy(lookback_period=5)


@pytest.fixture
def uptrend_data():
    """Create data with clear uptrend (price crossing above MA)."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    # Start below MA, then cross above
    prices = [100, 99, 98, 97, 96, 95, 100, 105, 110, 115]
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
def downtrend_data():
    """Create data with clear downtrend (price crossing below MA)."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    # Start above MA, then cross below
    prices = [100, 101, 102, 103, 104, 105, 100, 95, 90, 85]
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
def sideways_data():
    """Create data with no clear trend."""
    dates = pd.date_range("2023-01-01", periods=10, freq="D")
    prices = [100, 100, 100, 100, 100, 100, 100, 100, 100, 100]
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 1 for p in prices],
            "low": [p - 1 for p in prices],
            "close": prices,
            "volume": [1000000] * 10,
        },
        index=dates,
    )
    return OHLCVSeries(ticker="TEST", data=df)


class TestMomentumStrategy:
    def test_init_default_lookback(self):
        strategy = MomentumStrategy()
        assert strategy.lookback_period == 20

    def test_init_custom_lookback(self):
        strategy = MomentumStrategy(lookback_period=10)
        assert strategy.lookback_period == 10

    def test_init_invalid_lookback(self):
        with pytest.raises(ValueError, match="lookback_period must be at least 1"):
            MomentumStrategy(lookback_period=0)

    def test_info_property(self, strategy):
        info = strategy.info
        assert info.name == "momentum"
        assert "momentum" in info.description.lower()
        assert "lookback_period" in info.parameters

    def test_generate_signals_insufficient_data(self, strategy):
        """Test that strategy returns empty list with insufficient data."""
        dates = pd.date_range("2023-01-01", periods=3, freq="D")
        df = pd.DataFrame(
            {
                "open": [100, 101, 102],
                "high": [102, 103, 104],
                "low": [99, 100, 101],
                "close": [101, 102, 103],
                "volume": [1000000, 1100000, 1200000],
            },
            index=dates,
        )
        data = OHLCVSeries(ticker="TEST", data=df)
        signals = strategy.generate_signals(data)
        assert signals == []

    def test_generate_buy_signal(self, strategy, uptrend_data):
        """Test that strategy generates buy signal on uptrend crossover."""
        signals = strategy.generate_signals(uptrend_data)
        buy_signals = [s for s in signals if s.signal == Signal.BUY]
        assert len(buy_signals) > 0
        assert all(s.ticker == "TEST" for s in buy_signals)

    def test_generate_sell_signal(self, strategy, downtrend_data):
        """Test that strategy generates sell signal on downtrend crossover."""
        signals = strategy.generate_signals(downtrend_data)
        sell_signals = [s for s in signals if s.signal == Signal.SELL]
        assert len(sell_signals) > 0
        assert all(s.ticker == "TEST" for s in sell_signals)

    def test_no_duplicate_signals(self, strategy, uptrend_data):
        """Test that consecutive signals of same type are not duplicated."""
        signals = strategy.generate_signals(uptrend_data)
        # Check no consecutive signals of same type
        for i in range(1, len(signals)):
            assert signals[i].signal != signals[i - 1].signal

    def test_signal_has_metadata(self, strategy, uptrend_data):
        """Test that signals include MA and lookback in metadata."""
        signals = strategy.generate_signals(uptrend_data)
        if signals:
            assert "ma" in signals[0].metadata
            assert "lookback_period" in signals[0].metadata
            assert signals[0].metadata["lookback_period"] == 5

    def test_signal_strength_bounded(self, strategy, uptrend_data):
        """Test that signal strength is between 0 and 1."""
        signals = strategy.generate_signals(uptrend_data)
        for signal in signals:
            assert 0 <= signal.strength <= 1

    def test_sideways_market_minimal_signals(self, strategy, sideways_data):
        """Test that sideways market generates few or no signals."""
        signals = strategy.generate_signals(sideways_data)
        # In a flat market with constant prices, there should be no crossovers
        assert len(signals) == 0
