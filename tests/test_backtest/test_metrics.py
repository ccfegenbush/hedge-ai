from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import (
    calculate_sharpe_ratio,
    calculate_max_drawdown,
    calculate_sortino_ratio,
    calculate_cagr,
    calculate_win_rate,
    calculate_profit_factor,
    calculate_all_metrics,
)
from src.models.schemas import BacktestResult, Trade, OrderSide


class TestSharpeRatio:
    def test_sharpe_ratio_positive_returns(self):
        """Test Sharpe ratio with positive average returns (with variance)."""
        returns = pd.Series([0.02, 0.01, 0.03, 0.01, 0.02])
        sharpe = calculate_sharpe_ratio(returns)
        # Positive average returns with variance should give positive Sharpe
        assert sharpe > 0

    def test_sharpe_ratio_negative_returns(self):
        """Test Sharpe ratio with negative average returns (with variance)."""
        returns = pd.Series([-0.02, -0.01, -0.03, -0.01, -0.02])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe < 0

    def test_sharpe_ratio_zero_std(self):
        """Test Sharpe ratio with zero standard deviation."""
        returns = pd.Series([0.0, 0.0, 0.0, 0.0])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_sharpe_ratio_insufficient_data(self):
        """Test Sharpe ratio with insufficient data."""
        returns = pd.Series([0.01])
        sharpe = calculate_sharpe_ratio(returns)
        assert sharpe == 0.0

    def test_sharpe_ratio_with_risk_free_rate(self):
        """Test Sharpe ratio with non-zero risk-free rate."""
        returns = pd.Series([0.02, 0.01, 0.03, 0.01, 0.02])
        sharpe_no_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.0)
        sharpe_with_rf = calculate_sharpe_ratio(returns, risk_free_rate=0.05)
        # Higher risk-free rate should lower Sharpe
        assert sharpe_with_rf < sharpe_no_rf


class TestMaxDrawdown:
    def test_max_drawdown_no_drawdown(self):
        """Test max drawdown with monotonically increasing values."""
        values = pd.Series([100, 110, 120, 130, 140])
        mdd = calculate_max_drawdown(values)
        assert mdd == 0.0

    def test_max_drawdown_simple_drawdown(self):
        """Test max drawdown with simple 20% drawdown."""
        values = pd.Series([100, 120, 100, 110])  # 20/120 = 16.67% drawdown
        mdd = calculate_max_drawdown(values)
        assert 0.16 < mdd < 0.17

    def test_max_drawdown_multiple_drawdowns(self):
        """Test max drawdown picks largest drawdown."""
        values = pd.Series([100, 110, 100, 120, 90])  # Two drawdowns
        mdd = calculate_max_drawdown(values)
        # Max drawdown is from 120 to 90 = 25%
        assert mdd == 0.25

    def test_max_drawdown_insufficient_data(self):
        """Test max drawdown with insufficient data."""
        values = pd.Series([100])
        mdd = calculate_max_drawdown(values)
        assert mdd == 0.0


class TestSortinoRatio:
    def test_sortino_ratio_only_positive_returns(self):
        """Test Sortino ratio with only positive returns."""
        returns = pd.Series([0.01, 0.02, 0.01, 0.03, 0.02])
        sortino = calculate_sortino_ratio(returns)
        # No downside, should be very high
        assert sortino == float("inf")

    def test_sortino_ratio_mixed_returns(self):
        """Test Sortino ratio with mixed returns."""
        returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.01])
        sortino = calculate_sortino_ratio(returns)
        # Should be positive with overall positive returns
        assert sortino > 0

    def test_sortino_ratio_insufficient_data(self):
        """Test Sortino ratio with insufficient data."""
        returns = pd.Series([0.01])
        sortino = calculate_sortino_ratio(returns)
        assert sortino == 0.0


class TestCAGR:
    def test_cagr_positive_growth(self):
        """Test CAGR with positive growth."""
        cagr = calculate_cagr(initial_value=100, final_value=200, years=5)
        # (200/100)^(1/5) - 1 = 0.1487 or about 14.87%
        assert 0.14 < cagr < 0.15

    def test_cagr_negative_growth(self):
        """Test CAGR with negative growth."""
        cagr = calculate_cagr(initial_value=100, final_value=50, years=5)
        assert cagr < 0

    def test_cagr_one_year(self):
        """Test CAGR over one year equals simple return."""
        cagr = calculate_cagr(initial_value=100, final_value=120, years=1)
        assert cagr == pytest.approx(0.2)

    def test_cagr_zero_years(self):
        """Test CAGR with zero years."""
        cagr = calculate_cagr(initial_value=100, final_value=120, years=0)
        assert cagr == 0.0

    def test_cagr_zero_initial(self):
        """Test CAGR with zero initial value."""
        cagr = calculate_cagr(initial_value=0, final_value=120, years=1)
        assert cagr == 0.0


class TestWinRate:
    @pytest.fixture
    def winning_trades(self):
        """Create winning round-trip trades."""
        return [
            Trade(
                ticker="TEST",
                side=OrderSide.BUY,
                quantity=10,
                price=100,
                timestamp=datetime(2023, 1, 1),
            ),
            Trade(
                ticker="TEST",
                side=OrderSide.SELL,
                quantity=10,
                price=120,
                timestamp=datetime(2023, 1, 5),
            ),
        ]

    @pytest.fixture
    def losing_trades(self):
        """Create losing round-trip trades."""
        return [
            Trade(
                ticker="TEST",
                side=OrderSide.BUY,
                quantity=10,
                price=100,
                timestamp=datetime(2023, 1, 1),
            ),
            Trade(
                ticker="TEST",
                side=OrderSide.SELL,
                quantity=10,
                price=80,
                timestamp=datetime(2023, 1, 5),
            ),
        ]

    def test_win_rate_all_winning(self, winning_trades):
        """Test win rate with all winning trades."""
        result = BacktestResult(
            initial_capital=10000,
            final_value=10200,
            trades=winning_trades,
            portfolio_values=pd.Series([10000, 10200]),
        )
        win_rate = calculate_win_rate(result)
        assert win_rate == 1.0

    def test_win_rate_all_losing(self, losing_trades):
        """Test win rate with all losing trades."""
        result = BacktestResult(
            initial_capital=10000,
            final_value=9800,
            trades=losing_trades,
            portfolio_values=pd.Series([10000, 9800]),
        )
        win_rate = calculate_win_rate(result)
        assert win_rate == 0.0

    def test_win_rate_no_trades(self):
        """Test win rate with no trades."""
        result = BacktestResult(
            initial_capital=10000,
            final_value=10000,
            trades=[],
            portfolio_values=pd.Series([10000]),
        )
        win_rate = calculate_win_rate(result)
        assert win_rate == 0.0


class TestProfitFactor:
    def test_profit_factor_all_profits(self):
        """Test profit factor with only profitable trades."""
        trades = [
            Trade("TEST", OrderSide.BUY, 10, 100, datetime(2023, 1, 1)),
            Trade("TEST", OrderSide.SELL, 10, 120, datetime(2023, 1, 5)),
        ]
        result = BacktestResult(
            initial_capital=10000,
            final_value=10200,
            trades=trades,
            portfolio_values=pd.Series([10000, 10200]),
        )
        pf = calculate_profit_factor(result)
        assert pf == float("inf")

    def test_profit_factor_mixed(self):
        """Test profit factor with mixed results."""
        trades = [
            Trade("TEST", OrderSide.BUY, 10, 100, datetime(2023, 1, 1)),
            Trade("TEST", OrderSide.SELL, 10, 120, datetime(2023, 1, 2)),  # +200
            Trade("TEST", OrderSide.BUY, 10, 120, datetime(2023, 1, 3)),
            Trade("TEST", OrderSide.SELL, 10, 110, datetime(2023, 1, 4)),  # -100
        ]
        result = BacktestResult(
            initial_capital=10000,
            final_value=10100,
            trades=trades,
            portfolio_values=pd.Series([10000, 10100]),
        )
        pf = calculate_profit_factor(result)
        assert pf == 2.0  # 200 / 100


class TestCalculateAllMetrics:
    def test_calculate_all_metrics_returns_dict(self):
        """Test that calculate_all_metrics returns expected keys."""
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        result = BacktestResult(
            initial_capital=10000,
            final_value=11000,
            trades=[],
            portfolio_values=pd.Series([10000, 10200, 10400, 10800, 11000], index=dates),
        )
        metrics = calculate_all_metrics(result)

        expected_keys = [
            "total_return_pct",
            "cagr",
            "sharpe_ratio",
            "sortino_ratio",
            "max_drawdown_pct",
            "win_rate_pct",
            "profit_factor",
            "num_trades",
        ]
        for key in expected_keys:
            assert key in metrics
