"""Tests for Polymarket-specific metrics."""

from datetime import datetime, timedelta

import pandas as pd
import pytest

from src.backtest.polymarket_metrics import (
    PolymarketMetrics,
    calculate_all_polymarket_metrics,
    calculate_average_hold_duration,
    calculate_brier_score,
    calculate_calibration_score,
    calculate_edge_captured,
    calculate_max_drawdown,
    calculate_sharpe_ratio,
    compare_strategies,
    format_metrics_summary,
)
from src.models.polymarket_schemas import (
    PolymarketBacktestResult,
    PolymarketTrade,
    TransactionType,
)


class TestBrierScore:
    """Tests for Brier score calculation."""

    def test_perfect_predictions(self):
        """Test Brier score with perfect predictions."""
        predictions = [
            (1.0, True),  # Predicted 100%, happened
            (0.0, False),  # Predicted 0%, didn't happen
        ]
        score = calculate_brier_score(predictions)
        assert score == 0.0

    def test_worst_predictions(self):
        """Test Brier score with completely wrong predictions."""
        predictions = [
            (1.0, False),  # Predicted 100%, didn't happen
            (0.0, True),  # Predicted 0%, happened
        ]
        score = calculate_brier_score(predictions)
        assert score == 1.0

    def test_moderate_predictions(self):
        """Test Brier score with moderate predictions."""
        predictions = [
            (0.7, True),  # Predicted 70%, happened
            (0.3, False),  # Predicted 30%, didn't happen
        ]
        score = calculate_brier_score(predictions)
        # (0.7-1)^2 + (0.3-0)^2 = 0.09 + 0.09 = 0.18, /2 = 0.09
        assert abs(score - 0.09) < 0.01

    def test_empty_predictions(self):
        """Test Brier score with no predictions."""
        score = calculate_brier_score([])
        assert score == 0.0


class TestCalibrationScore:
    """Tests for calibration score calculation."""

    def test_perfect_calibration(self):
        """Test calibration with perfectly calibrated predictions."""
        # 10 predictions at 80% that hit 80% of the time
        predictions = []
        for i in range(100):
            if i < 80:
                predictions.append((0.8, True))
            else:
                predictions.append((0.8, False))

        score = calculate_calibration_score(predictions, num_bins=5)
        # Should be close to 1.0 (perfect calibration)
        assert score > 0.8

    def test_poor_calibration(self):
        """Test calibration with poorly calibrated predictions."""
        # All predictions at 90% but only 10% happen
        predictions = [(0.9, i < 10) for i in range(100)]

        score = calculate_calibration_score(predictions, num_bins=5)
        # Should be lower due to poor calibration
        assert score < 0.5

    def test_too_few_predictions(self):
        """Test calibration with too few predictions."""
        predictions = [(0.5, True), (0.5, False)]
        score = calculate_calibration_score(predictions, num_bins=10)
        assert score == 0.0


class TestHoldDuration:
    """Tests for average hold duration calculation."""

    def test_simple_round_trip(self):
        """Test with a single buy/sell pair."""
        now = datetime.now()
        trades = [
            PolymarketTrade(
                market_id="m1",
                outcome_id="yes",
                side=TransactionType.BUY,
                amount=1000,
                price=0.5,
                timestamp=now,
                signal_strength=0.8,
            ),
            PolymarketTrade(
                market_id="m1",
                outcome_id="yes",
                side=TransactionType.SELL,
                amount=1000,
                price=0.6,
                timestamp=now + timedelta(hours=24),
                signal_strength=0.7,
            ),
        ]

        duration = calculate_average_hold_duration(trades)
        assert abs(duration - 24.0) < 0.1

    def test_no_trades(self):
        """Test with no trades."""
        duration = calculate_average_hold_duration([])
        assert duration == 0.0

    def test_only_buys(self):
        """Test with only buy trades (no sells)."""
        now = datetime.now()
        trades = [
            PolymarketTrade(
                market_id="m1",
                outcome_id="yes",
                side=TransactionType.BUY,
                amount=1000,
                price=0.5,
                timestamp=now,
                signal_strength=0.8,
            ),
        ]

        duration = calculate_average_hold_duration(trades)
        assert duration == 0.0


class TestEdgeCaptured:
    """Tests for edge captured calculation."""

    def test_winning_trade(self):
        """Test edge captured on a winning trade."""
        trades = [
            PolymarketTrade(
                market_id="m1",
                outcome_id="yes",
                side=TransactionType.BUY,
                amount=100,
                price=0.6,  # Bought at 60%
                timestamp=datetime.now(),
                signal_strength=0.8,
            ),
        ]
        resolutions = {"m1": True}  # Resolved YES (we won)

        edge = calculate_edge_captured(trades, resolutions)
        # We bought at 0.6, resolved to 1.0, edge = 0.4
        assert edge > 0

    def test_losing_trade(self):
        """Test edge captured on a losing trade."""
        trades = [
            PolymarketTrade(
                market_id="m1",
                outcome_id="yes",
                side=TransactionType.BUY,
                amount=100,
                price=0.6,
                timestamp=datetime.now(),
                signal_strength=0.8,
            ),
        ]
        resolutions = {"m1": False}  # Resolved NO (we lost)

        edge = calculate_edge_captured(trades, resolutions)
        # Negative edge since we lost
        assert edge < 0


class TestSharpeRatio:
    """Tests for Sharpe ratio calculation."""

    def test_constant_returns(self):
        """Test Sharpe with constant returns (zero std)."""
        values = pd.Series([100, 100, 100, 100, 100])
        sharpe = calculate_sharpe_ratio(values)
        assert sharpe == 0.0

    def test_positive_returns(self):
        """Test Sharpe with positive returns."""
        values = pd.Series([100, 101, 102, 103, 104, 105])
        sharpe = calculate_sharpe_ratio(values)
        assert sharpe > 0

    def test_volatile_returns(self):
        """Test Sharpe with volatile returns."""
        values = pd.Series([100, 110, 90, 115, 85, 120])
        sharpe = calculate_sharpe_ratio(values)
        # Lower due to volatility
        assert isinstance(sharpe, float)


class TestMaxDrawdown:
    """Tests for max drawdown calculation."""

    def test_no_drawdown(self):
        """Test with monotonically increasing values."""
        values = pd.Series([100, 101, 102, 103, 104])
        dd = calculate_max_drawdown(values)
        assert dd == 0.0

    def test_simple_drawdown(self):
        """Test with a simple drawdown."""
        values = pd.Series([100, 110, 90, 100])  # 20% drawdown from 110 to 90
        dd = calculate_max_drawdown(values)
        assert abs(dd - 18.18) < 1  # ~18.18% drawdown


class TestCalculateAllMetrics:
    """Tests for the all-metrics function."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample backtest result."""
        now = datetime.now()
        return PolymarketBacktestResult(
            initial_capital=10000,
            final_value=11000,
            trades=[
                PolymarketTrade(
                    market_id="m1",
                    outcome_id="yes",
                    side=TransactionType.BUY,
                    amount=1000,
                    price=0.6,
                    timestamp=now - timedelta(days=5),
                    signal_strength=0.8,
                ),
                PolymarketTrade(
                    market_id="m1",
                    outcome_id="yes",
                    side=TransactionType.SELL,
                    amount=1000,
                    price=0.7,
                    timestamp=now,
                    signal_strength=0.7,
                ),
            ],
            portfolio_values=pd.Series({
                now - timedelta(days=5): 10000,
                now - timedelta(days=4): 10100,
                now - timedelta(days=3): 10200,
                now - timedelta(days=2): 10150,
                now - timedelta(days=1): 10800,
                now: 11000,
            }),
            markets_traded=1,
            resolutions={"m1": True},
        )

    def test_returns_metrics_object(self, sample_result):
        """Test that function returns PolymarketMetrics."""
        metrics = calculate_all_polymarket_metrics(sample_result)
        assert isinstance(metrics, PolymarketMetrics)

    def test_total_return(self, sample_result):
        """Test total return calculation."""
        metrics = calculate_all_polymarket_metrics(sample_result)
        # 10000 -> 11000 = 10% return
        assert abs(metrics.total_return_pct - 10.0) < 0.1

    def test_win_count(self, sample_result):
        """Test win/loss counts."""
        metrics = calculate_all_polymarket_metrics(sample_result)
        assert metrics.win_count == 1
        assert metrics.loss_count == 0


class TestFormatMetricsSummary:
    """Tests for the metrics summary formatter."""

    def test_format_summary(self):
        """Test that summary formats correctly."""
        metrics = PolymarketMetrics(
            total_return_pct=10.5,
            sharpe_ratio=1.5,
            max_drawdown_pct=5.0,
            brier_score=0.15,
            calibration_score=0.85,
            resolution_accuracy_pct=70.0,
            average_hold_duration_hours=48.5,
            edge_captured_pct=25.0,
            markets_traded=5,
            win_count=4,
            loss_count=1,
        )

        summary = format_metrics_summary(metrics)

        assert "10.5" in summary or "10.50" in summary
        assert "1.5" in summary or "1.50" in summary
        assert "Brier" in summary
        assert "Calibration" in summary


class TestCompareStrategies:
    """Tests for strategy comparison."""

    def test_compare_two_strategies(self):
        """Test comparing two strategy results."""
        now = datetime.now()

        result1 = PolymarketBacktestResult(
            initial_capital=10000,
            final_value=11000,
            trades=[],
            portfolio_values=pd.Series({now: 11000}),
            markets_traded=1,
            resolutions={},
        )

        result2 = PolymarketBacktestResult(
            initial_capital=10000,
            final_value=12000,
            trades=[],
            portfolio_values=pd.Series({now: 12000}),
            markets_traded=2,
            resolutions={},
        )

        comparison = compare_strategies({
            "Strategy A": result1,
            "Strategy B": result2,
        })

        assert len(comparison) == 2
        assert "Strategy" in comparison.columns
        assert "Return %" in comparison.columns
