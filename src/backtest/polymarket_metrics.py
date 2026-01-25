"""Polymarket-specific performance metrics.

These metrics are designed for prediction markets and include:
- Brier Score: Measures prediction accuracy (lower is better)
- Calibration Score: Measures prediction reliability
- Resolution Accuracy: % of markets resolved in our favor
- Edge Captured: Actual vs expected value captured
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.models.polymarket_schemas import (
    PolymarketBacktestResult,
    PolymarketTrade,
    TransactionType,
)

logger = logging.getLogger(__name__)


@dataclass
class PolymarketMetrics:
    """Collection of Polymarket-specific metrics."""

    # Standard metrics
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float

    # Polymarket-specific
    brier_score: float
    calibration_score: float
    resolution_accuracy_pct: float
    average_hold_duration_hours: float
    edge_captured_pct: float
    markets_traded: int
    win_count: int
    loss_count: int


def calculate_brier_score(
    predictions: list[tuple[float, bool]],
) -> float:
    """
    Calculate Brier Score for prediction accuracy.

    The Brier Score measures the accuracy of probabilistic predictions.
    Lower is better (0 = perfect, 1 = completely wrong).

    Args:
        predictions: List of (predicted_probability, actual_outcome) tuples
            - predicted_probability: Our probability estimate (0-1)
            - actual_outcome: True if YES won, False if NO won

    Returns:
        Brier Score (0-1, lower is better)
    """
    if not predictions:
        return 0.0

    total_squared_error = 0.0

    for prob, outcome in predictions:
        actual = 1.0 if outcome else 0.0
        squared_error = (prob - actual) ** 2
        total_squared_error += squared_error

    return total_squared_error / len(predictions)


def calculate_calibration_score(
    predictions: list[tuple[float, bool]],
    num_bins: int = 10,
) -> float:
    """
    Calculate calibration score measuring prediction reliability.

    A well-calibrated predictor should have:
    - Events predicted at 70% happening ~70% of the time
    - Events predicted at 30% happening ~30% of the time

    Args:
        predictions: List of (predicted_probability, actual_outcome) tuples
        num_bins: Number of probability bins for calibration

    Returns:
        Calibration score (0-1, higher is better)
    """
    if len(predictions) < num_bins:
        return 0.0

    # Group predictions into bins
    bin_size = 1.0 / num_bins
    bins = [[] for _ in range(num_bins)]

    for prob, outcome in predictions:
        bin_idx = min(int(prob / bin_size), num_bins - 1)
        bins[bin_idx].append((prob, outcome))

    # Calculate calibration error for each bin
    total_error = 0.0
    bins_used = 0

    for i, bin_predictions in enumerate(bins):
        if len(bin_predictions) < 2:
            continue

        bins_used += 1
        bin_center = (i + 0.5) * bin_size

        # Actual frequency in this bin
        actual_freq = sum(1 for _, o in bin_predictions if o) / len(bin_predictions)

        # Error is difference between expected and actual
        total_error += abs(bin_center - actual_freq)

    if bins_used == 0:
        return 0.0

    # Return as score (1 - normalized error)
    avg_error = total_error / bins_used
    return max(0.0, 1.0 - avg_error)


def calculate_average_hold_duration(
    trades: list[PolymarketTrade],
) -> float:
    """
    Calculate average position hold duration in hours.

    Args:
        trades: List of trades

    Returns:
        Average hold duration in hours
    """
    if len(trades) < 2:
        return 0.0

    # Group by market_id
    market_trades: dict[str, list[PolymarketTrade]] = {}
    for trade in trades:
        if trade.market_id not in market_trades:
            market_trades[trade.market_id] = []
        market_trades[trade.market_id].append(trade)

    total_duration = 0.0
    count = 0

    for market_id, market_trade_list in market_trades.items():
        # Sort by timestamp
        sorted_trades = sorted(market_trade_list, key=lambda t: t.timestamp)

        # Find buy/sell pairs
        for i, trade in enumerate(sorted_trades):
            if trade.side == TransactionType.BUY:
                # Look for matching sell
                for j in range(i + 1, len(sorted_trades)):
                    if sorted_trades[j].side == TransactionType.SELL:
                        duration = sorted_trades[j].timestamp - trade.timestamp
                        total_duration += duration.total_seconds() / 3600
                        count += 1
                        break

    return total_duration / count if count > 0 else 0.0


def calculate_edge_captured(
    trades: list[PolymarketTrade],
    resolutions: dict[str, bool],
) -> float:
    """
    Calculate edge captured as percentage of theoretical maximum.

    Edge = difference between entry price and fair value (resolution outcome)

    Args:
        trades: List of trades
        resolutions: Market resolutions (market_id -> won True/False)

    Returns:
        Edge captured as percentage
    """
    if not trades or not resolutions:
        return 0.0

    theoretical_edge = 0.0
    captured_edge = 0.0

    # Group buys by market
    market_buys: dict[str, list[PolymarketTrade]] = {}
    for trade in trades:
        if trade.side == TransactionType.BUY:
            if trade.market_id not in market_buys:
                market_buys[trade.market_id] = []
            market_buys[trade.market_id].append(trade)

    for market_id, buys in market_buys.items():
        if market_id not in resolutions:
            continue

        resolution = resolutions[market_id]
        fair_value = 1.0 if resolution else 0.0

        for buy in buys:
            # Edge is always fair_value - entry_price
            # Positive if we were on the winning side, negative if losing
            edge = fair_value - buy.price

            # Theoretical edge is max possible gain (amount weighted)
            theoretical_edge += abs(1.0 - buy.price) * buy.amount
            captured_edge += edge * buy.amount

    if theoretical_edge == 0:
        return 0.0

    return (captured_edge / theoretical_edge) * 100


def calculate_sharpe_ratio(
    portfolio_values: pd.Series,
    periods_per_year: int = 365,  # Daily for PM
) -> float:
    """
    Calculate annualized Sharpe ratio for Polymarket.

    Args:
        portfolio_values: Time series of portfolio values
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sharpe ratio
    """
    if len(portfolio_values) < 2:
        return 0.0

    returns = portfolio_values.pct_change().dropna()

    if returns.std() == 0:
        return 0.0

    sharpe = returns.mean() / returns.std()
    return sharpe * np.sqrt(periods_per_year)


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Calculate maximum drawdown.

    Args:
        portfolio_values: Time series of portfolio values

    Returns:
        Maximum drawdown as percentage
    """
    if len(portfolio_values) < 2:
        return 0.0

    running_max = portfolio_values.expanding().max()
    drawdowns = (running_max - portfolio_values) / running_max

    return drawdowns.max() * 100


def calculate_all_polymarket_metrics(
    result: PolymarketBacktestResult,
) -> PolymarketMetrics:
    """
    Calculate all Polymarket-specific metrics.

    Args:
        result: Backtest result

    Returns:
        PolymarketMetrics with all calculated values
    """
    # Build predictions list for Brier/calibration scores
    predictions: list[tuple[float, bool]] = []

    # Group trades by market to get entry prices
    market_entry_prices: dict[str, float] = {}
    for trade in result.trades:
        if trade.side == TransactionType.BUY:
            if trade.market_id not in market_entry_prices:
                market_entry_prices[trade.market_id] = trade.price

    # Build predictions from resolutions
    for market_id, won in result.resolutions.items():
        if market_id in market_entry_prices:
            entry_price = market_entry_prices[market_id]
            predictions.append((entry_price, won))

    # Calculate all metrics
    brier = calculate_brier_score(predictions)
    calibration = calculate_calibration_score(predictions)
    hold_duration = calculate_average_hold_duration(result.trades)
    edge = calculate_edge_captured(result.trades, result.resolutions)
    sharpe = calculate_sharpe_ratio(result.portfolio_values)
    max_dd = calculate_max_drawdown(result.portfolio_values)

    return PolymarketMetrics(
        total_return_pct=result.total_return_pct,
        sharpe_ratio=round(sharpe, 2),
        max_drawdown_pct=round(max_dd, 2),
        brier_score=round(brier, 4),
        calibration_score=round(calibration, 4),
        resolution_accuracy_pct=round(result.resolution_accuracy, 2),
        average_hold_duration_hours=round(hold_duration, 2),
        edge_captured_pct=round(edge, 2),
        markets_traded=result.markets_traded,
        win_count=result.win_count,
        loss_count=result.loss_count,
    )


def format_metrics_summary(metrics: PolymarketMetrics) -> str:
    """
    Format metrics as a human-readable summary.

    Args:
        metrics: Calculated metrics

    Returns:
        Formatted string summary
    """
    lines = [
        "=" * 50,
        "POLYMARKET BACKTEST RESULTS",
        "=" * 50,
        "",
        "PERFORMANCE",
        f"  Total Return:       {metrics.total_return_pct:+.2f}%",
        f"  Sharpe Ratio:       {metrics.sharpe_ratio:.2f}",
        f"  Max Drawdown:       {metrics.max_drawdown_pct:.2f}%",
        "",
        "PREDICTION QUALITY",
        f"  Brier Score:        {metrics.brier_score:.4f} (lower is better)",
        f"  Calibration:        {metrics.calibration_score:.2%}",
        f"  Resolution Accuracy: {metrics.resolution_accuracy_pct:.1f}%",
        "",
        "TRADING ACTIVITY",
        f"  Markets Traded:     {metrics.markets_traded}",
        f"  Wins/Losses:        {metrics.win_count}/{metrics.loss_count}",
        f"  Avg Hold Duration:  {metrics.average_hold_duration_hours:.1f} hours",
        f"  Edge Captured:      {metrics.edge_captured_pct:.1f}%",
        "",
        "=" * 50,
    ]

    return "\n".join(lines)


def compare_strategies(
    results: dict[str, PolymarketBacktestResult],
) -> pd.DataFrame:
    """
    Compare metrics across multiple strategy backtests.

    Args:
        results: Dict of strategy_name -> backtest result

    Returns:
        DataFrame with comparison
    """
    comparison_data = []

    for name, result in results.items():
        metrics = calculate_all_polymarket_metrics(result)
        comparison_data.append(
            {
                "Strategy": name,
                "Return %": metrics.total_return_pct,
                "Sharpe": metrics.sharpe_ratio,
                "Max DD %": metrics.max_drawdown_pct,
                "Brier": metrics.brier_score,
                "Calibration": metrics.calibration_score,
                "Resolution %": metrics.resolution_accuracy_pct,
                "Markets": metrics.markets_traded,
                "W/L": f"{metrics.win_count}/{metrics.loss_count}",
            }
        )

    return pd.DataFrame(comparison_data)
