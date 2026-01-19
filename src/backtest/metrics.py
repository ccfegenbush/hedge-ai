import numpy as np
import pandas as pd

from src.models.schemas import BacktestResult


def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default: 0)
        periods_per_year: Number of trading periods per year (default: 252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    if excess_returns.std() == 0:
        return 0.0

    sharpe = excess_returns.mean() / excess_returns.std()
    # Annualize
    return sharpe * np.sqrt(periods_per_year)


def calculate_max_drawdown(portfolio_values: pd.Series) -> float:
    """
    Calculate maximum drawdown from portfolio values.

    Args:
        portfolio_values: Time series of portfolio values

    Returns:
        Maximum drawdown as a positive decimal (e.g., 0.20 for 20% drawdown)
    """
    if len(portfolio_values) < 2:
        return 0.0

    # Calculate running maximum
    running_max = portfolio_values.expanding().max()

    # Calculate drawdown at each point
    drawdowns = (running_max - portfolio_values) / running_max

    return drawdowns.max()


def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sortino ratio (uses downside deviation instead of std).

    Args:
        returns: Series of periodic returns
        risk_free_rate: Annual risk-free rate (default: 0)
        periods_per_year: Number of trading periods per year (default: 252 for daily)

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)

    # Calculate downside deviation (only negative returns)
    negative_returns = excess_returns[excess_returns < 0]
    if len(negative_returns) == 0:
        return float("inf") if excess_returns.mean() > 0 else 0.0

    downside_std = np.sqrt((negative_returns**2).mean())
    if downside_std == 0:
        return 0.0

    sortino = excess_returns.mean() / downside_std
    return sortino * np.sqrt(periods_per_year)


def calculate_win_rate(result: BacktestResult) -> float:
    """
    Calculate win rate from trades.

    Args:
        result: BacktestResult containing trades

    Returns:
        Win rate as decimal (e.g., 0.6 for 60%)
    """
    if len(result.trades) < 2:
        return 0.0

    # Group trades into round trips (buy + sell pairs)
    # Simple approach: calculate P&L for each pair of trades
    wins = 0
    total_round_trips = 0

    from src.models.schemas import OrderSide

    i = 0
    while i < len(result.trades) - 1:
        if result.trades[i].side == OrderSide.BUY:
            buy_trade = result.trades[i]
            # Look for matching sell
            for j in range(i + 1, len(result.trades)):
                if (
                    result.trades[j].side == OrderSide.SELL
                    and result.trades[j].ticker == buy_trade.ticker
                ):
                    sell_trade = result.trades[j]
                    pnl = (sell_trade.price - buy_trade.price) * min(
                        buy_trade.quantity, sell_trade.quantity
                    )
                    if pnl > 0:
                        wins += 1
                    total_round_trips += 1
                    break
        i += 1

    return wins / total_round_trips if total_round_trips > 0 else 0.0


def calculate_profit_factor(result: BacktestResult) -> float:
    """
    Calculate profit factor (gross profits / gross losses).

    Args:
        result: BacktestResult containing trades

    Returns:
        Profit factor (>1 is profitable)
    """
    if len(result.trades) < 2:
        return 0.0

    from src.models.schemas import OrderSide

    gross_profit = 0.0
    gross_loss = 0.0

    i = 0
    while i < len(result.trades) - 1:
        if result.trades[i].side == OrderSide.BUY:
            buy_trade = result.trades[i]
            for j in range(i + 1, len(result.trades)):
                if (
                    result.trades[j].side == OrderSide.SELL
                    and result.trades[j].ticker == buy_trade.ticker
                ):
                    sell_trade = result.trades[j]
                    pnl = (sell_trade.price - buy_trade.price) * min(
                        buy_trade.quantity, sell_trade.quantity
                    )
                    if pnl > 0:
                        gross_profit += pnl
                    else:
                        gross_loss += abs(pnl)
                    break
        i += 1

    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0

    return gross_profit / gross_loss


def calculate_cagr(
    initial_value: float,
    final_value: float,
    years: float,
) -> float:
    """
    Calculate Compound Annual Growth Rate.

    Args:
        initial_value: Starting portfolio value
        final_value: Ending portfolio value
        years: Number of years

    Returns:
        CAGR as decimal (e.g., 0.15 for 15%)
    """
    if years <= 0 or initial_value <= 0:
        return 0.0

    return (final_value / initial_value) ** (1 / years) - 1


def calculate_all_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
) -> dict:
    """
    Calculate all performance metrics for a backtest result.

    Args:
        result: BacktestResult from backtesting engine
        risk_free_rate: Annual risk-free rate for Sharpe calculation

    Returns:
        Dictionary of metric name to value
    """
    # Calculate daily returns
    returns = result.portfolio_values.pct_change().dropna()

    # Calculate time period in years
    if len(result.portfolio_values) > 1:
        start_date = result.portfolio_values.index[0]
        end_date = result.portfolio_values.index[-1]
        years = (end_date - start_date).days / 365.25
    else:
        years = 0

    return {
        "total_return_pct": result.total_return_pct,
        "cagr": calculate_cagr(result.initial_capital, result.final_value, years) * 100,
        "sharpe_ratio": calculate_sharpe_ratio(returns, risk_free_rate),
        "sortino_ratio": calculate_sortino_ratio(returns, risk_free_rate),
        "max_drawdown_pct": calculate_max_drawdown(result.portfolio_values) * 100,
        "win_rate_pct": calculate_win_rate(result) * 100,
        "profit_factor": calculate_profit_factor(result),
        "num_trades": result.num_trades,
    }
