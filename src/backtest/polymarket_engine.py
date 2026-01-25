"""Polymarket-specific backtesting engine.

Unlike stock backtesting, Polymarket has unique characteristics:
- Binary outcomes (markets resolve to 0 or 1)
- Time-limited markets with expiration dates
- Resolution timing considerations
- Prediction-specific metrics (Brier score, calibration)
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.data.polymarket_provider import PolymarketProvider
from src.models.polymarket_schemas import (
    Market,
    MarketStatus,
    PolymarketBacktestResult,
    PolymarketPosition,
    PolymarketTrade,
    PriceHistory,
    SignalDirection,
    TransactionType,
)
from src.strategies.polymarket_base import PolymarketStrategy

logger = logging.getLogger(__name__)


@dataclass
class PolymarketPortfolio:
    """Portfolio state for Polymarket backtesting."""

    cash: float
    positions: dict[str, PolymarketPosition] = field(default_factory=dict)

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions at current cost)."""
        positions_value = sum(
            pos.tokens * pos.avg_cost for pos in self.positions.values()
        )
        return self.cash + positions_value

    def value_at_prices(self, prices: dict[str, float]) -> float:
        """Calculate portfolio value at given market prices."""
        value = self.cash
        for market_id, position in self.positions.items():
            if market_id in prices:
                value += position.tokens * prices[market_id]
            else:
                value += position.cost_basis
        return value


class PolymarketBacktestEngine:
    """
    Backtesting engine for Polymarket prediction markets.

    Key differences from stock backtesting:
    - Trades in outcome tokens (YES/NO tokens)
    - Markets resolve to 0 or 1
    - Position value depends on resolution outcome
    - Simulates realistic trading conditions
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        max_position_pct: float = 0.1,
        min_signal_strength: float = 0.3,
        slippage_bps: float = 10.0,  # Basis points of slippage
    ):
        """
        Initialize the backtest engine.

        Args:
            initial_capital: Starting cash in USDC
            max_position_pct: Maximum position size as % of portfolio
            min_signal_strength: Minimum signal strength to trade
            slippage_bps: Slippage in basis points (0.01% = 1 bp)
        """
        self.initial_capital = initial_capital
        self.max_position_pct = max_position_pct
        self.min_signal_strength = min_signal_strength
        self.slippage_bps = slippage_bps

    async def run(
        self,
        strategy: PolymarketStrategy,
        markets: list[Market],
        price_histories: dict[str, PriceHistory],
        start_date: datetime,
        end_date: datetime,
        signal_interval_hours: int = 24,
    ) -> PolymarketBacktestResult:
        """
        Run a backtest over the specified period.

        Args:
            strategy: The strategy to backtest
            markets: List of markets to trade
            price_histories: Price history for each market
            start_date: Backtest start date
            end_date: Backtest end date
            signal_interval_hours: How often to generate signals

        Returns:
            PolymarketBacktestResult with performance data
        """
        logger.info(
            f"Starting Polymarket backtest from {start_date} to {end_date} "
            f"on {len(markets)} markets"
        )

        # Initialize portfolio
        portfolio = PolymarketPortfolio(cash=self.initial_capital)
        trades: list[PolymarketTrade] = []
        portfolio_values: dict[datetime, float] = {}
        resolutions: dict[str, bool] = {}

        # Create market lookup
        market_lookup = {m.condition_id: m for m in markets}

        # Create price lookup for each market
        price_lookup: dict[str, dict[datetime, float]] = {}
        for market_id, history in price_histories.items():
            if len(history.data) > 0:
                price_lookup[market_id] = {}
                for _, row in history.data.iterrows():
                    ts = row["timestamp"]
                    if isinstance(ts, pd.Timestamp):
                        ts = ts.to_pydatetime()
                    price_lookup[market_id][ts] = row["price"]

        # Simulate trading at each interval
        current_time = start_date
        interval = timedelta(hours=signal_interval_hours)

        while current_time <= end_date:
            # Get current prices
            current_prices = self._get_prices_at_time(price_lookup, current_time)

            # Record portfolio value
            portfolio_values[current_time] = portfolio.value_at_prices(current_prices)

            # Generate signals and trade for each active market
            for market in markets:
                market_id = market.condition_id

                # Skip if market is already resolved
                if market_id in resolutions:
                    continue

                # Skip if market has ended
                if market.end_date and current_time > market.end_date:
                    # Resolve the market
                    resolution = self._simulate_resolution(market, portfolio)
                    if resolution is not None:
                        resolutions[market_id] = resolution
                    continue

                # Get price history up to current time
                if market_id in price_histories:
                    filtered_history = self._filter_history(
                        price_histories[market_id], current_time
                    )
                else:
                    filtered_history = None

                # Generate signal
                signal = await strategy.generate_signal(
                    market, filtered_history, current_time
                )

                # Process signal if strong enough
                if signal.strength >= self.min_signal_strength:
                    trade = self._execute_trade(
                        portfolio=portfolio,
                        market=market,
                        signal_direction=signal.direction,
                        signal_strength=signal.strength,
                        current_price=current_prices.get(market_id, 0.5),
                        timestamp=current_time,
                    )
                    if trade:
                        trades.append(trade)

            current_time += interval

        # Resolve any remaining open positions at end date
        for market in markets:
            market_id = market.condition_id
            if market_id not in resolutions and market_id in portfolio.positions:
                resolution = self._simulate_resolution(market, portfolio)
                if resolution is not None:
                    resolutions[market_id] = resolution

        # Calculate final value (all positions resolved)
        final_value = portfolio.cash

        # Count unique markets traded
        markets_traded = len(set(t.market_id for t in trades))

        logger.info(
            f"Backtest complete: {len(trades)} trades, "
            f"{markets_traded} markets traded, "
            f"final value: ${final_value:.2f}"
        )

        return PolymarketBacktestResult(
            initial_capital=self.initial_capital,
            final_value=final_value,
            trades=trades,
            portfolio_values=pd.Series(portfolio_values),
            markets_traded=markets_traded,
            resolutions=resolutions,
        )

    def _get_prices_at_time(
        self,
        price_lookup: dict[str, dict[datetime, float]],
        timestamp: datetime,
    ) -> dict[str, float]:
        """Get prices for all markets at a specific time."""
        prices = {}
        for market_id, market_prices in price_lookup.items():
            # Find closest price at or before timestamp
            closest_time = None
            closest_price = None
            for t, p in market_prices.items():
                if t <= timestamp:
                    if closest_time is None or t > closest_time:
                        closest_time = t
                        closest_price = p
            if closest_price is not None:
                prices[market_id] = closest_price
        return prices

    def _filter_history(
        self,
        history: PriceHistory,
        end_time: datetime,
    ) -> PriceHistory:
        """Filter price history to only include data up to end_time."""
        filtered_data = history.data[history.data["timestamp"] <= end_time]
        return PriceHistory(
            condition_id=history.condition_id,
            outcome_id=history.outcome_id,
            data=filtered_data,
        )

    def _execute_trade(
        self,
        portfolio: PolymarketPortfolio,
        market: Market,
        signal_direction: SignalDirection,
        signal_strength: float,
        current_price: float,
        timestamp: datetime,
    ) -> Optional[PolymarketTrade]:
        """Execute a trade based on signal."""
        market_id = market.condition_id
        outcome_id = market.outcomes[0].token_id if market.outcomes else "yes"

        # Apply slippage
        slippage_factor = 1 + (self.slippage_bps / 10000)

        if signal_direction == SignalDirection.BUY:
            # Check if we already have a position
            if market_id in portfolio.positions:
                return None  # Don't add to existing position

            # Calculate position size
            max_amount = portfolio.cash * self.max_position_pct
            amount = max_amount * signal_strength

            if amount < 10:  # Minimum trade size
                return None

            # Apply slippage (pay more)
            execution_price = min(0.99, current_price * slippage_factor)

            # Calculate tokens received
            tokens = amount / execution_price

            # Update portfolio
            portfolio.cash -= amount
            portfolio.positions[market_id] = PolymarketPosition(
                market_id=market_id,
                outcome_id=outcome_id,
                tokens=tokens,
                avg_cost=execution_price,
                entry_time=timestamp,
            )

            return PolymarketTrade(
                market_id=market_id,
                outcome_id=outcome_id,
                side=TransactionType.BUY,
                amount=amount,
                price=execution_price,
                timestamp=timestamp,
                signal_strength=signal_strength,
            )

        elif signal_direction == SignalDirection.SELL:
            # Check if we have a position to sell
            if market_id not in portfolio.positions:
                return None

            position = portfolio.positions[market_id]

            # Apply slippage (receive less)
            execution_price = max(0.01, current_price / slippage_factor)

            # Calculate proceeds
            proceeds = position.tokens * execution_price

            # Update portfolio
            portfolio.cash += proceeds
            del portfolio.positions[market_id]

            return PolymarketTrade(
                market_id=market_id,
                outcome_id=outcome_id,
                side=TransactionType.SELL,
                amount=proceeds,
                price=execution_price,
                timestamp=timestamp,
                signal_strength=signal_strength,
            )

        return None

    def _simulate_resolution(
        self,
        market: Market,
        portfolio: PolymarketPortfolio,
    ) -> Optional[bool]:
        """
        Simulate market resolution for backtesting.

        In a real backtest with historical data, we would use actual resolutions.
        For simulation, we determine outcome based on final price.
        """
        market_id = market.condition_id

        if market_id not in portfolio.positions:
            return None

        position = portfolio.positions[market_id]

        # For simulation: if we bought YES at price < 0.5, assume we won
        # This is a simplified simulation - real backtest would use actual outcomes
        won = position.avg_cost < 0.5

        if won:
            # Position resolves to $1 per token
            payout = position.tokens * 1.0
        else:
            # Position resolves to $0
            payout = 0.0

        portfolio.cash += payout
        del portfolio.positions[market_id]

        return won


async def run_polymarket_backtest(
    strategy: PolymarketStrategy,
    markets: list[Market],
    price_histories: dict[str, PriceHistory],
    start_date: datetime,
    end_date: datetime,
    initial_capital: float = 10000.0,
    max_position_pct: float = 0.1,
    min_signal_strength: float = 0.3,
) -> PolymarketBacktestResult:
    """
    Convenience function to run a Polymarket backtest.

    Args:
        strategy: Strategy to backtest
        markets: Markets to trade
        price_histories: Historical prices
        start_date: Start date
        end_date: End date
        initial_capital: Starting capital
        max_position_pct: Max position size %
        min_signal_strength: Min signal to trade

    Returns:
        Backtest results
    """
    engine = PolymarketBacktestEngine(
        initial_capital=initial_capital,
        max_position_pct=max_position_pct,
        min_signal_strength=min_signal_strength,
    )

    return await engine.run(
        strategy=strategy,
        markets=markets,
        price_histories=price_histories,
        start_date=start_date,
        end_date=end_date,
    )
