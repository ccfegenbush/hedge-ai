"""Base class for Polymarket trading strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.models.polymarket_schemas import (
    CompositeSignal,
    Market,
    PriceHistory,
    SignalComponent,
    SignalDirection,
)


@dataclass
class PolymarketStrategyInfo:
    """Metadata about a Polymarket strategy."""

    name: str
    description: str
    signal_sources: list[str]  # e.g., ["sentiment", "whale", "momentum"]
    parameters: dict[str, str]  # param_name -> description


class PolymarketStrategy(ABC):
    """
    Abstract base class for Polymarket trading strategies.

    Unlike stock strategies that work with OHLCV data, Polymarket strategies
    generate signals based on sentiment, whale activity, momentum, or
    combinations thereof.
    """

    @property
    @abstractmethod
    def info(self) -> PolymarketStrategyInfo:
        """Return metadata about the strategy."""
        pass

    @abstractmethod
    async def generate_signal(
        self,
        market: Market,
        price_history: Optional[PriceHistory] = None,
        timestamp: Optional[datetime] = None,
    ) -> CompositeSignal:
        """
        Generate a trading signal for a market.

        Args:
            market: The market to analyze
            price_history: Optional price history for momentum analysis
            timestamp: Analysis timestamp (defaults to now)

        Returns:
            CompositeSignal with direction, strength, and components
        """
        pass

    async def generate_signals_batch(
        self,
        markets: list[Market],
        price_histories: Optional[dict[str, PriceHistory]] = None,
    ) -> list[CompositeSignal]:
        """
        Generate signals for multiple markets.

        Args:
            markets: List of markets to analyze
            price_histories: Optional dict of condition_id -> PriceHistory

        Returns:
            List of CompositeSignal objects
        """
        signals = []
        for market in markets:
            history = None
            if price_histories and market.condition_id in price_histories:
                history = price_histories[market.condition_id]

            signal = await self.generate_signal(market, history)
            signals.append(signal)

        return signals

    def _create_signal(
        self,
        market_id: str,
        components: list[SignalComponent],
        timestamp: Optional[datetime] = None,
        metadata: Optional[dict] = None,
    ) -> CompositeSignal:
        """
        Helper to create a CompositeSignal from components.

        Determines direction and strength from components.
        """
        if not timestamp:
            timestamp = datetime.now()

        if not metadata:
            metadata = {}

        # Calculate weighted average of component values
        total_weight = 0.0
        weighted_sum = 0.0

        for comp in components:
            weighted_sum += comp.value * comp.confidence
            total_weight += comp.confidence

        if total_weight > 0:
            avg_value = weighted_sum / total_weight
        else:
            avg_value = 0.0

        # Determine direction
        if avg_value > 0.1:
            direction = SignalDirection.BUY
        elif avg_value < -0.1:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.HOLD

        # Strength is the absolute value, scaled by confidence
        strength = min(1.0, abs(avg_value) * (total_weight / len(components)))

        return CompositeSignal(
            market_id=market_id,
            direction=direction,
            strength=round(strength, 4),
            components=components,
            timestamp=timestamp,
            metadata=metadata,
        )

    def _normalize_value(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize a value to the -1 to 1 range."""
        if max_val == min_val:
            return 0.0
        normalized = 2 * (value - min_val) / (max_val - min_val) - 1
        return max(-1.0, min(1.0, normalized))
