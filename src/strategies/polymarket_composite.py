"""Composite strategy combining multiple signal sources for Polymarket."""

import logging
from datetime import datetime
from enum import Enum
from typing import Optional

from src.data.polymarket_provider import PolymarketProvider
from src.data.sentiment_provider import SentimentProvider, get_sentiment_provider
from src.data.whale_tracker import WhaleTracker, get_whale_tracker
from src.models.polymarket_schemas import (
    CompositeSignal,
    Market,
    PriceHistory,
    SignalComponent,
    SignalDirection,
)
from src.strategies.polymarket_base import PolymarketStrategy, PolymarketStrategyInfo
from src.strategies.polymarket_momentum import MomentumStrategy
from src.strategies.polymarket_sentiment import SentimentStrategy
from src.strategies.polymarket_whale import WhaleStrategy

logger = logging.getLogger(__name__)


class CombinationMethod(Enum):
    """Methods for combining signals from multiple sources."""

    WEIGHTED = "weighted"  # Weighted average of signals
    UNANIMOUS = "unanimous"  # All signals must agree
    MAJORITY = "majority"  # Majority of signals must agree


class CompositeStrategy(PolymarketStrategy):
    """
    Combined strategy using sentiment, whale tracking, and momentum signals.

    Allows configurable weights and combination methods for flexible
    strategy composition.
    """

    def __init__(
        self,
        # Strategy weights (must sum to 1.0 or be normalized)
        sentiment_weight: float = 0.3,
        whale_weight: float = 0.4,
        momentum_weight: float = 0.3,
        # Combination method
        combination_method: CombinationMethod = CombinationMethod.WEIGHTED,
        # Signal thresholds
        min_signal_strength: float = 0.2,
        min_confidence: float = 0.3,
        # Data providers
        sentiment_provider: Optional[SentimentProvider] = None,
        whale_tracker: Optional[WhaleTracker] = None,
        polymarket_provider: Optional[PolymarketProvider] = None,
        # Sub-strategy parameters
        sentiment_lookback_hours: int = 24,
        whale_lookback_hours: int = 24,
        momentum_period: int = 24,
    ):
        """
        Initialize the composite strategy.

        Args:
            sentiment_weight: Weight for sentiment signals (0-1)
            whale_weight: Weight for whale signals (0-1)
            momentum_weight: Weight for momentum signals (0-1)
            combination_method: How to combine signals
            min_signal_strength: Minimum strength to trade
            min_confidence: Minimum average confidence to trade
            sentiment_provider: Provider for sentiment data
            whale_tracker: Provider for whale data
            polymarket_provider: Provider for market data
            sentiment_lookback_hours: Lookback for sentiment
            whale_lookback_hours: Lookback for whale activity
            momentum_period: Period for momentum calculation
        """
        # Normalize weights to sum to 1.0
        total_weight = sentiment_weight + whale_weight + momentum_weight
        if total_weight > 0:
            self.sentiment_weight = sentiment_weight / total_weight
            self.whale_weight = whale_weight / total_weight
            self.momentum_weight = momentum_weight / total_weight
        else:
            self.sentiment_weight = 1 / 3
            self.whale_weight = 1 / 3
            self.momentum_weight = 1 / 3

        self.combination_method = combination_method
        self.min_signal_strength = min_signal_strength
        self.min_confidence = min_confidence

        # Initialize sub-strategies
        self.sentiment_strategy = SentimentStrategy(
            sentiment_provider=sentiment_provider or get_sentiment_provider(use_mock=True),
            lookback_hours=sentiment_lookback_hours,
        )

        self.whale_strategy = WhaleStrategy(
            whale_tracker=whale_tracker or get_whale_tracker(use_mock=True),
            lookback_hours=whale_lookback_hours,
        )

        self.momentum_strategy = MomentumStrategy(
            polymarket_provider=polymarket_provider,
            momentum_period=momentum_period,
        )

    @property
    def info(self) -> PolymarketStrategyInfo:
        return PolymarketStrategyInfo(
            name="composite",
            description="Combined strategy using sentiment, whale tracking, and momentum",
            signal_sources=["sentiment", "whale", "momentum"],
            parameters={
                "sentiment_weight": f"Weight for sentiment signals ({self.sentiment_weight:.2f})",
                "whale_weight": f"Weight for whale signals ({self.whale_weight:.2f})",
                "momentum_weight": f"Weight for momentum signals ({self.momentum_weight:.2f})",
                "combination_method": self.combination_method.value,
                "min_signal_strength": str(self.min_signal_strength),
                "min_confidence": str(self.min_confidence),
            },
        )

    async def generate_signal(
        self,
        market: Market,
        price_history: Optional[PriceHistory] = None,
        timestamp: Optional[datetime] = None,
    ) -> CompositeSignal:
        """Generate a combined signal from all sources."""
        if not timestamp:
            timestamp = datetime.now()

        market_id = market.condition_id

        # Gather signals from all sub-strategies
        sentiment_signal = await self.sentiment_strategy.generate_signal(
            market, price_history, timestamp
        )
        whale_signal = await self.whale_strategy.generate_signal(
            market, price_history, timestamp
        )
        momentum_signal = await self.momentum_strategy.generate_signal(
            market, price_history, timestamp
        )

        # Extract components
        sentiment_comp = sentiment_signal.components[0] if sentiment_signal.components else None
        whale_comp = whale_signal.components[0] if whale_signal.components else None
        momentum_comp = momentum_signal.components[0] if momentum_signal.components else None

        # Combine based on method
        if self.combination_method == CombinationMethod.WEIGHTED:
            combined_signal = self._combine_weighted(
                sentiment_comp, whale_comp, momentum_comp
            )
        elif self.combination_method == CombinationMethod.UNANIMOUS:
            combined_signal = self._combine_unanimous(
                sentiment_comp, whale_comp, momentum_comp
            )
        else:  # MAJORITY
            combined_signal = self._combine_majority(
                sentiment_comp, whale_comp, momentum_comp
            )

        # Build components list with weights applied
        components = []
        if sentiment_comp:
            components.append(
                SignalComponent(
                    source="sentiment",
                    value=sentiment_comp.value,
                    confidence=sentiment_comp.confidence,
                    metadata={
                        "weight": self.sentiment_weight,
                        **sentiment_comp.metadata,
                    },
                )
            )
        if whale_comp:
            components.append(
                SignalComponent(
                    source="whale",
                    value=whale_comp.value,
                    confidence=whale_comp.confidence,
                    metadata={
                        "weight": self.whale_weight,
                        **whale_comp.metadata,
                    },
                )
            )
        if momentum_comp:
            components.append(
                SignalComponent(
                    source="momentum",
                    value=momentum_comp.value,
                    confidence=momentum_comp.confidence,
                    metadata={
                        "weight": self.momentum_weight,
                        **momentum_comp.metadata,
                    },
                )
            )

        # Determine direction and strength
        direction, strength = combined_signal

        # Apply minimum thresholds
        avg_confidence = (
            sum(c.confidence for c in components) / len(components)
            if components
            else 0.0
        )

        if strength < self.min_signal_strength or avg_confidence < self.min_confidence:
            direction = SignalDirection.HOLD
            strength = 0.0

        return CompositeSignal(
            market_id=market_id,
            direction=direction,
            strength=round(strength, 4),
            components=components,
            timestamp=timestamp,
            metadata={
                "strategy": "composite",
                "combination_method": self.combination_method.value,
                "weights": {
                    "sentiment": self.sentiment_weight,
                    "whale": self.whale_weight,
                    "momentum": self.momentum_weight,
                },
                "thresholds": {
                    "min_strength": self.min_signal_strength,
                    "min_confidence": self.min_confidence,
                },
            },
        )

    def _combine_weighted(
        self,
        sentiment: Optional[SignalComponent],
        whale: Optional[SignalComponent],
        momentum: Optional[SignalComponent],
    ) -> tuple[SignalDirection, float]:
        """Combine signals using weighted average."""
        weighted_sum = 0.0
        total_weight = 0.0

        if sentiment:
            weighted_sum += sentiment.value * self.sentiment_weight * sentiment.confidence
            total_weight += self.sentiment_weight * sentiment.confidence

        if whale:
            weighted_sum += whale.value * self.whale_weight * whale.confidence
            total_weight += self.whale_weight * whale.confidence

        if momentum:
            weighted_sum += momentum.value * self.momentum_weight * momentum.confidence
            total_weight += self.momentum_weight * momentum.confidence

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

        strength = abs(avg_value)

        return direction, strength

    def _combine_unanimous(
        self,
        sentiment: Optional[SignalComponent],
        whale: Optional[SignalComponent],
        momentum: Optional[SignalComponent],
    ) -> tuple[SignalDirection, float]:
        """Combine signals - all must agree for a signal."""
        components = [c for c in [sentiment, whale, momentum] if c is not None]

        if not components:
            return SignalDirection.HOLD, 0.0

        # All must be positive for BUY
        all_positive = all(c.value > 0.1 for c in components)
        # All must be negative for SELL
        all_negative = all(c.value < -0.1 for c in components)

        if all_positive:
            strength = sum(c.value * c.confidence for c in components) / len(components)
            return SignalDirection.BUY, abs(strength)
        elif all_negative:
            strength = sum(c.value * c.confidence for c in components) / len(components)
            return SignalDirection.SELL, abs(strength)
        else:
            return SignalDirection.HOLD, 0.0

    def _combine_majority(
        self,
        sentiment: Optional[SignalComponent],
        whale: Optional[SignalComponent],
        momentum: Optional[SignalComponent],
    ) -> tuple[SignalDirection, float]:
        """Combine signals - majority must agree for a signal."""
        components = [c for c in [sentiment, whale, momentum] if c is not None]

        if not components:
            return SignalDirection.HOLD, 0.0

        # Count votes
        positive = sum(1 for c in components if c.value > 0.1)
        negative = sum(1 for c in components if c.value < -0.1)

        majority_threshold = len(components) / 2

        if positive > majority_threshold:
            agreeing = [c for c in components if c.value > 0.1]
            strength = sum(c.value * c.confidence for c in agreeing) / len(agreeing)
            return SignalDirection.BUY, abs(strength)
        elif negative > majority_threshold:
            agreeing = [c for c in components if c.value < -0.1]
            strength = sum(c.value * c.confidence for c in agreeing) / len(agreeing)
            return SignalDirection.SELL, abs(strength)
        else:
            return SignalDirection.HOLD, 0.0

    async def get_detailed_analysis(
        self,
        market: Market,
        price_history: Optional[PriceHistory] = None,
    ) -> dict:
        """
        Get detailed analysis breakdown from all signal sources.

        Args:
            market: Market to analyze
            price_history: Optional price history

        Returns:
            Dict with detailed analysis from all sources
        """
        sentiment_breakdown = await self.sentiment_strategy.get_sentiment_breakdown(
            market
        )
        whale_breakdown = await self.whale_strategy.get_whale_breakdown(market)

        momentum_breakdown = {}
        if price_history and len(price_history.data) > 0:
            momentum_breakdown = await self.momentum_strategy.get_momentum_breakdown(
                market, price_history
            )

        # Generate the combined signal
        signal = await self.generate_signal(market, price_history)

        return {
            "market_id": market.condition_id,
            "question": market.question,
            "current_price": market.yes_price,
            "signal": {
                "direction": signal.direction.value,
                "strength": signal.strength,
            },
            "sentiment": sentiment_breakdown,
            "whale": whale_breakdown,
            "momentum": momentum_breakdown,
            "weights": {
                "sentiment": self.sentiment_weight,
                "whale": self.whale_weight,
                "momentum": self.momentum_weight,
            },
        }


def create_composite_strategy(
    sentiment_weight: float = 0.3,
    whale_weight: float = 0.4,
    momentum_weight: float = 0.3,
    combination_method: str = "weighted",
    use_mock: bool = True,
) -> CompositeStrategy:
    """
    Factory function to create a composite strategy with common configurations.

    Args:
        sentiment_weight: Weight for sentiment (0-1)
        whale_weight: Weight for whale signals (0-1)
        momentum_weight: Weight for momentum (0-1)
        combination_method: "weighted", "unanimous", or "majority"
        use_mock: Whether to use mock data providers

    Returns:
        Configured CompositeStrategy
    """
    method = CombinationMethod(combination_method.lower())

    return CompositeStrategy(
        sentiment_weight=sentiment_weight,
        whale_weight=whale_weight,
        momentum_weight=momentum_weight,
        combination_method=method,
        sentiment_provider=get_sentiment_provider(use_mock=use_mock),
        whale_tracker=get_whale_tracker(use_mock=use_mock),
    )
