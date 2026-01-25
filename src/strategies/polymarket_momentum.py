"""Momentum-based strategy for Polymarket."""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd

from src.data.polymarket_provider import PolymarketProvider
from src.models.polymarket_schemas import (
    CompositeSignal,
    Market,
    PriceHistory,
    SignalComponent,
)
from src.strategies.polymarket_base import PolymarketStrategy, PolymarketStrategyInfo

logger = logging.getLogger(__name__)


class MomentumStrategy(PolymarketStrategy):
    """
    Strategy based on price momentum and order book dynamics.

    Generates signals based on:
    - Price momentum (rate of change)
    - Price relative to moving averages
    - Order book imbalance
    """

    def __init__(
        self,
        polymarket_provider: Optional[PolymarketProvider] = None,
        momentum_period: int = 24,  # hours
        short_ma_period: int = 6,  # hours
        long_ma_period: int = 24,  # hours
        signal_threshold: float = 0.1,
    ):
        """
        Initialize the momentum strategy.

        Args:
            polymarket_provider: Provider for Polymarket data
            momentum_period: Period for momentum calculation (hours)
            short_ma_period: Short moving average period (hours)
            long_ma_period: Long moving average period (hours)
            signal_threshold: Minimum momentum to generate signal
        """
        self.provider = polymarket_provider
        self.momentum_period = momentum_period
        self.short_ma_period = short_ma_period
        self.long_ma_period = long_ma_period
        self.signal_threshold = signal_threshold

    @property
    def info(self) -> PolymarketStrategyInfo:
        return PolymarketStrategyInfo(
            name="momentum",
            description="Trade based on price momentum and order book dynamics",
            signal_sources=["momentum"],
            parameters={
                "momentum_period": "Period for momentum calculation (hours)",
                "short_ma_period": "Short moving average period (hours)",
                "long_ma_period": "Long moving average period (hours)",
                "signal_threshold": "Minimum momentum to trigger signal",
            },
        )

    def _calculate_momentum(self, prices: pd.Series) -> float:
        """Calculate price momentum as rate of change."""
        if len(prices) < 2:
            return 0.0

        current = prices.iloc[-1]
        past = prices.iloc[0]

        if past == 0:
            return 0.0

        return (current - past) / past

    def _calculate_ma_signal(self, prices: pd.Series) -> float:
        """Calculate signal from moving average crossover."""
        if len(prices) < self.long_ma_period:
            return 0.0

        short_ma = prices.rolling(window=self.short_ma_period).mean()
        long_ma = prices.rolling(window=self.long_ma_period).mean()

        if pd.isna(short_ma.iloc[-1]) or pd.isna(long_ma.iloc[-1]):
            return 0.0

        # Positive when short MA is above long MA
        diff = short_ma.iloc[-1] - long_ma.iloc[-1]

        # Normalize by current price
        current = prices.iloc[-1]
        if current > 0:
            return diff / current
        return 0.0

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI (Relative Strength Index)."""
        if len(prices) < period + 1:
            return 50.0  # Neutral

        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        if pd.isna(gain.iloc[-1]) or pd.isna(loss.iloc[-1]):
            return 50.0

        if loss.iloc[-1] == 0:
            return 100.0

        rs = gain.iloc[-1] / loss.iloc[-1]
        rsi = 100 - (100 / (1 + rs))

        return rsi

    async def _get_order_book_signal(
        self, market: Market
    ) -> tuple[float, float]:
        """
        Get signal from order book imbalance.

        Returns:
            Tuple of (signal_value, confidence)
        """
        if not self.provider:
            return 0.0, 0.0

        try:
            if market.outcomes:
                outcome = market.outcomes[0]  # Primary outcome (YES)
                order_book = await self.provider.get_order_book(
                    market.condition_id, outcome.token_id
                )

                if order_book.bids and order_book.asks:
                    # Calculate bid/ask volume imbalance
                    bid_volume = sum(level.size for level in order_book.bids[:5])
                    ask_volume = sum(level.size for level in order_book.asks[:5])

                    total = bid_volume + ask_volume
                    if total > 0:
                        imbalance = (bid_volume - ask_volume) / total
                        confidence = min(1.0, total / 10000)  # $10k = full confidence
                        return imbalance, confidence

        except Exception as e:
            logger.warning(f"Failed to get order book: {e}")

        return 0.0, 0.0

    async def generate_signal(
        self,
        market: Market,
        price_history: Optional[PriceHistory] = None,
        timestamp: Optional[datetime] = None,
    ) -> CompositeSignal:
        """Generate a signal based on price momentum."""
        if not timestamp:
            timestamp = datetime.now()

        market_id = market.condition_id

        # If no price history provided, try to fetch it
        if price_history is None and self.provider:
            try:
                start = timestamp - timedelta(hours=max(self.long_ma_period * 2, 48))
                if market.outcomes:
                    price_history = await self.provider.get_historical_prices(
                        market.condition_id,
                        market.outcomes[0].token_id,
                        start,
                        timestamp,
                    )
            except Exception as e:
                logger.warning(f"Failed to fetch price history: {e}")

        # Calculate momentum components
        momentum_value = 0.0
        ma_signal = 0.0
        rsi_signal = 0.0
        ob_signal = 0.0
        ob_confidence = 0.0

        if price_history is not None and len(price_history.data) > 0:
            prices = price_history.data["price"]

            # Raw momentum
            momentum_value = self._calculate_momentum(prices)

            # Moving average signal
            ma_signal = self._calculate_ma_signal(prices)

            # RSI converted to -1 to 1 scale
            rsi = self._calculate_rsi(prices)
            rsi_signal = (rsi - 50) / 50  # Convert 0-100 to -1 to 1

        # Order book signal
        ob_signal, ob_confidence = await self._get_order_book_signal(market)

        # Combine signals with weights
        # Momentum: 30%, MA: 30%, RSI: 20%, Order Book: 20%
        combined = (
            momentum_value * 0.30
            + ma_signal * 0.30
            + rsi_signal * 0.20
            + ob_signal * 0.20
        )

        # Clamp to -1 to 1
        combined = max(-1.0, min(1.0, combined))

        # Calculate confidence
        if price_history is not None and len(price_history.data) > 0:
            # More data = more confidence
            data_confidence = min(1.0, len(price_history.data) / 48)
        else:
            data_confidence = 0.2  # Low confidence with no historical data

        confidence = data_confidence * 0.7 + ob_confidence * 0.3

        component = SignalComponent(
            source="momentum",
            value=round(combined, 4),
            confidence=round(confidence, 4),
            metadata={
                "raw_momentum": round(momentum_value, 4),
                "ma_signal": round(ma_signal, 4),
                "rsi_signal": round(rsi_signal, 4),
                "order_book_signal": round(ob_signal, 4),
                "data_points": len(price_history.data) if price_history else 0,
            },
        )

        return self._create_signal(
            market_id=market_id,
            components=[component],
            timestamp=timestamp,
            metadata={
                "strategy": "momentum",
                "momentum_period": self.momentum_period,
                "short_ma": self.short_ma_period,
                "long_ma": self.long_ma_period,
            },
        )

    async def get_momentum_breakdown(
        self,
        market: Market,
        price_history: PriceHistory,
    ) -> dict:
        """
        Get detailed momentum breakdown for a market.

        Args:
            market: Market to analyze
            price_history: Price history data

        Returns:
            Dict with momentum details
        """
        if len(price_history.data) == 0:
            return {
                "error": "No price data available",
                "momentum": 0.0,
                "ma_signal": 0.0,
                "rsi": 50.0,
            }

        prices = price_history.data["price"]

        momentum = self._calculate_momentum(prices)
        ma_signal = self._calculate_ma_signal(prices)
        rsi = self._calculate_rsi(prices)

        # Calculate support/resistance levels
        high = prices.max()
        low = prices.min()
        current = prices.iloc[-1]

        # Trend strength
        if len(prices) >= 10:
            recent_prices = prices.tail(10)
            trend_direction = 1 if recent_prices.iloc[-1] > recent_prices.iloc[0] else -1
            trend_strength = abs(recent_prices.iloc[-1] - recent_prices.iloc[0]) / max(
                0.01, recent_prices.std()
            )
        else:
            trend_direction = 0
            trend_strength = 0.0

        return {
            "momentum": round(momentum, 4),
            "ma_signal": round(ma_signal, 4),
            "rsi": round(rsi, 2),
            "current_price": round(current, 4),
            "high": round(high, 4),
            "low": round(low, 4),
            "range": round(high - low, 4),
            "position_in_range": round((current - low) / max(0.01, high - low), 4),
            "trend_direction": "bullish" if trend_direction > 0 else "bearish" if trend_direction < 0 else "neutral",
            "trend_strength": round(min(1.0, trend_strength), 4),
            "volatility": round(prices.std(), 4),
        }
