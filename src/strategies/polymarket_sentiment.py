"""Sentiment-based strategy for Polymarket."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from src.data.sentiment_provider import SentimentProvider, get_sentiment_provider
from src.models.polymarket_schemas import (
    CompositeSignal,
    Market,
    PriceHistory,
    SignalComponent,
)
from src.strategies.polymarket_base import PolymarketStrategy, PolymarketStrategyInfo

logger = logging.getLogger(__name__)


class SentimentStrategy(PolymarketStrategy):
    """
    Strategy based on X/Twitter sentiment analysis.

    Generates buy signals when sentiment is strongly positive and
    sell signals when sentiment is strongly negative.
    """

    def __init__(
        self,
        sentiment_provider: Optional[SentimentProvider] = None,
        lookback_hours: int = 24,
        signal_threshold: float = 0.3,
        use_trend: bool = True,
    ):
        """
        Initialize the sentiment strategy.

        Args:
            sentiment_provider: Provider for sentiment data (uses mock if None)
            lookback_hours: Hours of sentiment history to consider
            signal_threshold: Minimum absolute sentiment to generate signal
            use_trend: Whether to consider sentiment trend in addition to level
        """
        self.sentiment_provider = sentiment_provider or get_sentiment_provider(
            use_mock=True
        )
        self.lookback_hours = lookback_hours
        self.signal_threshold = signal_threshold
        self.use_trend = use_trend

    @property
    def info(self) -> PolymarketStrategyInfo:
        return PolymarketStrategyInfo(
            name="sentiment",
            description="Trade based on X/Twitter sentiment analysis",
            signal_sources=["sentiment"],
            parameters={
                "lookback_hours": "Hours of sentiment history to analyze",
                "signal_threshold": "Minimum sentiment level to trigger signal",
                "use_trend": "Whether to consider sentiment trend",
            },
        )

    async def generate_signal(
        self,
        market: Market,
        price_history: Optional[PriceHistory] = None,
        timestamp: Optional[datetime] = None,
    ) -> CompositeSignal:
        """Generate a signal based on sentiment analysis."""
        if not timestamp:
            timestamp = datetime.now()

        market_id = market.condition_id

        try:
            # Get current sentiment
            current = await self.sentiment_provider.get_real_time_sentiment(market_id)
            current_score = current.score
            volume = current.volume

            # Calculate trend if enabled
            trend_score = 0.0
            if self.use_trend:
                start = timestamp - timedelta(hours=self.lookback_hours)
                history = await self.sentiment_provider.get_sentiment(
                    market_id, start, timestamp
                )

                if len(history) >= 2:
                    # Calculate trend as difference between recent and older sentiment
                    recent_readings = history[-len(history) // 3 :]
                    older_readings = history[: len(history) // 3]

                    recent_avg = sum(r.score for r in recent_readings) / len(
                        recent_readings
                    )
                    older_avg = sum(r.score for r in older_readings) / len(
                        older_readings
                    )

                    trend_score = recent_avg - older_avg

            # Combine current sentiment and trend
            combined_score = current_score * 0.7 + trend_score * 0.3

            # Calculate confidence based on volume and consistency
            base_confidence = min(1.0, volume / 100)  # More posts = more confidence

            if self.use_trend and len(history) >= 2:
                # Higher confidence if current and trend agree
                if (current_score > 0 and trend_score > 0) or (
                    current_score < 0 and trend_score < 0
                ):
                    base_confidence *= 1.2
                base_confidence = min(1.0, base_confidence)

            component = SignalComponent(
                source="sentiment",
                value=round(combined_score, 4),
                confidence=round(base_confidence, 4),
                metadata={
                    "current_score": current_score,
                    "trend_score": trend_score,
                    "volume": volume,
                    "sample_posts": current.sample_posts[:3],
                },
            )

            return self._create_signal(
                market_id=market_id,
                components=[component],
                timestamp=timestamp,
                metadata={
                    "strategy": "sentiment",
                    "lookback_hours": self.lookback_hours,
                    "threshold": self.signal_threshold,
                },
            )

        except Exception as e:
            logger.warning(f"Failed to get sentiment for {market_id}: {e}")
            # Return neutral signal on error
            return self._create_signal(
                market_id=market_id,
                components=[
                    SignalComponent(
                        source="sentiment",
                        value=0.0,
                        confidence=0.0,
                        metadata={"error": str(e)},
                    )
                ],
                timestamp=timestamp,
                metadata={"strategy": "sentiment", "error": str(e)},
            )

    async def get_sentiment_breakdown(
        self,
        market: Market,
        hours: int = 24,
    ) -> dict:
        """
        Get detailed sentiment breakdown for a market.

        Args:
            market: Market to analyze
            hours: Hours of history

        Returns:
            Dict with sentiment details
        """
        end = datetime.now()
        start = end - timedelta(hours=hours)

        history = await self.sentiment_provider.get_sentiment(
            market.condition_id, start, end
        )

        if not history:
            return {
                "current_score": 0.0,
                "average_score": 0.0,
                "min_score": 0.0,
                "max_score": 0.0,
                "total_volume": 0,
                "readings_count": 0,
                "trend": "neutral",
            }

        scores = [r.score for r in history]
        volumes = [r.volume for r in history]

        avg_score = sum(scores) / len(scores)
        recent_scores = scores[-len(scores) // 4 :] if len(scores) >= 4 else scores
        recent_avg = sum(recent_scores) / len(recent_scores)

        if recent_avg > avg_score + 0.1:
            trend = "improving"
        elif recent_avg < avg_score - 0.1:
            trend = "declining"
        else:
            trend = "stable"

        return {
            "current_score": history[-1].score,
            "average_score": round(avg_score, 4),
            "min_score": min(scores),
            "max_score": max(scores),
            "total_volume": sum(volumes),
            "readings_count": len(history),
            "trend": trend,
        }
