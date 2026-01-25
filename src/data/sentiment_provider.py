"""Sentiment data provider for Polymarket markets.

This module provides sentiment analysis data from X/Twitter posts about prediction markets.
Data is sourced from n8n workflows that scrape and analyze posts, storing results in Supabase.

For development/testing, a mock provider is included that generates realistic test data.
"""

import logging
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

from src.exceptions import SentimentUnavailableError
from src.models.polymarket_schemas import InfluencerPost, SentimentReading

logger = logging.getLogger(__name__)


class SentimentProvider(ABC):
    """Abstract base class for sentiment data providers."""

    @abstractmethod
    async def get_sentiment(
        self,
        market_id: str,
        start: datetime,
        end: datetime,
    ) -> list[SentimentReading]:
        """
        Get historical sentiment readings for a market.

        Args:
            market_id: The market's condition ID
            start: Start datetime
            end: End datetime

        Returns:
            List of SentimentReading objects
        """
        pass

    @abstractmethod
    async def get_real_time_sentiment(
        self,
        market_id: str,
    ) -> SentimentReading:
        """
        Get the current/real-time sentiment reading for a market.

        Args:
            market_id: The market's condition ID

        Returns:
            Current SentimentReading

        Raises:
            SentimentUnavailableError: If no sentiment data is available
        """
        pass

    @abstractmethod
    async def get_influencer_posts(
        self,
        market_id: Optional[str] = None,
        account_handle: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[InfluencerPost]:
        """
        Get posts from influential accounts.

        Args:
            market_id: Optional filter by market
            account_handle: Optional filter by account
            start: Optional start datetime
            end: Optional end datetime
            limit: Maximum posts to return

        Returns:
            List of InfluencerPost objects
        """
        pass


class MockSentimentProvider(SentimentProvider):
    """
    Mock sentiment provider for testing and development.

    Generates realistic-looking sentiment data with configurable patterns.
    """

    # Sample influential accounts for mock data
    INFLUENTIAL_ACCOUNTS = [
        "NateSilver538",
        "ElectionAnalyst",
        "PredictItPro",
        "MarketWatcher",
        "PolyWhale",
    ]

    # Sample post templates
    POST_TEMPLATES = [
        "Market {market_id} looking {sentiment} based on recent data",
        "Seeing {sentiment} momentum in {market_id}",
        "The odds are shifting {direction} on this one",
        "Smart money appears to be {action} at these levels",
        "Historical patterns suggest {sentiment} outcome",
    ]

    def __init__(
        self,
        base_sentiment: float = 0.0,
        volatility: float = 0.3,
        trend_strength: float = 0.1,
        seed: Optional[int] = None,
    ):
        """
        Initialize the mock provider.

        Args:
            base_sentiment: Baseline sentiment (-1 to 1)
            volatility: How much sentiment fluctuates
            trend_strength: How strong trends are
            seed: Random seed for reproducibility
        """
        self.base_sentiment = base_sentiment
        self.volatility = volatility
        self.trend_strength = trend_strength

        if seed is not None:
            random.seed(seed)

        self._sentiment_cache: dict[str, float] = {}

    def _generate_sentiment_value(
        self,
        market_id: str,
        timestamp: datetime,
    ) -> float:
        """Generate a realistic sentiment value."""
        # Use market_id hash for consistent per-market behavior
        market_hash = hash(market_id) % 1000 / 1000.0

        # Add time-based trend
        days_factor = (timestamp - datetime(2024, 1, 1)).days / 365.0
        trend = self.trend_strength * (market_hash - 0.5) * days_factor

        # Add random noise
        noise = random.gauss(0, self.volatility)

        # Combine components
        sentiment = self.base_sentiment + trend + noise

        # Clamp to valid range
        return max(-1.0, min(1.0, sentiment))

    def _generate_volume(self, base: int = 50) -> int:
        """Generate a realistic post volume."""
        return max(1, int(random.gauss(base, base * 0.3)))

    def _generate_sample_posts(
        self,
        market_id: str,
        sentiment: float,
        count: int = 3,
    ) -> list[str]:
        """Generate sample post texts based on sentiment."""
        sentiment_word = (
            "bullish" if sentiment > 0.2 else "bearish" if sentiment < -0.2 else "neutral"
        )
        direction = "up" if sentiment > 0 else "down" if sentiment < 0 else "sideways"
        action = (
            "accumulating"
            if sentiment > 0.3
            else "distributing"
            if sentiment < -0.3
            else "holding"
        )

        posts = []
        for _ in range(count):
            template = random.choice(self.POST_TEMPLATES)
            post = template.format(
                market_id=market_id[:8] + "...",
                sentiment=sentiment_word,
                direction=direction,
                action=action,
            )
            posts.append(post)

        return posts

    async def get_sentiment(
        self,
        market_id: str,
        start: datetime,
        end: datetime,
    ) -> list[SentimentReading]:
        """Generate mock historical sentiment readings."""
        logger.info(f"Generating mock sentiment for {market_id} from {start} to {end}")

        readings = []
        current = start

        # Generate readings at 15-minute intervals
        while current <= end:
            sentiment = self._generate_sentiment_value(market_id, current)
            volume = self._generate_volume()

            readings.append(
                SentimentReading(
                    market_id=market_id,
                    timestamp=current,
                    score=round(sentiment, 4),
                    volume=volume,
                    sample_posts=self._generate_sample_posts(market_id, sentiment),
                )
            )

            current += timedelta(minutes=15)

        return readings

    async def get_real_time_sentiment(
        self,
        market_id: str,
    ) -> SentimentReading:
        """Generate mock current sentiment reading."""
        now = datetime.now()
        sentiment = self._generate_sentiment_value(market_id, now)

        # Cache for consistency in repeated calls
        if market_id in self._sentiment_cache:
            # Slight drift from cached value
            cached = self._sentiment_cache[market_id]
            sentiment = cached + random.gauss(0, 0.05)
            sentiment = max(-1.0, min(1.0, sentiment))

        self._sentiment_cache[market_id] = sentiment

        return SentimentReading(
            market_id=market_id,
            timestamp=now,
            score=round(sentiment, 4),
            volume=self._generate_volume(100),
            sample_posts=self._generate_sample_posts(market_id, sentiment, 5),
        )

    async def get_influencer_posts(
        self,
        market_id: Optional[str] = None,
        account_handle: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[InfluencerPost]:
        """Generate mock influencer posts."""
        logger.info(f"Generating mock influencer posts (limit={limit})")

        posts = []
        now = datetime.now()

        for i in range(min(limit, 20)):
            # Random account
            if account_handle:
                handle = account_handle
            else:
                handle = random.choice(self.INFLUENTIAL_ACCOUNTS)

            # Random timestamp within the last 24 hours
            post_time = now - timedelta(hours=random.uniform(0, 24))

            if start and post_time < start:
                continue
            if end and post_time > end:
                continue

            # Generate sentiment
            sentiment = random.gauss(0, 0.4)
            sentiment = max(-1.0, min(1.0, sentiment))

            # Generate prediction if sentiment is strong
            prediction = None
            if abs(sentiment) > 0.5:
                prediction = "YES" if sentiment > 0 else "NO"

            post_text = self._generate_sample_posts(
                market_id or "general", sentiment, 1
            )[0]

            posts.append(
                InfluencerPost(
                    account_handle=handle,
                    market_id=market_id,
                    post_text=post_text,
                    sentiment_score=round(sentiment, 4),
                    prediction_extracted=prediction,
                    timestamp=post_time,
                )
            )

        # Sort by timestamp descending
        posts.sort(key=lambda p: p.timestamp, reverse=True)

        return posts


class SupabaseSentimentProvider(SentimentProvider):
    """
    Sentiment provider that reads from Supabase.

    This is the production provider that reads sentiment data
    collected by n8n workflows and stored in Supabase.
    """

    def __init__(
        self,
        supabase_url: str,
        supabase_key: str,
    ):
        """
        Initialize the Supabase sentiment provider.

        Args:
            supabase_url: Supabase project URL
            supabase_key: Supabase API key (anon or service role)
        """
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self._client = None

    async def _get_client(self):
        """Get or create the Supabase client."""
        if self._client is None:
            # Lazy import to avoid requiring supabase in all environments
            try:
                from supabase import create_client
                self._client = create_client(self.supabase_url, self.supabase_key)
            except ImportError:
                raise SentimentUnavailableError(
                    "supabase",
                    "supabase-py package not installed",
                )
        return self._client

    async def get_sentiment(
        self,
        market_id: str,
        start: datetime,
        end: datetime,
    ) -> list[SentimentReading]:
        """Fetch historical sentiment from Supabase."""
        try:
            client = await self._get_client()

            # Query the polymarket_sentiment table
            response = (
                client.table("polymarket_sentiment")
                .select("*")
                .eq("market_id", market_id)
                .gte("timestamp", start.isoformat())
                .lte("timestamp", end.isoformat())
                .order("timestamp", desc=False)
                .execute()
            )

            readings = []
            for row in response.data:
                readings.append(
                    SentimentReading(
                        market_id=row["market_id"],
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                        score=row["score"],
                        volume=row["volume"],
                        sample_posts=row.get("sample_posts", []),
                    )
                )

            return readings

        except Exception as e:
            logger.error(f"Failed to fetch sentiment from Supabase: {e}")
            raise SentimentUnavailableError(market_id, str(e))

    async def get_real_time_sentiment(
        self,
        market_id: str,
    ) -> SentimentReading:
        """Fetch the most recent sentiment reading from Supabase."""
        try:
            client = await self._get_client()

            # Get the most recent reading
            response = (
                client.table("polymarket_sentiment")
                .select("*")
                .eq("market_id", market_id)
                .order("timestamp", desc=True)
                .limit(1)
                .execute()
            )

            if not response.data:
                raise SentimentUnavailableError(market_id, "No sentiment data found")

            row = response.data[0]
            return SentimentReading(
                market_id=row["market_id"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                score=row["score"],
                volume=row["volume"],
                sample_posts=row.get("sample_posts", []),
            )

        except SentimentUnavailableError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch real-time sentiment: {e}")
            raise SentimentUnavailableError(market_id, str(e))

    async def get_influencer_posts(
        self,
        market_id: Optional[str] = None,
        account_handle: Optional[str] = None,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: int = 50,
    ) -> list[InfluencerPost]:
        """Fetch influencer posts from Supabase."""
        try:
            client = await self._get_client()

            # Build query
            query = client.table("polymarket_influencer_posts").select("*")

            if market_id:
                query = query.eq("market_id", market_id)
            if account_handle:
                query = query.eq("account_handle", account_handle)
            if start:
                query = query.gte("timestamp", start.isoformat())
            if end:
                query = query.lte("timestamp", end.isoformat())

            query = query.order("timestamp", desc=True).limit(limit)

            response = query.execute()

            posts = []
            for row in response.data:
                posts.append(
                    InfluencerPost(
                        account_handle=row["account_handle"],
                        market_id=row.get("market_id"),
                        post_text=row["post_text"],
                        sentiment_score=row.get("sentiment_score"),
                        prediction_extracted=row.get("prediction_extracted"),
                        timestamp=datetime.fromisoformat(row["timestamp"]),
                    )
                )

            return posts

        except Exception as e:
            logger.error(f"Failed to fetch influencer posts: {e}")
            return []


def get_sentiment_provider(
    use_mock: bool = False,
    supabase_url: Optional[str] = None,
    supabase_key: Optional[str] = None,
) -> SentimentProvider:
    """
    Factory function to get the appropriate sentiment provider.

    Args:
        use_mock: If True, return mock provider
        supabase_url: Supabase URL for production provider
        supabase_key: Supabase key for production provider

    Returns:
        SentimentProvider instance
    """
    if use_mock:
        return MockSentimentProvider()

    if supabase_url and supabase_key:
        return SupabaseSentimentProvider(supabase_url, supabase_key)

    # Default to mock if no credentials provided
    logger.warning("No Supabase credentials provided, using mock sentiment provider")
    return MockSentimentProvider()
