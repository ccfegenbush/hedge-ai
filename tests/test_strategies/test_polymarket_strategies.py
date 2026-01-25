"""Tests for Polymarket trading strategies."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.data.sentiment_provider import MockSentimentProvider
from src.data.whale_tracker import MockWhaleTracker
from src.models.polymarket_schemas import (
    Market,
    MarketCategory,
    MarketStatus,
    Outcome,
    PriceHistory,
    SignalDirection,
)
from src.strategies.polymarket_composite import CompositeStrategy, CombinationMethod
from src.strategies.polymarket_momentum import MomentumStrategy
from src.strategies.polymarket_sentiment import SentimentStrategy
from src.strategies.polymarket_whale import WhaleStrategy


@pytest.fixture
def sample_market():
    """Create a sample market for testing."""
    return Market(
        condition_id="test_market_123",
        question="Will this test pass?",
        description="Test market for unit testing",
        category=MarketCategory.OTHER,
        status=MarketStatus.ACTIVE,
        outcomes=[
            Outcome(
                outcome_id="yes",
                name="Yes",
                price=0.65,
                token_id="token_yes",
            ),
            Outcome(
                outcome_id="no",
                name="No",
                price=0.35,
                token_id="token_no",
            ),
        ],
        volume=1000000,
        liquidity=50000,
        created_at=datetime.now() - timedelta(days=30),
        end_date=datetime.now() + timedelta(days=30),
    )


@pytest.fixture
def sample_price_history():
    """Create sample price history for testing."""
    dates = pd.date_range(end=datetime.now(), periods=50, freq="h")
    prices = [0.5 + 0.01 * i + 0.02 * (i % 5 - 2) for i in range(50)]
    volumes = [1000 + 100 * i for i in range(50)]

    df = pd.DataFrame({
        "timestamp": dates,
        "price": prices,
        "volume": volumes,
    })

    return PriceHistory(
        condition_id="test_market_123",
        outcome_id="yes",
        data=df,
    )


class TestSentimentStrategy:
    """Tests for SentimentStrategy."""

    @pytest.fixture
    def strategy(self):
        return SentimentStrategy(
            sentiment_provider=MockSentimentProvider(seed=42),
            lookback_hours=24,
        )

    @pytest.mark.asyncio
    async def test_generate_signal(self, strategy, sample_market):
        """Test signal generation."""
        signal = await strategy.generate_signal(sample_market)

        assert signal.market_id == sample_market.condition_id
        assert signal.direction in [
            SignalDirection.BUY,
            SignalDirection.SELL,
            SignalDirection.HOLD,
        ]
        assert 0 <= signal.strength <= 1
        assert len(signal.components) == 1
        assert signal.components[0].source == "sentiment"

    @pytest.mark.asyncio
    async def test_signal_has_metadata(self, strategy, sample_market):
        """Test that signals include metadata."""
        signal = await strategy.generate_signal(sample_market)

        assert "strategy" in signal.metadata
        assert signal.metadata["strategy"] == "sentiment"

    @pytest.mark.asyncio
    async def test_sentiment_breakdown(self, strategy, sample_market):
        """Test sentiment breakdown analysis."""
        breakdown = await strategy.get_sentiment_breakdown(sample_market)

        assert "current_score" in breakdown
        assert "average_score" in breakdown
        assert "trend" in breakdown
        assert breakdown["trend"] in ["improving", "declining", "stable"]

    @pytest.mark.asyncio
    async def test_strategy_info(self, strategy):
        """Test strategy info property."""
        info = strategy.info

        assert info.name == "sentiment"
        assert "sentiment" in info.signal_sources


class TestWhaleStrategy:
    """Tests for WhaleStrategy."""

    @pytest.fixture
    def strategy(self):
        return WhaleStrategy(
            whale_tracker=MockWhaleTracker(seed=42),
            lookback_hours=24,
        )

    @pytest.mark.asyncio
    async def test_generate_signal(self, strategy, sample_market):
        """Test signal generation."""
        signal = await strategy.generate_signal(sample_market)

        assert signal.market_id == sample_market.condition_id
        assert signal.direction in [
            SignalDirection.BUY,
            SignalDirection.SELL,
            SignalDirection.HOLD,
        ]
        assert len(signal.components) == 1
        assert signal.components[0].source == "whale"

    @pytest.mark.asyncio
    async def test_whale_breakdown(self, strategy, sample_market):
        """Test whale activity breakdown."""
        breakdown = await strategy.get_whale_breakdown(sample_market)

        assert "net_flow" in breakdown
        assert "buy_volume" in breakdown
        assert "sell_volume" in breakdown
        assert "unique_wallets" in breakdown

    @pytest.mark.asyncio
    async def test_strategy_info(self, strategy):
        """Test strategy info property."""
        info = strategy.info

        assert info.name == "whale"
        assert "whale" in info.signal_sources


class TestMomentumStrategy:
    """Tests for MomentumStrategy."""

    @pytest.fixture
    def strategy(self):
        return MomentumStrategy(
            momentum_period=24,
            short_ma_period=6,
            long_ma_period=24,
        )

    @pytest.mark.asyncio
    async def test_generate_signal_with_history(
        self, strategy, sample_market, sample_price_history
    ):
        """Test signal generation with price history."""
        signal = await strategy.generate_signal(sample_market, sample_price_history)

        assert signal.market_id == sample_market.condition_id
        assert len(signal.components) == 1
        assert signal.components[0].source == "momentum"

    @pytest.mark.asyncio
    async def test_generate_signal_without_history(self, strategy, sample_market):
        """Test signal generation without price history."""
        signal = await strategy.generate_signal(sample_market)

        # Should still return a signal, just with low confidence
        assert signal.market_id == sample_market.condition_id
        assert signal.components[0].confidence <= 0.5

    @pytest.mark.asyncio
    async def test_momentum_breakdown(
        self, strategy, sample_market, sample_price_history
    ):
        """Test momentum breakdown analysis."""
        breakdown = await strategy.get_momentum_breakdown(
            sample_market, sample_price_history
        )

        assert "momentum" in breakdown
        assert "rsi" in breakdown
        assert "current_price" in breakdown

    @pytest.mark.asyncio
    async def test_strategy_info(self, strategy):
        """Test strategy info property."""
        info = strategy.info

        assert info.name == "momentum"
        assert "momentum" in info.signal_sources


class TestCompositeStrategy:
    """Tests for CompositeStrategy."""

    @pytest.fixture
    def strategy(self):
        return CompositeStrategy(
            sentiment_weight=0.3,
            whale_weight=0.4,
            momentum_weight=0.3,
            combination_method=CombinationMethod.WEIGHTED,
        )

    @pytest.mark.asyncio
    async def test_generate_signal(
        self, strategy, sample_market, sample_price_history
    ):
        """Test composite signal generation."""
        signal = await strategy.generate_signal(sample_market, sample_price_history)

        assert signal.market_id == sample_market.condition_id
        assert len(signal.components) == 3

        sources = {c.source for c in signal.components}
        assert sources == {"sentiment", "whale", "momentum"}

    @pytest.mark.asyncio
    async def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1."""
        strategy = CompositeStrategy(
            sentiment_weight=1.0,
            whale_weight=1.0,
            momentum_weight=1.0,
        )

        # Each should be normalized to 1/3
        total = (
            strategy.sentiment_weight
            + strategy.whale_weight
            + strategy.momentum_weight
        )
        assert abs(total - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_weighted_combination(
        self, strategy, sample_market, sample_price_history
    ):
        """Test weighted signal combination."""
        signal = await strategy.generate_signal(sample_market, sample_price_history)

        # Check that weights are in metadata
        for comp in signal.components:
            assert "weight" in comp.metadata

    @pytest.mark.asyncio
    async def test_unanimous_combination(self, sample_market, sample_price_history):
        """Test unanimous combination method."""
        strategy = CompositeStrategy(
            combination_method=CombinationMethod.UNANIMOUS,
        )

        signal = await strategy.generate_signal(sample_market, sample_price_history)

        # With mock data, result depends on whether all agree
        assert signal.direction in [
            SignalDirection.BUY,
            SignalDirection.SELL,
            SignalDirection.HOLD,
        ]

    @pytest.mark.asyncio
    async def test_detailed_analysis(
        self, strategy, sample_market, sample_price_history
    ):
        """Test detailed analysis method."""
        analysis = await strategy.get_detailed_analysis(
            sample_market, sample_price_history
        )

        assert "market_id" in analysis
        assert "sentiment" in analysis
        assert "whale" in analysis
        assert "momentum" in analysis
        assert "signal" in analysis

    @pytest.mark.asyncio
    async def test_strategy_info(self, strategy):
        """Test strategy info property."""
        info = strategy.info

        assert info.name == "composite"
        assert set(info.signal_sources) == {"sentiment", "whale", "momentum"}
