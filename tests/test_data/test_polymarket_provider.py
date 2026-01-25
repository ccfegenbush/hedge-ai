"""Tests for the Polymarket data provider."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from src.data.polymarket_provider import PolymarketProvider, SyncPolymarketProvider
from src.exceptions import MarketNotFoundError
from src.models.polymarket_schemas import MarketCategory, MarketStatus


# Load test fixtures
FIXTURES_PATH = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    with open(FIXTURES_PATH / name) as f:
        return json.load(f)


@pytest.fixture
def mock_markets():
    return load_fixture("polymarket_markets.json")["markets"]


@pytest.fixture
def mock_prices():
    return load_fixture("polymarket_prices.json")["prices"]


class TestPolymarketProvider:
    """Tests for PolymarketProvider."""

    @pytest.mark.asyncio
    async def test_parse_market_basic(self, mock_markets):
        """Test that markets are parsed correctly."""
        provider = PolymarketProvider()
        market = provider._parse_market(mock_markets[0])

        assert market.condition_id == "0x1234567890abcdef1234567890abcdef12345678"
        assert "election" in market.question.lower()
        assert market.category == MarketCategory.POLITICS
        assert market.status == MarketStatus.ACTIVE
        assert len(market.outcomes) == 2
        assert market.is_binary is True

    @pytest.mark.asyncio
    async def test_parse_market_outcomes(self, mock_markets):
        """Test that market outcomes are parsed correctly."""
        provider = PolymarketProvider()
        market = provider._parse_market(mock_markets[0])

        yes_outcome = market.outcomes[0]
        assert yes_outcome.name == "Yes"
        assert yes_outcome.price == 0.65
        assert yes_outcome.token_id == "token_yes_123"

    @pytest.mark.asyncio
    async def test_parse_category_politics(self):
        """Test category parsing for politics."""
        provider = PolymarketProvider()
        assert provider._parse_category("politics") == MarketCategory.POLITICS
        assert provider._parse_category("political") == MarketCategory.POLITICS

    @pytest.mark.asyncio
    async def test_parse_category_unknown(self):
        """Test category parsing for unknown category."""
        provider = PolymarketProvider()
        assert provider._parse_category("unknown") == MarketCategory.OTHER
        assert provider._parse_category(None) == MarketCategory.OTHER

    @pytest.mark.asyncio
    async def test_get_active_markets_mock(self, mock_markets):
        """Test fetching active markets with mocked API."""
        provider = PolymarketProvider()

        with patch.object(provider, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_markets
            markets = await provider.get_active_markets(limit=10)

            assert len(markets) > 0
            assert all(m.is_active for m in markets)

        await provider.close()

    @pytest.mark.asyncio
    async def test_get_market_by_id_mock(self, mock_markets):
        """Test fetching a specific market."""
        provider = PolymarketProvider()

        with patch.object(provider, "_request", new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_markets[0]
            market = await provider.get_market_by_id("0x1234")

            assert market.condition_id == mock_markets[0]["condition_id"]

        await provider.close()

    @pytest.mark.asyncio
    async def test_market_yes_price(self, mock_markets):
        """Test yes_price property for binary markets."""
        provider = PolymarketProvider()
        market = provider._parse_market(mock_markets[0])

        assert market.yes_price == 0.65

    @pytest.mark.asyncio
    async def test_order_book_properties(self):
        """Test OrderBook calculated properties."""
        from src.models.polymarket_schemas import OrderBook, OrderBookLevel

        order_book = OrderBook(
            condition_id="test",
            outcome_id="yes",
            bids=[
                OrderBookLevel(price=0.60, size=1000),
                OrderBookLevel(price=0.59, size=500),
            ],
            asks=[
                OrderBookLevel(price=0.62, size=800),
                OrderBookLevel(price=0.63, size=600),
            ],
            timestamp=datetime.now(),
        )

        assert order_book.best_bid == 0.60
        assert order_book.best_ask == 0.62
        assert order_book.spread == 0.02
        assert order_book.mid_price == 0.61


class TestSyncPolymarketProvider:
    """Tests for synchronous wrapper."""

    def test_sync_wrapper_exists(self):
        """Test that sync wrapper can be instantiated."""
        provider = SyncPolymarketProvider()
        assert provider is not None

    def test_sync_wrapper_has_methods(self):
        """Test that sync wrapper has expected methods."""
        provider = SyncPolymarketProvider()
        assert hasattr(provider, "get_active_markets")
        assert hasattr(provider, "get_market_by_id")
        assert hasattr(provider, "search_markets")
