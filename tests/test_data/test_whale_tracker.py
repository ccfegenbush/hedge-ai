"""Tests for the whale tracker."""

import json
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from src.data.whale_tracker import (
    MockWhaleTracker,
    TheGraphWhaleTracker,
    WhaleTracker,
    get_whale_tracker,
)
from src.models.polymarket_schemas import TransactionType


# Load test fixtures
FIXTURES_PATH = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    with open(FIXTURES_PATH / name) as f:
        return json.load(f)


class TestMockWhaleTracker:
    """Tests for MockWhaleTracker."""

    @pytest.fixture
    def tracker(self):
        return MockWhaleTracker(seed=42)

    @pytest.mark.asyncio
    async def test_get_whale_wallets(self, tracker):
        """Test getting list of whale wallets."""
        wallets = await tracker.get_whale_wallets(min_volume=100000)

        assert len(wallets) > 0
        assert all(w.total_volume >= 100000 for w in wallets)
        assert all(hasattr(w, "address") for w in wallets)
        assert all(hasattr(w, "label") for w in wallets)

    @pytest.mark.asyncio
    async def test_get_wallet_activity(self, tracker):
        """Test getting activity for a specific wallet."""
        end = datetime.now()
        start = end - timedelta(days=7)
        wallet_address = "0x1234567890abcdef1234567890abcdef12345678"

        transactions = await tracker.get_wallet_activity(
            wallet_address=wallet_address,
            start=start,
            end=end,
        )

        for tx in transactions:
            assert tx.wallet_address == wallet_address
            assert start <= tx.timestamp <= end

    @pytest.mark.asyncio
    async def test_get_market_whale_flow(self, tracker):
        """Test getting whale flow for a market."""
        end = datetime.now()
        start = end - timedelta(days=3)
        market_id = "test_market"

        flow = await tracker.get_market_whale_flow(
            market_id=market_id,
            start=start,
            end=end,
            min_transaction_size=1000,
        )

        assert flow.market_id == market_id
        assert flow.period_start == start
        assert flow.period_end == end
        # Use approximate comparison due to floating point rounding
        assert abs(flow.net_flow - (flow.buy_volume - flow.sell_volume)) < 0.01
        assert flow.unique_wallets >= 0

    @pytest.mark.asyncio
    async def test_get_recent_large_transactions(self, tracker):
        """Test getting recent large transactions."""
        transactions = await tracker.get_recent_large_transactions(
            min_size=10000,
            limit=20,
        )

        assert len(transactions) <= 20
        assert all(tx.amount >= 10000 for tx in transactions)

    @pytest.mark.asyncio
    async def test_transaction_types(self, tracker):
        """Test that transactions have valid types."""
        transactions = await tracker.get_recent_large_transactions(limit=50)

        for tx in transactions:
            assert tx.transaction_type in [TransactionType.BUY, TransactionType.SELL]

    @pytest.mark.asyncio
    async def test_whale_flow_aggregation(self, tracker):
        """Test that whale flow correctly aggregates buy/sell volumes."""
        end = datetime.now()
        start = end - timedelta(hours=24)

        flow = await tracker.get_market_whale_flow(
            market_id="test",
            start=start,
            end=end,
        )

        # Recalculate from transactions
        buy_vol = sum(
            tx.amount
            for tx in flow.transactions
            if tx.transaction_type == TransactionType.BUY
        )
        sell_vol = sum(
            tx.amount
            for tx in flow.transactions
            if tx.transaction_type == TransactionType.SELL
        )

        assert abs(flow.buy_volume - buy_vol) < 0.01
        assert abs(flow.sell_volume - sell_vol) < 0.01

    def test_generate_tx_hash(self, tracker):
        """Test transaction hash generation."""
        tx_hash = tracker._generate_tx_hash()

        assert tx_hash.startswith("0x")
        assert len(tx_hash) == 66  # 0x + 64 hex chars


class TestWhaleTrackerFactory:
    """Tests for the factory function."""

    def test_get_mock_tracker(self):
        """Test getting mock tracker explicitly."""
        tracker = get_whale_tracker(use_mock=True)
        assert isinstance(tracker, MockWhaleTracker)

    def test_get_graph_tracker(self):
        """Test getting The Graph tracker."""
        tracker = get_whale_tracker(use_mock=False)
        assert isinstance(tracker, TheGraphWhaleTracker)

    def test_graph_tracker_with_api_key(self):
        """Test The Graph tracker accepts API key."""
        tracker = get_whale_tracker(
            use_mock=False,
            graph_api_key="test_key",
        )
        assert isinstance(tracker, TheGraphWhaleTracker)
        assert tracker.api_key == "test_key"
