"""Tests for Polymarket API endpoints."""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


# Load test fixtures
FIXTURES_PATH = Path(__file__).parent.parent / "fixtures"


def load_fixture(name: str) -> dict:
    with open(FIXTURES_PATH / name) as f:
        return json.load(f)


class TestMarketsEndpoints:
    """Tests for market-related endpoints."""

    def test_list_markets(self):
        """Test listing markets endpoint."""
        with patch(
            "src.data.polymarket_provider.PolymarketProvider.get_active_markets",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = []

            response = client.get("/api/polymarket/markets")

            assert response.status_code == 200
            data = response.json()
            assert "markets" in data
            assert "total" in data

    def test_list_markets_with_category(self):
        """Test listing markets with category filter."""
        with patch(
            "src.data.polymarket_provider.PolymarketProvider.get_active_markets",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = []

            response = client.get("/api/polymarket/markets?category=politics")

            assert response.status_code == 200

    def test_get_market_by_id(self):
        """Test getting a specific market."""
        from src.models.polymarket_schemas import (
            Market,
            MarketCategory,
            MarketStatus,
            Outcome,
        )

        mock_market = Market(
            condition_id="test123",
            question="Test?",
            description="Test",
            category=MarketCategory.POLITICS,
            status=MarketStatus.ACTIVE,
            outcomes=[
                Outcome(outcome_id="yes", name="Yes", price=0.5, token_id="t1")
            ],
            volume=1000,
            liquidity=100,
            created_at=datetime.now(),
        )

        with patch(
            "src.data.polymarket_provider.PolymarketProvider.get_market_by_id",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = mock_market

            response = client.get("/api/polymarket/markets/test123")

            assert response.status_code == 200
            data = response.json()
            assert data["condition_id"] == "test123"

    def test_search_markets(self):
        """Test market search endpoint."""
        with patch(
            "src.data.polymarket_provider.PolymarketProvider.search_markets",
            new_callable=AsyncMock,
        ) as mock:
            mock.return_value = []

            response = client.post(
                "/api/polymarket/markets/search",
                json={"query": "election", "limit": 10},
            )

            assert response.status_code == 200


class TestSignalsEndpoints:
    """Tests for signal-related endpoints."""

    def test_generate_signals(self):
        """Test signal generation endpoint."""
        response = client.post(
            "/api/polymarket/signals",
            json={
                "market_ids": ["market1", "market2"],
                "include_sentiment": True,
                "include_whale": True,
                "include_momentum": True,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "signals" in data
        assert "generated_at" in data

    def test_get_market_signals(self):
        """Test getting signals for a specific market."""
        response = client.get("/api/polymarket/signals/test_market")

        assert response.status_code == 200
        data = response.json()
        assert "signals" in data


class TestWhaleEndpoints:
    """Tests for whale tracking endpoints."""

    def test_list_whale_wallets(self):
        """Test listing whale wallets."""
        response = client.get("/api/polymarket/whales")

        assert response.status_code == 200
        data = response.json()
        assert "wallets" in data
        assert "total" in data

    def test_get_whale_activity(self):
        """Test getting whale activity."""
        response = client.post(
            "/api/polymarket/whales/activity",
            json={"limit": 20},
        )

        assert response.status_code == 200
        data = response.json()
        assert "transactions" in data
        assert "net_flow" in data


class TestSentimentEndpoints:
    """Tests for sentiment endpoints."""

    def test_get_market_sentiment(self):
        """Test getting market sentiment."""
        response = client.get("/api/polymarket/sentiment/test_market")

        assert response.status_code == 200
        data = response.json()
        assert "market_id" in data
        assert "current_score" in data
        assert "trend" in data


class TestBacktestEndpoints:
    """Tests for backtest endpoints."""

    def test_run_backtest(self):
        """Test running a backtest."""
        response = client.post(
            "/api/polymarket/backtest",
            json={
                "market_ids": ["market1"],
                "strategy": "composite",
                "start_date": "2024-01-01",
                "end_date": "2024-01-31",
                "initial_capital": 10000,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert "initial_capital" in data
        assert "final_value" in data
        assert "metrics" in data

    def test_backtest_metrics_structure(self):
        """Test that backtest returns all expected metrics."""
        response = client.post(
            "/api/polymarket/backtest",
            json={
                "market_ids": ["market1"],
                "strategy": "sentiment",
                "start_date": "2024-01-01",
                "initial_capital": 10000,
            },
        )

        assert response.status_code == 200
        metrics = response.json()["metrics"]

        # Check standard metrics
        assert "total_return_pct" in metrics
        assert "sharpe_ratio" in metrics
        assert "max_drawdown_pct" in metrics

        # Check PM-specific metrics
        assert "brier_score" in metrics
        assert "calibration_score" in metrics
        assert "resolution_accuracy_pct" in metrics


class TestStrategiesEndpoint:
    """Tests for strategies list endpoint."""

    def test_list_strategies(self):
        """Test listing available strategies."""
        response = client.get("/api/polymarket/strategies")

        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert len(data["strategies"]) >= 4  # At least 4 strategies

        # Check strategy structure
        strategy = data["strategies"][0]
        assert "name" in strategy
        assert "description" in strategy
        assert "signal_sources" in strategy


class TestErrorHandling:
    """Tests for API error handling."""

    def test_market_not_found(self):
        """Test 404 error for non-existent market."""
        from src.exceptions import MarketNotFoundError

        with patch(
            "src.data.polymarket_provider.PolymarketProvider.get_market_by_id",
            new_callable=AsyncMock,
        ) as mock:
            mock.side_effect = MarketNotFoundError("nonexistent")

            response = client.get("/api/polymarket/markets/nonexistent")

            assert response.status_code == 404

    def test_invalid_backtest_request(self):
        """Test validation error for invalid backtest request."""
        response = client.post(
            "/api/polymarket/backtest",
            json={
                "market_ids": [],  # Empty list should fail validation
                "strategy": "composite",
                "start_date": "2024-01-01",
            },
        )

        assert response.status_code == 422  # Validation error
