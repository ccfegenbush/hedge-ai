from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.models.schemas import OHLCVSeries

client = TestClient(app)


@pytest.fixture
def mock_ohlcv_data():
    """Create mock OHLCV data for testing."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    prices = list(range(100, 115)) + list(range(115, 100, -1))
    df = pd.DataFrame(
        {
            "open": prices,
            "high": [p + 2 for p in prices],
            "low": [p - 2 for p in prices],
            "close": prices,
            "volume": [1000000] * 30,
        },
        index=dates,
    )
    return OHLCVSeries(ticker="TEST", data=df)


class TestHealthEndpoint:
    def test_root_returns_ok(self):
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestStrategiesEndpoint:
    def test_list_strategies(self):
        response = client.get("/api/strategies")
        assert response.status_code == 200
        data = response.json()
        assert "strategies" in data
        assert len(data["strategies"]) > 0
        assert data["strategies"][0]["name"] == "momentum"


class TestBacktestEndpoint:
    def test_backtest_success(self, mock_ohlcv_data):
        with patch("src.api.main.YahooFinanceProvider") as mock_provider:
            mock_instance = MagicMock()
            mock_instance.get_multiple_tickers.return_value = {"TEST": mock_ohlcv_data}
            mock_provider.return_value = mock_instance

            response = client.post(
                "/api/backtest",
                json={
                    "strategy": "momentum",
                    "tickers": ["TEST"],
                    "start_date": "2023-01-01",
                    "initial_capital": 10000,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert "initial_capital" in data
            assert "final_value" in data
            assert "metrics" in data
            assert "trades" in data
            assert "portfolio_values" in data

    def test_backtest_invalid_strategy(self):
        response = client.post(
            "/api/backtest",
            json={
                "strategy": "nonexistent",
                "tickers": ["AAPL"],
                "start_date": "2023-01-01",
            },
        )
        assert response.status_code == 400
        assert "Unknown strategy" in response.json()["detail"]

    def test_backtest_invalid_date_format(self):
        response = client.post(
            "/api/backtest",
            json={
                "strategy": "momentum",
                "tickers": ["AAPL"],
                "start_date": "invalid-date",
            },
        )
        assert response.status_code == 400
        assert "Invalid date format" in response.json()["detail"]

    def test_backtest_no_data(self):
        with patch("src.api.main.YahooFinanceProvider") as mock_provider:
            mock_instance = MagicMock()
            mock_instance.get_multiple_tickers.return_value = {}
            mock_provider.return_value = mock_instance

            response = client.post(
                "/api/backtest",
                json={
                    "strategy": "momentum",
                    "tickers": ["INVALID"],
                    "start_date": "2023-01-01",
                },
            )

            assert response.status_code == 404
            assert "No data found" in response.json()["detail"]

    def test_backtest_with_all_params(self, mock_ohlcv_data):
        with patch("src.api.main.YahooFinanceProvider") as mock_provider:
            mock_instance = MagicMock()
            mock_instance.get_multiple_tickers.return_value = {"TEST": mock_ohlcv_data}
            mock_provider.return_value = mock_instance

            response = client.post(
                "/api/backtest",
                json={
                    "strategy": "momentum",
                    "tickers": ["TEST"],
                    "start_date": "2023-01-01",
                    "end_date": "2023-01-30",
                    "initial_capital": 50000,
                    "commission": 5.0,
                    "lookback_period": 10,
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["initial_capital"] == 50000
