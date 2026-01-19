from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.data.yahoo import YahooFinanceProvider
from src.models.schemas import OHLCVSeries


@pytest.fixture
def provider():
    return YahooFinanceProvider()


@pytest.fixture
def sample_df():
    """Create sample OHLCV DataFrame."""
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    return pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0, 101.5, 103.0],
            "High": [102.0, 103.0, 104.0, 103.5, 105.0],
            "Low": [99.0, 100.0, 101.0, 100.5, 102.0],
            "Close": [101.0, 102.0, 103.0, 102.5, 104.0],
            "Volume": [1000000, 1100000, 1200000, 1150000, 1300000],
        },
        index=dates,
    )


class TestYahooFinanceProvider:
    def test_get_historical_data_returns_ohlcv_series(self, provider, sample_df):
        """Test that get_historical_data returns an OHLCVSeries."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.history.return_value = sample_df
            mock_ticker.return_value = mock_instance

            result = provider.get_historical_data(
                ticker="AAPL",
                start_date=datetime(2023, 1, 1),
                end_date=datetime(2023, 1, 5),
            )

            assert isinstance(result, OHLCVSeries)
            assert result.ticker == "AAPL"
            assert len(result) == 5

    def test_get_historical_data_normalizes_columns(self, provider, sample_df):
        """Test that column names are normalized to lowercase."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.history.return_value = sample_df
            mock_ticker.return_value = mock_instance

            result = provider.get_historical_data(
                ticker="AAPL",
                start_date=datetime(2023, 1, 1),
            )

            expected_columns = ["open", "high", "low", "close", "volume"]
            assert list(result.data.columns) == expected_columns

    def test_get_historical_data_raises_on_empty(self, provider):
        """Test that ValueError is raised for empty data."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.history.return_value = pd.DataFrame()
            mock_ticker.return_value = mock_instance

            with pytest.raises(ValueError, match="No data returned"):
                provider.get_historical_data(
                    ticker="INVALID",
                    start_date=datetime(2023, 1, 1),
                )

    def test_get_multiple_tickers(self, provider, sample_df):
        """Test fetching data for multiple tickers."""
        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.history.return_value = sample_df
            mock_ticker.return_value = mock_instance

            result = provider.get_multiple_tickers(
                tickers=["AAPL", "MSFT"],
                start_date=datetime(2023, 1, 1),
            )

            assert "AAPL" in result
            assert "MSFT" in result
            assert len(result) == 2

    def test_get_multiple_tickers_skips_invalid(self, provider, sample_df):
        """Test that invalid tickers are skipped."""
        call_count = 0

        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return sample_df
            return pd.DataFrame()

        with patch("yfinance.Ticker") as mock_ticker:
            mock_instance = MagicMock()
            mock_instance.history.side_effect = side_effect
            mock_ticker.return_value = mock_instance

            result = provider.get_multiple_tickers(
                tickers=["AAPL", "INVALID"],
                start_date=datetime(2023, 1, 1),
            )

            assert "AAPL" in result
            assert "INVALID" not in result
