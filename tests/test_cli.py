from datetime import datetime
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from src.cli import app
from src.models.schemas import OHLCVSeries

runner = CliRunner()


@pytest.fixture
def mock_ohlcv_data():
    """Create mock OHLCV data for testing."""
    dates = pd.date_range("2023-01-01", periods=30, freq="D")
    # Create data with clear momentum crossover
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


class TestBacktestCommand:
    def test_backtest_basic(self, mock_ohlcv_data):
        """Test basic backtest command."""
        with patch("src.cli.YahooFinanceProvider") as mock_provider:
            mock_instance = MagicMock()
            mock_instance.get_multiple_tickers.return_value = {
                "TEST": mock_ohlcv_data
            }
            mock_provider.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "backtest",
                    "--strategy",
                    "momentum",
                    "--tickers",
                    "TEST",
                    "--start",
                    "2023-01-01",
                    "--capital",
                    "10000",
                ],
            )

            assert result.exit_code == 0
            assert "Performance Summary" in result.stdout

    def test_backtest_invalid_strategy(self):
        """Test backtest with invalid strategy name."""
        result = runner.invoke(
            app,
            [
                "backtest",
                "--strategy",
                "nonexistent",
                "--tickers",
                "AAPL",
                "--start",
                "2023-01-01",
            ],
        )

        assert result.exit_code == 1
        assert "Unknown strategy" in result.stdout

    def test_backtest_no_data(self):
        """Test backtest when no data is returned."""
        with patch("src.cli.YahooFinanceProvider") as mock_provider:
            mock_instance = MagicMock()
            mock_instance.get_multiple_tickers.return_value = {}
            mock_provider.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "backtest",
                    "--strategy",
                    "momentum",
                    "--tickers",
                    "INVALID",
                    "--start",
                    "2023-01-01",
                ],
            )

            assert result.exit_code == 1
            assert "No data retrieved" in result.stdout

    def test_backtest_multiple_tickers(self, mock_ohlcv_data):
        """Test backtest with multiple tickers."""
        mock_data_2 = OHLCVSeries(ticker="TEST2", data=mock_ohlcv_data.data.copy())

        with patch("src.cli.YahooFinanceProvider") as mock_provider:
            mock_instance = MagicMock()
            mock_instance.get_multiple_tickers.return_value = {
                "TEST": mock_ohlcv_data,
                "TEST2": mock_data_2,
            }
            mock_provider.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "backtest",
                    "--strategy",
                    "momentum",
                    "--tickers",
                    "TEST,TEST2",
                    "--start",
                    "2023-01-01",
                ],
            )

            assert result.exit_code == 0
            assert "TEST, TEST2" in result.stdout

    def test_backtest_with_custom_params(self, mock_ohlcv_data):
        """Test backtest with custom parameters."""
        with patch("src.cli.YahooFinanceProvider") as mock_provider:
            mock_instance = MagicMock()
            mock_instance.get_multiple_tickers.return_value = {
                "TEST": mock_ohlcv_data
            }
            mock_provider.return_value = mock_instance

            result = runner.invoke(
                app,
                [
                    "backtest",
                    "--strategy",
                    "momentum",
                    "--tickers",
                    "TEST",
                    "--start",
                    "2023-01-01",
                    "--capital",
                    "50000",
                    "--commission",
                    "5",
                    "--lookback",
                    "10",
                ],
            )

            assert result.exit_code == 0
            assert "$50,000.00" in result.stdout


class TestListStrategiesCommand:
    def test_list_strategies(self):
        """Test list-strategies command."""
        result = runner.invoke(app, ["list-strategies"])

        assert result.exit_code == 0
        assert "Available Strategies" in result.stdout
        assert "momentum" in result.stdout

    def test_list_strategies_shows_params(self):
        """Test that list-strategies shows parameters."""
        result = runner.invoke(app, ["list-strategies"])

        assert result.exit_code == 0
        assert "lookback_period" in result.stdout


class TestHelpCommand:
    def test_main_help(self):
        """Test main help output."""
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "Hedge AI" in result.stdout
        assert "backtest" in result.stdout
        assert "list-strategies" in result.stdout

    def test_backtest_help(self):
        """Test backtest command help."""
        result = runner.invoke(app, ["backtest", "--help"])

        assert result.exit_code == 0
        assert "--strategy" in result.stdout
        assert "--tickers" in result.stdout
        assert "--start" in result.stdout
        assert "--capital" in result.stdout
