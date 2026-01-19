from abc import ABC, abstractmethod
from datetime import datetime

from src.models.schemas import OHLCVSeries


class DataProvider(ABC):
    """Abstract base class for data providers."""

    @abstractmethod
    def get_historical_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> OHLCVSeries:
        """
        Fetch historical OHLCV data for a ticker.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            start_date: Start date for historical data
            end_date: End date for historical data (defaults to today)

        Returns:
            OHLCVSeries containing the historical data
        """
        pass

    @abstractmethod
    def get_multiple_tickers(
        self,
        tickers: list[str],
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> dict[str, OHLCVSeries]:
        """
        Fetch historical data for multiple tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data (defaults to today)

        Returns:
            Dictionary mapping ticker to OHLCVSeries
        """
        pass
