from datetime import datetime

import pandas as pd
import yfinance as yf

from src.data.provider import DataProvider
from src.models.schemas import OHLCVSeries


class YahooFinanceProvider(DataProvider):
    """Data provider using Yahoo Finance via yfinance."""

    def get_historical_data(
        self,
        ticker: str,
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> OHLCVSeries:
        """
        Fetch historical OHLCV data for a ticker from Yahoo Finance.

        Args:
            ticker: Stock ticker symbol (e.g., "AAPL")
            start_date: Start date for historical data
            end_date: End date for historical data (defaults to today)

        Returns:
            OHLCVSeries containing the historical data

        Raises:
            ValueError: If no data is returned for the ticker
        """
        if end_date is None:
            end_date = datetime.now()

        # yfinance expects string dates
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # Fetch data
        stock = yf.Ticker(ticker)
        df = stock.history(start=start_str, end=end_str)

        if df.empty:
            raise ValueError(f"No data returned for ticker: {ticker}")

        # Normalize column names to lowercase
        df.columns = df.columns.str.lower()

        # Keep only OHLCV columns
        df = df[["open", "high", "low", "close", "volume"]].copy()

        # Ensure index is datetime
        df.index = pd.to_datetime(df.index)

        # Remove timezone info if present (normalize to UTC-naive)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)

        return OHLCVSeries(ticker=ticker, data=df)

    def get_multiple_tickers(
        self,
        tickers: list[str],
        start_date: datetime,
        end_date: datetime | None = None,
    ) -> dict[str, OHLCVSeries]:
        """
        Fetch historical data for multiple tickers from Yahoo Finance.

        Args:
            tickers: List of ticker symbols
            start_date: Start date for historical data
            end_date: End date for historical data (defaults to today)

        Returns:
            Dictionary mapping ticker to OHLCVSeries
        """
        result = {}
        for ticker in tickers:
            try:
                result[ticker] = self.get_historical_data(ticker, start_date, end_date)
            except ValueError:
                # Skip tickers with no data
                continue
        return result
