from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.models.schemas import OHLCVSeries, TradeSignal


@dataclass
class StrategyInfo:
    """Metadata about a strategy."""
    name: str
    description: str
    parameters: dict[str, str]  # param_name -> description


class Strategy(ABC):
    """Abstract base class for trading strategies."""

    @property
    @abstractmethod
    def info(self) -> StrategyInfo:
        """Return metadata about the strategy."""
        pass

    @abstractmethod
    def generate_signals(self, data: OHLCVSeries) -> list[TradeSignal]:
        """
        Generate trading signals from historical price data.

        Args:
            data: Historical OHLCV data for a single ticker

        Returns:
            List of TradeSignal objects indicating buy/sell/hold signals
        """
        pass

    def validate_data(self, data: OHLCVSeries, min_periods: int) -> bool:
        """
        Validate that data has sufficient history for the strategy.

        Args:
            data: Historical OHLCV data
            min_periods: Minimum number of periods required

        Returns:
            True if data is valid, False otherwise
        """
        return len(data) >= min_periods
