import pandas as pd

from src.models.schemas import OHLCVSeries, Signal, TradeSignal
from src.strategies.base import Strategy, StrategyInfo


class MomentumStrategy(Strategy):
    """
    Simple momentum strategy based on moving average crossover.

    Buy signal: Price closes above N-day moving average
    Sell signal: Price closes below N-day moving average
    """

    def __init__(self, lookback_period: int = 20):
        """
        Initialize momentum strategy.

        Args:
            lookback_period: Number of days for moving average calculation (default: 20)
        """
        if lookback_period < 1:
            raise ValueError("lookback_period must be at least 1")
        self.lookback_period = lookback_period

    @property
    def info(self) -> StrategyInfo:
        return StrategyInfo(
            name="momentum",
            description="Simple momentum strategy using moving average crossover",
            parameters={
                "lookback_period": f"Number of days for MA calculation (current: {self.lookback_period})"
            },
        )

    def generate_signals(self, data: OHLCVSeries) -> list[TradeSignal]:
        """
        Generate trading signals based on price vs moving average.

        Args:
            data: Historical OHLCV data for a single ticker

        Returns:
            List of TradeSignal objects
        """
        if not self.validate_data(data, self.lookback_period + 1):
            return []

        df = data.data.copy()
        ticker = data.ticker

        # Calculate moving average
        df["ma"] = df["close"].rolling(window=self.lookback_period).mean()

        # Generate signals starting after we have enough data for MA
        signals = []
        prev_position = None  # Track previous signal to avoid duplicate signals

        for i in range(self.lookback_period, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i - 1]

            if pd.isna(row["ma"]) or pd.isna(prev_row["ma"]):
                continue

            close = row["close"]
            ma = row["ma"]
            prev_close = prev_row["close"]
            prev_ma = prev_row["ma"]

            # Determine signal based on crossover
            signal = Signal.HOLD

            # Buy: price crosses above MA
            if close > ma and prev_close <= prev_ma:
                signal = Signal.BUY
            # Sell: price crosses below MA
            elif close < ma and prev_close >= prev_ma:
                signal = Signal.SELL

            # Only emit signal on state change
            if signal != Signal.HOLD and signal != prev_position:
                # Calculate signal strength based on distance from MA
                distance_pct = abs(close - ma) / ma
                strength = min(distance_pct * 10, 1.0)  # Cap at 1.0

                signals.append(
                    TradeSignal(
                        ticker=ticker,
                        signal=signal,
                        timestamp=df.index[i].to_pydatetime(),
                        price=close,
                        strength=strength,
                        metadata={
                            "ma": ma,
                            "lookback_period": self.lookback_period,
                        },
                    )
                )
                prev_position = signal

        return signals
