"""Rule-based strategy implementation using JSON definitions."""

from typing import Any, Optional

import numpy as np
import pandas as pd

from src.models.schemas import OHLCVSeries, Signal, TradeSignal
from src.strategies.base import Strategy, StrategyInfo
from src.strategies.indicators import (
    atr,
    bollinger_bands,
    ema,
    macd,
    obv,
    rsi,
    sma,
    volume_sma,
)
from src.strategies.validator import validate_strategy_definition


class RuleBasedStrategy(Strategy):
    """
    A flexible rule-based strategy driven by JSON definitions.

    Supports technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, volume)
    and various condition types (crossover, crossunder, above, below, between).
    """

    def __init__(self, definition: dict[str, Any]):
        """
        Initialize rule-based strategy from a definition.

        Args:
            definition: Strategy definition dictionary containing:
                - name: Strategy name
                - description: Optional strategy description
                - indicators: List of indicator definitions
                - rules: Buy and sell rules with conditions

        Raises:
            ValueError: If definition is invalid
        """
        is_valid, errors = validate_strategy_definition(definition)
        if not is_valid:
            raise ValueError(f"Invalid strategy definition: {'; '.join(errors)}")

        self.definition = definition
        self._name = definition["name"]
        self._description = definition.get("description", "Custom rule-based strategy")
        self._indicators = definition.get("indicators", [])
        self._rules = definition.get("rules", {})

    @property
    def info(self) -> StrategyInfo:
        """Return metadata about the strategy."""
        # Build parameters dict from indicator configs
        params = {}
        for ind in self._indicators:
            ind_params = ind.get("params", {})
            for param, value in ind_params.items():
                params[f"{ind['id']}.{param}"] = str(value)

        return StrategyInfo(
            name=self._name,
            description=self._description,
            parameters=params,
        )

    def generate_signals(self, data: OHLCVSeries) -> list[TradeSignal]:
        """
        Generate trading signals based on the rule definition.

        Args:
            data: Historical OHLCV data for a single ticker

        Returns:
            List of TradeSignal objects
        """
        df = data.data.copy()
        ticker = data.ticker

        # Compute all indicators
        indicators = self._compute_indicators(df)

        # Determine minimum required periods based on indicators
        min_periods = self._get_min_periods()

        if len(df) < min_periods:
            return []

        signals = []
        prev_signal: Optional[Signal] = None

        # Evaluate rules for each bar after warmup period
        for idx in range(min_periods - 1, len(df)):
            buy_triggered = False
            sell_triggered = False

            # Check buy rule
            if "buy" in self._rules:
                buy_triggered = self._evaluate_rule(
                    self._rules["buy"], indicators, idx
                )

            # Check sell rule
            if "sell" in self._rules:
                sell_triggered = self._evaluate_rule(
                    self._rules["sell"], indicators, idx
                )

            # Determine signal (buy takes precedence if both trigger)
            current_signal = Signal.HOLD
            if buy_triggered and not sell_triggered:
                current_signal = Signal.BUY
            elif sell_triggered and not buy_triggered:
                current_signal = Signal.SELL

            # Only emit signal on state change
            if current_signal != Signal.HOLD and current_signal != prev_signal:
                close_price = df.iloc[idx]["close"]
                timestamp = df.index[idx].to_pydatetime()

                # Calculate signal strength based on how many conditions matched strongly
                strength = self._calculate_signal_strength(
                    current_signal, indicators, idx
                )

                signals.append(
                    TradeSignal(
                        ticker=ticker,
                        signal=current_signal,
                        timestamp=timestamp,
                        price=close_price,
                        strength=strength,
                        metadata=self._get_signal_metadata(indicators, idx),
                    )
                )
                prev_signal = current_signal

        return signals

    def _compute_indicators(self, df: pd.DataFrame) -> dict[str, pd.Series]:
        """
        Compute all indicators defined in the strategy.

        Args:
            df: OHLCV DataFrame

        Returns:
            Dictionary mapping indicator IDs to their computed values
        """
        indicators: dict[str, pd.Series] = {}

        # Add built-in references
        indicators["close"] = df["close"]
        indicators["open"] = df["open"]
        indicators["high"] = df["high"]
        indicators["low"] = df["low"]
        indicators["volume"] = df["volume"]

        # Compute each defined indicator
        for ind in self._indicators:
            ind_id = ind["id"]
            ind_type = ind["type"]
            params = ind.get("params", {})

            if ind_type == "sma":
                period = params.get("period", 20)
                indicators[ind_id] = sma(df["close"], period)

            elif ind_type == "ema":
                period = params.get("period", 20)
                indicators[ind_id] = ema(df["close"], period)

            elif ind_type == "rsi":
                period = params.get("period", 14)
                indicators[ind_id] = rsi(df["close"], period)

            elif ind_type == "macd":
                fast = params.get("fast", 12)
                slow = params.get("slow", 26)
                signal = params.get("signal", 9)
                macd_line, signal_line, histogram = macd(df["close"], fast, slow, signal)
                indicators[f"{ind_id}.line"] = macd_line
                indicators[f"{ind_id}.signal"] = signal_line
                indicators[f"{ind_id}.histogram"] = histogram
                # Also store the main MACD line under the base ID for simple references
                indicators[ind_id] = macd_line

            elif ind_type == "bollinger_bands":
                period = params.get("period", 20)
                std_dev = params.get("std_dev", 2.0)
                upper, middle, lower = bollinger_bands(df["close"], period, std_dev)
                indicators[f"{ind_id}.upper"] = upper
                indicators[f"{ind_id}.middle"] = middle
                indicators[f"{ind_id}.lower"] = lower
                # Store middle as the base reference
                indicators[ind_id] = middle

            elif ind_type == "volume_sma":
                period = params.get("period", 20)
                indicators[ind_id] = volume_sma(df["volume"], period)

            elif ind_type == "obv":
                indicators[ind_id] = obv(df["close"], df["volume"])

            elif ind_type == "atr":
                period = params.get("period", 14)
                indicators[ind_id] = atr(df["high"], df["low"], df["close"], period)

        return indicators

    def _evaluate_rule(
        self, rule: dict[str, Any], indicators: dict[str, pd.Series], idx: int
    ) -> bool:
        """
        Evaluate a rule (buy or sell) at a specific index.

        Args:
            rule: Rule definition with logic and conditions
            indicators: Computed indicator values
            idx: Current bar index

        Returns:
            True if rule conditions are met
        """
        logic = rule.get("logic", "and")
        conditions = rule.get("conditions", [])

        if not conditions:
            return False

        results = [
            self._evaluate_condition(cond, indicators, idx) for cond in conditions
        ]

        if logic == "and":
            return all(results)
        else:  # or
            return any(results)

    def _evaluate_condition(
        self, condition: dict[str, Any], indicators: dict[str, pd.Series], idx: int
    ) -> bool:
        """
        Evaluate a single condition at a specific index.

        Args:
            condition: Condition definition
            indicators: Computed indicator values
            idx: Current bar index

        Returns:
            True if condition is met
        """
        cond_type = condition["type"]
        left_ref = condition["left"]

        # Get left value
        left_series = indicators.get(left_ref)
        if left_series is None:
            return False

        left_val = left_series.iloc[idx]
        if pd.isna(left_val):
            return False

        # Handle different condition types
        if cond_type == "between":
            lower = condition["lower"]
            upper = condition["upper"]
            return lower <= left_val <= upper

        # Get right value for comparison conditions
        right_ref = condition.get("right")
        if right_ref is None:
            return False

        # Right can be a numeric constant or an indicator reference
        if isinstance(right_ref, (int, float)):
            right_val = right_ref
            right_prev = right_ref
        else:
            right_series = indicators.get(right_ref)
            if right_series is None:
                return False
            right_val = right_series.iloc[idx]
            if pd.isna(right_val):
                return False
            right_prev = right_series.iloc[idx - 1] if idx > 0 else np.nan

        if cond_type == "above":
            return left_val > right_val

        elif cond_type == "below":
            return left_val < right_val

        elif cond_type == "crossover":
            # Current: left > right, Previous: left <= right
            if idx == 0:
                return False
            left_prev = left_series.iloc[idx - 1]
            if pd.isna(left_prev):
                return False
            return left_val > right_val and left_prev <= right_prev

        elif cond_type == "crossunder":
            # Current: left < right, Previous: left >= right
            if idx == 0:
                return False
            left_prev = left_series.iloc[idx - 1]
            if pd.isna(left_prev):
                return False
            return left_val < right_val and left_prev >= right_prev

        return False

    def _get_min_periods(self) -> int:
        """Calculate minimum periods needed for all indicators."""
        min_periods = 1

        for ind in self._indicators:
            ind_type = ind["type"]
            params = ind.get("params", {})

            if ind_type == "sma":
                min_periods = max(min_periods, params.get("period", 20))
            elif ind_type == "ema":
                # EMA needs about 3x the period for stability
                min_periods = max(min_periods, params.get("period", 20) * 2)
            elif ind_type == "rsi":
                min_periods = max(min_periods, params.get("period", 14) + 1)
            elif ind_type == "macd":
                slow = params.get("slow", 26)
                signal = params.get("signal", 9)
                min_periods = max(min_periods, slow + signal)
            elif ind_type == "bollinger_bands":
                min_periods = max(min_periods, params.get("period", 20))
            elif ind_type == "volume_sma":
                min_periods = max(min_periods, params.get("period", 20))
            elif ind_type == "atr":
                min_periods = max(min_periods, params.get("period", 14) + 1)

        return min_periods

    def _calculate_signal_strength(
        self, signal: Signal, indicators: dict[str, pd.Series], idx: int
    ) -> float:
        """
        Calculate signal strength based on indicator values.

        Args:
            signal: The signal type (BUY or SELL)
            indicators: Computed indicator values
            idx: Current bar index

        Returns:
            Signal strength between 0 and 1
        """
        # Simple strength calculation based on distance from trigger
        # Can be enhanced with more sophisticated logic
        base_strength = 0.5

        # Boost strength if RSI confirms the signal
        for ind in self._indicators:
            if ind["type"] == "rsi":
                rsi_val = indicators.get(ind["id"])
                if rsi_val is not None:
                    current_rsi = rsi_val.iloc[idx]
                    if not pd.isna(current_rsi):
                        if signal == Signal.BUY and current_rsi < 30:
                            base_strength += 0.3  # Oversold confirms buy
                        elif signal == Signal.SELL and current_rsi > 70:
                            base_strength += 0.3  # Overbought confirms sell

        return min(base_strength, 1.0)

    def _get_signal_metadata(
        self, indicators: dict[str, pd.Series], idx: int
    ) -> dict[str, Any]:
        """
        Get metadata about the current state at signal generation.

        Args:
            indicators: Computed indicator values
            idx: Current bar index

        Returns:
            Dictionary of indicator values at the signal point
        """
        metadata: dict[str, Any] = {}

        for ind in self._indicators:
            ind_id = ind["id"]
            ind_type = ind["type"]

            if ind_type == "macd":
                for suffix in [".line", ".signal", ".histogram"]:
                    key = f"{ind_id}{suffix}"
                    if key in indicators:
                        val = indicators[key].iloc[idx]
                        if not pd.isna(val):
                            metadata[key] = round(val, 4)
            elif ind_type == "bollinger_bands":
                for suffix in [".upper", ".middle", ".lower"]:
                    key = f"{ind_id}{suffix}"
                    if key in indicators:
                        val = indicators[key].iloc[idx]
                        if not pd.isna(val):
                            metadata[key] = round(val, 2)
            else:
                if ind_id in indicators:
                    val = indicators[ind_id].iloc[idx]
                    if not pd.isna(val):
                        metadata[ind_id] = round(val, 4)

        return metadata
