"""Technical indicators library for rule-based strategies."""

import numpy as np
import pandas as pd


def sma(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Simple Moving Average.

    Args:
        data: Price series (typically close prices)
        period: Number of periods for the moving average

    Returns:
        Series with SMA values
    """
    return data.rolling(window=period).mean()


def ema(data: pd.Series, period: int) -> pd.Series:
    """
    Calculate Exponential Moving Average.

    Args:
        data: Price series (typically close prices)
        period: Number of periods for the EMA

    Returns:
        Series with EMA values
    """
    return data.ewm(span=period, adjust=False).mean()


def rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Relative Strength Index.

    Args:
        data: Price series (typically close prices)
        period: Number of periods for RSI calculation (default: 14)

    Returns:
        Series with RSI values (0-100)
    """
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi_values = 100 - (100 / (1 + rs))

    return rsi_values


def macd(
    data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate MACD (Moving Average Convergence Divergence).

    Args:
        data: Price series (typically close prices)
        fast: Fast EMA period (default: 12)
        slow: Slow EMA period (default: 26)
        signal: Signal line period (default: 9)

    Returns:
        Tuple of (macd_line, signal_line, histogram)
    """
    fast_ema = ema(data, fast)
    slow_ema = ema(data, slow)

    macd_line = fast_ema - slow_ema
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def bollinger_bands(
    data: pd.Series, period: int = 20, std_dev: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """
    Calculate Bollinger Bands.

    Args:
        data: Price series (typically close prices)
        period: Number of periods for the moving average (default: 20)
        std_dev: Number of standard deviations for bands (default: 2.0)

    Returns:
        Tuple of (upper_band, middle_band, lower_band)
    """
    middle = sma(data, period)
    std = data.rolling(window=period).std()

    upper = middle + (std * std_dev)
    lower = middle - (std * std_dev)

    return upper, middle, lower


def volume_sma(volume: pd.Series, period: int) -> pd.Series:
    """
    Calculate Volume Simple Moving Average.

    Args:
        volume: Volume series
        period: Number of periods for the moving average

    Returns:
        Series with volume SMA values
    """
    return volume.rolling(window=period).mean()


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Calculate On-Balance Volume.

    Args:
        close: Close price series
        volume: Volume series

    Returns:
        Series with OBV values
    """
    direction = np.sign(close.diff())
    direction.iloc[0] = 0
    obv_values = (direction * volume).cumsum()
    return obv_values


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.

    Args:
        high: High price series
        low: Low price series
        close: Close price series
        period: Number of periods for ATR (default: 14)

    Returns:
        Series with ATR values
    """
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = abs(high - prev_close)
    tr3 = abs(low - prev_close)

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


# Indicator registry for easy lookup
INDICATOR_FUNCTIONS = {
    "sma": sma,
    "ema": ema,
    "rsi": rsi,
    "macd": macd,
    "bollinger_bands": bollinger_bands,
    "volume_sma": volume_sma,
    "obv": obv,
    "atr": atr,
}

# Parameter bounds for validation
INDICATOR_PARAMS = {
    "sma": {"period": {"min": 1, "max": 500, "type": "int"}},
    "ema": {"period": {"min": 1, "max": 500, "type": "int"}},
    "rsi": {"period": {"min": 1, "max": 100, "type": "int"}},
    "macd": {
        "fast": {"min": 1, "max": 100, "type": "int"},
        "slow": {"min": 1, "max": 200, "type": "int"},
        "signal": {"min": 1, "max": 100, "type": "int"},
    },
    "bollinger_bands": {
        "period": {"min": 1, "max": 500, "type": "int"},
        "std_dev": {"min": 0.1, "max": 5.0, "type": "float"},
    },
    "volume_sma": {"period": {"min": 1, "max": 500, "type": "int"}},
    "obv": {},  # No parameters
    "atr": {"period": {"min": 1, "max": 100, "type": "int"}},
}
