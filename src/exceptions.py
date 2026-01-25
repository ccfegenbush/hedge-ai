"""Custom exception classes for the hedge-ai platform."""


class HedgeAIError(Exception):
    """Base exception for all hedge-ai errors."""

    pass


# =============================================================================
# Polymarket Exceptions
# =============================================================================


class PolymarketError(HedgeAIError):
    """Base exception for Polymarket-related errors."""

    pass


class MarketNotFoundError(PolymarketError):
    """Raised when a market cannot be found."""

    def __init__(self, market_id: str):
        self.market_id = market_id
        super().__init__(f"Market not found: {market_id}")


class DataProviderError(PolymarketError):
    """Raised when a data provider fails to fetch data."""

    def __init__(self, provider: str, message: str):
        self.provider = provider
        super().__init__(f"{provider} error: {message}")


class SentimentUnavailableError(PolymarketError):
    """Raised when sentiment data is unavailable."""

    def __init__(self, query: str, reason: str | None = None):
        self.query = query
        self.reason = reason
        msg = f"Sentiment unavailable for: {query}"
        if reason:
            msg += f" ({reason})"
        super().__init__(msg)


class WhaleTrackerError(PolymarketError):
    """Raised when whale tracking fails."""

    def __init__(self, message: str):
        super().__init__(f"Whale tracker error: {message}")


class RateLimitError(PolymarketError):
    """Raised when an API rate limit is hit."""

    def __init__(self, service: str, retry_after: int | None = None):
        self.service = service
        self.retry_after = retry_after
        msg = f"Rate limit exceeded for {service}"
        if retry_after:
            msg += f" (retry after {retry_after}s)"
        super().__init__(msg)


class InvalidMarketStateError(PolymarketError):
    """Raised when a market is in an invalid state for the operation."""

    def __init__(self, market_id: str, state: str, expected: str):
        self.market_id = market_id
        self.state = state
        self.expected = expected
        super().__init__(
            f"Market {market_id} is in state '{state}', expected '{expected}'"
        )


# =============================================================================
# Backtest Exceptions
# =============================================================================


class BacktestError(HedgeAIError):
    """Base exception for backtest-related errors."""

    pass


class InsufficientDataError(BacktestError):
    """Raised when there's not enough data for backtesting."""

    def __init__(self, market_id: str, required: int, available: int):
        self.market_id = market_id
        self.required = required
        self.available = available
        super().__init__(
            f"Insufficient data for {market_id}: need {required} points, have {available}"
        )


class StrategyError(BacktestError):
    """Raised when a strategy fails to execute."""

    def __init__(self, strategy_name: str, message: str):
        self.strategy_name = strategy_name
        super().__init__(f"Strategy '{strategy_name}' failed: {message}")
