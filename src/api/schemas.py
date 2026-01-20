from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# =============================================================================
# Strategy Definition Schemas
# =============================================================================


class IndicatorDefinition(BaseModel):
    """Definition of a technical indicator."""

    id: str = Field(..., description="Unique identifier for this indicator")
    type: str = Field(
        ..., description="Indicator type (sma, ema, rsi, macd, bollinger_bands, etc.)"
    )
    params: dict[str, Any] = Field(
        default_factory=dict, description="Indicator parameters"
    )


class ConditionDefinition(BaseModel):
    """Definition of a trading condition."""

    type: str = Field(
        ..., description="Condition type (crossover, crossunder, above, below, between)"
    )
    left: str = Field(..., description="Left side reference (indicator id or built-in)")
    right: str | float | None = Field(
        default=None, description="Right side reference or numeric value"
    )
    lower: float | None = Field(default=None, description="Lower bound for 'between'")
    upper: float | None = Field(default=None, description="Upper bound for 'between'")


class RuleDefinition(BaseModel):
    """Definition of a buy or sell rule."""

    logic: str = Field(default="and", description="Logic operator ('and' or 'or')")
    conditions: list[ConditionDefinition] = Field(
        default_factory=list, description="List of conditions"
    )


class RulesDefinition(BaseModel):
    """Definition of buy and sell rules."""

    buy: RuleDefinition | None = Field(default=None, description="Buy rule")
    sell: RuleDefinition | None = Field(default=None, description="Sell rule")


class StrategyDefinition(BaseModel):
    """Complete strategy definition."""

    name: str = Field(..., description="Strategy name")
    description: str | None = Field(default=None, description="Strategy description")
    indicators: list[IndicatorDefinition] = Field(
        default_factory=list, description="List of indicator definitions"
    )
    rules: RulesDefinition = Field(..., description="Buy and sell rules")


class StrategyValidationRequest(BaseModel):
    """Request to validate a strategy definition."""

    definition: StrategyDefinition


class StrategyValidationResponse(BaseModel):
    """Response from strategy validation."""

    is_valid: bool
    errors: list[str] = Field(default_factory=list)


# =============================================================================
# Custom Backtest Schemas
# =============================================================================


class CustomBacktestRequest(BaseModel):
    """Request schema for running a custom strategy backtest."""

    definition: StrategyDefinition = Field(..., description="Strategy definition")
    tickers: list[str] = Field(default=["AAPL"], description="List of ticker symbols")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str | None = Field(default=None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, ge=0, description="Initial capital")
    commission: float = Field(default=0.0, ge=0, description="Commission per trade")


# =============================================================================
# Strategy Comparison Schemas
# =============================================================================


class CompareStrategiesRequest(BaseModel):
    """Request schema for comparing multiple strategies."""

    strategies: list[StrategyDefinition] = Field(
        ..., min_length=2, max_length=4, description="2-4 strategies to compare"
    )
    tickers: list[str] = Field(default=["AAPL"], description="List of ticker symbols")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str | None = Field(default=None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, ge=0, description="Initial capital")
    commission: float = Field(default=0.0, ge=0, description="Commission per trade")


class StrategyComparisonResult(BaseModel):
    """Results for a single strategy in a comparison."""

    name: str
    initial_capital: float
    final_value: float
    metrics: "MetricsResponse"
    trades: list["TradeResponse"]
    portfolio_values: list["PortfolioValuePoint"]


class CompareStrategiesResponse(BaseModel):
    """Response schema for strategy comparison."""

    results: list[StrategyComparisonResult]


# =============================================================================
# Original Backtest Schemas
# =============================================================================


class BacktestRequest(BaseModel):
    """Request schema for running a backtest."""
    strategy: str = Field(default="momentum", description="Strategy name")
    tickers: list[str] = Field(default=["AAPL"], description="List of ticker symbols")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: str | None = Field(default=None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, ge=0, description="Initial capital")
    commission: float = Field(default=0.0, ge=0, description="Commission per trade")
    lookback_period: int = Field(default=20, ge=1, description="Lookback period for momentum")


class TradeResponse(BaseModel):
    """Response schema for a single trade."""
    ticker: str
    side: str
    quantity: int
    price: float
    timestamp: str
    total_value: float


class MetricsResponse(BaseModel):
    """Response schema for performance metrics."""
    total_return_pct: float
    cagr: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown_pct: float
    win_rate_pct: float
    profit_factor: float | str
    num_trades: int


class PortfolioValuePoint(BaseModel):
    """A single point in the portfolio value time series."""
    date: str
    value: float


class BacktestResponse(BaseModel):
    """Response schema for backtest results."""
    initial_capital: float
    final_value: float
    metrics: MetricsResponse
    trades: list[TradeResponse]
    portfolio_values: list[PortfolioValuePoint]


class StrategyInfo(BaseModel):
    """Response schema for strategy information."""
    name: str
    description: str
    parameters: dict[str, str]


class StrategiesResponse(BaseModel):
    """Response schema for list of strategies."""
    strategies: list[StrategyInfo]


class ErrorResponse(BaseModel):
    """Response schema for errors."""
    error: str
    detail: str | None = None


# =============================================================================
# Ticker Price Schemas
# =============================================================================


class TickerPricesRequest(BaseModel):
    """Request schema for fetching ticker prices."""
    tickers: list[str] = Field(..., min_length=1, description="List of ticker symbols")


class TickerPrice(BaseModel):
    """Price data for a single ticker."""
    ticker: str
    price: float
    change_24h: float
    change_percent_24h: float
    previous_close: float
    fetched_at: str


class TickerPricesResponse(BaseModel):
    """Response schema for ticker prices."""
    prices: list[TickerPrice]
    errors: list[str] = Field(default_factory=list)
