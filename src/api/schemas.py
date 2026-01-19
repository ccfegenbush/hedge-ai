from datetime import datetime
from pydantic import BaseModel, Field


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
