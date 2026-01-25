"""Pydantic schemas for Polymarket API endpoints."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


# =============================================================================
# Market Schemas
# =============================================================================


class OutcomeResponse(BaseModel):
    """Response schema for a market outcome."""

    outcome_id: str
    name: str
    price: float = Field(ge=0, le=1, description="Current price (probability)")
    token_id: str


class MarketResponse(BaseModel):
    """Response schema for a prediction market."""

    condition_id: str
    question: str
    description: str
    category: str
    status: str
    outcomes: list[OutcomeResponse]
    volume: float
    liquidity: float
    created_at: str
    end_date: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    image_url: Optional[str] = None

    # Computed fields
    is_binary: bool = True
    yes_price: Optional[float] = None


class MarketsListResponse(BaseModel):
    """Response schema for list of markets."""

    markets: list[MarketResponse]
    total: int
    offset: int
    limit: int


class MarketSearchRequest(BaseModel):
    """Request schema for market search."""

    query: str = Field(min_length=1, description="Search query")
    limit: int = Field(default=20, ge=1, le=100, description="Maximum results")


# =============================================================================
# Signal Schemas
# =============================================================================


class SignalComponentResponse(BaseModel):
    """Response schema for a signal component."""

    source: str = Field(description="Signal source: sentiment, whale, momentum")
    value: float = Field(ge=-1, le=1, description="Signal value")
    confidence: float = Field(ge=0, le=1, description="Confidence level")
    metadata: dict[str, Any] = Field(default_factory=dict)


class CompositeSignalResponse(BaseModel):
    """Response schema for a composite signal."""

    market_id: str
    direction: str = Field(description="buy, sell, or hold")
    strength: float = Field(ge=0, le=1, description="Signal strength")
    components: list[SignalComponentResponse]
    timestamp: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class GenerateSignalsRequest(BaseModel):
    """Request schema for generating signals."""

    market_ids: list[str] = Field(
        min_length=1, max_length=20, description="Market IDs to analyze"
    )
    include_sentiment: bool = Field(default=True, description="Include sentiment signals")
    include_whale: bool = Field(default=True, description="Include whale signals")
    include_momentum: bool = Field(default=True, description="Include momentum signals")
    sentiment_weight: float = Field(default=0.3, ge=0, le=1)
    whale_weight: float = Field(default=0.4, ge=0, le=1)
    momentum_weight: float = Field(default=0.3, ge=0, le=1)


class SignalsResponse(BaseModel):
    """Response schema for generated signals."""

    signals: list[CompositeSignalResponse]
    generated_at: str


# =============================================================================
# Whale Tracking Schemas
# =============================================================================


class WhaleWalletResponse(BaseModel):
    """Response schema for a whale wallet."""

    address: str
    label: Optional[str] = None
    total_volume: float
    win_rate: Optional[float] = None
    first_seen: Optional[str] = None
    last_active: Optional[str] = None


class WhaleTransactionResponse(BaseModel):
    """Response schema for a whale transaction."""

    tx_hash: str
    wallet_address: str
    market_id: str
    outcome_id: str
    transaction_type: str  # buy or sell
    amount: float
    price: float
    timestamp: str


class WhaleActivityRequest(BaseModel):
    """Request schema for whale activity."""

    market_id: Optional[str] = Field(default=None, description="Filter by market")
    wallet_address: Optional[str] = Field(default=None, description="Filter by wallet")
    start_date: Optional[str] = Field(default=None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    limit: int = Field(default=50, ge=1, le=200)


class WhaleActivityResponse(BaseModel):
    """Response schema for whale activity."""

    transactions: list[WhaleTransactionResponse]
    net_flow: float = Field(description="Net buying/selling")
    buy_volume: float
    sell_volume: float
    unique_wallets: int


class WhaleWalletsResponse(BaseModel):
    """Response schema for list of whale wallets."""

    wallets: list[WhaleWalletResponse]
    total: int


# =============================================================================
# Sentiment Schemas
# =============================================================================


class SentimentReadingResponse(BaseModel):
    """Response schema for a sentiment reading."""

    market_id: str
    timestamp: str
    score: float = Field(ge=-1, le=1, description="Sentiment score")
    volume: int = Field(description="Posts analyzed")
    sample_posts: list[str] = Field(default_factory=list)


class MarketSentimentRequest(BaseModel):
    """Request schema for market sentiment."""

    start_date: Optional[str] = None
    end_date: Optional[str] = None


class MarketSentimentResponse(BaseModel):
    """Response schema for market sentiment."""

    market_id: str
    current_score: float
    readings: list[SentimentReadingResponse]
    trend: str = Field(description="improving, declining, or stable")


# =============================================================================
# Backtest Schemas
# =============================================================================


class PolymarketBacktestRequest(BaseModel):
    """Request schema for running a Polymarket backtest."""

    market_ids: list[str] = Field(
        min_length=1,
        max_length=50,
        description="Markets to include in backtest",
    )
    strategy: str = Field(
        default="composite",
        description="Strategy: sentiment, whale, momentum, or composite",
    )
    start_date: str = Field(description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(default=None, description="End date (YYYY-MM-DD)")
    initial_capital: float = Field(default=10000.0, ge=100, description="Starting capital")

    # Strategy weights (for composite strategy)
    sentiment_weight: float = Field(default=0.3, ge=0, le=1)
    whale_weight: float = Field(default=0.4, ge=0, le=1)
    momentum_weight: float = Field(default=0.3, ge=0, le=1)

    # Position sizing
    max_position_pct: float = Field(
        default=0.1, ge=0.01, le=1.0, description="Max position as % of portfolio"
    )
    min_signal_strength: float = Field(
        default=0.3, ge=0, le=1, description="Minimum signal strength to trade"
    )


class PolymarketTradeResponse(BaseModel):
    """Response schema for a Polymarket backtest trade."""

    market_id: str
    outcome_id: str
    side: str
    amount: float
    price: float
    timestamp: str
    signal_strength: float


class PolymarketMetricsResponse(BaseModel):
    """Response schema for Polymarket-specific metrics."""

    # Standard metrics
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float

    # Polymarket-specific metrics
    brier_score: float = Field(description="Prediction accuracy (lower is better)")
    calibration_score: float = Field(description="Prediction reliability")
    resolution_accuracy_pct: float = Field(
        description="% of markets resolved in our favor"
    )
    average_hold_duration_hours: float
    edge_captured_pct: float = Field(
        description="Actual vs expected value captured"
    )
    markets_traded: int
    win_count: int
    loss_count: int


class PolymarketPortfolioValuePoint(BaseModel):
    """A single point in the portfolio value time series."""

    date: str
    value: float


class PolymarketBacktestResponse(BaseModel):
    """Response schema for Polymarket backtest results."""

    initial_capital: float
    final_value: float
    metrics: PolymarketMetricsResponse
    trades: list[PolymarketTradeResponse]
    portfolio_values: list[PolymarketPortfolioValuePoint]
    strategy_used: str
    markets_analyzed: int


# =============================================================================
# Strategy Schemas
# =============================================================================


class PolymarketStrategyInfo(BaseModel):
    """Response schema for a Polymarket strategy."""

    name: str
    description: str
    signal_sources: list[str]
    default_weights: dict[str, float]
    parameters: dict[str, str]


class PolymarketStrategiesResponse(BaseModel):
    """Response schema for list of Polymarket strategies."""

    strategies: list[PolymarketStrategyInfo]


# =============================================================================
# Price History Schemas
# =============================================================================


class PricePointResponse(BaseModel):
    """Response schema for a price point."""

    timestamp: str
    price: float
    volume: float


class PriceHistoryRequest(BaseModel):
    """Request schema for price history."""

    outcome_id: str
    start_date: str
    end_date: Optional[str] = None
    interval: str = Field(default="1h", description="1m, 5m, 1h, or 1d")


class PriceHistoryResponse(BaseModel):
    """Response schema for price history."""

    condition_id: str
    outcome_id: str
    prices: list[PricePointResponse]


# =============================================================================
# Order Book Schemas
# =============================================================================


class OrderBookLevelResponse(BaseModel):
    """Response schema for an order book level."""

    price: float
    size: float


class OrderBookResponse(BaseModel):
    """Response schema for an order book."""

    condition_id: str
    outcome_id: str
    bids: list[OrderBookLevelResponse]
    asks: list[OrderBookLevelResponse]
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    spread: Optional[float] = None
    mid_price: Optional[float] = None
    timestamp: str
