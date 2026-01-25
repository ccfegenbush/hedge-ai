"""Core data models for Polymarket prediction markets."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

import pandas as pd


class MarketStatus(Enum):
    """Status of a prediction market."""

    ACTIVE = "active"
    CLOSED = "closed"
    RESOLVED = "resolved"


class MarketCategory(Enum):
    """Categories of prediction markets."""

    POLITICS = "politics"
    SPORTS = "sports"
    CRYPTO = "crypto"
    ENTERTAINMENT = "entertainment"
    SCIENCE = "science"
    ECONOMICS = "economics"
    OTHER = "other"


class OutcomeType(Enum):
    """Type of market outcome."""

    BINARY = "binary"  # Yes/No
    MULTIPLE = "multiple"  # Multiple choice


class SignalDirection(Enum):
    """Direction of a trading signal."""

    BUY = "buy"  # Buy YES tokens (price will increase)
    SELL = "sell"  # Sell YES tokens (price will decrease)
    HOLD = "hold"  # No action


@dataclass
class Outcome:
    """A single outcome in a prediction market."""

    outcome_id: str
    name: str
    price: float  # Current price (0-1, represents probability)
    token_id: str  # CLOB token ID


@dataclass
class Market:
    """A prediction market on Polymarket."""

    condition_id: str
    question: str
    description: str
    category: MarketCategory
    status: MarketStatus
    outcomes: list[Outcome]
    volume: float  # Total volume in USDC
    liquidity: float  # Current liquidity
    created_at: datetime
    end_date: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_outcome: Optional[str] = None  # Winning outcome ID
    tags: list[str] = field(default_factory=list)
    image_url: Optional[str] = None

    @property
    def is_active(self) -> bool:
        """Check if market is currently tradeable."""
        return self.status == MarketStatus.ACTIVE

    @property
    def is_binary(self) -> bool:
        """Check if this is a binary (Yes/No) market."""
        return len(self.outcomes) == 2

    @property
    def yes_price(self) -> Optional[float]:
        """Get the YES price for binary markets."""
        if not self.is_binary:
            return None
        for outcome in self.outcomes:
            if outcome.name.lower() in ["yes", "y"]:
                return outcome.price
        return self.outcomes[0].price if self.outcomes else None


@dataclass
class PricePoint:
    """A single price point in a market's history."""

    timestamp: datetime
    price: float
    volume: float


@dataclass
class PriceHistory:
    """Historical prices for a market outcome."""

    condition_id: str
    outcome_id: str
    data: pd.DataFrame  # DataFrame with columns: timestamp, price, volume

    @property
    def start_date(self) -> datetime:
        return self.data["timestamp"].min()

    @property
    def end_date(self) -> datetime:
        return self.data["timestamp"].max()

    def __len__(self) -> int:
        return len(self.data)


@dataclass
class OrderBookLevel:
    """A single level in an order book."""

    price: float
    size: float  # Amount at this price level


@dataclass
class OrderBook:
    """Order book for a market outcome."""

    condition_id: str
    outcome_id: str
    bids: list[OrderBookLevel]  # Sorted by price descending
    asks: list[OrderBookLevel]  # Sorted by price ascending
    timestamp: datetime

    @property
    def best_bid(self) -> Optional[float]:
        """Get the highest bid price."""
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        """Get the lowest ask price."""
        return self.asks[0].price if self.asks else None

    @property
    def spread(self) -> Optional[float]:
        """Get the bid-ask spread."""
        if self.best_bid is not None and self.best_ask is not None:
            return self.best_ask - self.best_bid
        return None

    @property
    def mid_price(self) -> Optional[float]:
        """Get the mid-market price."""
        if self.best_bid is not None and self.best_ask is not None:
            return (self.best_bid + self.best_ask) / 2
        return None


# =============================================================================
# Signal Models
# =============================================================================


@dataclass
class SignalComponent:
    """A component signal from a single source."""

    source: str  # "sentiment", "whale", "momentum"
    value: float  # -1 to 1
    confidence: float  # 0 to 1
    metadata: dict = field(default_factory=dict)


@dataclass
class CompositeSignal:
    """A combined signal from multiple sources."""

    market_id: str
    direction: SignalDirection
    strength: float  # 0 to 1
    components: list[SignalComponent]
    timestamp: datetime
    metadata: dict = field(default_factory=dict)

    @property
    def sentiment_score(self) -> Optional[float]:
        """Get the sentiment component score."""
        for comp in self.components:
            if comp.source == "sentiment":
                return comp.value
        return None

    @property
    def whale_score(self) -> Optional[float]:
        """Get the whale tracking component score."""
        for comp in self.components:
            if comp.source == "whale":
                return comp.value
        return None

    @property
    def momentum_score(self) -> Optional[float]:
        """Get the momentum component score."""
        for comp in self.components:
            if comp.source == "momentum":
                return comp.value
        return None


# =============================================================================
# Whale Tracking Models
# =============================================================================


@dataclass
class WhaleWallet:
    """A tracked whale wallet."""

    address: str
    label: Optional[str] = None  # e.g., "Whale #1", "Smart Money"
    total_volume: float = 0.0
    win_rate: Optional[float] = None
    first_seen: Optional[datetime] = None
    last_active: Optional[datetime] = None


class TransactionType(Enum):
    """Type of whale transaction."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class WhaleTransaction:
    """A single whale transaction."""

    tx_hash: str
    wallet_address: str
    market_id: str
    outcome_id: str
    transaction_type: TransactionType
    amount: float  # USDC amount
    price: float  # Price at execution
    timestamp: datetime
    block_number: int


@dataclass
class WhaleFlow:
    """Aggregated whale activity for a market."""

    market_id: str
    period_start: datetime
    period_end: datetime
    net_flow: float  # Positive = net buying, Negative = net selling
    buy_volume: float
    sell_volume: float
    unique_wallets: int
    transactions: list[WhaleTransaction]


# =============================================================================
# Sentiment Models
# =============================================================================


@dataclass
class SentimentReading:
    """A single sentiment reading."""

    market_id: str
    timestamp: datetime
    score: float  # -1 to 1
    volume: int  # Number of posts analyzed
    sample_posts: list[str] = field(default_factory=list)


@dataclass
class InfluencerPost:
    """A post from an influential account."""

    account_handle: str
    market_id: Optional[str]
    post_text: str
    sentiment_score: Optional[float]
    prediction_extracted: Optional[str]
    timestamp: datetime


# =============================================================================
# Backtest Models
# =============================================================================


@dataclass
class PolymarketTrade:
    """A trade in a Polymarket backtest."""

    market_id: str
    outcome_id: str
    side: TransactionType
    amount: float  # USDC amount
    price: float
    timestamp: datetime
    signal_strength: float = 1.0


@dataclass
class PolymarketPosition:
    """A position in a Polymarket market."""

    market_id: str
    outcome_id: str
    tokens: float  # Number of outcome tokens
    avg_cost: float  # Average cost per token
    entry_time: datetime

    @property
    def cost_basis(self) -> float:
        """Total cost basis of the position."""
        return self.tokens * self.avg_cost


@dataclass
class PolymarketBacktestResult:
    """Results from a Polymarket backtest."""

    initial_capital: float
    final_value: float
    trades: list[PolymarketTrade]
    portfolio_values: pd.Series  # Time series of portfolio values

    # Polymarket-specific fields
    markets_traded: int
    resolutions: dict[str, bool]  # market_id -> won (True) or lost (False)

    @property
    def total_return(self) -> float:
        """Total return as a decimal."""
        return (self.final_value - self.initial_capital) / self.initial_capital

    @property
    def total_return_pct(self) -> float:
        """Total return as a percentage."""
        return self.total_return * 100

    @property
    def win_count(self) -> int:
        """Number of winning market resolutions."""
        return sum(1 for won in self.resolutions.values() if won)

    @property
    def loss_count(self) -> int:
        """Number of losing market resolutions."""
        return sum(1 for won in self.resolutions.values() if not won)

    @property
    def resolution_accuracy(self) -> float:
        """Percentage of markets resolved in our favor."""
        if not self.resolutions:
            return 0.0
        return self.win_count / len(self.resolutions) * 100
