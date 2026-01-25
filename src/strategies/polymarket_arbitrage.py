"""
Polymarket Pure Arbitrage Strategy

This strategy exploits price inefficiencies in binary (Yes/No) prediction markets.
When the combined cost of buying both Yes and No shares is less than $1.00,
there's a guaranteed profit opportunity at market resolution.

Example:
    - Yes price: $0.48 (best ask)
    - No price: $0.50 (best ask)
    - Combined: $0.98
    - Buying $100 of each = $200 total investment
    - Guaranteed payout at resolution: $200 (since one side always wins)
    - But you only paid: $196 (98% of $200)
    - Guaranteed profit: $4 (2% risk-free return)

Strategy focuses on:
    1. Scanning all active binary markets
    2. Fetching order books for both Yes and No outcomes
    3. Calculating combined ask prices
    4. Identifying opportunities below threshold (accounting for fees)
    5. Sizing trades based on liquidity depth
    6. Risk management and position limits
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from src.data.polymarket_provider import PolymarketProvider
from src.exceptions import DataProviderError, RateLimitError
from src.models.polymarket_schemas import (
    CompositeSignal,
    Market,
    OrderBook,
    PriceHistory,
    SignalComponent,
    SignalDirection,
)
from src.strategies.polymarket_base import PolymarketStrategy, PolymarketStrategyInfo

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration Constants
# =============================================================================

# Arbitrage thresholds
ARB_THRESHOLD = 0.99  # Combined price must be below this (leaves 1% for fees)
MIN_EDGE_SMALL_TRADE = 0.02  # 2% edge for small trades
MIN_EDGE_LARGE_TRADE = 0.03  # 3% edge for larger trades (more buffer)
LARGE_TRADE_THRESHOLD_USDC = 500  # Trades above this need larger edge

# Liquidity requirements
MIN_LIQUIDITY_USDC = 100  # Minimum liquidity on each side
MIN_LIQUIDITY_DEPTH_LEVELS = 3  # Need at least this many order book levels
MAX_SLIPPAGE_PCT = 0.005  # Max 0.5% slippage tolerated

# Position sizing
MAX_TRADE_SIZE_PCT = 0.05  # Max 5% of capital per trade
MAX_POSITION_PER_MARKET_PCT = 0.10  # Max 10% of capital in any single market
MIN_TRADE_SIZE_USDC = 10  # Minimum trade size worth executing

# Market filters
MIN_MARKET_VOLUME_USDC = 1000  # Only consider markets with sufficient volume
MIN_HOURS_UNTIL_RESOLUTION = 24  # Avoid markets about to resolve (timing risk)
MAX_HOURS_UNTIL_RESOLUTION = 8760  # 1 year - avoid very long-dated markets

# Fee estimates (conservative)
ESTIMATED_FEE_PCT = 0.002  # 0.2% estimated fees per side
ESTIMATED_GAS_USDC = 0.50  # Estimated gas cost per transaction


# =============================================================================
# Arbitrage Opportunity Model
# =============================================================================


@dataclass
class ArbitrageOpportunity:
    """Represents a detected arbitrage opportunity."""

    market_id: str
    market_question: str
    yes_token_id: str
    no_token_id: str
    yes_ask_price: float
    no_ask_price: float
    combined_price: float
    edge_pct: float  # (1 - combined_price) * 100
    yes_liquidity_usdc: float  # Available liquidity at best ask
    no_liquidity_usdc: float
    max_trade_size_usdc: float  # Limited by liquidity
    expected_profit_usdc: float
    expected_profit_pct: float
    timestamp: datetime
    hours_until_resolution: Optional[float] = None
    market_volume: float = 0.0
    confidence: float = 1.0

    @property
    def is_profitable_after_fees(self) -> bool:
        """Check if opportunity is profitable after estimated fees."""
        total_fees = (
            (self.max_trade_size_usdc * 2 * ESTIMATED_FEE_PCT)  # Trading fees both sides
            + (ESTIMATED_GAS_USDC * 2)  # Gas for both transactions
        )
        return self.expected_profit_usdc > total_fees

    @property
    def net_profit_usdc(self) -> float:
        """Expected profit after fees."""
        total_fees = (
            (self.max_trade_size_usdc * 2 * ESTIMATED_FEE_PCT)
            + (ESTIMATED_GAS_USDC * 2)
        )
        return self.expected_profit_usdc - total_fees

    def to_dict(self) -> dict:
        """Convert to dictionary for logging/storage."""
        return {
            "market_id": self.market_id,
            "market_question": self.market_question[:100],  # Truncate
            "yes_price": round(self.yes_ask_price, 4),
            "no_price": round(self.no_ask_price, 4),
            "combined_price": round(self.combined_price, 4),
            "edge_pct": round(self.edge_pct, 2),
            "max_trade_usdc": round(self.max_trade_size_usdc, 2),
            "expected_profit": round(self.expected_profit_usdc, 2),
            "net_profit": round(self.net_profit_usdc, 2),
            "confidence": round(self.confidence, 2),
            "hours_until_resolution": self.hours_until_resolution,
        }


@dataclass
class TradeRecommendation:
    """A recommended arbitrage trade."""

    market_id: str
    action: str  # "buy_yes_and_no"
    yes_token_id: str
    no_token_id: str
    yes_amount_usdc: float
    no_amount_usdc: float
    total_amount_usdc: float
    expected_profit_usdc: float
    expected_profit_pct: float
    reason: str
    opportunity: ArbitrageOpportunity
    timestamp: datetime
    is_dry_run: bool = False

    def to_dict(self) -> dict:
        """Convert to dictionary for execution/logging."""
        return {
            "market_id": self.market_id,
            "action": self.action,
            "yes_token_id": self.yes_token_id,
            "no_token_id": self.no_token_id,
            "yes_amount_usdc": round(self.yes_amount_usdc, 2),
            "no_amount_usdc": round(self.no_amount_usdc, 2),
            "total_amount_usdc": round(self.total_amount_usdc, 2),
            "expected_profit_usdc": round(self.expected_profit_usdc, 4),
            "expected_profit_pct": round(self.expected_profit_pct, 2),
            "reason": self.reason,
            "is_dry_run": self.is_dry_run,
            "timestamp": self.timestamp.isoformat(),
        }


# =============================================================================
# Arbitrage Strategy Implementation
# =============================================================================


class ArbitrageStrategy(PolymarketStrategy):
    """
    Pure arbitrage strategy for Polymarket binary markets.

    Scans all active binary markets for price inefficiencies where
    the combined cost of Yes + No shares is less than $1.00.
    """

    def __init__(
        self,
        provider: PolymarketProvider,
        arb_threshold: float = ARB_THRESHOLD,
        min_liquidity_usdc: float = MIN_LIQUIDITY_USDC,
        max_trade_size_pct: float = MAX_TRADE_SIZE_PCT,
        min_trade_size_usdc: float = MIN_TRADE_SIZE_USDC,
        dry_run: bool = True,
    ):
        """
        Initialize the arbitrage strategy.

        Args:
            provider: Polymarket data provider
            arb_threshold: Maximum combined price to trigger (default 0.99)
            min_liquidity_usdc: Minimum liquidity required on each side
            max_trade_size_pct: Maximum trade size as % of capital
            min_trade_size_usdc: Minimum trade size worth executing
            dry_run: If True, only log opportunities without executing
        """
        self.provider = provider
        self.arb_threshold = arb_threshold
        self.min_liquidity_usdc = min_liquidity_usdc
        self.max_trade_size_pct = max_trade_size_pct
        self.min_trade_size_usdc = min_trade_size_usdc
        self.dry_run = dry_run

        # Tracking
        self._opportunities_found: list[ArbitrageOpportunity] = []
        self._last_scan_time: Optional[datetime] = None

    @property
    def info(self) -> PolymarketStrategyInfo:
        """Return strategy metadata."""
        return PolymarketStrategyInfo(
            name="arbitrage",
            description=(
                "Pure arbitrage strategy that exploits price inefficiencies "
                "in binary markets where Yes + No prices sum to less than $1.00"
            ),
            signal_sources=["order_book"],
            parameters={
                "arb_threshold": f"Combined price threshold ({self.arb_threshold})",
                "min_liquidity_usdc": f"Minimum liquidity per side (${self.min_liquidity_usdc})",
                "max_trade_size_pct": f"Max trade size as % of capital ({self.max_trade_size_pct * 100}%)",
                "dry_run": f"Dry run mode ({self.dry_run})",
            },
        )

    async def generate_signal(
        self,
        market: Market,
        price_history: Optional[PriceHistory] = None,
        timestamp: Optional[datetime] = None,
    ) -> CompositeSignal:
        """
        Generate arbitrage signal for a single market.

        This method analyzes a single market's order books to detect
        arbitrage opportunities.

        Args:
            market: Market to analyze
            price_history: Not used for arbitrage (order book only)
            timestamp: Signal timestamp

        Returns:
            CompositeSignal with arbitrage opportunity details
        """
        timestamp = timestamp or datetime.now()

        # Check if market is suitable for arbitrage
        if not self._is_market_eligible(market):
            return self._create_neutral_signal(market.condition_id, timestamp)

        try:
            # Fetch order books for both outcomes
            opportunity = await self._analyze_market(market)

            if opportunity is None:
                return self._create_neutral_signal(market.condition_id, timestamp)

            # Create signal based on opportunity
            signal_value = self._calculate_signal_value(opportunity)
            confidence = self._calculate_confidence(opportunity)

            component = SignalComponent(
                source="arbitrage",
                value=signal_value,
                confidence=confidence,
                metadata={
                    "combined_price": opportunity.combined_price,
                    "edge_pct": opportunity.edge_pct,
                    "yes_price": opportunity.yes_ask_price,
                    "no_price": opportunity.no_ask_price,
                    "max_trade_usdc": opportunity.max_trade_size_usdc,
                    "expected_profit": opportunity.expected_profit_usdc,
                },
            )

            # Determine direction (always BUY for arb - buy both sides)
            direction = SignalDirection.BUY if signal_value > 0 else SignalDirection.HOLD

            return self._create_signal(
                market_id=market.condition_id,
                components=[component],
                timestamp=timestamp,
                metadata={
                    "opportunity": opportunity.to_dict(),
                    "is_arbitrage": True,
                },
            )

        except (DataProviderError, RateLimitError) as e:
            logger.warning(f"Failed to analyze market {market.condition_id}: {e}")
            return self._create_neutral_signal(market.condition_id, timestamp)

    async def scan_all_markets(
        self,
        available_capital: float = 10000.0,
        max_markets: int = 100,
    ) -> list[ArbitrageOpportunity]:
        """
        Scan all active binary markets for arbitrage opportunities.

        This is the main method for the arbitrage bot loop.

        Args:
            available_capital: Capital available for trading
            max_markets: Maximum markets to scan (for rate limiting)

        Returns:
            List of detected arbitrage opportunities, sorted by edge
        """
        self._last_scan_time = datetime.now()
        self._opportunities_found = []

        logger.info(f"Starting arbitrage scan with ${available_capital:.2f} capital")

        try:
            # Fetch active markets
            markets = await self.provider.get_active_markets(limit=max_markets)
            logger.info(f"Fetched {len(markets)} active markets")

            # Filter to binary markets only
            binary_markets = [m for m in markets if m.is_binary]
            logger.info(f"Found {len(binary_markets)} binary markets")

            # Analyze each market
            for market in binary_markets:
                if not self._is_market_eligible(market):
                    continue

                try:
                    opportunity = await self._analyze_market(market)
                    if opportunity and opportunity.is_profitable_after_fees:
                        # Apply position sizing limits
                        max_from_capital = available_capital * self.max_trade_size_pct
                        opportunity.max_trade_size_usdc = min(
                            opportunity.max_trade_size_usdc,
                            max_from_capital,
                        )
                        # Recalculate profit with adjusted size
                        opportunity.expected_profit_usdc = (
                            opportunity.max_trade_size_usdc * opportunity.edge_pct / 100
                        )

                        if opportunity.max_trade_size_usdc >= self.min_trade_size_usdc:
                            self._opportunities_found.append(opportunity)
                            logger.info(
                                f"Found opportunity: {market.question[:50]}... "
                                f"edge={opportunity.edge_pct:.2f}% "
                                f"profit=${opportunity.expected_profit_usdc:.2f}"
                            )

                except (DataProviderError, RateLimitError) as e:
                    logger.debug(f"Skipping market {market.condition_id}: {e}")
                    continue

            # Sort by edge percentage (best opportunities first)
            self._opportunities_found.sort(key=lambda x: x.edge_pct, reverse=True)

            logger.info(
                f"Scan complete: found {len(self._opportunities_found)} opportunities"
            )
            return self._opportunities_found

        except Exception as e:
            logger.error(f"Arbitrage scan failed: {e}")
            return []

    def generate_trade_recommendations(
        self,
        opportunities: list[ArbitrageOpportunity],
        available_capital: float,
        max_recommendations: int = 5,
    ) -> list[TradeRecommendation]:
        """
        Generate trade recommendations from opportunities.

        Args:
            opportunities: List of detected opportunities
            available_capital: Capital available for trading
            max_recommendations: Maximum number of recommendations

        Returns:
            List of trade recommendations ready for execution
        """
        recommendations = []
        remaining_capital = available_capital

        for opp in opportunities[:max_recommendations]:
            if remaining_capital < self.min_trade_size_usdc * 2:
                break

            # Calculate trade size (buy equal amounts on both sides)
            max_trade = min(
                opp.max_trade_size_usdc,
                remaining_capital * self.max_trade_size_pct,
            )

            if max_trade < self.min_trade_size_usdc:
                continue

            # Size proportionally to prices
            # To lock in arb, buy: yes_shares = trade_amount / yes_price
            # and: no_shares = trade_amount / no_price
            # But we want equal USDC spend, so:
            yes_amount = max_trade / 2 * (1 / opp.yes_ask_price) * opp.yes_ask_price
            no_amount = max_trade / 2 * (1 / opp.no_ask_price) * opp.no_ask_price
            # Simplified: just split evenly
            yes_amount = max_trade / 2
            no_amount = max_trade / 2
            total_amount = yes_amount + no_amount

            expected_profit = total_amount * (1 - opp.combined_price)

            rec = TradeRecommendation(
                market_id=opp.market_id,
                action="buy_yes_and_no",
                yes_token_id=opp.yes_token_id,
                no_token_id=opp.no_token_id,
                yes_amount_usdc=yes_amount,
                no_amount_usdc=no_amount,
                total_amount_usdc=total_amount,
                expected_profit_usdc=expected_profit,
                expected_profit_pct=opp.edge_pct,
                reason=f"arbitrage: combined_price={opp.combined_price:.4f}",
                opportunity=opp,
                timestamp=datetime.now(),
                is_dry_run=self.dry_run,
            )

            recommendations.append(rec)
            remaining_capital -= total_amount

            logger.info(
                f"{'[DRY RUN] ' if self.dry_run else ''}"
                f"Recommendation: {opp.market_question[:40]}... "
                f"${total_amount:.2f} -> ${expected_profit:.2f} profit"
            )

        return recommendations

    async def _analyze_market(self, market: Market) -> Optional[ArbitrageOpportunity]:
        """
        Analyze a single market for arbitrage opportunity.

        Args:
            market: Market to analyze

        Returns:
            ArbitrageOpportunity if found, None otherwise
        """
        if len(market.outcomes) != 2:
            return None

        # Identify Yes and No outcomes
        yes_outcome = None
        no_outcome = None
        for outcome in market.outcomes:
            if outcome.name.lower() in ["yes", "y"]:
                yes_outcome = outcome
            elif outcome.name.lower() in ["no", "n"]:
                no_outcome = outcome

        # Fallback to first/second if not labeled
        if yes_outcome is None:
            yes_outcome = market.outcomes[0]
        if no_outcome is None:
            no_outcome = market.outcomes[1]

        # Fetch order books for both outcomes
        yes_book = await self.provider.get_order_book(
            market.condition_id, yes_outcome.token_id
        )
        no_book = await self.provider.get_order_book(
            market.condition_id, no_outcome.token_id
        )

        # Check for valid order books
        if not yes_book.asks or not no_book.asks:
            return None

        yes_best_ask = yes_book.best_ask
        no_best_ask = no_book.best_ask

        if yes_best_ask is None or no_best_ask is None:
            return None

        # Calculate combined price
        combined_price = yes_best_ask + no_best_ask

        # Check if below threshold
        if combined_price >= self.arb_threshold:
            return None

        edge_pct = (1 - combined_price) * 100

        # Calculate liquidity at best ask levels
        yes_liquidity = self._calculate_available_liquidity(yes_book, "ask")
        no_liquidity = self._calculate_available_liquidity(no_book, "ask")

        if yes_liquidity < self.min_liquidity_usdc or no_liquidity < self.min_liquidity_usdc:
            return None

        # Max trade size limited by minimum liquidity
        max_trade_size = min(yes_liquidity, no_liquidity)

        # Apply edge requirements based on trade size
        required_edge = MIN_EDGE_SMALL_TRADE
        if max_trade_size > LARGE_TRADE_THRESHOLD_USDC:
            required_edge = MIN_EDGE_LARGE_TRADE

        if edge_pct < required_edge * 100:
            return None

        # Calculate expected profit
        expected_profit = max_trade_size * edge_pct / 100

        # Calculate hours until resolution
        hours_until_resolution = None
        if market.end_date:
            delta = market.end_date - datetime.now()
            hours_until_resolution = delta.total_seconds() / 3600

        # Calculate confidence based on liquidity depth and edge
        confidence = self._calculate_opportunity_confidence(
            edge_pct, yes_liquidity, no_liquidity, hours_until_resolution
        )

        return ArbitrageOpportunity(
            market_id=market.condition_id,
            market_question=market.question,
            yes_token_id=yes_outcome.token_id,
            no_token_id=no_outcome.token_id,
            yes_ask_price=yes_best_ask,
            no_ask_price=no_best_ask,
            combined_price=combined_price,
            edge_pct=edge_pct,
            yes_liquidity_usdc=yes_liquidity,
            no_liquidity_usdc=no_liquidity,
            max_trade_size_usdc=max_trade_size,
            expected_profit_usdc=expected_profit,
            expected_profit_pct=edge_pct,
            timestamp=datetime.now(),
            hours_until_resolution=hours_until_resolution,
            market_volume=market.volume,
            confidence=confidence,
        )

    def _is_market_eligible(self, market: Market) -> bool:
        """Check if a market is eligible for arbitrage trading."""
        # Must be active and binary
        if not market.is_active or not market.is_binary:
            return False

        # Must have sufficient volume
        if market.volume < MIN_MARKET_VOLUME_USDC:
            return False

        # Check time until resolution
        if market.end_date:
            hours_until = (market.end_date - datetime.now()).total_seconds() / 3600
            if hours_until < MIN_HOURS_UNTIL_RESOLUTION:
                return False
            if hours_until > MAX_HOURS_UNTIL_RESOLUTION:
                return False

        return True

    def _calculate_available_liquidity(
        self, order_book: OrderBook, side: str = "ask"
    ) -> float:
        """
        Calculate available liquidity in USDC from order book.

        Args:
            order_book: The order book to analyze
            side: "ask" for buying, "bid" for selling

        Returns:
            Total liquidity in USDC
        """
        levels = order_book.asks if side == "ask" else order_book.bids

        if not levels:
            return 0.0

        # Sum up liquidity across top levels (limited by slippage)
        total_liquidity = 0.0
        base_price = levels[0].price

        for level in levels[:MIN_LIQUIDITY_DEPTH_LEVELS]:
            # Check if price is within slippage tolerance
            if side == "ask":
                slippage = (level.price - base_price) / base_price
            else:
                slippage = (base_price - level.price) / base_price

            if slippage > MAX_SLIPPAGE_PCT:
                break

            # Size is in tokens, convert to USDC
            level_usdc = level.size * level.price
            total_liquidity += level_usdc

        return total_liquidity

    def _calculate_signal_value(self, opportunity: ArbitrageOpportunity) -> float:
        """
        Convert opportunity to signal value (-1 to 1 scale).

        For arbitrage, higher edge = higher signal value.
        """
        # Normalize edge: 1% = 0.33, 2% = 0.66, 3%+ = 1.0
        normalized = min(1.0, opportunity.edge_pct / 3.0)
        return round(normalized, 4)

    def _calculate_confidence(self, opportunity: ArbitrageOpportunity) -> float:
        """Calculate confidence score for the opportunity."""
        return opportunity.confidence

    def _calculate_opportunity_confidence(
        self,
        edge_pct: float,
        yes_liquidity: float,
        no_liquidity: float,
        hours_until_resolution: Optional[float],
    ) -> float:
        """
        Calculate confidence score based on multiple factors.

        Factors:
        - Edge size (higher = more confident)
        - Liquidity depth (deeper = more confident)
        - Time to resolution (not too soon, not too far)
        """
        confidence = 1.0

        # Edge factor: 1% edge = 0.7 confidence, 3%+ = 1.0
        edge_factor = min(1.0, 0.4 + (edge_pct / 5.0))
        confidence *= edge_factor

        # Liquidity factor: $100 = 0.5, $1000+ = 1.0
        min_liquidity = min(yes_liquidity, no_liquidity)
        liquidity_factor = min(1.0, 0.5 + (min_liquidity / 2000.0))
        confidence *= liquidity_factor

        # Time factor: prefer 24-720 hours (1-30 days)
        if hours_until_resolution:
            if hours_until_resolution < 48:
                confidence *= 0.7  # Too soon
            elif hours_until_resolution > 2160:  # 90 days
                confidence *= 0.8  # Too far out

        return round(confidence, 4)

    def _create_neutral_signal(
        self, market_id: str, timestamp: datetime
    ) -> CompositeSignal:
        """Create a neutral (HOLD) signal."""
        return self._create_signal(
            market_id=market_id,
            components=[
                SignalComponent(
                    source="arbitrage",
                    value=0.0,
                    confidence=0.0,
                    metadata={"reason": "no_opportunity"},
                )
            ],
            timestamp=timestamp,
            metadata={"is_arbitrage": False},
        )


# =============================================================================
# Main Loop Integration Example
# =============================================================================


async def run_arbitrage_scan(
    capital: float = 10000.0,
    dry_run: bool = True,
    scan_interval_seconds: int = 60,
) -> list[TradeRecommendation]:
    """
    Example function showing how to integrate into main loop.

    Args:
        capital: Available trading capital
        dry_run: If True, only log opportunities
        scan_interval_seconds: Time between scans

    Returns:
        List of trade recommendations
    """
    provider = PolymarketProvider()
    strategy = ArbitrageStrategy(provider=provider, dry_run=dry_run)

    try:
        # Scan for opportunities
        opportunities = await strategy.scan_all_markets(
            available_capital=capital,
            max_markets=100,
        )

        if not opportunities:
            logger.info("No arbitrage opportunities found")
            return []

        # Generate trade recommendations
        recommendations = strategy.generate_trade_recommendations(
            opportunities=opportunities,
            available_capital=capital,
            max_recommendations=5,
        )

        # Log summary
        logger.info(
            f"Arbitrage scan complete: "
            f"{len(opportunities)} opportunities, "
            f"{len(recommendations)} recommendations"
        )

        for rec in recommendations:
            logger.info(f"  {rec.reason}: ${rec.total_amount_usdc:.2f} -> ${rec.expected_profit_usdc:.2f}")

        return recommendations

    finally:
        await provider.close()


# =============================================================================
# CLI Entry Point for Testing
# =============================================================================

if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    async def main():
        print("=" * 60)
        print("Polymarket Arbitrage Scanner")
        print("=" * 60)
        print()

        recommendations = await run_arbitrage_scan(
            capital=10000.0,
            dry_run=True,  # Set to False for live trading
        )

        print()
        print("=" * 60)
        if recommendations:
            print(f"Found {len(recommendations)} trade recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"\n{i}. {rec.opportunity.market_question[:60]}...")
                print(f"   Combined price: {rec.opportunity.combined_price:.4f}")
                print(f"   Edge: {rec.expected_profit_pct:.2f}%")
                print(f"   Trade size: ${rec.total_amount_usdc:.2f}")
                print(f"   Expected profit: ${rec.expected_profit_usdc:.2f}")
        else:
            print("No profitable arbitrage opportunities found.")
        print("=" * 60)

    asyncio.run(main())
