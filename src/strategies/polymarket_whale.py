"""Whale tracking strategy for Polymarket."""

import logging
from datetime import datetime, timedelta
from typing import Optional

from src.data.whale_tracker import WhaleTracker, get_whale_tracker
from src.models.polymarket_schemas import (
    CompositeSignal,
    Market,
    PriceHistory,
    SignalComponent,
    TransactionType,
)
from src.strategies.polymarket_base import PolymarketStrategy, PolymarketStrategyInfo

logger = logging.getLogger(__name__)


class WhaleStrategy(PolymarketStrategy):
    """
    Strategy based on whale wallet activity.

    Follows large wallet movements - buy when whales are accumulating,
    sell when they're distributing.
    """

    def __init__(
        self,
        whale_tracker: Optional[WhaleTracker] = None,
        lookback_hours: int = 24,
        min_transaction_size: float = 10000.0,
        signal_threshold: float = 0.2,
    ):
        """
        Initialize the whale strategy.

        Args:
            whale_tracker: Provider for whale data (uses mock if None)
            lookback_hours: Hours of whale activity to consider
            min_transaction_size: Minimum transaction size to track (USDC)
            signal_threshold: Minimum net flow ratio to generate signal
        """
        self.whale_tracker = whale_tracker or get_whale_tracker(use_mock=True)
        self.lookback_hours = lookback_hours
        self.min_transaction_size = min_transaction_size
        self.signal_threshold = signal_threshold

    @property
    def info(self) -> PolymarketStrategyInfo:
        return PolymarketStrategyInfo(
            name="whale",
            description="Follow large wallet trading activity",
            signal_sources=["whale"],
            parameters={
                "lookback_hours": "Hours of whale activity to analyze",
                "min_transaction_size": "Minimum transaction size to track (USDC)",
                "signal_threshold": "Minimum net flow ratio to trigger signal",
            },
        )

    async def generate_signal(
        self,
        market: Market,
        price_history: Optional[PriceHistory] = None,
        timestamp: Optional[datetime] = None,
    ) -> CompositeSignal:
        """Generate a signal based on whale activity."""
        if not timestamp:
            timestamp = datetime.now()

        market_id = market.condition_id
        start = timestamp - timedelta(hours=self.lookback_hours)

        try:
            # Get whale flow for this market
            whale_flow = await self.whale_tracker.get_market_whale_flow(
                market_id=market_id,
                start=start,
                end=timestamp,
                min_transaction_size=self.min_transaction_size,
            )

            # Calculate flow ratio
            total_volume = whale_flow.buy_volume + whale_flow.sell_volume

            if total_volume > 0:
                # Net flow normalized by total volume
                flow_ratio = whale_flow.net_flow / total_volume
            else:
                flow_ratio = 0.0

            # Analyze transaction patterns
            recent_txs = whale_flow.transactions[-10:] if whale_flow.transactions else []
            recent_buys = sum(
                1 for tx in recent_txs if tx.transaction_type == TransactionType.BUY
            )
            recent_sells = len(recent_txs) - recent_buys

            # Calculate momentum - are recent txs aligned with overall flow?
            if recent_txs:
                recent_ratio = (recent_buys - recent_sells) / len(recent_txs)
            else:
                recent_ratio = 0.0

            # Combine flow ratio with recent momentum
            combined_signal = flow_ratio * 0.6 + recent_ratio * 0.4

            # Calculate confidence based on volume and whale count
            if total_volume > 0:
                volume_factor = min(1.0, total_volume / 100000)  # $100k = full confidence
                wallet_factor = min(1.0, whale_flow.unique_wallets / 5)  # 5 wallets = full
                confidence = (volume_factor * 0.6 + wallet_factor * 0.4)
            else:
                confidence = 0.0

            # Clamp signal to -1 to 1
            combined_signal = max(-1.0, min(1.0, combined_signal))

            component = SignalComponent(
                source="whale",
                value=round(combined_signal, 4),
                confidence=round(confidence, 4),
                metadata={
                    "net_flow": whale_flow.net_flow,
                    "buy_volume": whale_flow.buy_volume,
                    "sell_volume": whale_flow.sell_volume,
                    "unique_wallets": whale_flow.unique_wallets,
                    "transaction_count": len(whale_flow.transactions),
                    "recent_buy_ratio": recent_ratio,
                },
            )

            return self._create_signal(
                market_id=market_id,
                components=[component],
                timestamp=timestamp,
                metadata={
                    "strategy": "whale",
                    "lookback_hours": self.lookback_hours,
                    "min_tx_size": self.min_transaction_size,
                },
            )

        except Exception as e:
            logger.warning(f"Failed to get whale data for {market_id}: {e}")
            return self._create_signal(
                market_id=market_id,
                components=[
                    SignalComponent(
                        source="whale",
                        value=0.0,
                        confidence=0.0,
                        metadata={"error": str(e)},
                    )
                ],
                timestamp=timestamp,
                metadata={"strategy": "whale", "error": str(e)},
            )

    async def get_whale_breakdown(
        self,
        market: Market,
        hours: int = 24,
    ) -> dict:
        """
        Get detailed whale activity breakdown for a market.

        Args:
            market: Market to analyze
            hours: Hours of history

        Returns:
            Dict with whale activity details
        """
        end = datetime.now()
        start = end - timedelta(hours=hours)

        try:
            whale_flow = await self.whale_tracker.get_market_whale_flow(
                market_id=market.condition_id,
                start=start,
                end=end,
                min_transaction_size=self.min_transaction_size,
            )

            # Categorize transactions by size
            small_txs = []  # < $25k
            medium_txs = []  # $25k - $100k
            large_txs = []  # > $100k

            for tx in whale_flow.transactions:
                if tx.amount < 25000:
                    small_txs.append(tx)
                elif tx.amount < 100000:
                    medium_txs.append(tx)
                else:
                    large_txs.append(tx)

            # Calculate buy/sell ratio for each tier
            def calc_ratio(txs):
                if not txs:
                    return 0.5
                buys = sum(
                    1 for tx in txs if tx.transaction_type == TransactionType.BUY
                )
                return buys / len(txs)

            return {
                "net_flow": whale_flow.net_flow,
                "buy_volume": whale_flow.buy_volume,
                "sell_volume": whale_flow.sell_volume,
                "unique_wallets": whale_flow.unique_wallets,
                "total_transactions": len(whale_flow.transactions),
                "small_tx_count": len(small_txs),
                "medium_tx_count": len(medium_txs),
                "large_tx_count": len(large_txs),
                "small_tx_buy_ratio": calc_ratio(small_txs),
                "medium_tx_buy_ratio": calc_ratio(medium_txs),
                "large_tx_buy_ratio": calc_ratio(large_txs),
                "flow_direction": (
                    "accumulating"
                    if whale_flow.net_flow > 0
                    else "distributing" if whale_flow.net_flow < 0 else "neutral"
                ),
            }

        except Exception as e:
            logger.error(f"Failed to get whale breakdown: {e}")
            return {
                "error": str(e),
                "net_flow": 0,
                "buy_volume": 0,
                "sell_volume": 0,
                "unique_wallets": 0,
            }

    async def get_top_whales(
        self,
        market: Market,
        limit: int = 10,
    ) -> list[dict]:
        """
        Get top whales for a specific market.

        Args:
            market: Market to analyze
            limit: Max whales to return

        Returns:
            List of whale wallet details
        """
        end = datetime.now()
        start = end - timedelta(days=7)  # Look at last 7 days

        try:
            whale_flow = await self.whale_tracker.get_market_whale_flow(
                market_id=market.condition_id,
                start=start,
                end=end,
                min_transaction_size=self.min_transaction_size,
            )

            # Aggregate by wallet
            wallet_stats: dict[str, dict] = {}

            for tx in whale_flow.transactions:
                addr = tx.wallet_address
                if addr not in wallet_stats:
                    wallet_stats[addr] = {
                        "address": addr,
                        "buy_volume": 0,
                        "sell_volume": 0,
                        "transaction_count": 0,
                    }

                if tx.transaction_type == TransactionType.BUY:
                    wallet_stats[addr]["buy_volume"] += tx.amount
                else:
                    wallet_stats[addr]["sell_volume"] += tx.amount

                wallet_stats[addr]["transaction_count"] += 1

            # Calculate net position and sort
            for stats in wallet_stats.values():
                stats["net_position"] = stats["buy_volume"] - stats["sell_volume"]
                stats["total_volume"] = stats["buy_volume"] + stats["sell_volume"]

            sorted_wallets = sorted(
                wallet_stats.values(),
                key=lambda x: x["total_volume"],
                reverse=True,
            )

            return sorted_wallets[:limit]

        except Exception as e:
            logger.error(f"Failed to get top whales: {e}")
            return []
