"""Whale wallet tracking for Polymarket markets.

This module tracks large wallet activity on Polygon for Polymarket contracts.
Data is sourced from The Graph (Polymarket subgraph) and/or direct RPC queries.

For development/testing, a mock provider is included that generates realistic test data.
"""

import logging
import random
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import httpx

from src.exceptions import WhaleTrackerError
from src.models.polymarket_schemas import (
    TransactionType,
    WhaleFlow,
    WhaleTransaction,
    WhaleWallet,
)

logger = logging.getLogger(__name__)


# Known Polymarket contract addresses on Polygon
POLYMARKET_CONTRACTS = {
    "ctf_exchange": "0x4bFb41d5B3570DeFd03C39a9A4D8dE6Bd8B8982E",
    "neg_risk_ctf_exchange": "0xC5d563A36AE78145C45a50134d48A1215220f80a",
    "conditional_tokens": "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045",
}

# Whale threshold in USDC
DEFAULT_WHALE_THRESHOLD = 10000.0


class WhaleTracker(ABC):
    """Abstract base class for whale tracking providers."""

    @abstractmethod
    async def get_whale_wallets(
        self,
        min_volume: float = 100000.0,
        limit: int = 100,
    ) -> list[WhaleWallet]:
        """
        Get list of tracked whale wallets.

        Args:
            min_volume: Minimum total volume to be considered a whale
            limit: Maximum wallets to return

        Returns:
            List of WhaleWallet objects
        """
        pass

    @abstractmethod
    async def get_wallet_activity(
        self,
        wallet_address: str,
        start: datetime,
        end: datetime,
        market_id: Optional[str] = None,
    ) -> list[WhaleTransaction]:
        """
        Get transaction history for a specific wallet.

        Args:
            wallet_address: The wallet's address
            start: Start datetime
            end: End datetime
            market_id: Optional filter by market

        Returns:
            List of WhaleTransaction objects
        """
        pass

    @abstractmethod
    async def get_market_whale_flow(
        self,
        market_id: str,
        start: datetime,
        end: datetime,
        min_transaction_size: float = 1000.0,
    ) -> WhaleFlow:
        """
        Get aggregated whale flow for a market.

        Args:
            market_id: The market's condition ID
            start: Start datetime
            end: End datetime
            min_transaction_size: Minimum transaction size to include

        Returns:
            WhaleFlow with aggregated data
        """
        pass

    @abstractmethod
    async def get_recent_large_transactions(
        self,
        min_size: float = 10000.0,
        limit: int = 50,
    ) -> list[WhaleTransaction]:
        """
        Get recent large transactions across all markets.

        Args:
            min_size: Minimum transaction size in USDC
            limit: Maximum transactions to return

        Returns:
            List of WhaleTransaction objects
        """
        pass


class MockWhaleTracker(WhaleTracker):
    """
    Mock whale tracker for testing and development.

    Generates realistic-looking whale activity data.
    """

    # Sample whale wallet addresses
    MOCK_WALLETS = [
        ("0x1234567890abcdef1234567890abcdef12345678", "Polymarket Whale #1"),
        ("0xabcdef1234567890abcdef1234567890abcdef12", "Smart Money Alpha"),
        ("0x9876543210fedcba9876543210fedcba98765432", "Early Adopter"),
        ("0xfedcba0987654321fedcba0987654321fedcba09", "Market Maker"),
        ("0x1111222233334444555566667777888899990000", "Institutional"),
    ]

    def __init__(
        self,
        seed: Optional[int] = None,
    ):
        """
        Initialize the mock whale tracker.

        Args:
            seed: Random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

    def _generate_tx_hash(self) -> str:
        """Generate a realistic-looking transaction hash."""
        chars = "0123456789abcdef"
        return "0x" + "".join(random.choice(chars) for _ in range(64))

    def _generate_transaction(
        self,
        wallet: tuple[str, str],
        market_id: str,
        timestamp: datetime,
        min_size: float = 1000.0,
    ) -> WhaleTransaction:
        """Generate a mock whale transaction."""
        tx_type = random.choice([TransactionType.BUY, TransactionType.SELL])

        # Larger transactions are rarer
        base_amount = random.uniform(min_size, min_size * 5)
        if random.random() < 0.1:  # 10% chance of very large tx
            base_amount *= random.uniform(3, 10)

        # Price between 0.1 and 0.9
        price = random.uniform(0.1, 0.9)

        return WhaleTransaction(
            tx_hash=self._generate_tx_hash(),
            wallet_address=wallet[0],
            market_id=market_id,
            outcome_id="yes" if random.random() > 0.5 else "no",
            transaction_type=tx_type,
            amount=round(base_amount, 2),
            price=round(price, 4),
            timestamp=timestamp,
            block_number=random.randint(50000000, 60000000),
        )

    async def get_whale_wallets(
        self,
        min_volume: float = 100000.0,
        limit: int = 100,
    ) -> list[WhaleWallet]:
        """Generate mock whale wallet list."""
        logger.info(f"Generating mock whale wallets (min_volume={min_volume})")

        wallets = []
        for address, label in self.MOCK_WALLETS[:limit]:
            total_volume = random.uniform(min_volume, min_volume * 50)
            win_rate = random.uniform(0.45, 0.75)

            first_seen = datetime.now() - timedelta(days=random.randint(30, 365))
            last_active = datetime.now() - timedelta(hours=random.randint(0, 72))

            wallets.append(
                WhaleWallet(
                    address=address,
                    label=label,
                    total_volume=round(total_volume, 2),
                    win_rate=round(win_rate, 4),
                    first_seen=first_seen,
                    last_active=last_active,
                )
            )

        # Sort by volume descending
        wallets.sort(key=lambda w: w.total_volume, reverse=True)

        return wallets

    async def get_wallet_activity(
        self,
        wallet_address: str,
        start: datetime,
        end: datetime,
        market_id: Optional[str] = None,
    ) -> list[WhaleTransaction]:
        """Generate mock wallet activity."""
        logger.info(f"Generating mock activity for {wallet_address[:10]}...")

        # Find or create wallet info
        wallet = None
        for w in self.MOCK_WALLETS:
            if w[0] == wallet_address:
                wallet = w
                break
        if not wallet:
            wallet = (wallet_address, "Unknown Whale")

        transactions = []
        current = start

        # Generate transactions at random intervals
        while current <= end:
            if random.random() < 0.3:  # 30% chance of transaction per hour
                market = market_id or f"market_{random.randint(1, 10)}"
                tx = self._generate_transaction(wallet, market, current)
                transactions.append(tx)

            current += timedelta(hours=1)

        return transactions

    async def get_market_whale_flow(
        self,
        market_id: str,
        start: datetime,
        end: datetime,
        min_transaction_size: float = 1000.0,
    ) -> WhaleFlow:
        """Generate mock whale flow for a market."""
        logger.info(f"Generating mock whale flow for {market_id}")

        transactions = []
        buy_volume = 0.0
        sell_volume = 0.0
        unique_wallets = set()

        current = start
        while current <= end:
            if random.random() < 0.4:  # 40% chance per hour
                wallet = random.choice(self.MOCK_WALLETS)
                tx = self._generate_transaction(
                    wallet, market_id, current, min_transaction_size
                )
                transactions.append(tx)
                unique_wallets.add(wallet[0])

                if tx.transaction_type == TransactionType.BUY:
                    buy_volume += tx.amount
                else:
                    sell_volume += tx.amount

            current += timedelta(hours=1)

        return WhaleFlow(
            market_id=market_id,
            period_start=start,
            period_end=end,
            net_flow=round(buy_volume - sell_volume, 2),
            buy_volume=round(buy_volume, 2),
            sell_volume=round(sell_volume, 2),
            unique_wallets=len(unique_wallets),
            transactions=transactions,
        )

    async def get_recent_large_transactions(
        self,
        min_size: float = 10000.0,
        limit: int = 50,
    ) -> list[WhaleTransaction]:
        """Generate mock recent large transactions."""
        logger.info(f"Generating mock large transactions (min_size={min_size})")

        transactions = []
        now = datetime.now()

        for i in range(limit):
            wallet = random.choice(self.MOCK_WALLETS)
            timestamp = now - timedelta(hours=random.uniform(0, 48))
            market_id = f"market_{random.randint(1, 20)}"

            tx = self._generate_transaction(wallet, market_id, timestamp, min_size)
            transactions.append(tx)

        # Sort by timestamp descending
        transactions.sort(key=lambda t: t.timestamp, reverse=True)

        return transactions


class TheGraphWhaleTracker(WhaleTracker):
    """
    Whale tracker using The Graph for on-chain data.

    Queries the Polymarket subgraph on The Graph to track
    large wallet activity on Polygon.
    """

    GRAPH_URL = "https://api.thegraph.com/subgraphs/name/polymarket/polymarket-matic"

    def __init__(
        self,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
    ):
        """
        Initialize The Graph whale tracker.

        Args:
            api_key: The Graph API key (optional, for higher rate limits)
            timeout: Request timeout in seconds
        """
        self.api_key = api_key
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None or self._client.is_closed:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                timeout=self.timeout,
                headers=headers,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _query_graph(self, query: str, variables: Optional[dict] = None) -> dict:
        """Execute a GraphQL query against The Graph."""
        client = await self._get_client()

        payload = {"query": query}
        if variables:
            payload["variables"] = variables

        try:
            response = await client.post(self.GRAPH_URL, json=payload)
            response.raise_for_status()
            data = response.json()

            if "errors" in data:
                logger.error(f"GraphQL errors: {data['errors']}")
                raise WhaleTrackerError(str(data["errors"]))

            return data.get("data", {})

        except httpx.HTTPStatusError as e:
            logger.error(f"Graph API error: {e}")
            raise WhaleTrackerError(f"Graph API request failed: {e}")
        except httpx.RequestError as e:
            logger.error(f"Graph request failed: {e}")
            raise WhaleTrackerError(f"Graph API connection error: {e}")

    async def get_whale_wallets(
        self,
        min_volume: float = 100000.0,
        limit: int = 100,
    ) -> list[WhaleWallet]:
        """Fetch whale wallets from The Graph."""
        query = """
        query GetWhaleWallets($minVolume: BigDecimal!, $limit: Int!) {
            users(
                first: $limit
                orderBy: totalVolume
                orderDirection: desc
                where: { totalVolume_gte: $minVolume }
            ) {
                id
                totalVolume
                totalPositions
                firstTradeTimestamp
                lastTradeTimestamp
            }
        }
        """

        try:
            data = await self._query_graph(
                query,
                {"minVolume": str(min_volume), "limit": limit},
            )

            wallets = []
            for user in data.get("users", []):
                first_seen = None
                if user.get("firstTradeTimestamp"):
                    first_seen = datetime.fromtimestamp(
                        int(user["firstTradeTimestamp"])
                    )

                last_active = None
                if user.get("lastTradeTimestamp"):
                    last_active = datetime.fromtimestamp(
                        int(user["lastTradeTimestamp"])
                    )

                wallets.append(
                    WhaleWallet(
                        address=user["id"],
                        label=None,  # Can be enriched from external sources
                        total_volume=float(user["totalVolume"]),
                        win_rate=None,  # Would need additional calculation
                        first_seen=first_seen,
                        last_active=last_active,
                    )
                )

            return wallets

        except WhaleTrackerError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch whale wallets: {e}")
            raise WhaleTrackerError(f"Failed to fetch whale wallets: {e}")

    async def get_wallet_activity(
        self,
        wallet_address: str,
        start: datetime,
        end: datetime,
        market_id: Optional[str] = None,
    ) -> list[WhaleTransaction]:
        """Fetch wallet activity from The Graph."""
        query = """
        query GetWalletActivity(
            $wallet: String!
            $startTs: Int!
            $endTs: Int!
            $market: String
        ) {
            trades(
                first: 1000
                orderBy: timestamp
                orderDirection: desc
                where: {
                    user: $wallet
                    timestamp_gte: $startTs
                    timestamp_lte: $endTs
                }
            ) {
                id
                transactionHash
                market {
                    id
                }
                outcome
                type
                amount
                price
                timestamp
                blockNumber
            }
        }
        """

        variables = {
            "wallet": wallet_address.lower(),
            "startTs": int(start.timestamp()),
            "endTs": int(end.timestamp()),
        }
        if market_id:
            variables["market"] = market_id

        try:
            data = await self._query_graph(query, variables)

            transactions = []
            for trade in data.get("trades", []):
                tx_type = (
                    TransactionType.BUY
                    if trade["type"].upper() == "BUY"
                    else TransactionType.SELL
                )

                transactions.append(
                    WhaleTransaction(
                        tx_hash=trade["transactionHash"],
                        wallet_address=wallet_address,
                        market_id=trade["market"]["id"],
                        outcome_id=trade["outcome"],
                        transaction_type=tx_type,
                        amount=float(trade["amount"]),
                        price=float(trade["price"]),
                        timestamp=datetime.fromtimestamp(int(trade["timestamp"])),
                        block_number=int(trade["blockNumber"]),
                    )
                )

            return transactions

        except WhaleTrackerError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch wallet activity: {e}")
            raise WhaleTrackerError(f"Failed to fetch wallet activity: {e}")

    async def get_market_whale_flow(
        self,
        market_id: str,
        start: datetime,
        end: datetime,
        min_transaction_size: float = 1000.0,
    ) -> WhaleFlow:
        """Fetch and aggregate whale flow from The Graph."""
        query = """
        query GetMarketTrades(
            $market: String!
            $startTs: Int!
            $endTs: Int!
            $minSize: BigDecimal!
        ) {
            trades(
                first: 1000
                orderBy: timestamp
                orderDirection: desc
                where: {
                    market: $market
                    timestamp_gte: $startTs
                    timestamp_lte: $endTs
                    amount_gte: $minSize
                }
            ) {
                id
                transactionHash
                user {
                    id
                }
                outcome
                type
                amount
                price
                timestamp
                blockNumber
            }
        }
        """

        variables = {
            "market": market_id,
            "startTs": int(start.timestamp()),
            "endTs": int(end.timestamp()),
            "minSize": str(min_transaction_size),
        }

        try:
            data = await self._query_graph(query, variables)

            transactions = []
            buy_volume = 0.0
            sell_volume = 0.0
            unique_wallets = set()

            for trade in data.get("trades", []):
                wallet = trade["user"]["id"]
                tx_type = (
                    TransactionType.BUY
                    if trade["type"].upper() == "BUY"
                    else TransactionType.SELL
                )
                amount = float(trade["amount"])

                if tx_type == TransactionType.BUY:
                    buy_volume += amount
                else:
                    sell_volume += amount

                unique_wallets.add(wallet)

                transactions.append(
                    WhaleTransaction(
                        tx_hash=trade["transactionHash"],
                        wallet_address=wallet,
                        market_id=market_id,
                        outcome_id=trade["outcome"],
                        transaction_type=tx_type,
                        amount=amount,
                        price=float(trade["price"]),
                        timestamp=datetime.fromtimestamp(int(trade["timestamp"])),
                        block_number=int(trade["blockNumber"]),
                    )
                )

            return WhaleFlow(
                market_id=market_id,
                period_start=start,
                period_end=end,
                net_flow=round(buy_volume - sell_volume, 2),
                buy_volume=round(buy_volume, 2),
                sell_volume=round(sell_volume, 2),
                unique_wallets=len(unique_wallets),
                transactions=transactions,
            )

        except WhaleTrackerError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch market whale flow: {e}")
            raise WhaleTrackerError(f"Failed to fetch market whale flow: {e}")

    async def get_recent_large_transactions(
        self,
        min_size: float = 10000.0,
        limit: int = 50,
    ) -> list[WhaleTransaction]:
        """Fetch recent large transactions from The Graph."""
        query = """
        query GetLargeTransactions($minSize: BigDecimal!, $limit: Int!) {
            trades(
                first: $limit
                orderBy: timestamp
                orderDirection: desc
                where: { amount_gte: $minSize }
            ) {
                id
                transactionHash
                user {
                    id
                }
                market {
                    id
                }
                outcome
                type
                amount
                price
                timestamp
                blockNumber
            }
        }
        """

        try:
            data = await self._query_graph(
                query,
                {"minSize": str(min_size), "limit": limit},
            )

            transactions = []
            for trade in data.get("trades", []):
                tx_type = (
                    TransactionType.BUY
                    if trade["type"].upper() == "BUY"
                    else TransactionType.SELL
                )

                transactions.append(
                    WhaleTransaction(
                        tx_hash=trade["transactionHash"],
                        wallet_address=trade["user"]["id"],
                        market_id=trade["market"]["id"],
                        outcome_id=trade["outcome"],
                        transaction_type=tx_type,
                        amount=float(trade["amount"]),
                        price=float(trade["price"]),
                        timestamp=datetime.fromtimestamp(int(trade["timestamp"])),
                        block_number=int(trade["blockNumber"]),
                    )
                )

            return transactions

        except WhaleTrackerError:
            raise
        except Exception as e:
            logger.error(f"Failed to fetch large transactions: {e}")
            raise WhaleTrackerError(f"Failed to fetch large transactions: {e}")


def get_whale_tracker(
    use_mock: bool = False,
    graph_api_key: Optional[str] = None,
) -> WhaleTracker:
    """
    Factory function to get the appropriate whale tracker.

    Args:
        use_mock: If True, return mock tracker
        graph_api_key: The Graph API key for production tracker

    Returns:
        WhaleTracker instance
    """
    if use_mock:
        return MockWhaleTracker()

    return TheGraphWhaleTracker(api_key=graph_api_key)
