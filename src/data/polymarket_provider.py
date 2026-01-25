"""Polymarket CLOB API data provider."""

import logging
from datetime import datetime
from typing import Optional

import httpx
import pandas as pd

from src.exceptions import DataProviderError, MarketNotFoundError, RateLimitError
from src.models.polymarket_schemas import (
    Market,
    MarketCategory,
    MarketStatus,
    OrderBook,
    OrderBookLevel,
    Outcome,
    PriceHistory,
)

logger = logging.getLogger(__name__)


class PolymarketProvider:
    """
    Data provider for Polymarket CLOB API.

    Polymarket uses two main APIs:
    - CLOB API: For order books, trades, markets (https://clob.polymarket.com)
    - Gamma API: For market metadata, images (https://gamma-api.polymarket.com)
    """

    CLOB_BASE_URL = "https://clob.polymarket.com"
    GAMMA_BASE_URL = "https://gamma-api.polymarket.com"

    def __init__(self, timeout: float = 30.0):
        """
        Initialize the Polymarket provider.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create the HTTP client."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        url: str,
        **kwargs,
    ) -> dict:
        """
        Make an HTTP request with error handling.

        Args:
            method: HTTP method
            url: Full URL
            **kwargs: Additional arguments for httpx

        Returns:
            JSON response as dict

        Raises:
            DataProviderError: If the request fails
            RateLimitError: If rate limited
        """
        client = await self._get_client()

        try:
            response = await client.request(method, url, **kwargs)

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                raise RateLimitError(
                    "Polymarket",
                    int(retry_after) if retry_after else None,
                )

            response.raise_for_status()
            return response.json()

        except httpx.HTTPStatusError as e:
            logger.error(f"Polymarket API error: {e.response.status_code} - {e}")
            raise DataProviderError("Polymarket", str(e))
        except httpx.RequestError as e:
            logger.error(f"Polymarket request failed: {e}")
            raise DataProviderError("Polymarket", str(e))

    def _parse_category(self, category_str: str | None) -> MarketCategory:
        """Parse a category string to MarketCategory enum."""
        if not category_str:
            return MarketCategory.OTHER

        category_lower = category_str.lower()
        category_map = {
            "politics": MarketCategory.POLITICS,
            "political": MarketCategory.POLITICS,
            "sports": MarketCategory.SPORTS,
            "sport": MarketCategory.SPORTS,
            "crypto": MarketCategory.CRYPTO,
            "cryptocurrency": MarketCategory.CRYPTO,
            "entertainment": MarketCategory.ENTERTAINMENT,
            "science": MarketCategory.SCIENCE,
            "economics": MarketCategory.ECONOMICS,
            "economy": MarketCategory.ECONOMICS,
        }

        return category_map.get(category_lower, MarketCategory.OTHER)

    def _parse_market(self, data: dict) -> Market:
        """Parse market data from API response."""
        # Parse outcomes/tokens
        outcomes = []
        tokens = data.get("tokens", [])
        for token in tokens:
            outcomes.append(
                Outcome(
                    outcome_id=token.get("outcome", ""),
                    name=token.get("outcome", ""),
                    price=float(token.get("price", 0.5)),
                    token_id=token.get("token_id", ""),
                )
            )

        # Parse status
        status_str = data.get("active", True)
        closed = data.get("closed", False)
        resolved = data.get("resolved", False)

        if resolved:
            status = MarketStatus.RESOLVED
        elif closed or not status_str:
            status = MarketStatus.CLOSED
        else:
            status = MarketStatus.ACTIVE

        # Parse dates
        created_at = datetime.now()  # Default if not provided
        if data.get("created_at"):
            try:
                created_at = datetime.fromisoformat(
                    data["created_at"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        end_date = None
        if data.get("end_date_iso"):
            try:
                end_date = datetime.fromisoformat(
                    data["end_date_iso"].replace("Z", "+00:00")
                )
            except (ValueError, AttributeError):
                pass

        return Market(
            condition_id=data.get("condition_id", ""),
            question=data.get("question", ""),
            description=data.get("description", ""),
            category=self._parse_category(data.get("category")),
            status=status,
            outcomes=outcomes,
            volume=float(data.get("volume", 0)),
            liquidity=float(data.get("liquidity", 0)),
            created_at=created_at,
            end_date=end_date,
            resolved_at=None,  # Parsed if resolved
            resolution_outcome=data.get("resolution"),
            tags=data.get("tags", []),
            image_url=data.get("image"),
        )

    async def get_active_markets(
        self,
        category: Optional[MarketCategory] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[Market]:
        """
        Fetch all active prediction markets.

        Args:
            category: Optional category filter
            limit: Maximum number of markets to return
            offset: Pagination offset

        Returns:
            List of active Market objects
        """
        url = f"{self.GAMMA_BASE_URL}/markets"
        params = {
            "active": "true",
            "closed": "false",
            "limit": limit,
            "offset": offset,
        }

        if category:
            params["tag"] = category.value

        logger.info(f"Fetching active markets from Polymarket (limit={limit})")

        try:
            data = await self._request("GET", url, params=params)
        except DataProviderError:
            # Fallback: return empty list if API fails
            logger.warning("Failed to fetch markets, returning empty list")
            return []

        markets = []
        for market_data in data:
            try:
                market = self._parse_market(market_data)
                if market.is_active:
                    markets.append(market)
            except Exception as e:
                logger.warning(f"Failed to parse market: {e}")
                continue

        logger.info(f"Fetched {len(markets)} active markets")
        return markets

    async def get_market_by_id(self, condition_id: str) -> Market:
        """
        Get a specific market by its condition ID.

        Args:
            condition_id: The market's condition ID

        Returns:
            Market object

        Raises:
            MarketNotFoundError: If market doesn't exist
        """
        url = f"{self.GAMMA_BASE_URL}/markets/{condition_id}"

        logger.info(f"Fetching market: {condition_id}")

        try:
            data = await self._request("GET", url)
        except DataProviderError as e:
            if "404" in str(e):
                raise MarketNotFoundError(condition_id)
            raise

        return self._parse_market(data)

    async def get_historical_prices(
        self,
        condition_id: str,
        outcome_id: str,
        start: datetime,
        end: datetime,
        interval: str = "1h",
    ) -> PriceHistory:
        """
        Get historical prices for a market outcome.

        Args:
            condition_id: Market condition ID
            outcome_id: Outcome/token ID
            start: Start datetime
            end: End datetime
            interval: Price interval (1m, 5m, 1h, 1d)

        Returns:
            PriceHistory with DataFrame of prices
        """
        # CLOB API endpoint for price history
        url = f"{self.CLOB_BASE_URL}/prices-history"
        params = {
            "market": condition_id,
            "startTs": int(start.timestamp()),
            "endTs": int(end.timestamp()),
            "interval": interval,
        }

        logger.info(
            f"Fetching price history for {condition_id}/{outcome_id} "
            f"from {start} to {end}"
        )

        try:
            data = await self._request("GET", url, params=params)
        except DataProviderError:
            # Return empty history on failure
            logger.warning("Failed to fetch price history, returning empty")
            return PriceHistory(
                condition_id=condition_id,
                outcome_id=outcome_id,
                data=pd.DataFrame(columns=["timestamp", "price", "volume"]),
            )

        # Parse the response into a DataFrame
        history_data = data.get("history", [])
        if not history_data:
            return PriceHistory(
                condition_id=condition_id,
                outcome_id=outcome_id,
                data=pd.DataFrame(columns=["timestamp", "price", "volume"]),
            )

        records = []
        for point in history_data:
            records.append(
                {
                    "timestamp": datetime.fromtimestamp(point.get("t", 0)),
                    "price": float(point.get("p", 0)),
                    "volume": float(point.get("v", 0)),
                }
            )

        df = pd.DataFrame(records)

        return PriceHistory(
            condition_id=condition_id,
            outcome_id=outcome_id,
            data=df,
        )

    async def get_order_book(
        self,
        condition_id: str,
        outcome_id: str,
    ) -> OrderBook:
        """
        Get the current order book for a market outcome.

        Args:
            condition_id: Market condition ID
            outcome_id: Outcome/token ID

        Returns:
            OrderBook with current bids and asks
        """
        url = f"{self.CLOB_BASE_URL}/book"
        params = {
            "token_id": outcome_id,
        }

        logger.info(f"Fetching order book for {condition_id}/{outcome_id}")

        try:
            data = await self._request("GET", url, params=params)
        except DataProviderError:
            # Return empty order book on failure
            return OrderBook(
                condition_id=condition_id,
                outcome_id=outcome_id,
                bids=[],
                asks=[],
                timestamp=datetime.now(),
            )

        # Parse bids (sorted by price descending)
        bids = []
        for bid in data.get("bids", []):
            bids.append(
                OrderBookLevel(
                    price=float(bid.get("price", 0)),
                    size=float(bid.get("size", 0)),
                )
            )
        bids.sort(key=lambda x: x.price, reverse=True)

        # Parse asks (sorted by price ascending)
        asks = []
        for ask in data.get("asks", []):
            asks.append(
                OrderBookLevel(
                    price=float(ask.get("price", 0)),
                    size=float(ask.get("size", 0)),
                )
            )
        asks.sort(key=lambda x: x.price)

        return OrderBook(
            condition_id=condition_id,
            outcome_id=outcome_id,
            bids=bids,
            asks=asks,
            timestamp=datetime.now(),
        )

    async def search_markets(
        self,
        query: str,
        limit: int = 20,
    ) -> list[Market]:
        """
        Search markets by keyword.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching markets
        """
        url = f"{self.GAMMA_BASE_URL}/markets"
        params = {
            "q": query,
            "limit": limit,
        }

        logger.info(f"Searching markets for: {query}")

        try:
            data = await self._request("GET", url, params=params)
        except DataProviderError:
            return []

        markets = []
        for market_data in data:
            try:
                markets.append(self._parse_market(market_data))
            except Exception as e:
                logger.warning(f"Failed to parse search result: {e}")
                continue

        return markets

    async def get_markets_by_category(
        self,
        category: MarketCategory,
        limit: int = 50,
    ) -> list[Market]:
        """
        Get markets filtered by category.

        Args:
            category: Market category
            limit: Maximum results

        Returns:
            List of markets in the category
        """
        return await self.get_active_markets(category=category, limit=limit)


# Synchronous wrapper for non-async contexts
class SyncPolymarketProvider:
    """
    Synchronous wrapper for PolymarketProvider.

    Use this when you need to call the provider from non-async code.
    """

    def __init__(self, timeout: float = 30.0):
        self.timeout = timeout
        self._async_provider = PolymarketProvider(timeout=timeout)

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        import asyncio

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(coro)

    def get_active_markets(
        self,
        category: Optional[MarketCategory] = None,
        limit: int = 100,
    ) -> list[Market]:
        """Fetch all active prediction markets."""
        return self._run_async(
            self._async_provider.get_active_markets(category=category, limit=limit)
        )

    def get_market_by_id(self, condition_id: str) -> Market:
        """Get a specific market by its condition ID."""
        return self._run_async(self._async_provider.get_market_by_id(condition_id))

    def get_historical_prices(
        self,
        condition_id: str,
        outcome_id: str,
        start: datetime,
        end: datetime,
        interval: str = "1h",
    ) -> PriceHistory:
        """Get historical prices for a market outcome."""
        return self._run_async(
            self._async_provider.get_historical_prices(
                condition_id, outcome_id, start, end, interval
            )
        )

    def get_order_book(self, condition_id: str, outcome_id: str) -> OrderBook:
        """Get the current order book for a market outcome."""
        return self._run_async(
            self._async_provider.get_order_book(condition_id, outcome_id)
        )

    def search_markets(self, query: str, limit: int = 20) -> list[Market]:
        """Search markets by keyword."""
        return self._run_async(self._async_provider.search_markets(query, limit))

    def close(self) -> None:
        """Close the HTTP client."""
        self._run_async(self._async_provider.close())
