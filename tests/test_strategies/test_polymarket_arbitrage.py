"""Tests for the Polymarket arbitrage strategy."""

import pytest
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

from src.models.polymarket_schemas import (
    Market,
    MarketCategory,
    MarketStatus,
    OrderBook,
    OrderBookLevel,
    Outcome,
    SignalDirection,
)
from src.strategies.polymarket_arbitrage import (
    ArbitrageOpportunity,
    ArbitrageStrategy,
    TradeRecommendation,
    ARB_THRESHOLD,
    MIN_LIQUIDITY_USDC,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_provider():
    """Create a mock Polymarket provider."""
    provider = MagicMock()
    provider.get_active_markets = AsyncMock(return_value=[])
    provider.get_order_book = AsyncMock()
    provider.close = AsyncMock()
    return provider


@pytest.fixture
def sample_binary_market():
    """Create a sample binary market for testing."""
    return Market(
        condition_id="test-market-123",
        question="Will Bitcoin reach $100k by end of 2026?",
        description="Market resolves YES if BTC >= $100,000 USD",
        category=MarketCategory.CRYPTO,
        status=MarketStatus.ACTIVE,
        outcomes=[
            Outcome(
                outcome_id="yes",
                name="Yes",
                price=0.48,
                token_id="token-yes-123",
            ),
            Outcome(
                outcome_id="no",
                name="No",
                price=0.50,
                token_id="token-no-456",
            ),
        ],
        volume=50000.0,
        liquidity=10000.0,
        created_at=datetime.now() - timedelta(days=30),
        end_date=datetime.now() + timedelta(days=180),
    )


@pytest.fixture
def arb_order_book_yes():
    """Order book with arbitrage opportunity - YES side."""
    return OrderBook(
        condition_id="test-market-123",
        outcome_id="token-yes-123",
        bids=[
            OrderBookLevel(price=0.47, size=500),
            OrderBookLevel(price=0.46, size=1000),
        ],
        asks=[
            OrderBookLevel(price=0.48, size=800),
            OrderBookLevel(price=0.49, size=1200),
            OrderBookLevel(price=0.50, size=2000),
        ],
        timestamp=datetime.now(),
    )


@pytest.fixture
def arb_order_book_no():
    """Order book with arbitrage opportunity - NO side."""
    return OrderBook(
        condition_id="test-market-123",
        outcome_id="token-no-456",
        bids=[
            OrderBookLevel(price=0.49, size=500),
            OrderBookLevel(price=0.48, size=1000),
        ],
        asks=[
            OrderBookLevel(price=0.50, size=600),
            OrderBookLevel(price=0.51, size=1000),
            OrderBookLevel(price=0.52, size=1500),
        ],
        timestamp=datetime.now(),
    )


@pytest.fixture
def no_arb_order_book_yes():
    """Order book without arbitrage opportunity - YES side."""
    return OrderBook(
        condition_id="test-market-456",
        outcome_id="token-yes-789",
        bids=[OrderBookLevel(price=0.54, size=500)],
        asks=[OrderBookLevel(price=0.55, size=800)],
        timestamp=datetime.now(),
    )


@pytest.fixture
def no_arb_order_book_no():
    """Order book without arbitrage opportunity - NO side."""
    return OrderBook(
        condition_id="test-market-456",
        outcome_id="token-no-012",
        bids=[OrderBookLevel(price=0.44, size=500)],
        asks=[OrderBookLevel(price=0.46, size=600)],
        timestamp=datetime.now(),
    )


# =============================================================================
# ArbitrageOpportunity Tests
# =============================================================================


class TestArbitrageOpportunity:
    """Tests for the ArbitrageOpportunity dataclass."""

    def test_profitable_after_fees(self):
        """Test is_profitable_after_fees property."""
        # Large profitable opportunity
        opp = ArbitrageOpportunity(
            market_id="test",
            market_question="Test question",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask_price=0.48,
            no_ask_price=0.50,
            combined_price=0.98,
            edge_pct=2.0,
            yes_liquidity_usdc=500,
            no_liquidity_usdc=500,
            max_trade_size_usdc=500,
            expected_profit_usdc=10.0,  # 2% of 500
            expected_profit_pct=2.0,
            timestamp=datetime.now(),
        )
        assert opp.is_profitable_after_fees is True

        # Small unprofitable opportunity
        opp_small = ArbitrageOpportunity(
            market_id="test",
            market_question="Test question",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask_price=0.495,
            no_ask_price=0.495,
            combined_price=0.99,
            edge_pct=1.0,
            yes_liquidity_usdc=50,
            no_liquidity_usdc=50,
            max_trade_size_usdc=50,
            expected_profit_usdc=0.5,  # 1% of 50 - too small
            expected_profit_pct=1.0,
            timestamp=datetime.now(),
        )
        assert opp_small.is_profitable_after_fees is False

    def test_net_profit_calculation(self):
        """Test net_profit_usdc property accounts for fees."""
        opp = ArbitrageOpportunity(
            market_id="test",
            market_question="Test question",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask_price=0.48,
            no_ask_price=0.50,
            combined_price=0.98,
            edge_pct=2.0,
            yes_liquidity_usdc=1000,
            no_liquidity_usdc=1000,
            max_trade_size_usdc=1000,
            expected_profit_usdc=20.0,
            expected_profit_pct=2.0,
            timestamp=datetime.now(),
        )
        # Net profit should be less than expected profit
        assert opp.net_profit_usdc < opp.expected_profit_usdc
        # But still positive for this size
        assert opp.net_profit_usdc > 0

    def test_to_dict(self):
        """Test serialization to dictionary."""
        opp = ArbitrageOpportunity(
            market_id="test-123",
            market_question="Will X happen?",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask_price=0.48,
            no_ask_price=0.50,
            combined_price=0.98,
            edge_pct=2.0,
            yes_liquidity_usdc=1000,
            no_liquidity_usdc=1000,
            max_trade_size_usdc=500,
            expected_profit_usdc=10.0,
            expected_profit_pct=2.0,
            timestamp=datetime.now(),
            hours_until_resolution=720,
        )
        result = opp.to_dict()

        assert result["market_id"] == "test-123"
        assert result["combined_price"] == 0.98
        assert result["edge_pct"] == 2.0
        assert result["hours_until_resolution"] == 720


# =============================================================================
# TradeRecommendation Tests
# =============================================================================


class TestTradeRecommendation:
    """Tests for the TradeRecommendation dataclass."""

    def test_to_dict(self):
        """Test serialization to dictionary."""
        opp = ArbitrageOpportunity(
            market_id="test",
            market_question="Test",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask_price=0.48,
            no_ask_price=0.50,
            combined_price=0.98,
            edge_pct=2.0,
            yes_liquidity_usdc=1000,
            no_liquidity_usdc=1000,
            max_trade_size_usdc=500,
            expected_profit_usdc=10.0,
            expected_profit_pct=2.0,
            timestamp=datetime.now(),
        )

        rec = TradeRecommendation(
            market_id="test",
            action="buy_yes_and_no",
            yes_token_id="yes",
            no_token_id="no",
            yes_amount_usdc=250,
            no_amount_usdc=250,
            total_amount_usdc=500,
            expected_profit_usdc=10.0,
            expected_profit_pct=2.0,
            reason="arbitrage: combined_price=0.98",
            opportunity=opp,
            timestamp=datetime.now(),
            is_dry_run=True,
        )

        result = rec.to_dict()
        assert result["action"] == "buy_yes_and_no"
        assert result["total_amount_usdc"] == 500
        assert result["is_dry_run"] is True


# =============================================================================
# ArbitrageStrategy Tests
# =============================================================================


class TestArbitrageStrategy:
    """Tests for the ArbitrageStrategy class."""

    def test_init(self, mock_provider):
        """Test strategy initialization."""
        strategy = ArbitrageStrategy(
            provider=mock_provider,
            arb_threshold=0.98,
            min_liquidity_usdc=200,
            max_trade_size_pct=0.1,
            dry_run=True,
        )

        assert strategy.arb_threshold == 0.98
        assert strategy.min_liquidity_usdc == 200
        assert strategy.max_trade_size_pct == 0.1
        assert strategy.dry_run is True

    def test_info_property(self, mock_provider):
        """Test strategy info metadata."""
        strategy = ArbitrageStrategy(provider=mock_provider)
        info = strategy.info

        assert info.name == "arbitrage"
        assert "arbitrage" in info.description.lower()
        assert "order_book" in info.signal_sources

    def test_market_eligibility_active_binary(self, mock_provider, sample_binary_market):
        """Test that active binary markets are eligible."""
        strategy = ArbitrageStrategy(provider=mock_provider)
        assert strategy._is_market_eligible(sample_binary_market) is True

    def test_market_eligibility_inactive(self, mock_provider, sample_binary_market):
        """Test that inactive markets are not eligible."""
        strategy = ArbitrageStrategy(provider=mock_provider)
        sample_binary_market.status = MarketStatus.CLOSED
        assert strategy._is_market_eligible(sample_binary_market) is False

    def test_market_eligibility_non_binary(self, mock_provider, sample_binary_market):
        """Test that non-binary markets are not eligible."""
        strategy = ArbitrageStrategy(provider=mock_provider)
        # Add third outcome
        sample_binary_market.outcomes.append(
            Outcome(outcome_id="maybe", name="Maybe", price=0.02, token_id="token-maybe")
        )
        assert strategy._is_market_eligible(sample_binary_market) is False

    def test_market_eligibility_low_volume(self, mock_provider, sample_binary_market):
        """Test that low volume markets are not eligible."""
        strategy = ArbitrageStrategy(provider=mock_provider)
        sample_binary_market.volume = 500  # Below MIN_MARKET_VOLUME_USDC
        assert strategy._is_market_eligible(sample_binary_market) is False

    def test_market_eligibility_near_resolution(self, mock_provider, sample_binary_market):
        """Test that markets near resolution are not eligible."""
        strategy = ArbitrageStrategy(provider=mock_provider)
        sample_binary_market.end_date = datetime.now() + timedelta(hours=12)  # Too soon
        assert strategy._is_market_eligible(sample_binary_market) is False

    @pytest.mark.asyncio
    async def test_analyze_market_with_opportunity(
        self, mock_provider, sample_binary_market, arb_order_book_yes, arb_order_book_no
    ):
        """Test analyzing a market with an arbitrage opportunity."""
        mock_provider.get_order_book = AsyncMock(
            side_effect=[arb_order_book_yes, arb_order_book_no]
        )

        strategy = ArbitrageStrategy(provider=mock_provider)
        opportunity = await strategy._analyze_market(sample_binary_market)

        assert opportunity is not None
        assert opportunity.combined_price == pytest.approx(0.98, rel=1e-6)
        assert opportunity.edge_pct == pytest.approx(2.0, rel=1e-6)
        assert opportunity.yes_ask_price == 0.48
        assert opportunity.no_ask_price == 0.50

    @pytest.mark.asyncio
    async def test_analyze_market_no_opportunity(
        self, mock_provider, sample_binary_market, no_arb_order_book_yes, no_arb_order_book_no
    ):
        """Test analyzing a market without an arbitrage opportunity."""
        # Combined price = 0.55 + 0.46 = 1.01 > threshold
        mock_provider.get_order_book = AsyncMock(
            side_effect=[no_arb_order_book_yes, no_arb_order_book_no]
        )

        strategy = ArbitrageStrategy(provider=mock_provider)
        opportunity = await strategy._analyze_market(sample_binary_market)

        assert opportunity is None

    @pytest.mark.asyncio
    async def test_generate_signal_with_opportunity(
        self, mock_provider, sample_binary_market, arb_order_book_yes, arb_order_book_no
    ):
        """Test signal generation when opportunity exists."""
        mock_provider.get_order_book = AsyncMock(
            side_effect=[arb_order_book_yes, arb_order_book_no]
        )

        strategy = ArbitrageStrategy(provider=mock_provider)
        signal = await strategy.generate_signal(sample_binary_market)

        assert signal.direction == SignalDirection.BUY
        assert signal.strength > 0
        assert len(signal.components) == 1
        assert signal.components[0].source == "arbitrage"
        assert signal.metadata.get("is_arbitrage") is True

    @pytest.mark.asyncio
    async def test_generate_signal_no_opportunity(
        self, mock_provider, sample_binary_market, no_arb_order_book_yes, no_arb_order_book_no
    ):
        """Test signal generation when no opportunity exists."""
        mock_provider.get_order_book = AsyncMock(
            side_effect=[no_arb_order_book_yes, no_arb_order_book_no]
        )

        strategy = ArbitrageStrategy(provider=mock_provider)
        signal = await strategy.generate_signal(sample_binary_market)

        assert signal.direction == SignalDirection.HOLD
        assert signal.strength == 0

    def test_calculate_available_liquidity(self, mock_provider, arb_order_book_yes):
        """Test liquidity calculation from order book."""
        strategy = ArbitrageStrategy(provider=mock_provider)

        liquidity = strategy._calculate_available_liquidity(arb_order_book_yes, "ask")

        # Should include top levels within slippage tolerance
        # Level 1: 800 * 0.48 = 384
        # Level 2: 1200 * 0.49 = 588
        # Level 3: 2000 * 0.50 = 1000
        assert liquidity > 300  # At least first level

    def test_calculate_signal_value(self, mock_provider):
        """Test signal value normalization."""
        strategy = ArbitrageStrategy(provider=mock_provider)

        opp_small = ArbitrageOpportunity(
            market_id="test",
            market_question="Test",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask_price=0.495,
            no_ask_price=0.495,
            combined_price=0.99,
            edge_pct=1.0,  # 1% edge
            yes_liquidity_usdc=1000,
            no_liquidity_usdc=1000,
            max_trade_size_usdc=500,
            expected_profit_usdc=5.0,
            expected_profit_pct=1.0,
            timestamp=datetime.now(),
        )

        opp_large = ArbitrageOpportunity(
            market_id="test",
            market_question="Test",
            yes_token_id="yes",
            no_token_id="no",
            yes_ask_price=0.47,
            no_ask_price=0.50,
            combined_price=0.97,
            edge_pct=3.0,  # 3% edge
            yes_liquidity_usdc=1000,
            no_liquidity_usdc=1000,
            max_trade_size_usdc=500,
            expected_profit_usdc=15.0,
            expected_profit_pct=3.0,
            timestamp=datetime.now(),
        )

        signal_small = strategy._calculate_signal_value(opp_small)
        signal_large = strategy._calculate_signal_value(opp_large)

        # Larger edge should produce larger signal
        assert signal_large > signal_small
        # Both should be between 0 and 1
        assert 0 <= signal_small <= 1
        assert 0 <= signal_large <= 1
        # 3% edge should be max (1.0)
        assert signal_large == 1.0

    def test_generate_trade_recommendations(self, mock_provider):
        """Test trade recommendation generation."""
        strategy = ArbitrageStrategy(provider=mock_provider, dry_run=True)

        opportunities = [
            ArbitrageOpportunity(
                market_id="market-1",
                market_question="Question 1",
                yes_token_id="yes1",
                no_token_id="no1",
                yes_ask_price=0.48,
                no_ask_price=0.50,
                combined_price=0.98,
                edge_pct=2.0,
                yes_liquidity_usdc=1000,
                no_liquidity_usdc=1000,
                max_trade_size_usdc=500,
                expected_profit_usdc=10.0,
                expected_profit_pct=2.0,
                timestamp=datetime.now(),
            ),
            ArbitrageOpportunity(
                market_id="market-2",
                market_question="Question 2",
                yes_token_id="yes2",
                no_token_id="no2",
                yes_ask_price=0.47,
                no_ask_price=0.50,
                combined_price=0.97,
                edge_pct=3.0,
                yes_liquidity_usdc=800,
                no_liquidity_usdc=800,
                max_trade_size_usdc=400,
                expected_profit_usdc=12.0,
                expected_profit_pct=3.0,
                timestamp=datetime.now(),
            ),
        ]

        recommendations = strategy.generate_trade_recommendations(
            opportunities=opportunities,
            available_capital=10000,
            max_recommendations=5,
        )

        assert len(recommendations) == 2
        assert all(r.is_dry_run for r in recommendations)
        assert all(r.action == "buy_yes_and_no" for r in recommendations)
        # Each recommendation should have balanced amounts
        for rec in recommendations:
            assert rec.yes_amount_usdc == rec.no_amount_usdc


# =============================================================================
# Integration Tests
# =============================================================================


class TestArbitrageIntegration:
    """Integration tests for the arbitrage strategy."""

    @pytest.mark.asyncio
    async def test_full_scan_with_mock_data(self, mock_provider, sample_binary_market):
        """Test full market scan with mock data."""
        # Setup mock to return markets and order books
        mock_provider.get_active_markets = AsyncMock(return_value=[sample_binary_market])

        # Create order books with arbitrage opportunity
        yes_book = OrderBook(
            condition_id="test-market-123",
            outcome_id="token-yes-123",
            bids=[OrderBookLevel(price=0.47, size=500)],
            asks=[
                OrderBookLevel(price=0.48, size=800),
                OrderBookLevel(price=0.49, size=1200),
                OrderBookLevel(price=0.50, size=2000),
            ],
            timestamp=datetime.now(),
        )
        no_book = OrderBook(
            condition_id="test-market-123",
            outcome_id="token-no-456",
            bids=[OrderBookLevel(price=0.49, size=500)],
            asks=[
                OrderBookLevel(price=0.50, size=600),
                OrderBookLevel(price=0.51, size=1000),
            ],
            timestamp=datetime.now(),
        )

        mock_provider.get_order_book = AsyncMock(side_effect=[yes_book, no_book])

        strategy = ArbitrageStrategy(provider=mock_provider, dry_run=True)
        opportunities = await strategy.scan_all_markets(
            available_capital=10000,
            max_markets=10,
        )

        assert len(opportunities) == 1
        assert opportunities[0].combined_price == pytest.approx(0.98, rel=1e-6)
        assert opportunities[0].edge_pct == pytest.approx(2.0, rel=1e-6)

    @pytest.mark.asyncio
    async def test_scan_filters_ineligible_markets(self, mock_provider):
        """Test that scan properly filters ineligible markets."""
        # Create markets with various issues
        markets = [
            # Good market
            Market(
                condition_id="good",
                question="Good market",
                description="",
                category=MarketCategory.CRYPTO,
                status=MarketStatus.ACTIVE,
                outcomes=[
                    Outcome(outcome_id="yes", name="Yes", price=0.48, token_id="yes1"),
                    Outcome(outcome_id="no", name="No", price=0.50, token_id="no1"),
                ],
                volume=50000,
                liquidity=10000,
                created_at=datetime.now() - timedelta(days=30),
                end_date=datetime.now() + timedelta(days=180),
            ),
            # Closed market
            Market(
                condition_id="closed",
                question="Closed market",
                description="",
                category=MarketCategory.CRYPTO,
                status=MarketStatus.CLOSED,
                outcomes=[
                    Outcome(outcome_id="yes", name="Yes", price=0.48, token_id="yes2"),
                    Outcome(outcome_id="no", name="No", price=0.50, token_id="no2"),
                ],
                volume=50000,
                liquidity=10000,
                created_at=datetime.now() - timedelta(days=30),
                end_date=datetime.now() + timedelta(days=180),
            ),
            # Low volume market
            Market(
                condition_id="lowvol",
                question="Low volume market",
                description="",
                category=MarketCategory.CRYPTO,
                status=MarketStatus.ACTIVE,
                outcomes=[
                    Outcome(outcome_id="yes", name="Yes", price=0.48, token_id="yes3"),
                    Outcome(outcome_id="no", name="No", price=0.50, token_id="no3"),
                ],
                volume=500,  # Too low
                liquidity=100,
                created_at=datetime.now() - timedelta(days=30),
                end_date=datetime.now() + timedelta(days=180),
            ),
        ]

        mock_provider.get_active_markets = AsyncMock(return_value=markets)

        # Return order books only for the good market
        yes_book = OrderBook(
            condition_id="good",
            outcome_id="yes1",
            bids=[OrderBookLevel(price=0.47, size=500)],
            asks=[
                OrderBookLevel(price=0.48, size=800),
                OrderBookLevel(price=0.49, size=1000),
                OrderBookLevel(price=0.50, size=1500),
            ],
            timestamp=datetime.now(),
        )
        no_book = OrderBook(
            condition_id="good",
            outcome_id="no1",
            bids=[OrderBookLevel(price=0.49, size=500)],
            asks=[
                OrderBookLevel(price=0.50, size=600),
                OrderBookLevel(price=0.51, size=800),
            ],
            timestamp=datetime.now(),
        )

        mock_provider.get_order_book = AsyncMock(side_effect=[yes_book, no_book])

        strategy = ArbitrageStrategy(provider=mock_provider)
        opportunities = await strategy.scan_all_markets(
            available_capital=10000,
            max_markets=10,
        )

        # Should only find opportunity in the "good" market
        assert len(opportunities) == 1
        assert opportunities[0].market_id == "good"
