from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    CompareStrategiesRequest,
    CompareStrategiesResponse,
    CustomBacktestRequest,
    ErrorResponse,
    MetricsResponse,
    PortfolioValuePoint,
    StrategiesResponse,
    StrategyComparisonResult,
    StrategyInfo,
    StrategyValidationRequest,
    StrategyValidationResponse,
    TickerPrice,
    TickerPricesRequest,
    TickerPricesResponse,
    TradeResponse,
)
from src.api.polymarket_schemas import (
    GenerateSignalsRequest,
    MarketResponse,
    MarketSearchRequest,
    MarketsListResponse,
    MarketSentimentRequest,
    MarketSentimentResponse,
    OrderBookResponse,
    OutcomeResponse,
    PolymarketBacktestRequest,
    PolymarketBacktestResponse,
    PolymarketStrategiesResponse,
    PolymarketStrategyInfo,
    PriceHistoryRequest,
    PriceHistoryResponse,
    PricePointResponse,
    SentimentReadingResponse,
    SignalsResponse,
    WhaleActivityRequest,
    WhaleActivityResponse,
    WhaleWalletsResponse,
)
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import calculate_all_metrics
from src.data.yahoo import YahooFinanceProvider
from src.strategies.momentum import MomentumStrategy
from src.strategies.rule_based import RuleBasedStrategy
from src.strategies.validator import validate_strategy_definition

app = FastAPI(
    title="Hedge AI API",
    description="API for backtesting investment strategies",
    version="0.1.0",
)

# Configure CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://*.vercel.app",
        "https://hedge-ai-ui.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Strategy registry
STRATEGIES = {
    "momentum": lambda params: MomentumStrategy(
        lookback_period=params.get("lookback_period", 20)
    ),
}


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "hedge-ai-api"}


@app.get("/api/strategies", response_model=StrategiesResponse)
async def list_strategies():
    """List all available trading strategies."""
    strategies = []
    for name, factory in STRATEGIES.items():
        strategy = factory({})
        info = strategy.info
        strategies.append(
            StrategyInfo(
                name=info.name,
                description=info.description,
                parameters=info.parameters,
            )
        )
    return StrategiesResponse(strategies=strategies)


@app.post("/api/backtest", response_model=BacktestResponse)
async def run_backtest(request: BacktestRequest):
    """
    Run a backtest with the specified strategy and parameters.
    """
    # Validate strategy
    if request.strategy not in STRATEGIES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy: {request.strategy}. Available: {list(STRATEGIES.keys())}",
        )

    # Parse dates
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = (
            datetime.strptime(request.end_date, "%Y-%m-%d")
            if request.end_date
            else datetime.now()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    # Fetch data
    try:
        provider = YahooFinanceProvider()
        data = provider.get_multiple_tickers(request.tickers, start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {e}")

    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for tickers: {request.tickers}",
        )

    # Create strategy
    strategy_params = {"lookback_period": request.lookback_period}
    strategy = STRATEGIES[request.strategy](strategy_params)

    # Run backtest
    engine = BacktestEngine(
        initial_capital=request.initial_capital,
        commission=request.commission,
    )

    if len(data) == 1:
        ticker = list(data.keys())[0]
        result = engine.run(strategy, data[ticker])
    else:
        result = engine.run_multiple(strategy, data)

    # Calculate metrics
    metrics = calculate_all_metrics(result)

    # Format response
    trades = [
        TradeResponse(
            ticker=t.ticker,
            side=t.side.value,
            quantity=t.quantity,
            price=round(t.price, 2),
            timestamp=t.timestamp.strftime("%Y-%m-%d"),
            total_value=round(t.total_value, 2),
        )
        for t in result.trades
    ]

    portfolio_values = [
        PortfolioValuePoint(
            date=date.strftime("%Y-%m-%d"),
            value=round(value, 2),
        )
        for date, value in result.portfolio_values.items()
    ]

    # Handle infinity for profit factor
    profit_factor = metrics["profit_factor"]
    if profit_factor == float("inf"):
        profit_factor = "Infinity"

    return BacktestResponse(
        initial_capital=result.initial_capital,
        final_value=round(result.final_value, 2),
        metrics=MetricsResponse(
            total_return_pct=round(metrics["total_return_pct"], 2),
            cagr=round(metrics["cagr"], 2),
            sharpe_ratio=round(metrics["sharpe_ratio"], 2),
            sortino_ratio=round(metrics["sortino_ratio"], 2),
            max_drawdown_pct=round(metrics["max_drawdown_pct"], 2),
            win_rate_pct=round(metrics["win_rate_pct"], 1),
            profit_factor=profit_factor,
            num_trades=metrics["num_trades"],
        ),
        trades=trades,
        portfolio_values=portfolio_values,
    )


@app.post("/api/strategies/validate", response_model=StrategyValidationResponse)
async def validate_strategy(request: StrategyValidationRequest):
    """
    Validate a custom strategy definition.
    """
    # Convert Pydantic model to dict for validator
    definition_dict = request.definition.model_dump()

    # Convert rules format to match validator expectations
    rules = {}
    if definition_dict.get("rules"):
        if definition_dict["rules"].get("buy"):
            rules["buy"] = definition_dict["rules"]["buy"]
        if definition_dict["rules"].get("sell"):
            rules["sell"] = definition_dict["rules"]["sell"]
    definition_dict["rules"] = rules

    is_valid, errors = validate_strategy_definition(definition_dict)

    return StrategyValidationResponse(is_valid=is_valid, errors=errors)


@app.post("/api/backtest/custom", response_model=BacktestResponse)
async def run_custom_backtest(request: CustomBacktestRequest):
    """
    Run a backtest with a custom strategy definition.
    """
    # Convert Pydantic model to dict for RuleBasedStrategy
    definition_dict = request.definition.model_dump()

    # Convert rules format
    rules = {}
    if definition_dict.get("rules"):
        if definition_dict["rules"].get("buy"):
            rules["buy"] = definition_dict["rules"]["buy"]
        if definition_dict["rules"].get("sell"):
            rules["sell"] = definition_dict["rules"]["sell"]
    definition_dict["rules"] = rules

    # Validate strategy
    is_valid, errors = validate_strategy_definition(definition_dict)
    if not is_valid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy definition: {'; '.join(errors)}",
        )

    # Parse dates
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = (
            datetime.strptime(request.end_date, "%Y-%m-%d")
            if request.end_date
            else datetime.now()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    # Fetch data
    try:
        provider = YahooFinanceProvider()
        data = provider.get_multiple_tickers(request.tickers, start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {e}")

    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for tickers: {request.tickers}",
        )

    # Create strategy
    try:
        strategy = RuleBasedStrategy(definition_dict)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Run backtest
    engine = BacktestEngine(
        initial_capital=request.initial_capital,
        commission=request.commission,
    )

    if len(data) == 1:
        ticker = list(data.keys())[0]
        result = engine.run(strategy, data[ticker])
    else:
        result = engine.run_multiple(strategy, data)

    # Calculate metrics
    metrics = calculate_all_metrics(result)

    # Format response
    trades = [
        TradeResponse(
            ticker=t.ticker,
            side=t.side.value,
            quantity=t.quantity,
            price=round(t.price, 2),
            timestamp=t.timestamp.strftime("%Y-%m-%d"),
            total_value=round(t.total_value, 2),
        )
        for t in result.trades
    ]

    portfolio_values = [
        PortfolioValuePoint(
            date=date.strftime("%Y-%m-%d"),
            value=round(value, 2),
        )
        for date, value in result.portfolio_values.items()
    ]

    # Handle infinity for profit factor
    profit_factor = metrics["profit_factor"]
    if profit_factor == float("inf"):
        profit_factor = "Infinity"

    return BacktestResponse(
        initial_capital=result.initial_capital,
        final_value=round(result.final_value, 2),
        metrics=MetricsResponse(
            total_return_pct=round(metrics["total_return_pct"], 2),
            cagr=round(metrics["cagr"], 2),
            sharpe_ratio=round(metrics["sharpe_ratio"], 2),
            sortino_ratio=round(metrics["sortino_ratio"], 2),
            max_drawdown_pct=round(metrics["max_drawdown_pct"], 2),
            win_rate_pct=round(metrics["win_rate_pct"], 1),
            profit_factor=profit_factor,
            num_trades=metrics["num_trades"],
        ),
        trades=trades,
        portfolio_values=portfolio_values,
    )


@app.post("/api/backtest/compare", response_model=CompareStrategiesResponse)
async def compare_strategies(request: CompareStrategiesRequest):
    """
    Compare multiple strategies side-by-side.
    """
    # Parse dates
    try:
        start_date = datetime.strptime(request.start_date, "%Y-%m-%d")
        end_date = (
            datetime.strptime(request.end_date, "%Y-%m-%d")
            if request.end_date
            else datetime.now()
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid date format: {e}")

    # Fetch data once for all strategies
    try:
        provider = YahooFinanceProvider()
        data = provider.get_multiple_tickers(request.tickers, start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch market data: {e}")

    if not data:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for tickers: {request.tickers}",
        )

    results = []

    for strategy_def in request.strategies:
        # Convert Pydantic model to dict
        definition_dict = strategy_def.model_dump()

        # Convert rules format
        rules = {}
        if definition_dict.get("rules"):
            if definition_dict["rules"].get("buy"):
                rules["buy"] = definition_dict["rules"]["buy"]
            if definition_dict["rules"].get("sell"):
                rules["sell"] = definition_dict["rules"]["sell"]
        definition_dict["rules"] = rules

        # Validate strategy
        is_valid, errors = validate_strategy_definition(definition_dict)
        if not is_valid:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid strategy '{definition_dict.get('name', 'unnamed')}': {'; '.join(errors)}",
            )

        # Create strategy
        try:
            strategy = RuleBasedStrategy(definition_dict)
        except ValueError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Error creating strategy '{definition_dict.get('name', 'unnamed')}': {e}",
            )

        # Run backtest
        engine = BacktestEngine(
            initial_capital=request.initial_capital,
            commission=request.commission,
        )

        if len(data) == 1:
            ticker = list(data.keys())[0]
            result = engine.run(strategy, data[ticker])
        else:
            result = engine.run_multiple(strategy, data)

        # Calculate metrics
        metrics = calculate_all_metrics(result)

        # Format trades
        trades = [
            TradeResponse(
                ticker=t.ticker,
                side=t.side.value,
                quantity=t.quantity,
                price=round(t.price, 2),
                timestamp=t.timestamp.strftime("%Y-%m-%d"),
                total_value=round(t.total_value, 2),
            )
            for t in result.trades
        ]

        # Format portfolio values
        portfolio_values = [
            PortfolioValuePoint(
                date=date.strftime("%Y-%m-%d"),
                value=round(value, 2),
            )
            for date, value in result.portfolio_values.items()
        ]

        # Handle infinity for profit factor
        profit_factor = metrics["profit_factor"]
        if profit_factor == float("inf"):
            profit_factor = "Infinity"

        results.append(
            StrategyComparisonResult(
                name=definition_dict["name"],
                initial_capital=result.initial_capital,
                final_value=round(result.final_value, 2),
                metrics=MetricsResponse(
                    total_return_pct=round(metrics["total_return_pct"], 2),
                    cagr=round(metrics["cagr"], 2),
                    sharpe_ratio=round(metrics["sharpe_ratio"], 2),
                    sortino_ratio=round(metrics["sortino_ratio"], 2),
                    max_drawdown_pct=round(metrics["max_drawdown_pct"], 2),
                    win_rate_pct=round(metrics["win_rate_pct"], 1),
                    profit_factor=profit_factor,
                    num_trades=metrics["num_trades"],
                ),
                trades=trades,
                portfolio_values=portfolio_values,
            )
        )

    return CompareStrategiesResponse(results=results)


@app.post("/api/tickers/prices", response_model=TickerPricesResponse)
async def get_ticker_prices(request: TickerPricesRequest):
    """
    Fetch current prices and 24h changes for a list of tickers.
    Uses yfinance fast_info for quick price lookups.
    """
    import yfinance as yf

    prices = []
    errors = []
    fetched_at = datetime.now().isoformat()

    for ticker_symbol in request.tickers:
        try:
            ticker = yf.Ticker(ticker_symbol.upper())
            info = ticker.fast_info

            # Get current price and previous close
            current_price = info.get("lastPrice") or info.get("regularMarketPrice", 0)
            previous_close = info.get("previousClose", 0) or info.get(
                "regularMarketPreviousClose", 0
            )

            # Calculate 24h change
            if previous_close and previous_close > 0:
                change_24h = current_price - previous_close
                change_percent_24h = (change_24h / previous_close) * 100
            else:
                change_24h = 0
                change_percent_24h = 0

            if current_price and current_price > 0:
                prices.append(
                    TickerPrice(
                        ticker=ticker_symbol.upper(),
                        price=round(current_price, 2),
                        change_24h=round(change_24h, 2),
                        change_percent_24h=round(change_percent_24h, 2),
                        previous_close=round(previous_close, 2),
                        fetched_at=fetched_at,
                    )
                )
            else:
                errors.append(f"No price data for {ticker_symbol}")
        except Exception as e:
            errors.append(f"Error fetching {ticker_symbol}: {str(e)}")

    return TickerPricesResponse(prices=prices, errors=errors)


# =============================================================================
# Polymarket Endpoints
# =============================================================================


@app.get("/api/polymarket/markets", response_model=MarketsListResponse)
async def list_polymarket_markets(
    category: str | None = None,
    limit: int = 100,
    offset: int = 0,
):
    """
    List active Polymarket prediction markets.

    Args:
        category: Optional category filter (politics, sports, crypto, etc.)
        limit: Maximum number of markets to return
        offset: Pagination offset
    """
    from src.data.polymarket_provider import PolymarketProvider
    from src.models.polymarket_schemas import MarketCategory

    provider = PolymarketProvider()

    try:
        # Parse category if provided
        cat_filter = None
        if category:
            try:
                cat_filter = MarketCategory(category.lower())
            except ValueError:
                pass  # Ignore invalid category

        markets = await provider.get_active_markets(
            category=cat_filter,
            limit=limit,
            offset=offset,
        )

        market_responses = []
        for m in markets:
            market_responses.append(
                MarketResponse(
                    condition_id=m.condition_id,
                    question=m.question,
                    description=m.description,
                    category=m.category.value,
                    status=m.status.value,
                    outcomes=[
                        OutcomeResponse(
                            outcome_id=o.outcome_id,
                            name=o.name,
                            price=o.price,
                            token_id=o.token_id,
                        )
                        for o in m.outcomes
                    ],
                    volume=m.volume,
                    liquidity=m.liquidity,
                    created_at=m.created_at.isoformat(),
                    end_date=m.end_date.isoformat() if m.end_date else None,
                    tags=m.tags,
                    image_url=m.image_url,
                    is_binary=m.is_binary,
                    yes_price=m.yes_price,
                )
            )

        return MarketsListResponse(
            markets=market_responses,
            total=len(market_responses),
            offset=offset,
            limit=limit,
        )

    finally:
        await provider.close()


@app.get("/api/polymarket/markets/{market_id}", response_model=MarketResponse)
async def get_polymarket_market(market_id: str):
    """
    Get details for a specific Polymarket market.

    Args:
        market_id: The market's condition ID
    """
    from src.data.polymarket_provider import PolymarketProvider
    from src.exceptions import MarketNotFoundError

    provider = PolymarketProvider()

    try:
        market = await provider.get_market_by_id(market_id)

        return MarketResponse(
            condition_id=market.condition_id,
            question=market.question,
            description=market.description,
            category=market.category.value,
            status=market.status.value,
            outcomes=[
                OutcomeResponse(
                    outcome_id=o.outcome_id,
                    name=o.name,
                    price=o.price,
                    token_id=o.token_id,
                )
                for o in market.outcomes
            ],
            volume=market.volume,
            liquidity=market.liquidity,
            created_at=market.created_at.isoformat(),
            end_date=market.end_date.isoformat() if market.end_date else None,
            tags=market.tags,
            image_url=market.image_url,
            is_binary=market.is_binary,
            yes_price=market.yes_price,
        )

    except MarketNotFoundError:
        raise HTTPException(status_code=404, detail=f"Market not found: {market_id}")
    finally:
        await provider.close()


@app.post("/api/polymarket/markets/search", response_model=MarketsListResponse)
async def search_polymarket_markets(request: MarketSearchRequest):
    """
    Search Polymarket markets by keyword.
    """
    from src.data.polymarket_provider import PolymarketProvider

    provider = PolymarketProvider()

    try:
        markets = await provider.search_markets(
            query=request.query,
            limit=request.limit,
        )

        market_responses = []
        for m in markets:
            market_responses.append(
                MarketResponse(
                    condition_id=m.condition_id,
                    question=m.question,
                    description=m.description,
                    category=m.category.value,
                    status=m.status.value,
                    outcomes=[
                        OutcomeResponse(
                            outcome_id=o.outcome_id,
                            name=o.name,
                            price=o.price,
                            token_id=o.token_id,
                        )
                        for o in m.outcomes
                    ],
                    volume=m.volume,
                    liquidity=m.liquidity,
                    created_at=m.created_at.isoformat(),
                    end_date=m.end_date.isoformat() if m.end_date else None,
                    tags=m.tags,
                    image_url=m.image_url,
                    is_binary=m.is_binary,
                    yes_price=m.yes_price,
                )
            )

        return MarketsListResponse(
            markets=market_responses,
            total=len(market_responses),
            offset=0,
            limit=request.limit,
        )

    finally:
        await provider.close()


@app.post("/api/polymarket/signals", response_model=SignalsResponse)
async def generate_polymarket_signals(request: GenerateSignalsRequest):
    """
    Generate trading signals for Polymarket markets.

    Combines sentiment, whale tracking, and momentum signals
    based on the specified weights.
    """
    # TODO: Implement when strategy layer is complete
    # For now, return placeholder response
    from datetime import datetime

    from src.api.polymarket_schemas import CompositeSignalResponse, SignalComponentResponse

    signals = []
    for market_id in request.market_ids:
        components = []

        if request.include_sentiment:
            components.append(
                SignalComponentResponse(
                    source="sentiment",
                    value=0.0,  # Neutral - no data yet
                    confidence=0.0,
                    metadata={"status": "mock_data"},
                )
            )

        if request.include_whale:
            components.append(
                SignalComponentResponse(
                    source="whale",
                    value=0.0,
                    confidence=0.0,
                    metadata={"status": "mock_data"},
                )
            )

        if request.include_momentum:
            components.append(
                SignalComponentResponse(
                    source="momentum",
                    value=0.0,
                    confidence=0.0,
                    metadata={"status": "mock_data"},
                )
            )

        signals.append(
            CompositeSignalResponse(
                market_id=market_id,
                direction="hold",
                strength=0.0,
                components=components,
                timestamp=datetime.now().isoformat(),
                metadata={"status": "mock_data"},
            )
        )

    return SignalsResponse(
        signals=signals,
        generated_at=datetime.now().isoformat(),
    )


@app.get("/api/polymarket/signals/{market_id}", response_model=SignalsResponse)
async def get_market_signals(market_id: str):
    """
    Get current signals for a specific market.
    """
    from datetime import datetime

    from src.api.polymarket_schemas import CompositeSignalResponse, SignalComponentResponse

    # TODO: Implement when strategy layer is complete
    return SignalsResponse(
        signals=[
            CompositeSignalResponse(
                market_id=market_id,
                direction="hold",
                strength=0.0,
                components=[
                    SignalComponentResponse(
                        source="sentiment",
                        value=0.0,
                        confidence=0.0,
                        metadata={"status": "mock_data"},
                    ),
                    SignalComponentResponse(
                        source="whale",
                        value=0.0,
                        confidence=0.0,
                        metadata={"status": "mock_data"},
                    ),
                    SignalComponentResponse(
                        source="momentum",
                        value=0.0,
                        confidence=0.0,
                        metadata={"status": "mock_data"},
                    ),
                ],
                timestamp=datetime.now().isoformat(),
                metadata={"status": "mock_data"},
            )
        ],
        generated_at=datetime.now().isoformat(),
    )


@app.get("/api/polymarket/whales", response_model=WhaleWalletsResponse)
async def list_whale_wallets():
    """
    List tracked whale wallets.
    """
    from src.api.polymarket_schemas import WhaleWalletResponse

    # TODO: Implement when whale tracker is complete
    return WhaleWalletsResponse(
        wallets=[
            WhaleWalletResponse(
                address="0x1234...abcd",
                label="Whale #1 (Mock)",
                total_volume=1000000.0,
                win_rate=0.65,
                first_seen="2024-01-01T00:00:00Z",
                last_active="2024-01-20T00:00:00Z",
            ),
        ],
        total=1,
    )


@app.post("/api/polymarket/whales/activity", response_model=WhaleActivityResponse)
async def get_whale_activity(request: WhaleActivityRequest):
    """
    Get recent whale trading activity.
    """
    from src.api.polymarket_schemas import WhaleTransactionResponse

    # TODO: Implement when whale tracker is complete
    return WhaleActivityResponse(
        transactions=[
            WhaleTransactionResponse(
                tx_hash="0xabc123...",
                wallet_address="0x1234...abcd",
                market_id=request.market_id or "mock_market",
                outcome_id="yes",
                transaction_type="buy",
                amount=50000.0,
                price=0.65,
                timestamp="2024-01-20T12:00:00Z",
            ),
        ],
        net_flow=50000.0,
        buy_volume=50000.0,
        sell_volume=0.0,
        unique_wallets=1,
    )


@app.get("/api/polymarket/sentiment/{market_id}", response_model=MarketSentimentResponse)
async def get_market_sentiment(
    market_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    Get sentiment data for a specific market.
    """
    from datetime import datetime

    # TODO: Implement when sentiment provider is complete
    return MarketSentimentResponse(
        market_id=market_id,
        current_score=0.0,
        readings=[
            SentimentReadingResponse(
                market_id=market_id,
                timestamp=datetime.now().isoformat(),
                score=0.0,
                volume=0,
                sample_posts=[],
            )
        ],
        trend="stable",
    )


@app.post("/api/polymarket/backtest", response_model=PolymarketBacktestResponse)
async def run_polymarket_backtest(request: PolymarketBacktestRequest):
    """
    Run a backtest on Polymarket markets.
    """
    from datetime import datetime

    from src.api.polymarket_schemas import (
        PolymarketMetricsResponse,
        PolymarketPortfolioValuePoint,
        PolymarketTradeResponse,
    )

    # TODO: Implement when backtest engine is complete
    # Return placeholder response
    return PolymarketBacktestResponse(
        initial_capital=request.initial_capital,
        final_value=request.initial_capital,  # No trades = no change
        metrics=PolymarketMetricsResponse(
            total_return_pct=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            brier_score=0.0,
            calibration_score=0.0,
            resolution_accuracy_pct=0.0,
            average_hold_duration_hours=0.0,
            edge_captured_pct=0.0,
            markets_traded=0,
            win_count=0,
            loss_count=0,
        ),
        trades=[],
        portfolio_values=[
            PolymarketPortfolioValuePoint(
                date=request.start_date,
                value=request.initial_capital,
            ),
        ],
        strategy_used=request.strategy,
        markets_analyzed=len(request.market_ids),
    )


@app.get("/api/polymarket/strategies", response_model=PolymarketStrategiesResponse)
async def list_polymarket_strategies():
    """
    List available Polymarket trading strategies.
    """
    return PolymarketStrategiesResponse(
        strategies=[
            PolymarketStrategyInfo(
                name="sentiment",
                description="Trade based on X/Twitter sentiment analysis",
                signal_sources=["sentiment"],
                default_weights={"sentiment": 1.0},
                parameters={"lookback_hours": "24"},
            ),
            PolymarketStrategyInfo(
                name="whale",
                description="Follow large wallet trading activity",
                signal_sources=["whale"],
                default_weights={"whale": 1.0},
                parameters={"min_transaction_size": "10000"},
            ),
            PolymarketStrategyInfo(
                name="momentum",
                description="Trade based on price momentum and order book dynamics",
                signal_sources=["momentum"],
                default_weights={"momentum": 1.0},
                parameters={"momentum_period": "24h"},
            ),
            PolymarketStrategyInfo(
                name="composite",
                description="Combined strategy using all three signal sources",
                signal_sources=["sentiment", "whale", "momentum"],
                default_weights={
                    "sentiment": 0.3,
                    "whale": 0.4,
                    "momentum": 0.3,
                },
                parameters={
                    "combination_method": "weighted",
                    "min_signal_strength": "0.3",
                },
            ),
        ]
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
