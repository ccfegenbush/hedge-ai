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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
