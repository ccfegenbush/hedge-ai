from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    BacktestRequest,
    BacktestResponse,
    ErrorResponse,
    MetricsResponse,
    PortfolioValuePoint,
    StrategiesResponse,
    StrategyInfo,
    TradeResponse,
)
from src.backtest.engine import BacktestEngine
from src.backtest.metrics import calculate_all_metrics
from src.data.yahoo import YahooFinanceProvider
from src.strategies.momentum import MomentumStrategy

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


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
