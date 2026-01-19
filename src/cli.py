from datetime import datetime
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.backtest.engine import BacktestEngine
from src.backtest.metrics import calculate_all_metrics
from src.data.yahoo import YahooFinanceProvider
from src.strategies.momentum import MomentumStrategy

app = typer.Typer(
    name="hedge",
    help="Hedge AI - Investment strategy backtesting and paper trading platform",
)
console = Console()

# Registry of available strategies
STRATEGIES = {
    "momentum": lambda params: MomentumStrategy(
        lookback_period=params.get("lookback_period", 20)
    ),
}


@app.command()
def backtest(
    strategy: str = typer.Option(
        "momentum",
        "--strategy",
        "-s",
        help="Strategy to backtest",
    ),
    tickers: str = typer.Option(
        "AAPL",
        "--tickers",
        "-t",
        help="Comma-separated list of ticker symbols",
    ),
    start: str = typer.Option(
        ...,
        "--start",
        help="Start date (YYYY-MM-DD)",
    ),
    end: Optional[str] = typer.Option(
        None,
        "--end",
        help="End date (YYYY-MM-DD), defaults to today",
    ),
    capital: float = typer.Option(
        10000.0,
        "--capital",
        "-c",
        help="Initial capital",
    ),
    commission: float = typer.Option(
        0.0,
        "--commission",
        help="Commission per trade",
    ),
    lookback: int = typer.Option(
        20,
        "--lookback",
        "-l",
        help="Lookback period for momentum strategy",
    ),
):
    """
    Run a backtest with the specified strategy and parameters.

    Example:
        hedge backtest --strategy momentum --tickers AAPL,MSFT --start 2023-01-01 --capital 10000
    """
    # Parse dates
    start_date = datetime.strptime(start, "%Y-%m-%d")
    end_date = datetime.strptime(end, "%Y-%m-%d") if end else datetime.now()

    # Parse tickers
    ticker_list = [t.strip().upper() for t in tickers.split(",")]

    # Validate strategy
    if strategy not in STRATEGIES:
        console.print(f"[red]Error: Unknown strategy '{strategy}'[/red]")
        console.print(f"Available strategies: {', '.join(STRATEGIES.keys())}")
        raise typer.Exit(1)

    console.print(f"\n[bold]Hedge AI Backtest[/bold]")
    console.print(f"Strategy: {strategy}")
    console.print(f"Tickers: {', '.join(ticker_list)}")
    console.print(f"Period: {start} to {end or 'today'}")
    console.print(f"Initial Capital: ${capital:,.2f}")
    console.print()

    # Fetch data
    with console.status("[bold green]Fetching market data..."):
        provider = YahooFinanceProvider()
        data = provider.get_multiple_tickers(ticker_list, start_date, end_date)

    if not data:
        console.print("[red]Error: No data retrieved for any tickers[/red]")
        raise typer.Exit(1)

    console.print(f"[green]✓[/green] Fetched data for {len(data)} ticker(s)")

    # Create strategy
    strategy_params = {"lookback_period": lookback}
    strategy_instance = STRATEGIES[strategy](strategy_params)

    # Run backtest
    with console.status("[bold green]Running backtest..."):
        engine = BacktestEngine(
            initial_capital=capital,
            commission=commission,
        )

        if len(data) == 1:
            ticker = list(data.keys())[0]
            result = engine.run(strategy_instance, data[ticker])
        else:
            result = engine.run_multiple(strategy_instance, data)

    console.print(f"[green]✓[/green] Backtest complete")
    console.print()

    # Calculate metrics
    metrics = calculate_all_metrics(result)

    # Display results
    _display_results(result, metrics, ticker_list)


def _display_results(result, metrics, tickers):
    """Display backtest results in formatted tables."""
    # Summary table
    summary_table = Table(title="Performance Summary")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", justify="right")

    summary_table.add_row("Initial Capital", f"${result.initial_capital:,.2f}")
    summary_table.add_row("Final Value", f"${result.final_value:,.2f}")
    summary_table.add_row("Total Return", f"{metrics['total_return_pct']:.2f}%")
    summary_table.add_row("CAGR", f"{metrics['cagr']:.2f}%")
    summary_table.add_row("Sharpe Ratio", f"{metrics['sharpe_ratio']:.2f}")
    summary_table.add_row("Sortino Ratio", f"{metrics['sortino_ratio']:.2f}")
    summary_table.add_row("Max Drawdown", f"{metrics['max_drawdown_pct']:.2f}%")
    summary_table.add_row("Win Rate", f"{metrics['win_rate_pct']:.1f}%")
    summary_table.add_row(
        "Profit Factor",
        f"{metrics['profit_factor']:.2f}"
        if metrics["profit_factor"] != float("inf")
        else "∞",
    )
    summary_table.add_row("Total Trades", str(metrics["num_trades"]))

    console.print(summary_table)

    # Trades table (show last 10)
    if result.trades:
        console.print()
        trades_table = Table(title="Recent Trades (Last 10)")
        trades_table.add_column("Date", style="cyan")
        trades_table.add_column("Ticker")
        trades_table.add_column("Side")
        trades_table.add_column("Quantity", justify="right")
        trades_table.add_column("Price", justify="right")
        trades_table.add_column("Value", justify="right")

        for trade in result.trades[-10:]:
            side_color = "green" if trade.side.value == "BUY" else "red"
            trades_table.add_row(
                trade.timestamp.strftime("%Y-%m-%d"),
                trade.ticker,
                f"[{side_color}]{trade.side.value}[/{side_color}]",
                str(trade.quantity),
                f"${trade.price:.2f}",
                f"${trade.total_value:,.2f}",
            )

        console.print(trades_table)


@app.command("list-strategies")
def list_strategies():
    """List all available trading strategies."""
    table = Table(title="Available Strategies")
    table.add_column("Name", style="cyan")
    table.add_column("Description")
    table.add_column("Parameters")

    for name, factory in STRATEGIES.items():
        # Create instance to get info
        strategy = factory({})
        info = strategy.info
        params = ", ".join(f"{k}" for k in info.parameters.keys())
        table.add_row(name, info.description, params or "None")

    console.print(table)


@app.callback()
def main():
    """
    Hedge AI - Investment strategy backtesting and paper trading platform.
    """
    pass


if __name__ == "__main__":
    app()
