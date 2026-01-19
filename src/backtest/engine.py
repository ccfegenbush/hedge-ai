from datetime import datetime

import pandas as pd

from src.models.schemas import (
    BacktestResult,
    OHLCVSeries,
    OrderSide,
    Portfolio,
    Position,
    Signal,
    Trade,
    TradeSignal,
)
from src.strategies.base import Strategy


class BacktestEngine:
    """
    Backtesting engine for running strategies against historical data.

    Assumptions:
    - Trades execute at the close price of the signal day
    - Long-only strategies (no shorting)
    - No slippage or market impact
    - Commission can be specified per trade
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.0,
        position_size_pct: float = 1.0,
    ):
        """
        Initialize backtest engine.

        Args:
            initial_capital: Starting cash amount
            commission: Commission per trade in dollars
            position_size_pct: Fraction of portfolio to use per trade (0-1)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.position_size_pct = min(max(position_size_pct, 0), 1)

    def run(
        self,
        strategy: Strategy,
        data: OHLCVSeries,
    ) -> BacktestResult:
        """
        Run backtest for a single ticker.

        Args:
            strategy: Strategy instance to backtest
            data: Historical OHLCV data

        Returns:
            BacktestResult with performance data
        """
        # Generate signals
        signals = strategy.generate_signals(data)

        # Initialize portfolio
        portfolio = Portfolio(cash=self.initial_capital)
        trades: list[Trade] = []
        portfolio_values: dict[datetime, float] = {}

        # Create price lookup
        prices = data.data["close"].to_dict()

        # Track portfolio value at each timestamp
        for timestamp in data.data.index:
            ts = timestamp.to_pydatetime()
            value = self._calculate_portfolio_value(portfolio, data.ticker, prices[timestamp])
            portfolio_values[ts] = value

        # Process signals
        signal_map = {s.timestamp: s for s in signals}

        for timestamp in data.data.index:
            ts = timestamp.to_pydatetime()
            if ts not in signal_map:
                continue

            signal = signal_map[ts]
            trade = self._process_signal(portfolio, signal, prices[timestamp])
            if trade:
                trades.append(trade)
                # Update portfolio value after trade
                portfolio_values[ts] = self._calculate_portfolio_value(
                    portfolio, data.ticker, prices[timestamp]
                )

        # Calculate final value
        final_price = data.data["close"].iloc[-1]
        final_value = self._calculate_portfolio_value(portfolio, data.ticker, final_price)

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_value=final_value,
            trades=trades,
            portfolio_values=pd.Series(portfolio_values),
        )

    def run_multiple(
        self,
        strategy: Strategy,
        data: dict[str, OHLCVSeries],
    ) -> BacktestResult:
        """
        Run backtest for multiple tickers (portfolio-level).

        Args:
            strategy: Strategy instance to backtest
            data: Dictionary of ticker to OHLCV data

        Returns:
            Combined BacktestResult
        """
        # Initialize portfolio
        portfolio = Portfolio(cash=self.initial_capital)
        all_trades: list[Trade] = []
        portfolio_values: dict[datetime, float] = {}

        # Collect all signals with timestamps
        all_signals: list[TradeSignal] = []
        for ticker, ohlcv in data.items():
            signals = strategy.generate_signals(ohlcv)
            all_signals.extend(signals)

        # Sort signals by timestamp
        all_signals.sort(key=lambda s: s.timestamp)

        # Get union of all timestamps
        all_dates = set()
        for ohlcv in data.values():
            all_dates.update(ohlcv.data.index.to_pydatetime())
        all_dates = sorted(all_dates)

        # Process each day
        for date in all_dates:
            # Get current prices for all tickers
            current_prices = {}
            for ticker, ohlcv in data.items():
                if date in ohlcv.data.index:
                    current_prices[ticker] = ohlcv.data.loc[date, "close"]

            # Calculate portfolio value
            value = portfolio.cash
            for ticker, position in portfolio.positions.items():
                if ticker in current_prices:
                    value += position.quantity * current_prices[ticker]
            portfolio_values[date] = value

            # Process any signals for this date
            day_signals = [s for s in all_signals if s.timestamp == date]
            for signal in day_signals:
                if signal.ticker in current_prices:
                    trade = self._process_signal(
                        portfolio, signal, current_prices[signal.ticker]
                    )
                    if trade:
                        all_trades.append(trade)

        # Calculate final value
        final_value = portfolio.cash
        for ticker, position in portfolio.positions.items():
            if ticker in data:
                final_price = data[ticker].data["close"].iloc[-1]
                final_value += position.quantity * final_price

        return BacktestResult(
            initial_capital=self.initial_capital,
            final_value=final_value,
            trades=all_trades,
            portfolio_values=pd.Series(portfolio_values),
        )

    def _process_signal(
        self,
        portfolio: Portfolio,
        signal: TradeSignal,
        price: float,
    ) -> Trade | None:
        """
        Process a trading signal and execute trade if valid.

        Args:
            portfolio: Current portfolio state
            signal: Trading signal to process
            price: Current price of the security

        Returns:
            Trade if executed, None otherwise
        """
        ticker = signal.ticker

        if signal.signal == Signal.BUY:
            # Calculate position size
            available_cash = portfolio.cash * self.position_size_pct
            max_shares = int((available_cash - self.commission) / price)

            if max_shares <= 0:
                return None

            # Execute buy
            trade = Trade(
                ticker=ticker,
                side=OrderSide.BUY,
                quantity=max_shares,
                price=price,
                timestamp=signal.timestamp,
                commission=self.commission,
            )

            # Update portfolio
            portfolio.cash -= trade.total_value
            if ticker in portfolio.positions:
                portfolio.positions[ticker].update(trade)
            else:
                portfolio.positions[ticker] = Position(
                    ticker=ticker,
                    quantity=max_shares,
                    avg_cost=price,
                )

            return trade

        elif signal.signal == Signal.SELL:
            # Check if we have a position to sell
            position = portfolio.get_position(ticker)
            if position is None or position.quantity <= 0:
                return None

            # Execute sell (sell entire position)
            trade = Trade(
                ticker=ticker,
                side=OrderSide.SELL,
                quantity=position.quantity,
                price=price,
                timestamp=signal.timestamp,
                commission=self.commission,
            )

            # Update portfolio
            portfolio.cash += (trade.quantity * trade.price) - self.commission
            position.update(trade)

            # Remove position if fully sold
            if position.quantity == 0:
                del portfolio.positions[ticker]

            return trade

        return None

    def _calculate_portfolio_value(
        self,
        portfolio: Portfolio,
        ticker: str,
        current_price: float,
    ) -> float:
        """Calculate total portfolio value at current prices."""
        value = portfolio.cash
        position = portfolio.get_position(ticker)
        if position:
            value += position.quantity * current_price
        return value
