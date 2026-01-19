from datetime import datetime

import pandas as pd
import pytest

from src.models.schemas import (
    BacktestResult,
    OrderSide,
    Position,
    Portfolio,
    Trade,
    OHLCVSeries,
)


class TestTrade:
    def test_trade_total_value_without_commission(self):
        trade = Trade(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            price=150.0,
            timestamp=datetime.now(),
        )
        assert trade.total_value == 1500.0

    def test_trade_total_value_with_commission(self):
        trade = Trade(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            price=150.0,
            timestamp=datetime.now(),
            commission=5.0,
        )
        assert trade.total_value == 1505.0


class TestPosition:
    def test_position_market_value(self):
        position = Position(ticker="AAPL", quantity=10, avg_cost=150.0)
        assert position.market_value == 1500.0

    def test_position_update_buy(self):
        position = Position(ticker="AAPL", quantity=10, avg_cost=100.0)
        trade = Trade(
            ticker="AAPL",
            side=OrderSide.BUY,
            quantity=10,
            price=120.0,
            timestamp=datetime.now(),
        )
        position.update(trade)
        assert position.quantity == 20
        assert position.avg_cost == 110.0  # (10*100 + 10*120) / 20

    def test_position_update_sell(self):
        position = Position(ticker="AAPL", quantity=20, avg_cost=100.0)
        trade = Trade(
            ticker="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            price=120.0,
            timestamp=datetime.now(),
        )
        position.update(trade)
        assert position.quantity == 10
        assert position.avg_cost == 100.0  # Avg cost unchanged on sell

    def test_position_update_sell_all(self):
        position = Position(ticker="AAPL", quantity=10, avg_cost=100.0)
        trade = Trade(
            ticker="AAPL",
            side=OrderSide.SELL,
            quantity=10,
            price=120.0,
            timestamp=datetime.now(),
        )
        position.update(trade)
        assert position.quantity == 0
        assert position.avg_cost == 0


class TestPortfolio:
    def test_portfolio_total_value_cash_only(self):
        portfolio = Portfolio(cash=10000.0)
        assert portfolio.total_value == 10000.0

    def test_portfolio_total_value_with_positions(self):
        portfolio = Portfolio(
            cash=5000.0,
            positions={
                "AAPL": Position(ticker="AAPL", quantity=10, avg_cost=150.0),
                "MSFT": Position(ticker="MSFT", quantity=5, avg_cost=300.0),
            },
        )
        # 5000 + (10*150) + (5*300) = 5000 + 1500 + 1500 = 8000
        assert portfolio.total_value == 8000.0

    def test_get_position_exists(self):
        portfolio = Portfolio(
            cash=10000.0,
            positions={"AAPL": Position(ticker="AAPL", quantity=10, avg_cost=150.0)},
        )
        position = portfolio.get_position("AAPL")
        assert position is not None
        assert position.quantity == 10

    def test_get_position_not_exists(self):
        portfolio = Portfolio(cash=10000.0)
        position = portfolio.get_position("AAPL")
        assert position is None


class TestOHLCVSeries:
    def test_ohlcv_series_properties(self):
        dates = pd.date_range("2023-01-01", periods=5, freq="D")
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 102.0, 101.5, 103.0],
                "high": [102.0, 103.0, 104.0, 103.5, 105.0],
                "low": [99.0, 100.0, 101.0, 100.5, 102.0],
                "close": [101.0, 102.0, 103.0, 102.5, 104.0],
                "volume": [1000000, 1100000, 1200000, 1150000, 1300000],
            },
            index=dates,
        )
        series = OHLCVSeries(ticker="AAPL", data=df)

        assert series.ticker == "AAPL"
        assert len(series) == 5
        assert series.start_date == dates[0]
        assert series.end_date == dates[-1]


class TestBacktestResult:
    def test_total_return(self):
        result = BacktestResult(
            initial_capital=10000.0,
            final_value=12000.0,
            trades=[],
            portfolio_values=pd.Series([10000, 11000, 12000]),
        )
        assert result.total_return == 0.2
        assert result.total_return_pct == 20.0

    def test_num_trades(self):
        trades = [
            Trade(
                ticker="AAPL",
                side=OrderSide.BUY,
                quantity=10,
                price=150.0,
                timestamp=datetime.now(),
            ),
            Trade(
                ticker="AAPL",
                side=OrderSide.SELL,
                quantity=10,
                price=160.0,
                timestamp=datetime.now(),
            ),
        ]
        result = BacktestResult(
            initial_capital=10000.0,
            final_value=10100.0,
            trades=trades,
            portfolio_values=pd.Series([10000, 10100]),
        )
        assert result.num_trades == 2
