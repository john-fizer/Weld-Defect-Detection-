"""Portfolio management and tracking."""

from typing import Dict, List, Any
import numpy as np
from datetime import datetime
from .db import get_session
from .models import Position, Trade, PerformanceMetric


class Portfolio:
    """Portfolio manager for tracking positions and risk."""

    def __init__(self, initial_capital: float = 100000.0, is_live: bool = False):
        """Initialize portfolio.

        Args:
            initial_capital: Starting capital
            is_live: Whether this is live trading
        """
        self.initial_capital = initial_capital
        self.is_live = is_live
        self.positions = []
        self.closed_positions = []
        self.equity_curve = [initial_capital]
        self.daily_returns = []

    def get_account_value(self) -> float:
        """Get current account value.

        Returns:
            Total account value
        """
        if self.equity_curve:
            return self.equity_curve[-1]
        return self.initial_capital

    def get_total_pnl(self) -> float:
        """Get total P&L.

        Returns:
            Total profit/loss
        """
        return self.get_account_value() - self.initial_capital

    def get_total_pnl_pct(self) -> float:
        """Get total P&L percentage.

        Returns:
            Total P&L as percentage
        """
        return (self.get_total_pnl() / self.initial_capital)

    def get_daily_pnl(self) -> float:
        """Get today's P&L.

        Returns:
            Today's profit/loss
        """
        if len(self.equity_curve) < 2:
            return 0.0
        return self.equity_curve[-1] - self.equity_curve[-2]

    def get_daily_pnl_pct(self) -> float:
        """Get today's P&L percentage.

        Returns:
            Today's P&L as percentage
        """
        if len(self.equity_curve) < 2:
            return 0.0
        return (self.equity_curve[-1] - self.equity_curve[-2]) / self.equity_curve[-2]

    def get_open_positions_count(self) -> int:
        """Get number of open positions.

        Returns:
            Count of open positions
        """
        return len(self.positions)

    def get_ticker_weight(self, ticker: str) -> float:
        """Get position weight for ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Weight as percentage of account value
        """
        ticker_value = sum(
            p.get("value", 0) for p in self.positions
            if p.get("ticker") == ticker
        )
        return ticker_value / self.get_account_value()

    def get_strategy_position_count(self, strategy: str) -> int:
        """Get number of positions for strategy.

        Args:
            strategy: Strategy name

        Returns:
            Count of positions
        """
        return sum(1 for p in self.positions if p.get("strategy") == strategy)

    def get_strategy_history(self) -> List[str]:
        """Get list of strategies that have been used.

        Returns:
            List of strategy names
        """
        strategies = set()
        for p in self.positions + self.closed_positions:
            strategies.add(p.get("strategy", "unknown"))
        return list(strategies)

    def is_live_trading(self) -> bool:
        """Check if this is live trading.

        Returns:
            True if live trading enabled
        """
        return self.is_live

    def get_var_95(self) -> float:
        """Get 95% Value at Risk.

        Returns:
            VaR as percentage
        """
        if len(self.daily_returns) < 30:
            return 0.0

        returns = np.array(self.daily_returns[-252:])  # Last year
        var = np.percentile(returns, 5)  # 5th percentile for 95% VaR
        return abs(float(var))

    def get_margin_usage(self) -> float:
        """Get margin usage percentage.

        Returns:
            Margin used as percentage of buying power
        """
        # Simplified: assume all positions use some margin
        total_margin = sum(p.get("margin", 0) for p in self.positions)
        return total_margin / self.get_account_value()

    def get_max_drawdown(self) -> float:
        """Get maximum drawdown.

        Returns:
            Max drawdown as percentage
        """
        if len(self.equity_curve) < 2:
            return 0.0

        equity = np.array(self.equity_curve)
        running_max = np.maximum.accumulate(equity)
        drawdown = (equity - running_max) / running_max

        return abs(float(np.min(drawdown)))

    def get_consecutive_losses(self) -> int:
        """Get count of consecutive losing trades.

        Returns:
            Number of consecutive losses
        """
        if not self.closed_positions:
            return 0

        consecutive = 0
        for pos in reversed(self.closed_positions):
            if pos.get("pnl", 0) < 0:
                consecutive += 1
            else:
                break

        return consecutive

    def add_position(self, position: Dict[str, Any]):
        """Add a new position.

        Args:
            position: Position dict
        """
        self.positions.append(position)

    def close_position(self, position_id: int, exit_price: float) -> Dict[str, Any]:
        """Close a position.

        Args:
            position_id: Position ID
            exit_price: Exit price

        Returns:
            Closed position dict
        """
        # Find and remove position
        position = None
        for i, p in enumerate(self.positions):
            if p.get("id") == position_id:
                position = self.positions.pop(i)
                break

        if not position:
            return {}

        # Calculate P&L
        entry_price = position.get("entry_price", 0)
        quantity = position.get("quantity", 1)
        pnl = (exit_price - entry_price) * quantity * 100

        position["exit_price"] = exit_price
        position["pnl"] = pnl
        position["closed_at"] = datetime.utcnow()

        self.closed_positions.append(position)

        # Update equity
        new_equity = self.get_account_value() + pnl
        self.equity_curve.append(new_equity)

        # Calculate return
        if self.equity_curve[-2] > 0:
            daily_return = pnl / self.equity_curve[-2]
            self.daily_returns.append(daily_return)

        return position

    def update_from_database(self):
        """Update portfolio from database.

        Loads all open positions from database.
        """
        session = get_session()
        try:
            db_positions = session.query(Position).filter(Position.status == "open").all()

            self.positions = []
            for pos in db_positions:
                self.positions.append({
                    "id": pos.id,
                    "ticker": pos.ticker,
                    "strategy": pos.strategy,
                    "quantity": pos.quantity,
                    "entry_price": pos.entry_price,
                    "delta": pos.delta,
                    "gamma": pos.gamma,
                    "theta": pos.theta,
                    "vega": pos.vega,
                    "opened_at": pos.opened_at,
                })

        finally:
            session.close()

    def get_summary(self) -> Dict[str, Any]:
        """Get portfolio summary.

        Returns:
            Summary dict
        """
        return {
            "account_value": self.get_account_value(),
            "total_pnl": self.get_total_pnl(),
            "total_pnl_pct": self.get_total_pnl_pct(),
            "daily_pnl": self.get_daily_pnl(),
            "daily_pnl_pct": self.get_daily_pnl_pct(),
            "open_positions": self.get_open_positions_count(),
            "var_95": self.get_var_95(),
            "max_drawdown": self.get_max_drawdown(),
            "margin_usage": self.get_margin_usage(),
        }
