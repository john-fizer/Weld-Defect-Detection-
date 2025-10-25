"""Performance Analyst agent - tracks and analyzes performance."""

from typing import Any, Dict
from datetime import datetime
from .base import Agent
from storage.db import get_session
from storage.models import Trade, PerformanceMetric


class PerformanceAnalyst(Agent):
    """Tracks performance and generates insights.

    Responsibilities:
    - Log all trades to database
    - Calculate daily/weekly/monthly P&L
    - Generate attribution by strategy
    - Identify winning/losing patterns
    - Write post-mortems for major trades
    - Update policy weights based on performance
    """

    name = "performance_analyst"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze performance and log trades.

        Args:
            context: Context with 'fills', 'portfolio'

        Returns:
            Context enriched with 'performance_report'
        """
        fills = context.get("fills", [])

        if fills:
            self.log(f"Logging {len(fills)} fills to database")
            self._log_trades(fills)

        # Generate performance report
        report = self._generate_report(context.get("portfolio"))

        self.add_to_context(context, "performance_report", report)
        self.log(f"Performance report generated")

        return context

    def _log_trades(self, fills: list):
        """Log filled trades to database.

        Args:
            fills: List of fill dicts
        """
        session = get_session()

        try:
            for fill in fills:
                trade = Trade(
                    timestamp=datetime.utcnow(),
                    ticker=fill.get("ticker"),
                    strategy=fill.get("strategy"),
                    action=fill.get("type"),
                    quantity=fill.get("quantity", 1),
                    fill_price=fill.get("fill_price", 0.0),
                    legs=str(fill.get("legs", [])),
                    status="filled",
                )
                session.add(trade)

            session.commit()
            self.log(f"Logged {len(fills)} trades to database")

        except Exception as e:
            session.rollback()
            self.log(f"Error logging trades: {e}", level="error")
        finally:
            session.close()

    def _generate_report(self, portfolio) -> Dict[str, Any]:
        """Generate performance report.

        Args:
            portfolio: Portfolio object

        Returns:
            Performance report dict
        """
        if not portfolio:
            return {}

        session = get_session()

        try:
            # Calculate metrics
            total_pnl = portfolio.get_total_pnl()
            daily_pnl = portfolio.get_daily_pnl()
            open_positions = portfolio.get_open_positions_count()

            # Get strategy attribution
            strategy_pnl = self._calculate_strategy_attribution(session)

            # Calculate win rate
            win_rate = self._calculate_win_rate(session)

            report = {
                "timestamp": datetime.utcnow().isoformat(),
                "total_pnl": total_pnl,
                "total_pnl_pct": portfolio.get_total_pnl_pct(),
                "daily_pnl": daily_pnl,
                "daily_pnl_pct": portfolio.get_daily_pnl_pct(),
                "open_positions": open_positions,
                "strategy_attribution": strategy_pnl,
                "win_rate": win_rate,
                "sharpe_ratio": self._calculate_sharpe(session),
                "max_drawdown": portfolio.get_max_drawdown(),
            }

            # Log metrics to database
            self._log_metrics(session, report)

            return report

        except Exception as e:
            self.log(f"Error generating report: {e}", level="error")
            return {}
        finally:
            session.close()

    def _calculate_strategy_attribution(self, session) -> Dict[str, float]:
        """Calculate P&L attribution by strategy."""
        try:
            # Query trades grouped by strategy
            # Simplified: would need to track position closes
            strategies = session.query(Trade.strategy).distinct().all()

            attribution = {}
            for (strategy,) in strategies:
                # Placeholder logic
                attribution[strategy] = 0.0

            return attribution

        except Exception as e:
            self.log(f"Error calculating attribution: {e}", level="error")
            return {}

    def _calculate_win_rate(self, session) -> float:
        """Calculate overall win rate."""
        try:
            # Simplified: would need closed positions
            total_trades = session.query(Trade).count()
            if total_trades == 0:
                return 0.0

            # Placeholder
            return 0.0

        except Exception as e:
            self.log(f"Error calculating win rate: {e}", level="error")
            return 0.0

    def _calculate_sharpe(self, session) -> float:
        """Calculate Sharpe ratio."""
        # Placeholder - would need daily returns
        return 0.0

    def _log_metrics(self, session, report: Dict):
        """Log performance metrics to database."""
        try:
            metric = PerformanceMetric(
                timestamp=datetime.utcnow(),
                total_pnl=report.get("total_pnl", 0.0),
                daily_pnl=report.get("daily_pnl", 0.0),
                open_positions=report.get("open_positions", 0),
                win_rate=report.get("win_rate", 0.0),
                sharpe_ratio=report.get("sharpe_ratio", 0.0),
                max_drawdown=report.get("max_drawdown", 0.0),
            )
            session.add(metric)
            session.commit()

        except Exception as e:
            session.rollback()
            self.log(f"Error logging metrics: {e}", level="error")
