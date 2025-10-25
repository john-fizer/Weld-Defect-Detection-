"""Wheel strategy implementation."""

from typing import Dict, Any, Optional
from .base import Strategy


class WheelStrategy(Strategy):
    """Wheel: Sell cash-secured puts, if assigned sell covered calls.

    Income generation strategy for quality underlyings.
    Accept assignment, sell calls against shares.
    """

    name = "wheel"

    def evaluate(self, ticker: str, chain, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate if wheel strategy conditions are met.

        Criteria:
        - Quality underlying (market cap, liquidity)
        - Decent IV for premium
        - Acceptable assignment risk
        """
        if not chain or not chain.contracts:
            return None

        # Check if ticker is in quality list
        # In production, would check fundamentals
        quality_tickers = ["SPY", "QQQ", "AAPL", "MSFT", "AMZN", "GOOGL"]
        if ticker not in quality_tickers:
            return None

        # Check IV for premium
        ivr = features.get("ivr", 0)
        if ivr < 30:
            return None

        # Find appropriate expiration
        expirations = chain.get_expirations()
        if not expirations:
            return None

        target_dte = self.config.get("put_dte", 30)
        expiration = min(expirations, key=lambda x: abs((x - chain.contracts[0].expiration).days - target_dte))

        # Find put at target delta
        target_delta = self.config.get("put_delta", -0.25)
        puts = chain.get_puts(expiration)

        if not puts:
            return None

        # Filter puts with delta info
        puts_with_delta = [p for p in puts if p.delta]

        if puts_with_delta:
            # Find closest to target delta
            selected_put = min(puts_with_delta, key=lambda x: abs(x.delta - target_delta))
        else:
            # Fallback to strike-based selection (~25% OTM)
            target_strike = chain.spot_price * 0.75
            selected_put = min(puts, key=lambda x: abs(x.strike - target_strike))

        # Calculate annualized return
        dte = (expiration - chain.contracts[0].expiration).days
        if dte == 0:
            return None

        premium_pct = (selected_put.mid_price / selected_put.strike) * 100
        annualized_return = (premium_pct / dte) * 365

        # Check minimum return
        if annualized_return < self.config.get("min_premium_annual_pct", 0.12) * 100:
            return None

        setup = {
            "ticker": ticker,
            "strategy": self.name,
            "action": "sell_put",
            "expiration": expiration,
            "strike": selected_put.strike,
            "premium": selected_put.mid_price,
            "delta": selected_put.delta,
            "spot_price": chain.spot_price,
            "premium_pct": premium_pct,
            "annualized_return_pct": annualized_return,
            "ivr": ivr,
            "features": features,
        }

        return setup

    def calculate_position_size(self, account_value: float, max_risk: float) -> int:
        """Calculate number of contracts.

        For cash-secured puts, each contract requires strike * 100 capital
        """
        # Simplified: 1 contract
        return 1

    def get_exit_conditions(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Get exit conditions."""
        return {
            "close_put_at_profit_pct": self.config.get("close_put_at_profit_pct", 0.50),
            "roll_at_dte": self.config.get("roll_at_dte", 7),
            "accept_assignment": self.config.get("accept_assignment", True),
        }
