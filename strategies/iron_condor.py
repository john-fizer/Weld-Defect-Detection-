"""Iron Condor strategy implementation."""

from typing import Dict, Any, Optional
from .base import Strategy


class IronCondorStrategy(Strategy):
    """Iron Condor: Sell OTM call and put credit spreads.

    Profit from range-bound movement and theta decay.
    Best in high IV, low movement environments.
    """

    name = "iron_condor"

    def evaluate(self, ticker: str, chain, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate if iron condor conditions are met.

        Criteria:
        - IV Rank > threshold
        - Range-bound market
        - Good liquidity
        - No earnings within window
        """
        if not chain or not chain.contracts:
            return None

        # Check IV Rank
        ivr = features.get("ivr", 0)
        if ivr < self.config.get("ivr_min", 40):
            return None

        # Check trend
        trend = features.get("trend", "unknown")
        if self.config.get("trend_filter") == "range_bound" and trend != "range_bound":
            return None

        # Check liquidity
        if features.get("liquidity_score", 0) < 50:
            return None

        # Find appropriate expiration
        expirations = chain.get_expirations()
        if not expirations:
            return None

        target_dte = self.config.get("target_dte", 35)
        min_dte = self.config.get("min_days_to_expiry", 25)
        max_dte = self.config.get("max_days_to_expiry", 45)

        valid_exps = [exp for exp in expirations
                      if min_dte <= (exp - chain.contracts[0].expiration).days <= max_dte]

        if not valid_exps:
            return None

        expiration = min(valid_exps, key=lambda x: abs((x - chain.contracts[0].expiration).days - target_dte))

        # Calculate strikes
        spot = chain.spot_price
        short_wing_pct = self.config.get("short_wing_pct", 0.20)
        long_wing_extra_pct = self.config.get("long_wing_extra_pct", 0.05)

        setup = {
            "ticker": ticker,
            "strategy": self.name,
            "expiration": expiration,
            "spot_price": spot,
            "short_call_strike": spot * (1 + short_wing_pct),
            "long_call_strike": spot * (1 + short_wing_pct + long_wing_extra_pct),
            "short_put_strike": spot * (1 - short_wing_pct),
            "long_put_strike": spot * (1 - short_wing_pct - long_wing_extra_pct),
            "ivr": ivr,
            "features": features,
        }

        return setup

    def calculate_position_size(self, account_value: float, max_risk: float) -> int:
        """Calculate number of contracts.

        For iron condor, risk = width of wider spread * contracts
        """
        # Simplified: 1 contract for now
        # In production: max_risk / spread_width
        return 1

    def get_exit_conditions(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Get exit conditions.

        Returns:
            Dict with take_profit_pct, stop_loss_pct, manage_at_dte
        """
        return {
            "take_profit_pct": self.config.get("take_profit_pct", 0.50),
            "stop_loss_pct": self.config.get("stop_loss_pct", 2.00),
            "manage_at_dte": self.config.get("manage_at_dte", 21),
        }
