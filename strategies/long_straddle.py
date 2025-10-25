"""Long Straddle strategy implementation."""

from typing import Dict, Any, Optional
from .base import Strategy


class LongStraddleStrategy(Strategy):
    """Long Straddle: Buy ATM call + put before volatility events.

    Profit from large moves in either direction.
    Best before earnings, FOMC, or other catalyst events.
    """

    name = "long_straddle"

    def evaluate(self, ticker: str, chain, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate if long straddle conditions are met.

        Criteria:
        - Upcoming event (earnings, FOMC, etc)
        - Priced move < historical move
        - Sufficient IV level
        """
        if not chain or not chain.contracts:
            return None

        # Check if event-based entry required
        if self.config.get("event_required", True):
            # In production, would check earnings calendar
            # For MVP, skip this strategy unless explicitly signaled
            if not features.get("upcoming_event"):
                return None

        # Check minimum expected move
        priced_move = features.get("priced_move", 0)
        if priced_move < self.config.get("min_expected_move", 0.05) * 100:
            return None

        # Check IV rank
        ivr = features.get("ivr", 0)
        if ivr < self.config.get("min_iv_rank", 30):
            return None

        # Find appropriate expiration (near event)
        expirations = chain.get_expirations()
        if not expirations:
            return None

        # Use nearest expiration for event play
        expiration = expirations[0]

        # Get ATM straddle
        straddle = chain.get_atm_straddle(expiration)
        if not straddle:
            return None

        call, put = straddle
        cost = call.mid_price + put.mid_price

        setup = {
            "ticker": ticker,
            "strategy": self.name,
            "expiration": expiration,
            "spot_price": chain.spot_price,
            "strike": call.strike,
            "call_price": call.mid_price,
            "put_price": put.mid_price,
            "total_cost": cost,
            "priced_move": priced_move,
            "ivr": ivr,
            "features": features,
        }

        return setup

    def calculate_position_size(self, account_value: float, max_risk: float) -> int:
        """Calculate number of contracts.

        For long straddle, risk = total debit paid
        """
        # Simplified: 1 contract
        return 1

    def get_exit_conditions(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Get exit conditions."""
        return {
            "take_profit_pct": self.config.get("take_profit_pct", 0.30),
            "stop_loss_pct": self.config.get("stop_loss_pct", 0.50),
            "exit_after_event": self.config.get("exit_after_event", True),
            "hold_max_days": self.config.get("hold_max_days", 3),
        }
