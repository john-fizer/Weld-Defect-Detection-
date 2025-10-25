"""Paper trading broker simulation."""

from typing import Dict, Any
import random
from datetime import datetime


class PaperBroker:
    """Simulated broker for paper trading.

    Fills orders at mid-price with configurable slippage.
    No real money involved - perfect for testing.
    """

    def __init__(self, slippage_bps: float = 5.0, commission_per_contract: float = 0.65):
        """Initialize paper broker.

        Args:
            slippage_bps: Slippage in basis points
            commission_per_contract: Commission per contract
        """
        self.slippage_bps = slippage_bps
        self.commission_per_contract = commission_per_contract
        self.order_id = 1000

    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place an order (simulated fill).

        Args:
            order: Order dict with type, ticker, legs, mid_price

        Returns:
            Fill dict with status, fill_price, commission
        """
        self.order_id += 1

        # Simulate fill
        mid_price = order.get("mid_price", 0.0)
        order_type = order.get("type", "")

        # Apply slippage
        if "credit" in order_type or order.get("net_credit"):
            # Selling: get slightly worse price (less credit)
            fill_price = mid_price * (1 - self.slippage_bps / 10000)
        else:
            # Buying: get slightly worse price (more debit)
            fill_price = mid_price * (1 + self.slippage_bps / 10000)

        # Calculate commission
        num_legs = len(order.get("legs", []))
        commission = num_legs * self.commission_per_contract

        # Simulate small chance of rejection (liquidity, etc)
        if random.random() < 0.05:  # 5% rejection rate
            return {
                "order_id": self.order_id,
                "status": "rejected",
                "reason": "Insufficient liquidity (simulated)",
                "timestamp": datetime.utcnow().isoformat(),
            }

        fill = {
            "order_id": self.order_id,
            "status": "filled",
            "ticker": order.get("ticker"),
            "strategy": order.get("strategy"),
            "type": order_type,
            "mid_price": mid_price,
            "fill_price": fill_price,
            "commission": commission,
            "legs": order.get("legs", []),
            "quantity": order.get("quantity", 1),
            "timestamp": datetime.utcnow().isoformat(),
        }

        return fill

    def cancel_order(self, order_id: int) -> bool:
        """Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            True if canceled successfully
        """
        # In paper trading, always succeeds
        return True

    def get_positions(self) -> list:
        """Get current positions.

        Returns:
            List of position dicts
        """
        # In paper broker, positions would be tracked separately
        # This is a stub
        return []

    def get_account_info(self) -> Dict[str, Any]:
        """Get account information.

        Returns:
            Account info dict
        """
        return {
            "account_type": "paper",
            "balance": 100000.00,
            "buying_power": 100000.00,
            "margin_used": 0.00,
        }
