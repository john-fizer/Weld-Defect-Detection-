"""Executioner agent - routes orders to broker."""

from typing import Any, Dict, List
from .base import Agent


class Executioner(Agent):
    """Routes approved plans to broker for execution.

    Responsibilities:
    - Convert plans to executable orders
    - Route orders to paper/live broker
    - Handle order slicing for large positions
    - Implement smart order routing (mid-price bidding)
    - Track fills and update portfolio
    """

    name = "executioner"

    def __init__(self, broker: Any, config: Dict[str, Any] = None):
        """Initialize Executioner.

        Args:
            broker: Broker adapter (PaperBroker or IBKRAdapter)
            config: Agent configuration
        """
        super().__init__(config)
        self.broker = broker

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute approved trade plans.

        Args:
            context: Context with 'approved_plans', 'chains'

        Returns:
            Context enriched with 'orders', 'fills', 'failed_orders'
        """
        if not self.validate_context(context, ["approved_plans"]):
            return context

        approved_plans = context["approved_plans"]
        chains = context.get("chains", {})

        self.log(f"Executing {len(approved_plans)} approved plans")

        orders = []
        fills = []
        failed = []

        for plan in approved_plans:
            try:
                # Build order from plan
                order = self._build_order(plan, chains.get(plan["ticker"]))

                if not order:
                    self.log(f"Could not build order for {plan['ticker']}", level="warning")
                    failed.append({"plan": plan, "reason": "Order construction failed"})
                    continue

                orders.append(order)

                # Execute order
                fill = self.broker.place_order(order)

                if fill.get("status") == "filled":
                    fills.append(fill)
                    self.log(f"Filled {plan['ticker']} {plan['strategy']}: {fill}")
                else:
                    failed.append({"plan": plan, "reason": fill.get("reason", "Unknown")})
                    self.log(f"Failed to fill {plan['ticker']}: {fill.get('reason')}", level="warning")

            except Exception as e:
                self.log(f"Error executing {plan['ticker']}: {e}", level="error")
                failed.append({"plan": plan, "reason": str(e)})

        # Update context
        self.add_to_context(context, "orders", orders)
        self.add_to_context(context, "fills", fills)
        self.add_to_context(context, "failed_orders", failed)

        self.log(f"Execution complete: {len(fills)} fills, {len(failed)} failures")

        return context

    def _build_order(self, plan: Dict[str, Any], chain) -> Dict[str, Any]:
        """Build executable order from plan.

        Args:
            plan: Trade plan with strategy and config
            chain: Options chain for the ticker

        Returns:
            Order dict ready for broker
        """
        strategy = plan["strategy"]
        ticker = plan["ticker"]

        if strategy == "iron_condor":
            return self._build_iron_condor_order(plan, chain)
        elif strategy == "long_straddle":
            return self._build_straddle_order(plan, chain)
        elif strategy == "wheel":
            return self._build_wheel_order(plan, chain)
        else:
            self.log(f"Unknown strategy: {strategy}", level="error")
            return None

    def _build_iron_condor_order(self, plan: Dict, chain) -> Dict:
        """Build iron condor order (short call spread + short put spread)."""
        if not chain:
            return None

        config = plan["config"]
        spot = chain.spot_price
        expirations = chain.get_expirations()

        if not expirations:
            return None

        # Use nearest expiration in target DTE range
        target_dte = config.get("target_dte", 35)
        exp = min(expirations, key=lambda x: abs((x - chain.contracts[0].expiration).days - target_dte))

        # Calculate strikes
        short_call_strike = spot * (1 + config.get("short_wing_pct", 0.20))
        long_call_strike = short_call_strike * (1 + config.get("long_wing_extra_pct", 0.05))

        short_put_strike = spot * (1 - config.get("short_wing_pct", 0.20))
        long_put_strike = short_put_strike * (1 - config.get("long_wing_extra_pct", 0.05))

        # Find nearest actual strikes
        calls = chain.get_calls(exp)
        puts = chain.get_puts(exp)

        if not calls or not puts:
            return None

        short_call = min(calls, key=lambda x: abs(x.strike - short_call_strike))
        long_call = min(calls, key=lambda x: abs(x.strike - long_call_strike))
        short_put = min(puts, key=lambda x: abs(x.strike - short_put_strike))
        long_put = min(puts, key=lambda x: abs(x.strike - long_put_strike))

        # Calculate net credit
        credit = (short_call.mid_price + short_put.mid_price -
                  long_call.mid_price - long_put.mid_price)

        return {
            "type": "iron_condor",
            "ticker": plan["ticker"],
            "legs": [
                {"action": "sell", "contract": short_call, "quantity": 1},
                {"action": "buy", "contract": long_call, "quantity": 1},
                {"action": "sell", "contract": short_put, "quantity": 1},
                {"action": "buy", "contract": long_put, "quantity": 1},
            ],
            "net_credit": credit,
            "mid_price": credit,
            "strategy": "iron_condor",
        }

    def _build_straddle_order(self, plan: Dict, chain) -> Dict:
        """Build long straddle order."""
        if not chain:
            return None

        expirations = chain.get_expirations()
        if not expirations:
            return None

        exp = expirations[0]
        straddle = chain.get_atm_straddle(exp)

        if not straddle:
            return None

        call, put = straddle
        cost = call.mid_price + put.mid_price

        return {
            "type": "straddle",
            "ticker": plan["ticker"],
            "legs": [
                {"action": "buy", "contract": call, "quantity": 1},
                {"action": "buy", "contract": put, "quantity": 1},
            ],
            "net_debit": cost,
            "mid_price": cost,
            "strategy": "long_straddle",
        }

    def _build_wheel_order(self, plan: Dict, chain) -> Dict:
        """Build wheel order (cash-secured put)."""
        if not chain:
            return None

        config = plan["config"]
        expirations = chain.get_expirations()

        if not expirations:
            return None

        # Target DTE
        target_dte = config.get("put_dte", 30)
        exp = min(expirations, key=lambda x: abs((x - chain.contracts[0].expiration).days - target_dte))

        # Find put at target delta
        target_delta = config.get("put_delta", -0.25)
        puts = chain.get_puts(exp)

        if not puts:
            return None

        # Filter by delta
        puts_with_delta = [p for p in puts if p.delta and abs(p.delta - target_delta) < 0.05]

        if not puts_with_delta:
            # Fallback to strike-based selection
            spot = chain.spot_price
            target_strike = spot * 0.75  # ~25% OTM
            put = min(puts, key=lambda x: abs(x.strike - target_strike))
        else:
            put = puts_with_delta[0]

        return {
            "type": "cash_secured_put",
            "ticker": plan["ticker"],
            "legs": [
                {"action": "sell", "contract": put, "quantity": 1},
            ],
            "net_credit": put.mid_price,
            "mid_price": put.mid_price,
            "strategy": "wheel",
        }
