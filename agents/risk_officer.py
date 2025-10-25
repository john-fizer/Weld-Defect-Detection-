"""Risk Officer agent - validates and enforces risk constraints."""

from typing import Any, Dict, List, Tuple
from .base import Agent
from risk.constraints import check_plan, validate_portfolio_risk


class RiskOfficer(Agent):
    """Enforces risk constraints and validates trade plans.

    Responsibilities:
    - Validate trade plans against risk policies
    - Check portfolio-level risk limits
    - Enforce position sizing
    - Flag high-risk trades for human review
    - Implement kill-switch logic
    """

    name = "risk_officer"

    def __init__(self, risk_config: Dict[str, Any], portfolio: Any, config: Dict[str, Any] = None):
        """Initialize Risk Officer.

        Args:
            risk_config: Risk policies from YAML
            portfolio: Portfolio object with current positions
            config: Agent configuration
        """
        super().__init__(config)
        self.risk_config = risk_config
        self.portfolio = portfolio

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Validate plans and enforce risk constraints.

        Args:
            context: Context with 'plans'

        Returns:
            Context enriched with 'approved_plans', 'flagged_plans', 'rejected_plans'
        """
        if not self.validate_context(context, ["plans"]):
            return context

        plans = context["plans"]
        self.log(f"Reviewing {len(plans)} trade plans")

        # Check kill-switch conditions first
        if self._check_kill_switch():
            self.log("KILL SWITCH ACTIVATED - All plans rejected", level="error")
            self.add_to_context(context, "approved_plans", [])
            self.add_to_context(context, "flagged_plans", [])
            self.add_to_context(context, "rejected_plans", plans)
            self.add_to_context(context, "kill_switch_active", True)
            return context

        # Validate portfolio-level risk
        portfolio_risk_ok, portfolio_msg = validate_portfolio_risk(
            self.portfolio,
            self.risk_config
        )
        if not portfolio_risk_ok:
            self.log(f"Portfolio risk violation: {portfolio_msg}", level="warning")

        # Review each plan
        approved = []
        flagged = []
        rejected = []

        for plan in plans:
            ok, reason = check_plan(plan, self.risk_config, self.portfolio, context)

            enriched_plan = {**plan, "risk_review": reason}

            if ok:
                # Check if needs human approval
                if self._requires_human_approval(plan):
                    flagged.append(enriched_plan)
                    self.log(f"Flagged {plan['ticker']} {plan['strategy']} for human review: {reason}")
                else:
                    approved.append(enriched_plan)
                    self.log(f"Approved {plan['ticker']} {plan['strategy']}: {reason}")
            else:
                rejected.append(enriched_plan)
                self.log(f"Rejected {plan['ticker']} {plan['strategy']}: {reason}", level="warning")

        # Update context
        self.add_to_context(context, "approved_plans", approved)
        self.add_to_context(context, "flagged_plans", flagged)
        self.add_to_context(context, "rejected_plans", rejected)
        self.add_to_context(context, "kill_switch_active", False)

        self.log(f"Review complete: {len(approved)} approved, {len(flagged)} flagged, {len(rejected)} rejected")

        return context

    def _check_kill_switch(self) -> bool:
        """Check if kill-switch conditions are met.

        Returns:
            True if kill-switch should activate
        """
        kill_switch_config = self.risk_config.get("kill_switch", {})

        # Check daily loss
        daily_loss_pct = self.portfolio.get_daily_pnl_pct()
        max_daily_loss = kill_switch_config.get("max_daily_loss_pct", 0.05)
        if daily_loss_pct < -max_daily_loss:
            self.log(f"Kill switch: Daily loss {daily_loss_pct:.2%} exceeds limit", level="error")
            return True

        # Check consecutive losses
        consecutive_losses = self.portfolio.get_consecutive_losses()
        max_consecutive = kill_switch_config.get("max_consecutive_losses", 5)
        if consecutive_losses >= max_consecutive:
            self.log(f"Kill switch: {consecutive_losses} consecutive losses", level="error")
            return True

        # Check VaR breach
        current_var = self.portfolio.get_var_95()
        max_var = self.risk_config.get("portfolio", {}).get("max_var_95", 0.03)
        breach_multiplier = kill_switch_config.get("var_breach_multiplier", 1.5)
        if current_var > max_var * breach_multiplier:
            self.log(f"Kill switch: VaR {current_var:.2%} exceeds limit", level="error")
            return True

        return False

    def _requires_human_approval(self, plan: Dict[str, Any]) -> bool:
        """Check if plan requires human approval.

        Args:
            plan: Trade plan

        Returns:
            True if human approval required
        """
        approval_config = self.risk_config.get("human_approval", {})

        # New strategy requires approval
        if approval_config.get("required_for_new_strategy", True):
            if plan.get("strategy") not in self.portfolio.get_strategy_history():
                return True

        # Risk limit breach requires approval
        if approval_config.get("required_for_risk_limit_breach", True):
            # Would check if plan pushes limits
            pass

        # Live trading always requires approval
        if approval_config.get("required_for_live_trading", True):
            if self.portfolio.is_live_trading():
                return True

        return False
