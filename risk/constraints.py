"""Risk constraint validation."""

from typing import Dict, Any, Tuple


def check_plan(
    plan: Dict[str, Any],
    risk_config: Dict[str, Any],
    portfolio: Any,
    context: Dict[str, Any],
) -> Tuple[bool, str]:
    """Check if trade plan meets risk constraints.

    Args:
        plan: Trade plan to validate
        risk_config: Risk policy configuration
        portfolio: Portfolio object
        context: Shared context

    Returns:
        Tuple of (approved: bool, reason: str)
    """
    # Per-trade checks
    per_trade_config = risk_config.get("per_trade", {})

    # Check max loss
    max_loss_pct = per_trade_config.get("max_loss_pct", 0.01)
    max_loss_dollars = per_trade_config.get("max_loss_dollars", 1000)

    estimated_risk = estimate_trade_risk(plan, context)
    account_value = portfolio.get_account_value()

    if estimated_risk > account_value * max_loss_pct:
        return False, f"Risk ${estimated_risk:.2f} exceeds {max_loss_pct:.1%} of account"

    if estimated_risk > max_loss_dollars:
        return False, f"Risk ${estimated_risk:.2f} exceeds max ${max_loss_dollars}"

    # Check DTE bounds
    # (would need expiration from plan)

    # Check liquidity requirements
    if plan.get("features", {}).get("liquidity_score", 0) < 50:
        return False, "Insufficient liquidity"

    # Check exposure limits
    exposure_config = risk_config.get("exposure", {})
    ticker = plan["ticker"]

    current_ticker_weight = portfolio.get_ticker_weight(ticker)
    max_ticker_weight = exposure_config.get("max_ticker_weight", 0.15)

    if current_ticker_weight > max_ticker_weight:
        return False, f"Ticker weight {current_ticker_weight:.1%} exceeds max {max_ticker_weight:.1%}"

    # Check strategy limits
    strategy = plan["strategy"]
    strategy_limits = risk_config.get("strategy_limits", {}).get(strategy, {})
    max_positions = strategy_limits.get("max_positions", 10)
    current_positions = portfolio.get_strategy_position_count(strategy)

    if current_positions >= max_positions:
        return False, f"Strategy has {current_positions} positions (max {max_positions})"

    # All checks passed
    return True, "All risk checks passed"


def validate_portfolio_risk(
    portfolio: Any,
    risk_config: Dict[str, Any],
) -> Tuple[bool, str]:
    """Validate portfolio-level risk limits.

    Args:
        portfolio: Portfolio object
        risk_config: Risk policy configuration

    Returns:
        Tuple of (ok: bool, message: str)
    """
    portfolio_config = risk_config.get("portfolio", {})

    # Check VaR
    current_var = portfolio.get_var_95()
    max_var = portfolio_config.get("max_var_95", 0.03)

    if current_var > max_var:
        return False, f"VaR {current_var:.2%} exceeds limit {max_var:.2%}"

    # Check margin usage
    margin_usage = portfolio.get_margin_usage()
    max_margin = portfolio_config.get("max_margin_usage", 0.30)

    if margin_usage > max_margin:
        return False, f"Margin usage {margin_usage:.1%} exceeds limit {max_margin:.1%}"

    # Check drawdown
    drawdown = portfolio.get_max_drawdown()
    max_dd_alert = portfolio_config.get("max_drawdown_alert", 0.08)

    if drawdown > max_dd_alert:
        return False, f"Drawdown {drawdown:.1%} exceeds alert threshold {max_dd_alert:.1%}"

    return True, "Portfolio risk within limits"


def estimate_trade_risk(plan: Dict[str, Any], context: Dict[str, Any]) -> float:
    """Estimate maximum risk for a trade plan.

    Args:
        plan: Trade plan
        context: Shared context

    Returns:
        Estimated max loss in dollars
    """
    strategy = plan["strategy"]

    if strategy == "iron_condor":
        # Risk = width of wider spread * contracts * 100
        # Simplified: assume $5 wide spreads
        return 500.0  # $5 * 1 contract * 100

    elif strategy == "long_straddle":
        # Risk = total debit paid
        return plan.get("features", {}).get("straddle_cost", 0) * 100

    elif strategy == "wheel":
        # Risk = strike * 100 (cash-secured put)
        # But true risk depends on how far stock could fall
        # Simplified: 50% of strike
        chain = context.get("chains", {}).get(plan["ticker"])
        if chain:
            return chain.spot_price * 0.50 * 100
        return 5000.0  # Default estimate

    return 1000.0  # Default unknown risk
