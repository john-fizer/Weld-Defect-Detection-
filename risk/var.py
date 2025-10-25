"""VaR and portfolio risk calculations."""

import numpy as np
from typing import Dict, Any, List
from scipy import stats


def calculate_var(
    returns: np.ndarray,
    confidence: float = 0.95,
    method: str = "historical",
) -> float:
    """Calculate Value at Risk.

    Args:
        returns: Array of historical returns
        confidence: Confidence level (0.95 = 95% VaR)
        method: 'historical' or 'parametric'

    Returns:
        VaR as decimal (e.g., 0.03 = 3%)
    """
    if len(returns) == 0:
        return 0.0

    if method == "historical":
        # Historical VaR: percentile of historical returns
        var = np.percentile(returns, (1 - confidence) * 100)
        return abs(float(var))

    elif method == "parametric":
        # Parametric VaR: assumes normal distribution
        mean = np.mean(returns)
        std = np.std(returns)
        z_score = stats.norm.ppf(1 - confidence)
        var = mean + z_score * std
        return abs(float(var))

    return 0.0


def calculate_portfolio_greeks(positions: List[Dict[str, Any]]) -> Dict[str, float]:
    """Calculate aggregated portfolio Greeks.

    Args:
        positions: List of position dicts with greeks

    Returns:
        Dict with total delta, gamma, theta, vega
    """
    if not positions:
        return {
            "delta": 0.0,
            "gamma": 0.0,
            "theta": 0.0,
            "vega": 0.0,
        }

    total_delta = sum(p.get("delta", 0) * p.get("quantity", 0) for p in positions)
    total_gamma = sum(p.get("gamma", 0) * p.get("quantity", 0) for p in positions)
    total_theta = sum(p.get("theta", 0) * p.get("quantity", 0) for p in positions)
    total_vega = sum(p.get("vega", 0) * p.get("quantity", 0) for p in positions)

    return {
        "delta": float(total_delta),
        "gamma": float(total_gamma),
        "theta": float(total_theta),
        "vega": float(total_vega),
    }


def calculate_position_var(
    position: Dict[str, Any],
    underlying_volatility: float,
    confidence: float = 0.95,
) -> float:
    """Calculate VaR for a single position.

    Args:
        position: Position dict with greeks
        underlying_volatility: Underlying volatility (annualized)
        confidence: Confidence level

    Returns:
        VaR in dollars
    """
    # Simplified: use delta approximation
    delta = position.get("delta", 0)
    quantity = position.get("quantity", 1)
    underlying_price = position.get("underlying_price", 100)

    # Z-score for confidence level
    z_score = stats.norm.ppf(confidence)

    # Estimated move (1-day)
    daily_vol = underlying_volatility / np.sqrt(252)
    expected_move = underlying_price * daily_vol * z_score

    # Position VaR
    var = abs(delta * quantity * expected_move * 100)

    return float(var)
