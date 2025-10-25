"""Strategy Planner agent - matches market regime to strategies."""

from typing import Any, Dict, List
import yaml
from .base import Agent


class StrategyPlanner(Agent):
    """Matches market regime to appropriate strategies.

    Responsibilities:
    - Analyze features and market regime
    - Match conditions to strategy criteria
    - Generate trade proposals with reasoning
    - Prioritize opportunities
    """

    name = "strategy_planner"

    def __init__(self, strategies_config: Dict[str, Any], config: Dict[str, Any] = None):
        """Initialize Strategy Planner.

        Args:
            strategies_config: Strategy configurations from YAML
            config: Agent configuration
        """
        super().__init__(config)
        self.strategies_config = strategies_config

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate trade plans based on features.

        Args:
            context: Context with 'features', 'chains', 'spot_prices'

        Returns:
            Context enriched with 'plans'
        """
        if not self.validate_context(context, ["features", "chains"]):
            return context

        features = context["features"]
        chains = context["chains"]

        self.log(f"Planning strategies for {len(features)} tickers")

        plans = []

        for ticker, feat in features.items():
            # Try each strategy
            plans.extend(self._check_iron_condor(ticker, feat, chains.get(ticker)))
            plans.extend(self._check_long_straddle(ticker, feat, chains.get(ticker)))
            plans.extend(self._check_wheel(ticker, feat, chains.get(ticker)))

        # Sort by priority/score
        plans = sorted(plans, key=lambda x: x.get("priority", 0), reverse=True)

        self.add_to_context(context, "plans", plans)
        self.log(f"Generated {len(plans)} trade plans")

        return context

    def _check_iron_condor(self, ticker: str, features: Dict, chain) -> List[Dict]:
        """Check if Iron Condor conditions are met."""
        plans = []
        config = self.strategies_config.get("iron_condor", {})

        if not config.get("enabled", True):
            return plans

        # Check criteria
        ivr = features.get("ivr", 0)
        trend = features.get("trend", "unknown")
        liquidity = features.get("liquidity_score", 0)

        ivr_min = config.get("ivr_min", 40)
        ivr_max = config.get("ivr_max", 80)
        trend_filter = config.get("trend_filter", "range_bound")

        if ivr < ivr_min or ivr > ivr_max:
            return plans

        if trend_filter == "range_bound" and trend != "range_bound":
            return plans

        if liquidity < 50:  # Minimum liquidity threshold
            return plans

        # Conditions met - create plan
        plan = {
            "ticker": ticker,
            "strategy": "iron_condor",
            "reasoning": f"High IV rank ({ivr:.1f}), range-bound market, good liquidity",
            "priority": ivr,  # Higher IV = higher priority
            "config": config,
            "features": features,
        }

        plans.append(plan)
        return plans

    def _check_long_straddle(self, ticker: str, features: Dict, chain) -> List[Dict]:
        """Check if Long Straddle conditions are met."""
        plans = []
        config = self.strategies_config.get("long_straddle", {})

        if not config.get("enabled", True):
            return plans

        # For now, only enter straddles if explicitly signaled
        # In production, would check earnings calendar, FOMC dates, etc.
        # Stub: skip for MVP
        return plans

    def _check_wheel(self, ticker: str, features: Dict, chain) -> List[Dict]:
        """Check if Wheel strategy conditions are met."""
        plans = []
        config = self.strategies_config.get("wheel", {})

        if not config.get("enabled", True):
            return plans

        # Check quality filters (would need fundamental data)
        # For MVP, only enable on high-quality tickers
        quality_tickers = ["SPY", "QQQ", "AAPL", "MSFT"]
        if ticker not in quality_tickers:
            return plans

        ivr = features.get("ivr", 0)
        if ivr < 30:  # Want some IV for premium
            return plans

        plan = {
            "ticker": ticker,
            "strategy": "wheel",
            "reasoning": f"Quality underlying, decent IV ({ivr:.1f}), suitable for income",
            "priority": ivr * 0.5,  # Lower priority than directional trades
            "config": config,
            "features": features,
        }

        plans.append(plan)
        return plans
