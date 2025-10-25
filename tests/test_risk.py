"""Tests for risk management."""

import pytest
import numpy as np
from risk.constraints import check_plan, validate_portfolio_risk, estimate_trade_risk
from risk.var import calculate_var, calculate_portfolio_greeks
from storage.portfolio import Portfolio


@pytest.fixture
def mock_portfolio():
    """Create a mock portfolio for testing."""
    portfolio = Portfolio(initial_capital=100000.0)
    return portfolio


@pytest.fixture
def risk_config():
    """Create risk configuration."""
    return {
        "per_trade": {
            "max_loss_pct": 0.01,
            "max_loss_dollars": 1000,
        },
        "exposure": {
            "max_ticker_weight": 0.15,
        },
        "strategy_limits": {
            "iron_condor": {
                "max_positions": 10,
            }
        },
        "portfolio": {
            "max_var_95": 0.03,
            "max_margin_usage": 0.30,
            "max_drawdown_alert": 0.08,
        },
    }


class TestConstraints:
    """Tests for constraint checking."""

    def test_check_plan_passes(self, mock_portfolio, risk_config):
        """Test that valid plan passes all checks."""
        plan = {
            "ticker": "SPY",
            "strategy": "iron_condor",
            "features": {"liquidity_score": 80.0},
        }

        context = {"chains": {}}

        ok, reason = check_plan(plan, risk_config, mock_portfolio, context)

        assert ok
        assert "passed" in reason.lower()

    def test_check_plan_fails_low_liquidity(self, mock_portfolio, risk_config):
        """Test rejection due to low liquidity."""
        plan = {
            "ticker": "SPY",
            "strategy": "iron_condor",
            "features": {"liquidity_score": 30.0},  # Too low
        }

        context = {"chains": {}}

        ok, reason = check_plan(plan, risk_config, mock_portfolio, context)

        assert not ok
        assert "liquidity" in reason.lower()

    def test_validate_portfolio_risk_passes(self, mock_portfolio, risk_config):
        """Test that portfolio within limits passes."""
        ok, message = validate_portfolio_risk(mock_portfolio, risk_config)

        assert ok
        assert "within limits" in message.lower()

    def test_estimate_trade_risk_iron_condor(self):
        """Test risk estimation for iron condor."""
        plan = {"strategy": "iron_condor", "ticker": "SPY"}
        context = {}

        risk = estimate_trade_risk(plan, context)

        assert risk > 0
        assert isinstance(risk, float)


class TestVaR:
    """Tests for VaR calculations."""

    def test_calculate_var_historical(self):
        """Test historical VaR calculation."""
        returns = np.random.normal(0, 0.01, 252)  # Random returns

        var = calculate_var(returns, confidence=0.95, method="historical")

        assert var >= 0
        assert isinstance(var, float)

    def test_calculate_var_parametric(self):
        """Test parametric VaR calculation."""
        returns = np.random.normal(0, 0.01, 252)

        var = calculate_var(returns, confidence=0.95, method="parametric")

        assert var >= 0
        assert isinstance(var, float)

    def test_calculate_var_empty_returns(self):
        """Test VaR with empty returns."""
        returns = np.array([])

        var = calculate_var(returns)

        assert var == 0.0

    def test_calculate_portfolio_greeks(self):
        """Test portfolio greeks aggregation."""
        positions = [
            {"delta": 0.50, "gamma": 0.05, "theta": -0.10, "vega": 0.15, "quantity": 1},
            {"delta": -0.25, "gamma": 0.03, "theta": -0.05, "vega": 0.10, "quantity": 2},
        ]

        greeks = calculate_portfolio_greeks(positions)

        assert "delta" in greeks
        assert "gamma" in greeks
        assert "theta" in greeks
        assert "vega" in greeks

        # Check calculations
        expected_delta = 0.50 * 1 + (-0.25) * 2
        assert abs(greeks["delta"] - expected_delta) < 0.01

    def test_calculate_portfolio_greeks_empty(self):
        """Test greeks with no positions."""
        positions = []

        greeks = calculate_portfolio_greeks(positions)

        assert greeks["delta"] == 0.0
        assert greeks["gamma"] == 0.0
        assert greeks["theta"] == 0.0
        assert greeks["vega"] == 0.0
