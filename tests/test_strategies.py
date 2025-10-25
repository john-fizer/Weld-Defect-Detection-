"""Tests for trading strategies."""

import pytest
from datetime import date
from strategies import IronCondorStrategy, LongStraddleStrategy, WheelStrategy
from data_providers.base import OptionsChain, OptionContract


@pytest.fixture
def sample_chain():
    """Create a sample options chain for testing."""
    spot_price = 450.0
    expiration = date(2024, 12, 31)

    contracts = [
        # Calls
        OptionContract(
            symbol="SPY_C_450",
            underlying="SPY",
            expiration=expiration,
            strike=450.0,
            option_type="call",
            bid=10.0,
            ask=10.5,
            last=10.25,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.20,
            delta=0.50,
            gamma=0.05,
            theta=-0.10,
            vega=0.15,
        ),
        OptionContract(
            symbol="SPY_C_480",
            underlying="SPY",
            expiration=expiration,
            strike=480.0,
            option_type="call",
            bid=2.0,
            ask=2.2,
            last=2.1,
            volume=500,
            open_interest=2000,
            implied_volatility=0.22,
            delta=0.25,
        ),
        # Puts
        OptionContract(
            symbol="SPY_P_450",
            underlying="SPY",
            expiration=expiration,
            strike=450.0,
            option_type="put",
            bid=10.0,
            ask=10.5,
            last=10.25,
            volume=1000,
            open_interest=5000,
            implied_volatility=0.20,
            delta=-0.50,
        ),
        OptionContract(
            symbol="SPY_P_420",
            underlying="SPY",
            expiration=expiration,
            strike=420.0,
            option_type="put",
            bid=2.0,
            ask=2.2,
            last=2.1,
            volume=500,
            open_interest=2000,
            implied_volatility=0.22,
            delta=-0.25,
        ),
    ]

    return OptionsChain(underlying="SPY", spot_price=spot_price, contracts=contracts)


class TestIronCondorStrategy:
    """Tests for Iron Condor strategy."""

    def test_evaluate_with_good_conditions(self, sample_chain):
        """Test evaluation with ideal conditions."""
        config = {
            "enabled": True,
            "ivr_min": 40,
            "trend_filter": "range_bound",
            "target_dte": 35,
            "min_days_to_expiry": 25,
            "max_days_to_expiry": 45,
        }

        features = {
            "ivr": 55.0,
            "trend": "range_bound",
            "liquidity_score": 75.0,
        }

        strategy = IronCondorStrategy(config)
        setup = strategy.evaluate("SPY", sample_chain, features)

        assert setup is not None
        assert setup["ticker"] == "SPY"
        assert setup["strategy"] == "iron_condor"
        assert "short_call_strike" in setup
        assert "short_put_strike" in setup

    def test_evaluate_with_low_ivr(self, sample_chain):
        """Test that low IVR rejects the trade."""
        config = {"enabled": True, "ivr_min": 40}
        features = {"ivr": 30.0, "trend": "range_bound", "liquidity_score": 75.0}

        strategy = IronCondorStrategy(config)
        setup = strategy.evaluate("SPY", sample_chain, features)

        assert setup is None

    def test_calculate_position_size(self):
        """Test position sizing."""
        config = {}
        strategy = IronCondorStrategy(config)

        size = strategy.calculate_position_size(account_value=100000, max_risk=1000)
        assert size == 1  # Simplified implementation

    def test_exit_conditions(self):
        """Test exit conditions."""
        config = {
            "take_profit_pct": 0.50,
            "stop_loss_pct": 2.00,
            "manage_at_dte": 21,
        }

        strategy = IronCondorStrategy(config)
        exit_rules = strategy.get_exit_conditions({})

        assert exit_rules["take_profit_pct"] == 0.50
        assert exit_rules["stop_loss_pct"] == 2.00
        assert exit_rules["manage_at_dte"] == 21


class TestLongStraddleStrategy:
    """Tests for Long Straddle strategy."""

    def test_evaluate_without_event(self, sample_chain):
        """Test that strategy skips without event."""
        config = {"enabled": True, "event_required": True}
        features = {"ivr": 50.0, "priced_move": 5.0}

        strategy = LongStraddleStrategy(config)
        setup = strategy.evaluate("SPY", sample_chain, features)

        assert setup is None  # No upcoming_event in features

    def test_evaluate_with_low_priced_move(self, sample_chain):
        """Test rejection when priced move too low."""
        config = {"enabled": True, "event_required": False, "min_expected_move": 0.05}
        features = {"ivr": 50.0, "priced_move": 2.0}  # Only 2% move

        strategy = LongStraddleStrategy(config)
        setup = strategy.evaluate("SPY", sample_chain, features)

        assert setup is None


class TestWheelStrategy:
    """Tests for Wheel strategy."""

    def test_evaluate_quality_ticker(self, sample_chain):
        """Test evaluation on quality ticker."""
        config = {"enabled": True, "put_dte": 30, "put_delta": -0.25}
        features = {"ivr": 40.0}

        strategy = WheelStrategy(config)
        setup = strategy.evaluate("SPY", sample_chain, features)

        assert setup is not None
        assert setup["ticker"] == "SPY"
        assert setup["strategy"] == "wheel"
        assert setup["action"] == "sell_put"

    def test_evaluate_non_quality_ticker(self, sample_chain):
        """Test rejection of non-quality ticker."""
        config = {"enabled": True}
        features = {"ivr": 40.0}

        strategy = WheelStrategy(config)
        setup = strategy.evaluate("JUNK", sample_chain, features)

        assert setup is None  # Not in quality list

    def test_evaluate_low_iv(self, sample_chain):
        """Test rejection when IV too low."""
        config = {"enabled": True}
        features = {"ivr": 20.0}  # Too low

        strategy = WheelStrategy(config)
        setup = strategy.evaluate("SPY", sample_chain, features)

        assert setup is None
