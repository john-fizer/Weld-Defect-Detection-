"""Signal Engineer agent - computes features and signals."""

from typing import Any, Dict
import numpy as np
import pandas as pd
from .base import Agent


class SignalEngineer(Agent):
    """Computes features and trading signals from market data.

    Responsibilities:
    - Calculate IV rank, skew, term structure
    - Compute priced moves from straddles
    - Detect volatility regime
    - Calculate put/call ratios, gamma exposure
    - Identify flow anomalies
    """

    name = "signal_engineer"

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compute signals and features.

        Args:
            context: Context with 'chains', 'ohlc', 'market_metrics'

        Returns:
            Context enriched with 'features' and 'signals'
        """
        required = ["chains", "ohlc", "market_metrics"]
        if not self.validate_context(context, required):
            return context

        chains = context["chains"]
        ohlc = context["ohlc"]

        self.log(f"Engineering features for {len(chains)} tickers")

        features = {}
        for ticker, chain in chains.items():
            try:
                ticker_features = self._compute_ticker_features(ticker, chain, ohlc.get(ticker))
                features[ticker] = ticker_features
            except Exception as e:
                self.log(f"Error computing features for {ticker}: {e}", level="warning")
                continue

        self.add_to_context(context, "features", features)
        self.log(f"Engineered features for {len(features)} tickers")

        return context

    def _compute_ticker_features(self, ticker: str, chain, ohlc_df) -> Dict[str, Any]:
        """Compute all features for a single ticker."""
        features = {}

        # IV Rank - need historical IV
        features["ivr"] = self._calculate_iv_rank(chain, ohlc_df)

        # Priced move from ATM straddle
        expirations = chain.get_expirations()
        if expirations:
            nearest_exp = expirations[0]
            priced_move = chain.priced_move_pct(nearest_exp)
            features["priced_move"] = priced_move if priced_move else 0.0

            # Straddle cost
            straddle = chain.get_atm_straddle(nearest_exp)
            if straddle:
                call, put = straddle
                features["straddle_cost"] = call.mid_price + put.mid_price
                features["atm_iv"] = (call.implied_volatility + put.implied_volatility) / 2
            else:
                features["straddle_cost"] = 0.0
                features["atm_iv"] = 0.0
        else:
            features["priced_move"] = 0.0
            features["straddle_cost"] = 0.0
            features["atm_iv"] = 0.0

        # Put/Call ratio by volume and OI
        features.update(self._calculate_put_call_ratios(chain))

        # IV Skew
        features["iv_skew"] = self._calculate_iv_skew(chain)

        # Liquidity score
        features["liquidity_score"] = self._calculate_liquidity_score(chain)

        # Trend detection
        if ohlc_df is not None and not ohlc_df.empty:
            features.update(self._detect_trend(ohlc_df))

        return features

    def _calculate_iv_rank(self, chain, ohlc_df, lookback=252) -> float:
        """Calculate IV Rank.

        IV Rank = (Current IV - Min IV) / (Max IV - Min IV) * 100
        """
        if not hasattr(chain, 'contracts') or not chain.contracts:
            return 0.0

        # Use ATM IV as current IV
        current_iv = 0.0
        expirations = chain.get_expirations()
        if expirations:
            straddle = chain.get_atm_straddle(expirations[0])
            if straddle:
                call, put = straddle
                current_iv = (call.implied_volatility + put.implied_volatility) / 2

        # For simplicity, use historical volatility range from OHLCV
        # In production, you'd track historical IV
        if ohlc_df is not None and not ohlc_df.empty:
            returns = ohlc_df["close"].pct_change().dropna()
            if len(returns) > lookback:
                returns = returns.tail(lookback)

            rolling_vol = returns.rolling(window=20).std() * np.sqrt(252)
            min_vol = rolling_vol.min()
            max_vol = rolling_vol.max()

            if max_vol > min_vol:
                ivr = ((current_iv - min_vol) / (max_vol - min_vol)) * 100
                return float(np.clip(ivr, 0, 100))

        return 50.0  # Default to mid-range if can't calculate

    def _calculate_put_call_ratios(self, chain) -> Dict[str, float]:
        """Calculate put/call ratios."""
        puts = chain.get_puts()
        calls = chain.get_calls()

        total_put_volume = sum(p.volume for p in puts)
        total_call_volume = sum(c.volume for c in calls)

        total_put_oi = sum(p.open_interest for p in puts)
        total_call_oi = sum(c.open_interest for c in calls)

        pc_volume_ratio = total_put_volume / max(total_call_volume, 1)
        pc_oi_ratio = total_put_oi / max(total_call_oi, 1)

        return {
            "put_call_volume_ratio": pc_volume_ratio,
            "put_call_oi_ratio": pc_oi_ratio,
        }

    def _calculate_iv_skew(self, chain) -> float:
        """Calculate IV skew (OTM put IV - OTM call IV)."""
        expirations = chain.get_expirations()
        if not expirations:
            return 0.0

        exp = expirations[0]
        spot = chain.spot_price

        # Get 10% OTM puts and calls
        puts = chain.get_puts(exp)
        calls = chain.get_calls(exp)

        otm_puts = [p for p in puts if p.strike < spot * 0.90]
        otm_calls = [c for c in calls if c.strike > spot * 1.10]

        if otm_puts and otm_calls:
            avg_put_iv = np.mean([p.implied_volatility for p in otm_puts[:3]])
            avg_call_iv = np.mean([c.implied_volatility for c in otm_calls[:3]])
            return float(avg_put_iv - avg_call_iv)

        return 0.0

    def _calculate_liquidity_score(self, chain) -> float:
        """Calculate liquidity score (0-100) based on OI and spreads."""
        if not chain.contracts:
            return 0.0

        # Average open interest
        avg_oi = np.mean([c.open_interest for c in chain.contracts])

        # Average bid-ask spread
        avg_spread = np.mean([c.bid_ask_spread_pct for c in chain.contracts if c.mid_price > 0])

        # Score: higher OI = better, lower spread = better
        oi_score = min(avg_oi / 1000, 100)  # Normalize to 0-100
        spread_score = max(0, 100 - avg_spread)  # Lower spread = higher score

        return float((oi_score + spread_score) / 2)

    def _detect_trend(self, df) -> Dict[str, Any]:
        """Detect trend using moving averages."""
        if len(df) < 50:
            return {"trend": "unknown", "trend_strength": 0.0}

        close = df["close"]
        sma_20 = close.rolling(window=20).mean().iloc[-1]
        sma_50 = close.rolling(window=50).mean().iloc[-1]
        current = close.iloc[-1]

        if current > sma_20 > sma_50:
            trend = "bullish"
            strength = ((current - sma_50) / sma_50) * 100
        elif current < sma_20 < sma_50:
            trend = "bearish"
            strength = ((sma_50 - current) / sma_50) * 100
        else:
            trend = "range_bound"
            strength = 0.0

        return {
            "trend": trend,
            "trend_strength": float(abs(strength)),
        }
