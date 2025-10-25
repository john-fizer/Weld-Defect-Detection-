"""Market Scout agent - fetches and normalizes market data."""

from typing import Any, Dict
import numpy as np
from .base import Agent
from data_providers.base import MarketDataProvider, OptionsDataProvider


class MarketScout(Agent):
    """Fetches market data, options chains, and basic market metrics.

    Responsibilities:
    - Pull OHLCV data for universe
    - Fetch options chains with greeks
    - Calculate basic metrics (ATR, volume, IV levels)
    - Identify upcoming events (earnings, FOMC, etc)
    """

    name = "market_scout"

    def __init__(
        self,
        market_provider: MarketDataProvider,
        options_provider: OptionsDataProvider,
        config: Dict[str, Any] = None,
    ):
        """Initialize Market Scout.

        Args:
            market_provider: Market data provider instance
            options_provider: Options data provider instance
            config: Configuration dict
        """
        super().__init__(config)
        self.market = market_provider
        self.options = options_provider

    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch market data and enrich context.

        Args:
            context: Shared context (expects 'tickers' key)

        Returns:
            Context enriched with 'ohlc', 'chains', 'spot_prices'
        """
        if not self.validate_context(context, ["tickers"]):
            return context

        tickers = context["tickers"]
        self.log(f"Scouting market data for {len(tickers)} tickers")

        # Fetch OHLCV data
        ohlc_data = self.market.get_ohlcv(tickers, lookback_days=60)
        self.log(f"Fetched OHLCV for {len(ohlc_data)} tickers")

        # Fetch options chains
        chains = {}
        spot_prices = {}
        for ticker in tickers:
            try:
                chain = self.options.get_options_chain(
                    ticker,
                    min_dte=self.config.get("min_dte", 20),
                    max_dte=self.config.get("max_dte", 50),
                )
                chains[ticker] = chain
                spot_prices[ticker] = chain.spot_price
                self.log(f"Fetched {len(chain.contracts)} contracts for {ticker}")
            except Exception as e:
                self.log(f"Error fetching chain for {ticker}: {e}", level="warning")
                continue

        # Calculate basic metrics
        metrics = {}
        for ticker, df in ohlc_data.items():
            if df.empty:
                continue

            try:
                metrics[ticker] = {
                    "atr_14": self._calculate_atr(df, window=14),
                    "avg_volume": float(df["volume"].tail(20).mean()),
                    "price_change_1d": float((df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100),
                    "price_change_5d": float((df["close"].iloc[-1] / df["close"].iloc[-6] - 1) * 100),
                    "volatility_20d": float(df["close"].pct_change().tail(20).std() * np.sqrt(252) * 100),
                }
            except Exception as e:
                self.log(f"Error calculating metrics for {ticker}: {e}", level="warning")
                continue

        # Update context
        self.add_to_context(context, "ohlc", ohlc_data)
        self.add_to_context(context, "chains", chains)
        self.add_to_context(context, "spot_prices", spot_prices)
        self.add_to_context(context, "market_metrics", metrics)

        self.log(f"Scout complete: {len(chains)} chains, {len(metrics)} metrics")
        return context

    def _calculate_atr(self, df, window=14):
        """Calculate Average True Range."""
        high = df["high"]
        low = df["low"]
        close = df["close"]

        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=window).mean()

        return float(atr.iloc[-1]) if not atr.empty else 0.0


# Required import
import pandas as pd
