"""Base interfaces for data providers."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime, date
import pandas as pd
from dataclasses import dataclass


@dataclass
class OptionContract:
    """Represents a single option contract."""

    symbol: str
    underlying: str
    expiration: date
    strike: float
    option_type: str  # 'call' or 'put'
    bid: float
    ask: float
    last: float
    volume: int
    open_interest: int
    implied_volatility: float
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None

    @property
    def mid_price(self) -> float:
        """Calculate mid price from bid/ask."""
        return (self.bid + self.ask) / 2.0

    @property
    def bid_ask_spread_pct(self) -> float:
        """Calculate bid-ask spread as percentage of mid."""
        mid = self.mid_price
        if mid == 0:
            return float('inf')
        return ((self.ask - self.bid) / mid) * 100

    @property
    def days_to_expiry(self) -> int:
        """Calculate days to expiration."""
        return (self.expiration - date.today()).days


@dataclass
class OptionsChain:
    """Represents an options chain for an underlying."""

    underlying: str
    spot_price: float
    contracts: List[OptionContract]

    def get_calls(self, expiration: Optional[date] = None) -> List[OptionContract]:
        """Get call options, optionally filtered by expiration."""
        calls = [c for c in self.contracts if c.option_type == 'call']
        if expiration:
            calls = [c for c in calls if c.expiration == expiration]
        return sorted(calls, key=lambda x: x.strike)

    def get_puts(self, expiration: Optional[date] = None) -> List[OptionContract]:
        """Get put options, optionally filtered by expiration."""
        puts = [c for c in self.contracts if c.option_type == 'put']
        if expiration:
            puts = [c for c in puts if c.expiration == expiration]
        return sorted(puts, key=lambda x: x.strike)

    def get_expirations(self) -> List[date]:
        """Get unique expiration dates."""
        return sorted(list(set(c.expiration for c in self.contracts)))

    def get_atm_straddle(self, expiration: date) -> Optional[tuple]:
        """Get ATM straddle (call + put at nearest strike to spot).

        Returns:
            Tuple of (call_contract, put_contract) or None
        """
        calls = self.get_calls(expiration)
        puts = self.get_puts(expiration)

        if not calls or not puts:
            return None

        # Find strike closest to spot
        atm_strike = min(calls, key=lambda x: abs(x.strike - self.spot_price)).strike

        atm_call = next((c for c in calls if c.strike == atm_strike), None)
        atm_put = next((p for p in puts if p.strike == atm_strike), None)

        if atm_call and atm_put:
            return (atm_call, atm_put)
        return None

    def priced_move_pct(self, expiration: date) -> Optional[float]:
        """Calculate priced move percentage from ATM straddle.

        Returns:
            Priced move as percentage of spot price
        """
        straddle = self.get_atm_straddle(expiration)
        if not straddle:
            return None

        call, put = straddle
        straddle_price = call.mid_price + put.mid_price
        return (straddle_price / self.spot_price) * 100


class MarketDataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    def get_ohlcv(
        self,
        tickers: List[str],
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        lookback_days: int = 60,
    ) -> Dict[str, pd.DataFrame]:
        """Get OHLCV data for tickers.

        Args:
            tickers: List of ticker symbols
            start_date: Start date (optional)
            end_date: End date (optional)
            lookback_days: Days to look back if dates not specified

        Returns:
            Dict mapping ticker -> DataFrame with columns [open, high, low, close, volume]
        """
        pass

    @abstractmethod
    def get_latest_price(self, ticker: str) -> float:
        """Get latest price for ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Latest price
        """
        pass


class OptionsDataProvider(ABC):
    """Abstract base class for options data providers."""

    @abstractmethod
    def get_options_chain(
        self,
        ticker: str,
        min_dte: int = 0,
        max_dte: int = 60,
    ) -> OptionsChain:
        """Get options chain for ticker.

        Args:
            ticker: Underlying ticker symbol
            min_dte: Minimum days to expiration
            max_dte: Maximum days to expiration

        Returns:
            OptionsChain object
        """
        pass

    @abstractmethod
    def get_greeks(self, option_symbol: str) -> Dict[str, float]:
        """Get option greeks.

        Args:
            option_symbol: Option symbol

        Returns:
            Dict with delta, gamma, theta, vega, rho
        """
        pass


class FlowDataProvider(ABC):
    """Abstract base class for flow/unusual options activity providers."""

    @abstractmethod
    def get_unusual_activity(
        self,
        ticker: Optional[str] = None,
        min_premium: float = 100000,
        date: Optional[date] = None,
    ) -> List[Dict[str, Any]]:
        """Get unusual options activity.

        Args:
            ticker: Filter by ticker (optional)
            min_premium: Minimum premium threshold
            date: Date to query (defaults to today)

        Returns:
            List of activity dicts
        """
        pass

    @abstractmethod
    def get_dark_pool_flow(
        self,
        ticker: str,
        lookback_days: int = 5,
    ) -> pd.DataFrame:
        """Get dark pool / block trade flow.

        Args:
            ticker: Ticker symbol
            lookback_days: Days to look back

        Returns:
            DataFrame with flow data
        """
        pass


class SentimentProvider(ABC):
    """Abstract base class for sentiment/news providers."""

    @abstractmethod
    def get_news_sentiment(
        self,
        ticker: str,
        lookback_hours: int = 24,
    ) -> Dict[str, Any]:
        """Get aggregated news sentiment for ticker.

        Args:
            ticker: Ticker symbol
            lookback_hours: Hours to look back

        Returns:
            Dict with sentiment score and summary
        """
        pass

    @abstractmethod
    def get_social_sentiment(
        self,
        ticker: str,
        source: str = "twitter",
    ) -> Dict[str, Any]:
        """Get social media sentiment.

        Args:
            ticker: Ticker symbol
            source: Social platform (twitter, reddit, etc)

        Returns:
            Dict with sentiment metrics
        """
        pass
