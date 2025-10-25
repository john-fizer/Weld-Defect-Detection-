"""YFinance market data provider."""

from typing import List, Dict, Optional
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf
from .base import MarketDataProvider


class YFinanceProvider(MarketDataProvider):
    """Market data provider using yfinance (free, rate-limited)."""

    def __init__(self):
        """Initialize yfinance provider."""
        pass

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
            Dict mapping ticker -> DataFrame with columns [Open, High, Low, Close, Volume]
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=lookback_days)
        if not end_date:
            end_date = datetime.now()

        result = {}
        for ticker in tickers:
            try:
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )
                if not data.empty:
                    # Standardize column names
                    data.columns = [col.lower() for col in data.columns]
                    result[ticker] = data
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                continue

        return result

    def get_latest_price(self, ticker: str) -> float:
        """Get latest price for ticker.

        Args:
            ticker: Ticker symbol

        Returns:
            Latest price
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                return float(data['Close'].iloc[-1])
        except Exception as e:
            print(f"Error fetching latest price for {ticker}: {e}")
            raise

        raise ValueError(f"Could not fetch price for {ticker}")

    def get_historical_volatility(
        self,
        ticker: str,
        window: int = 30,
        annualize: bool = True,
    ) -> float:
        """Calculate historical volatility.

        Args:
            ticker: Ticker symbol
            window: Window in days
            annualize: Whether to annualize the volatility

        Returns:
            Historical volatility
        """
        data = self.get_ohlcv([ticker], lookback_days=window + 10)
        if ticker not in data or data[ticker].empty:
            raise ValueError(f"No data for {ticker}")

        df = data[ticker]
        returns = df['close'].pct_change().dropna()
        vol = returns.std()

        if annualize:
            vol *= (252 ** 0.5)  # Annualize using 252 trading days

        return float(vol)

    def get_info(self, ticker: str) -> Dict:
        """Get ticker info (market cap, sector, etc).

        Args:
            ticker: Ticker symbol

        Returns:
            Dict with ticker information
        """
        try:
            stock = yf.Ticker(ticker)
            return stock.info
        except Exception as e:
            print(f"Error fetching info for {ticker}: {e}")
            return {}
