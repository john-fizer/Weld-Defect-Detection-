"""Tradier options data provider."""

from typing import Dict, List, Optional
from datetime import date, datetime, timedelta
import os
import requests
from .base import OptionsDataProvider, OptionContract, OptionsChain


class TradierOptionsProvider(OptionsDataProvider):
    """Options data provider using Tradier API."""

    BASE_URL = "https://api.tradier.com/v1"

    def __init__(self, api_key: Optional[str] = None):
        """Initialize Tradier provider.

        Args:
            api_key: Tradier API key (reads from env if not provided)
        """
        self.api_key = api_key or os.getenv("TRADIER_API_KEY")
        if not self.api_key:
            raise ValueError("TRADIER_API_KEY not found in environment or constructor")

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make API request to Tradier.

        Args:
            endpoint: API endpoint
            params: Query parameters

        Returns:
            Response JSON
        """
        url = f"{self.BASE_URL}/{endpoint}"
        response = requests.get(url, headers=self.headers, params=params or {})
        response.raise_for_status()
        return response.json()

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
        # Get expirations
        exp_data = self._make_request(
            f"markets/options/expirations",
            params={"symbol": ticker, "includeAllRoots": "false"}
        )

        expirations = exp_data.get("expirations", {}).get("date", [])
        if not expirations:
            return OptionsChain(ticker, 0.0, [])

        # Filter by DTE
        today = date.today()
        valid_exps = []
        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            dte = (exp_date - today).days
            if min_dte <= dte <= max_dte:
                valid_exps.append(exp_str)

        if not valid_exps:
            return OptionsChain(ticker, 0.0, [])

        # Get spot price
        quote_data = self._make_request(
            "markets/quotes",
            params={"symbols": ticker}
        )
        quotes = quote_data.get("quotes", {}).get("quote", {})
        spot_price = float(quotes.get("last", 0.0))

        # Get chains for valid expirations
        contracts = []
        for exp in valid_exps:
            chain_data = self._make_request(
                "markets/options/chains",
                params={"symbol": ticker, "expiration": exp, "greeks": "true"}
            )

            options = chain_data.get("options", {}).get("option", [])
            if not options:
                continue

            # Handle single option case
            if isinstance(options, dict):
                options = [options]

            exp_date = datetime.strptime(exp, "%Y-%m-%d").date()

            for opt in options:
                try:
                    contract = OptionContract(
                        symbol=opt["symbol"],
                        underlying=ticker,
                        expiration=exp_date,
                        strike=float(opt["strike"]),
                        option_type=opt["option_type"].lower(),
                        bid=float(opt.get("bid", 0.0)),
                        ask=float(opt.get("ask", 0.0)),
                        last=float(opt.get("last", 0.0)),
                        volume=int(opt.get("volume", 0)),
                        open_interest=int(opt.get("open_interest", 0)),
                        implied_volatility=float(opt.get("greeks", {}).get("smv_vol", 0.0)),
                        delta=float(opt.get("greeks", {}).get("delta", 0.0)) if opt.get("greeks") else None,
                        gamma=float(opt.get("greeks", {}).get("gamma", 0.0)) if opt.get("greeks") else None,
                        theta=float(opt.get("greeks", {}).get("theta", 0.0)) if opt.get("greeks") else None,
                        vega=float(opt.get("greeks", {}).get("vega", 0.0)) if opt.get("greeks") else None,
                    )
                    contracts.append(contract)
                except (KeyError, ValueError, TypeError) as e:
                    # Skip malformed contracts
                    continue

        return OptionsChain(ticker, spot_price, contracts)

    def get_greeks(self, option_symbol: str) -> Dict[str, float]:
        """Get option greeks.

        Args:
            option_symbol: Option symbol

        Returns:
            Dict with delta, gamma, theta, vega, rho
        """
        quote_data = self._make_request(
            "markets/quotes",
            params={"symbols": option_symbol, "greeks": "true"}
        )

        quote = quote_data.get("quotes", {}).get("quote", {})
        greeks = quote.get("greeks", {})

        return {
            "delta": float(greeks.get("delta", 0.0)),
            "gamma": float(greeks.get("gamma", 0.0)),
            "theta": float(greeks.get("theta", 0.0)),
            "vega": float(greeks.get("vega", 0.0)),
            "rho": float(greeks.get("rho", 0.0)),
        }
