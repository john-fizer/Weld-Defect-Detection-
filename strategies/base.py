"""Base strategy interface."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import date


class Strategy(ABC):
    """Base class for all trading strategies."""

    name: str = "strategy"

    def __init__(self, config: Dict[str, Any]):
        """Initialize strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config

    @abstractmethod
    def evaluate(self, ticker: str, chain, features: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate if strategy conditions are met.

        Args:
            ticker: Ticker symbol
            chain: Options chain
            features: Computed features

        Returns:
            Trade setup dict if conditions met, None otherwise
        """
        pass

    @abstractmethod
    def calculate_position_size(self, account_value: float, max_risk: float) -> int:
        """Calculate position size.

        Args:
            account_value: Account value
            max_risk: Maximum risk per trade

        Returns:
            Number of contracts
        """
        pass

    @abstractmethod
    def get_exit_conditions(self, position: Dict[str, Any]) -> Dict[str, Any]:
        """Get exit conditions for position.

        Args:
            position: Position dict

        Returns:
            Exit conditions (take profit, stop loss, etc)
        """
        pass
