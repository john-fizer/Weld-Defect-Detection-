"""Base agent interface for the multi-agent trading system."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime
import logging


class Agent(ABC):
    """Base class for all trading agents.

    Each agent has a specific role in the trading system and processes
    context from previous agents, adds its own analysis, and passes
    the enriched context forward.
    """

    name: str = "agent"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize agent with optional configuration.

        Args:
            config: Agent-specific configuration dict
        """
        self.config = config or {}
        self.logger = logging.getLogger(f"agents.{self.name}")
        self.logger.setLevel(logging.INFO)

    @abstractmethod
    def run(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process context and return enriched context.

        Args:
            context: Shared context dict containing data from previous agents

        Returns:
            Enriched context dict with this agent's additions
        """
        raise NotImplementedError

    def log(self, message: str, level: str = "info"):
        """Log a message with the agent's name.

        Args:
            message: Message to log
            level: Log level (info, warning, error, debug)
        """
        log_fn = getattr(self.logger, level.lower(), self.logger.info)
        log_fn(f"[{self.name}] {message}")

    def add_to_context(self, context: Dict[str, Any], key: str, value: Any) -> Dict[str, Any]:
        """Safely add data to context.

        Args:
            context: Context dict
            key: Key to add
            value: Value to add

        Returns:
            Updated context
        """
        context[key] = value
        context.setdefault("_agent_trace", []).append({
            "agent": self.name,
            "timestamp": datetime.utcnow().isoformat(),
            "key": key,
        })
        return context

    def get_from_context(self, context: Dict[str, Any], key: str, default: Any = None) -> Any:
        """Safely get data from context.

        Args:
            context: Context dict
            key: Key to retrieve
            default: Default value if key not found

        Returns:
            Value from context or default
        """
        return context.get(key, default)

    def validate_context(self, context: Dict[str, Any], required_keys: list) -> bool:
        """Validate that required keys exist in context.

        Args:
            context: Context dict
            required_keys: List of required keys

        Returns:
            True if all required keys present, False otherwise
        """
        missing = [k for k in required_keys if k not in context]
        if missing:
            self.log(f"Missing required context keys: {missing}", level="error")
            return False
        return True

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}')>"
