"""Interactive Brokers adapter (stub for live trading)."""

from typing import Dict, Any, Optional
import os


class IBKRAdapter:
    """Adapter for Interactive Brokers TWS/Gateway.

    IMPORTANT: This is a STUB implementation for portfolio purposes.
    Real IBKR integration requires:
    - ib_insync library
    - TWS or IB Gateway running
    - Proper authentication and risk controls
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 7497,
        client_id: int = 1,
        account: Optional[str] = None,
    ):
        """Initialize IBKR adapter.

        Args:
            host: IB Gateway host
            port: IB Gateway port (7497 paper, 7496 live)
            client_id: Client ID
            account: Account number
        """
        self.host = host or os.getenv("IBKR_HOST", "127.0.0.1")
        self.port = port or int(os.getenv("IBKR_PORT", "7497"))
        self.client_id = client_id or int(os.getenv("IBKR_CLIENT_ID", "1"))
        self.account = account or os.getenv("IBKR_ACCOUNT")

        self.connected = False

        print("⚠️  IBKR Adapter is a STUB - not actually connecting to IB")
        print("⚠️  For live trading, implement full ib_insync integration")

    def connect(self) -> bool:
        """Connect to IB Gateway.

        Returns:
            True if connected
        """
        # Stub: would use ib_insync to connect
        print(f"[STUB] Would connect to IBKR at {self.host}:{self.port}")
        self.connected = False
        return False

    def disconnect(self):
        """Disconnect from IB Gateway."""
        print("[STUB] Would disconnect from IBKR")
        self.connected = False

    def place_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Place order (STUB - NEVER ACTUALLY EXECUTES).

        Args:
            order: Order dict

        Returns:
            Error dict indicating stub
        """
        return {
            "status": "error",
            "reason": "IBKR adapter is stub-only. Implement ib_insync for live trading.",
            "order": order,
        }

    def get_positions(self) -> list:
        """Get positions (STUB).

        Returns:
            Empty list
        """
        return []

    def get_account_info(self) -> Dict[str, Any]:
        """Get account info (STUB).

        Returns:
            Stub account info
        """
        return {
            "account_type": "ibkr_stub",
            "error": "Not implemented - stub only",
        }
