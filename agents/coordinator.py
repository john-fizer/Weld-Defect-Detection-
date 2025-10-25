"""Coordinator agent - orchestrates the multi-agent workflow."""

from typing import Any, Dict, List
import yaml
from datetime import datetime
from .base import Agent
from .market_scout import MarketScout
from .signal_engineer import SignalEngineer
from .strategy_planner import StrategyPlanner
from .risk_officer import RiskOfficer
from .executioner import Executioner
from .performance_analyst import PerformanceAnalyst


class Coordinator(Agent):
    """Orchestrates the multi-agent trading workflow.

    Responsibilities:
    - Schedule agents based on market hours
    - Maintain shared context/memory
    - Handle escalations and conflicts
    - Coordinate information flow
    - Manage agent lifecycle
    """

    name = "coordinator"

    def __init__(
        self,
        agents: Dict[str, Agent],
        config: Dict[str, Any] = None,
    ):
        """Initialize Coordinator.

        Args:
            agents: Dict mapping agent name to agent instance
            config: Configuration dict
        """
        super().__init__(config)
        self.agents = agents
        self.context = self._initialize_context()

    def _initialize_context(self) -> Dict[str, Any]:
        """Initialize shared context."""
        return {
            "session_id": datetime.utcnow().isoformat(),
            "tickers": self.config.get("tickers", ["SPY", "QQQ"]),
            "_agent_trace": [],
            "kill_switch_active": False,
        }

    def run_workflow(self, schedule: str = "open") -> Dict[str, Any]:
        """Run the full agent workflow.

        Args:
            schedule: Schedule phase (premarket, open, midday, close)

        Returns:
            Final context after all agents have run
        """
        self.log(f"Starting {schedule} workflow")

        # Define agent execution order
        workflows = {
            "premarket": ["market_scout"],
            "open": ["market_scout", "signal_engineer", "strategy_planner", "risk_officer", "executioner"],
            "midday": ["performance_analyst"],
            "close": ["performance_analyst"],
        }

        agent_sequence = workflows.get(schedule, [])

        # Execute agents in sequence
        for agent_name in agent_sequence:
            agent = self.agents.get(agent_name)

            if not agent:
                self.log(f"Agent {agent_name} not found", level="warning")
                continue

            try:
                self.log(f"Running {agent_name}")
                self.context = agent.run(self.context)

                # Check for kill switch
                if self.context.get("kill_switch_active"):
                    self.log("Kill switch activated, halting workflow", level="error")
                    break

            except Exception as e:
                self.log(f"Error running {agent_name}: {e}", level="error")
                # Continue with other agents
                continue

        self.log(f"Workflow {schedule} complete")
        return self.context

    def get_summary(self) -> str:
        """Get summary of last workflow run.

        Returns:
            Human-readable summary
        """
        lines = [
            f"Session: {self.context.get('session_id')}",
            f"Tickers: {', '.join(self.context.get('tickers', []))}",
            "",
            "Results:",
        ]

        approved = self.context.get("approved_plans", [])
        flagged = self.context.get("flagged_plans", [])
        rejected = self.context.get("rejected_plans", [])
        fills = self.context.get("fills", [])
        failed = self.context.get("failed_orders", [])

        lines.append(f"  Plans: {len(approved)} approved, {len(flagged)} flagged, {len(rejected)} rejected")
        lines.append(f"  Execution: {len(fills)} fills, {len(failed)} failures")

        if self.context.get("kill_switch_active"):
            lines.append("  ⚠️  KILL SWITCH ACTIVE")

        report = self.context.get("performance_report", {})
        if report:
            lines.append("")
            lines.append("Performance:")
            lines.append(f"  Total P&L: ${report.get('total_pnl', 0):.2f} ({report.get('total_pnl_pct', 0):.2%})")
            lines.append(f"  Daily P&L: ${report.get('daily_pnl', 0):.2f} ({report.get('daily_pnl_pct', 0):.2%})")
            lines.append(f"  Open Positions: {report.get('open_positions', 0)}")

        return "\n".join(lines)

    def reset(self):
        """Reset context for new session."""
        self.context = self._initialize_context()
        self.log("Context reset")
