"""LangGraph-based trading coordination graph."""

from typing import Dict, Any
from langgraph.graph import StateGraph, END
from agents.base import Agent


def create_trading_graph(agents: Dict[str, Agent]) -> StateGraph:
    """Create LangGraph workflow for trading system.

    Workflow:
        MarketScout -> SignalEngineer -> StrategyPlanner -> RiskOfficer
        -> [if approved] -> Executioner -> PerformanceAnalyst
        -> [if escalated] -> HumanReview

    Args:
        agents: Dict mapping agent name to agent instance

    Returns:
        Compiled StateGraph
    """
    # Define state schema
    def state_reducer(left: Dict, right: Dict) -> Dict:
        """Merge state updates."""
        return {**left, **right}

    # Create graph
    workflow = StateGraph(state_schema=Dict[str, Any])

    # Add agent nodes
    workflow.add_node("market_scout", agents["market_scout"].run)
    workflow.add_node("signal_engineer", agents["signal_engineer"].run)
    workflow.add_node("strategy_planner", agents["strategy_planner"].run)
    workflow.add_node("risk_officer", agents["risk_officer"].run)
    workflow.add_node("executioner", agents["executioner"].run)
    workflow.add_node("performance_analyst", agents["performance_analyst"].run)

    # Define edges
    workflow.add_edge("market_scout", "signal_engineer")
    workflow.add_edge("signal_engineer", "strategy_planner")
    workflow.add_edge("strategy_planner", "risk_officer")

    # Conditional edge from risk_officer
    def should_execute(state: Dict[str, Any]) -> str:
        """Decide whether to execute or escalate."""
        if state.get("kill_switch_active"):
            return "end"

        approved = state.get("approved_plans", [])
        flagged = state.get("flagged_plans", [])

        if flagged:
            # Has flagged plans - would escalate to human review
            # For now, skip execution
            return "performance_analyst"

        if approved:
            return "executioner"

        # No approved plans
        return "performance_analyst"

    workflow.add_conditional_edges(
        "risk_officer",
        should_execute,
        {
            "executioner": "executioner",
            "performance_analyst": "performance_analyst",
            "end": END,
        }
    )

    workflow.add_edge("executioner", "performance_analyst")
    workflow.add_edge("performance_analyst", END)

    # Set entry point
    workflow.set_entry_point("market_scout")

    # Compile
    return workflow.compile()


def run_trading_workflow(
    graph: StateGraph,
    initial_state: Dict[str, Any],
) -> Dict[str, Any]:
    """Run the trading workflow.

    Args:
        graph: Compiled StateGraph
        initial_state: Initial state dict with 'tickers', etc.

    Returns:
        Final state after workflow completes
    """
    # Run graph
    final_state = graph.invoke(initial_state)

    return final_state
