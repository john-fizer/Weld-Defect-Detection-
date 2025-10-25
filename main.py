"""Main entry point for multi-agent trading system."""

import os
import yaml
import argparse
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import components
from agents import (
    MarketScout, SignalEngineer, StrategyPlanner,
    RiskOfficer, Executioner, PerformanceAnalyst, Coordinator
)
from data_providers import YFinanceProvider, TradierOptionsProvider
from exec import PaperBroker, IBKRAdapter
from storage import init_db, Portfolio
from graphs import create_trading_graph, run_trading_workflow


def load_config(config_dir: str = "config") -> dict:
    """Load configuration files.

    Args:
        config_dir: Configuration directory

    Returns:
        Combined config dict
    """
    config = {}

    # Load settings
    with open(f"{config_dir}/settings.yaml") as f:
        config["settings"] = yaml.safe_load(f)

    # Load risk policies
    with open(f"{config_dir}/risk_policies.yaml") as f:
        config["risk"] = yaml.safe_load(f)

    # Load strategies
    with open(f"{config_dir}/strategies.yaml") as f:
        config["strategies"] = yaml.safe_load(f)

    return config


def create_agents(config: dict, portfolio: Portfolio) -> dict:
    """Create all agents.

    Args:
        config: Configuration dict
        portfolio: Portfolio instance

    Returns:
        Dict of agent instances
    """
    settings = config["settings"]
    risk_config = config["risk"]
    strategies_config = config["strategies"]

    # Initialize data providers
    market_provider = YFinanceProvider()

    # Try to use Tradier if API key available, else fallback
    try:
        options_provider = TradierOptionsProvider()
    except ValueError:
        print("⚠️  Tradier API key not found, options data will be limited")
        options_provider = None

    # Initialize broker
    environment = settings.get("environment", "paper")
    if environment == "live":
        print("⚠️  LIVE TRADING MODE - Use with extreme caution!")
        broker = IBKRAdapter()
    else:
        print("✓ Paper trading mode enabled")
        broker = PaperBroker()

    # Create agents
    agents = {
        "market_scout": MarketScout(
            market_provider=market_provider,
            options_provider=options_provider,
            config=settings,
        ),
        "signal_engineer": SignalEngineer(config=settings),
        "strategy_planner": StrategyPlanner(
            strategies_config=strategies_config,
            config=settings,
        ),
        "risk_officer": RiskOfficer(
            risk_config=risk_config,
            portfolio=portfolio,
            config=settings,
        ),
        "executioner": Executioner(broker=broker, config=settings),
        "performance_analyst": PerformanceAnalyst(config=settings),
    }

    return agents


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Multi-Agent Trading System")
    parser.add_argument(
        "--schedule",
        type=str,
        default="open",
        choices=["premarket", "open", "midday", "close"],
        help="Trading schedule phase"
    )
    parser.add_argument(
        "--init-db",
        action="store_true",
        help="Initialize database tables"
    )
    parser.add_argument(
        "--use-langgraph",
        action="store_true",
        help="Use LangGraph workflow instead of Coordinator"
    )

    args = parser.parse_args()

    # Initialize database if requested
    if args.init_db:
        print("Initializing database...")
        init_db()

    # Load configuration
    print("Loading configuration...")
    config = load_config()

    # Initialize portfolio
    initial_capital = config["settings"].get("backtest", {}).get("initial_capital", 100000)
    portfolio = Portfolio(initial_capital=initial_capital, is_live=False)

    # Create agents
    print("Creating agents...")
    agents = create_agents(config, portfolio)

    # Add portfolio to context
    initial_context = {
        "tickers": config["settings"].get("tickers", ["SPY", "QQQ"]),
        "portfolio": portfolio,
    }

    # Run workflow
    if args.use_langgraph:
        print(f"Running LangGraph workflow ({args.schedule})...")
        graph = create_trading_graph(agents)
        final_state = run_trading_workflow(graph, initial_context)
        print("\nWorkflow complete!")
        print(f"Approved plans: {len(final_state.get('approved_plans', []))}")
        print(f"Fills: {len(final_state.get('fills', []))}")
    else:
        print(f"Running Coordinator workflow ({args.schedule})...")
        coordinator = Coordinator(agents=agents, config=config["settings"])
        coordinator.context = initial_context
        final_state = coordinator.run_workflow(schedule=args.schedule)

        print("\n" + "="*60)
        print(coordinator.get_summary())
        print("="*60)


if __name__ == "__main__":
    main()
