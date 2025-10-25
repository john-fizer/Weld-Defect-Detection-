# 🤖 Self-Organizing AI Trading Team (Multi-Agent System)

A sophisticated multi-agent AI system for automated options trading, featuring specialized agents for market analysis, strategy planning, risk management, and execution. Built with LangGraph, this system demonstrates advanced AI coordination, rigorous risk controls, and professional software engineering practices.

[![Tests](https://github.com/yourusername/multi-agent-trading/actions/workflows/lint-tests.yaml/badge.svg)](https://github.com/yourusername/multi-agent-trading/actions)
[![Nightly Backtests](https://github.com/yourusername/multi-agent-trading/actions/workflows/nightly-backtest.yaml/badge.svg)](https://github.com/yourusername/multi-agent-trading/actions)

## 🎯 What This Shows

This project demonstrates:

- **Multi-Agent Coordination**: 7 specialized AI agents working together via LangGraph
- **Options Trading Strategies**: Iron Condor, Long Straddle, Wheel (cash-secured puts)
- **Risk Management**: Multi-layered risk controls, VaR calculations, kill switches
- **Production-Ready Architecture**: Clean code, tests, CI/CD, monitoring dashboard
- **Reasoning Traces**: Full audit trail of agent decisions
- **Paper Trading First**: Safe development with paper broker, strict gates for live trading

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      MULTI-AGENT WORKFLOW                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐      ┌────────────────┐      ┌─────────────┐ │
│  │ Market Scout │─────>│Signal Engineer │─────>│Strategy     │ │
│  │              │      │                │      │Planner      │ │
│  │ - OHLCV Data │      │ - IV Rank      │      │             │ │
│  │ - Options    │      │ - Priced Move  │      │ - Match     │ │
│  │ - Chains     │      │ - Put/Call     │      │   Regime    │ │
│  │ - Events     │      │ - Skew         │      │ - Generate  │ │
│  │              │      │ - Trend        │      │   Plans     │ │
│  └──────────────┘      └────────────────┘      └──────┬──────┘ │
│                                                        │        │
│                                                        v        │
│  ┌──────────────┐      ┌────────────────┐      ┌─────────────┐ │
│  │Performance   │<─────│  Executioner   │<─────│Risk Officer │ │
│  │Analyst       │      │                │      │             │ │
│  │              │      │ - Route Orders │      │ - Validate  │ │
│  │ - Track P&L  │      │ - Smart Fill   │      │ - Enforce   │ │
│  │ - Attribution│      │ - Manage Legs  │      │   Limits    │ │
│  │ - Learn      │      │                │      │ - Kill      │ │
│  │              │      │                │      │   Switch    │ │
│  └──────────────┘      └────────────────┘      └─────────────┘ │
│                                                                  │
│                     ┌──────────────────┐                        │
│                     │   Coordinator    │                        │
│                     │  (LangGraph)     │                        │
│                     │                  │                        │
│                     │ - Orchestrate    │                        │
│                     │ - Schedule       │                        │
│                     │ - Memory         │                        │
│                     └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                          DATA LAYER                              │
├─────────────────────────────────────────────────────────────────┤
│  Market Data     │  Options Data    │  Flow Data   │  Storage   │
│  - YFinance      │  - Tradier       │  - Quiver    │  - SQLite  │
│  - Polygon       │  - Polygon       │  - UW        │  - Postgres│
│  - Alpha Vantage │  - IBKR          │  - Cheddar   │  - Chroma  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                      EXECUTION LAYER                             │
├─────────────────────────────────────────────────────────────────┤
│  Paper Broker (Default)         │  IBKR Adapter (Live - Gated)  │
│  - Simulated fills              │  - Real orders                │
│  - No real money                │  - Risk flags required        │
│  - Perfect for development      │  - Human approval             │
└─────────────────────────────────────────────────────────────────┘
```

## 🎓 Strategies Implemented

### 1. Iron Condor
- **Type**: Premium selling, neutral strategy
- **Structure**: Short OTM call spread + Short OTM put spread
- **Profit**: Theta decay in range-bound markets
- **Entry**: IV Rank > 40, range-bound trend, 30-45 DTE
- **Exit**: 50% profit target, 200% stop loss, manage at 21 DTE

### 2. Long Straddle
- **Type**: Volatility play, directional agnostic
- **Structure**: Buy ATM call + Buy ATM put
- **Profit**: Large moves in either direction
- **Entry**: Before events (earnings, FOMC), priced move < historical
- **Exit**: 30% profit, 50% stop, or post-event

### 3. Wheel (Cash-Secured Puts)
- **Type**: Income generation, assignment-tolerant
- **Structure**: Sell cash-secured puts, if assigned sell covered calls
- **Profit**: Premium collection, potential share appreciation
- **Entry**: Quality underlyings, ~25 delta puts, 30 DTE
- **Exit**: 50% profit, roll at 7 DTE, accept assignment

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Poetry (for dependency management)
- API Keys (optional but recommended):
  - Tradier (options data)
  - OpenAI or Anthropic (for LLM features)

### Installation

```bash
# Install dependencies
make dev

# Copy environment template
cp .env.example .env

# Edit .env with your API keys (optional for basic features)
nano .env

# Initialize database
python main.py --init-db
```

### Run the System

```bash
# Run with Coordinator (recommended for start)
python main.py --schedule open

# Run with LangGraph workflow
python main.py --use-langgraph --schedule open

# Launch dashboard
make dashboard
# Open browser to http://localhost:8501
```

### Run Backtests

```bash
# Run all backtests
make backtest

# Or run specific notebook
jupyter notebook notebooks/backtest_iron_condor.ipynb
```

### Run Tests

```bash
# Run test suite
make test
```

## 📊 Dashboard

The Streamlit dashboard provides real-time monitoring:

- **Overview**: Account value, P&L, open positions, recent trades
- **Positions**: Position details, Greeks, strategy breakdown
- **Planner**: Proposed trades, risk checks, approval workflow
- **Analytics**: Performance metrics, strategy attribution, equity curve
- **Settings**: Risk limits, environment configuration

## 🛡️ Risk Management

### Multi-Layered Controls

1. **Per-Trade Limits**
   - Max loss: 1% of account
   - Min open interest: 500
   - Max bid-ask spread: 60 bps
   - DTE bounds: 20-50 days

2. **Portfolio Limits**
   - Max VaR (95%): 3%
   - Max drawdown alert: 8%
   - Max margin usage: 30%
   - Max ticker concentration: 15%

3. **Strategy Limits**
   - Max positions per strategy
   - Max per-ticker positions
   - Event proximity filters

4. **Kill Switch**
   - Daily loss > 5% → halt trading
   - 5 consecutive losses → review required
   - VaR breach 1.5x → stop new entries

5. **Human Approval Gates**
   - Live trading requires explicit approval
   - New strategies require review
   - Flagged trades escalate to human
   - Risk limit breaches require override

## 📁 Project Structure

```
multi-agent-trading/
├── agents/                  # Agent implementations
├── data_providers/          # Data source adapters
├── strategies/              # Trading strategies
├── risk/                    # Risk management
├── exec/                    # Execution layer
├── storage/                 # Data persistence
├── graphs/                  # LangGraph workflows
├── dashboards/              # Monitoring & visualization
├── notebooks/               # Backtesting & research
├── tests/                   # Test suite
├── config/                  # Configuration files
├── main.py                  # Main entry point
└── pyproject.toml          # Dependencies
```

## 🎯 Roadmap

### Phase 1: MVP ✅ (Current)
- [x] Agent architecture
- [x] Iron Condor & Wheel strategies
- [x] Paper broker
- [x] Risk management
- [x] Streamlit dashboard
- [x] Basic backtests

### Phase 2: Enhancement (Next)
- [ ] Long Straddle with earnings calendar
- [ ] Sentiment/flow integration
- [ ] Improved VaR-based position sizing
- [ ] Roll logic for options

### Phase 3: Advanced (Future)
- [ ] IBKR live integration (with extreme care)
- [ ] RL-based strategy weight optimization
- [ ] Bayesian event-move prediction

## ⚠️ Disclaimers & Safety

**CRITICAL WARNINGS:**

1. **Paper Trading by Default**: This system uses paper trading (simulated) by default. No real money at risk unless explicitly enabled.

2. **Live Trading Requires Extreme Caution**: Live trading is gated behind multiple warnings. Even then, start with TINY position sizes.

3. **Not Financial Advice**: This is an educational/portfolio project. Use at your own risk.

4. **Options Are Risky**: Options can result in total loss of capital.

5. **No Guarantees**: The system may have bugs or encounter edge cases.

## 📄 License

MIT License - see LICENSE file for details.

---

**Built with ❤️ for demonstrating multi-agent AI systems and production-grade software engineering.**

**⭐ Star this repo if you find it useful!**
