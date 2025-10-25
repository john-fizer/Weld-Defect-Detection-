# SAIRA - Self-Organizing AI Research & Applications

A collection of advanced AI systems demonstrating multi-agent coordination, specialized domain expertise, and production-grade software engineering. These projects will be combined into a unified SAIRA platform.

---

## 🌟 Projects

### 1. 🤖 Multi-Agent AI Trading System

A sophisticated multi-agent system for automated options trading, featuring specialized agents for market analysis, strategy planning, risk management, and execution.

**Status**: ✅ MVP Complete

**Key Features:**
- 7 specialized agents coordinated via LangGraph
- Options strategies: Iron Condor, Long Straddle, Wheel
- Multi-layer risk management with kill switches
- Paper trading (default) + live trading adapters
- Streamlit monitoring dashboard
- Comprehensive test suite + CI/CD

**Quick Start:**
```bash
make dev
python main.py --init-db
python main.py --schedule open
make dashboard  # Streamlit UI
```

[📖 Detailed Trading System Documentation](#multi-agent-trading-system)

---

### 2. 🔮 Holographic Astrology Platform

A next-generation astrology interpretation system that synthesizes multiple chart systems, predictive techniques, and machine learning feedback loops for precision astrological analysis.

**Status**: 🚧 In Development

**Key Features:**
- Multi-system chart synthesis (Western + Vedic)
- LLM-powered unified interpretation
- Advanced timing predictions (Progressions, Transits, ZR, LB)
- Event correlation engine with ML feedback loops
- Modular feedback per prediction technique

**Architecture:**
- FastAPI backend with Swiss Ephemeris
- PostgreSQL + ML training pipeline
- Holographic overlay visualization (planned)

[📖 Detailed Astrology Platform Documentation](#holographic-astrology-platform)

---

## 🎯 SAIRA Vision

These projects demonstrate complementary AI capabilities that will combine into SAIRA:

- **Multi-Agent Coordination**: Trading system's 7-agent architecture
- **Specialized Domain Expertise**: Options trading + astrological analysis
- **Machine Learning Integration**: Feedback loops, continuous learning
- **Production Engineering**: Tests, CI/CD, monitoring, databases
- **LLM Integration**: Natural language synthesis and reasoning

**Future Integration:**
- Unified agent framework across domains
- Shared memory and learning systems
- Cross-domain insights and patterns
- Single dashboard for all systems

---

## 📁 Repository Structure

```
SAIRA/
├── agents/                      # Trading system agents
├── strategies/                  # Trading strategies
├── risk/                        # Risk management
├── storage/                     # Database & portfolio
├── dashboards/                  # Streamlit UI
├── tests/                       # Test suite
├── config/                      # Configuration
├── main.py                      # Trading system entry
├── pyproject.toml              # Dependencies
└── README.md                   # This file
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- Poetry (dependency management)
- PostgreSQL (optional, SQLite works)

### Installation
```bash
# Install dependencies
make dev

# Initialize database
python main.py --init-db

# Run trading system
python main.py --schedule open

# Launch dashboard
make dashboard
```

### Run Tests
```bash
make test
make lint
```

---

# Multi-Agent Trading System

## System Architecture

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
│                     └──────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

## Agents

### 1. Market Scout
Fetches and normalizes market data.
- OHLCV data from multiple providers
- Options chains with Greeks
- Market metrics (ATR, volume, volatility)
- Upcoming events detection

### 2. Signal Engineer
Computes features and trading signals.
- IV Rank calculation
- Priced moves from straddles
- Put/call ratios and skew
- Trend detection (SMA crossovers)
- Liquidity scoring

### 3. Strategy Planner
Matches market regime to strategies.
- Analyzes features and conditions
- Generates trade proposals
- Prioritizes opportunities
- Provides reasoning for each plan

### 4. Risk Officer
Validates and enforces risk constraints.
- Per-trade limit validation
- Portfolio-level risk checks
- Kill switch monitoring
- Human approval escalation

### 5. Executioner
Routes orders to broker.
- Multi-leg order construction
- Smart order routing
- Slippage simulation
- Fill tracking

### 6. Performance Analyst
Tracks and analyzes performance.
- Trade logging to database
- P&L calculation
- Strategy attribution
- Performance metrics (Sharpe, win rate)

### 7. Coordinator
Orchestrates the workflow.
- Agent scheduling
- Context management
- LangGraph integration
- Summary generation

## Trading Strategies

### Iron Condor
**Type**: Premium selling, neutral
**Entry**: IV Rank > 40, range-bound, 30-45 DTE
**Exit**: 50% profit / 200% stop loss

### Long Straddle
**Type**: Volatility play
**Entry**: Before events, priced move < historical
**Exit**: 30% profit / 50% stop loss

### Wheel
**Type**: Income generation
**Entry**: Quality tickers, 25 delta puts, 30 DTE
**Exit**: 50% profit, roll at 7 DTE

## Risk Management

### Per-Trade Limits
- Max loss: 1% of account
- Min open interest: 500
- Max bid-ask: 60 bps
- DTE: 20-50 days

### Portfolio Limits
- Max VaR (95%): 3%
- Max drawdown: 8%
- Max margin: 30%
- Max ticker weight: 15%

### Kill Switch
- Daily loss > 5%
- 5 consecutive losses
- VaR breach 1.5x

## Tech Stack
- **Coordination**: LangGraph
- **Data**: YFinance, Tradier, QuiverQuant
- **Execution**: Paper Broker (default), IBKR (stub)
- **Storage**: SQLAlchemy, ChromaDB
- **Dashboard**: Streamlit
- **Testing**: Pytest, GitHub Actions

---

# Holographic Astrology Platform

## Core Philosophy: Holographic Synthesis

Different house systems and astrological techniques provide complementary perspectives that clarify and reinforce each other. This platform layers:

- **Western Systems**: Whole Sign, Placidus
- **Vedic Astrology**: Nakshatras, Sidereal positions
- **Refinement Techniques**: Decans, Degree Theory
- **Predictive Methods**: Progressions, Transits, Zodiacal Releasing (ZR), Loosening of Bonds (LB)

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Frontend Layer                            │
│  • Multi-chart visualization (holographic overlay)           │
│  • Event timeline & prediction dashboard                     │
│  • Feedback interface (modular by technique)                 │
└─────────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                       │
│  ┌────────────┐ ┌──────────┐ ┌──────────────────────────┐  │
│  │ Chart      │ │ LLM      │ │ Prediction Engine        │  │
│  │ Calculator │ │ Narrator │ │ (Progressions/Transits/  │  │
│  │            │ │          │ │  ZR/LB modules)          │  │
│  └────────────┘ └──────────┘ └──────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                           ↕
┌─────────────────────────────────────────────────────────────┐
│           Database & ML Training Pipeline                    │
│  • PostgreSQL: Charts, Events, Feedback (per technique)     │
│  • Training datasets: Isolated per prediction method         │
│  • Model registry: Version control for ML models            │
└─────────────────────────────────────────────────────────────┘
```

## Modular Feedback Loops

Each predictive technique has isolated feedback collection:
- **Progressions Feedback**: User validates progression-based predictions
- **Transit Feedback**: User validates transit timing accuracy
- **ZR Feedback**: Zodiacal Releasing event correlation
- **LB Feedback**: Loosening of Bonds accuracy metrics
- **House System Feedback**: Comparative resonance ratings

## Key Features

### 1. Multi-System Chart Synthesis
- Calculate charts in multiple house systems simultaneously
- Vedic Nakshatra integration with planetary rulers
- Decan and degree theory overlays

### 2. LLM-Powered Unified Interpretation
- AI synthesizes insights across all systems
- Natural language interpretation of complex configurations
- Holistic narrative generation

### 3. Advanced Timing Predictions
- **Secondary Progressions**: Aspects to natal chart
- **Transits**: Real-time planetary positions vs natal
- **Zodiacal Releasing**: Peak/loosening periods with event flags
- **Loosening of Bonds**: Refinement of ZR timing

### 4. Event Correlation Engine
Event categories tracked:
- Career changes (new job, promotion, termination)
- Relationships (start, marriage, breakup, divorce)
- Family events (birth, death, health crisis)
- Financial shifts (windfall, loss, investment)
- Relocations and major life transitions

### 5. Closed-Loop Machine Learning
- User feedback on prediction accuracy
- Continuous model retraining
- A/B testing of interpretation strategies
- Performance metrics per technique

## Tech Stack
- **Backend**: Python 3.11+, FastAPI
- **Astrology Engine**: Swiss Ephemeris (pyswisseph)
- **Database**: PostgreSQL + SQLAlchemy
- **ML/AI**: Transformers, PyTorch, OpenAI/Anthropic APIs
- **Frontend**: React/Next.js (TBD)

## Development Roadmap
- [x] Project initialization
- [ ] Core chart calculation engine
- [ ] Multi-house system implementation
- [ ] Vedic/Nakshatra integration
- [ ] Secondary progressions engine
- [ ] Transit prediction engine
- [ ] Zodiacal Releasing implementation
- [ ] LLM integration & prompt engineering
- [ ] Feedback collection system
- [ ] ML training pipeline

---

## ⚠️ Disclaimers

### Trading System
- **Paper trading by default** - no real money at risk
- **Not financial advice** - educational/portfolio project
- **Options are risky** - can result in total loss
- Use extreme caution if enabling live trading

### Astrology Platform
- **For entertainment and research purposes**
- **Not a substitute for professional advice**
- Astrological predictions are interpretive

---

## 📄 License

MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- LangGraph for agent coordination
- Swiss Ephemeris for astronomical calculations
- Tradier & YFinance for market data
- Open source community

---

**SAIRA - Demonstrating the future of specialized AI systems**

**⭐ Star this repo if you find it useful!**
