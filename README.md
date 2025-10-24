# Holographic Astrology Platform

A next-generation astrology interpretation system that synthesizes multiple chart systems, predictive techniques, and machine learning feedback loops for precision astrological analysis.

## Core Philosophy: Holographic Synthesis

Different house systems and astrological techniques provide complementary perspectives that clarify and reinforce each other, rather than contradict. This platform layers:

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

This allows independent refinement, validation, or removal of underperforming techniques.

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

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up database
python scripts/init_db.py

# Run development server
uvicorn app.main:app --reload
```

## Project Structure

```
├── app/
│   ├── api/                 # FastAPI routes
│   ├── core/                # Core calculation engines
│   │   ├── chart_calculator.py
│   │   ├── house_systems.py
│   │   ├── vedic.py
│   │   └── aspects.py
│   ├── predictions/         # Modular prediction engines
│   │   ├── progressions.py
│   │   ├── transits.py
│   │   ├── zodiacal_releasing.py
│   │   └── loosening_bonds.py
│   ├── llm/                 # LLM integration
│   │   ├── narrator.py
│   │   └── prompts.py
│   ├── ml/                  # Machine learning pipelines
│   │   ├── feedback_processor.py
│   │   ├── training/
│   │   └── models/
│   ├── models/              # Database models
│   └── schemas/             # Pydantic schemas
├── data/                    # Training data & ephemeris files
├── scripts/                 # Utility scripts
├── tests/                   # Test suite
└── requirements.txt
```

## Development Roadmap

- [x] Project initialization
- [ ] Core chart calculation engine
- [ ] Multi-house system implementation
- [ ] Vedic/Nakshatra integration
- [ ] Secondary progressions engine
- [ ] Transit prediction engine
- [ ] Zodiacal Releasing implementation
- [ ] Loosening of Bonds implementation
- [ ] LLM integration & prompt engineering
- [ ] Feedback collection system
- [ ] ML training pipeline
- [ ] Event correlation analytics
- [ ] API endpoints
- [ ] Frontend development

## License

MIT
