# AI-Powered Application Suite

A collection of intelligent, ethically-designed AI applications leveraging LLMs, machine learning, and automation to solve real-world problems.

## Projects

### 1. [Holographic Astrology Platform](./app/)
**Status**: In Development

A next-generation astrology interpretation system that synthesizes multiple chart systems, predictive techniques, and machine learning feedback loops for precision astrological analysis.

**Key Features**:
- Multi-system chart synthesis (Western, Vedic, Decans)
- LLM-powered unified interpretation
- Advanced timing predictions (Progressions, Transits, Zodiacal Releasing)
- Event correlation engine with feedback loops
- Modular ML training per prediction technique

**Tech Stack**: FastAPI, Swiss Ephemeris, PostgreSQL, PyTorch, Claude/GPT-4

[📖 Read More](./app/README.md) | [🏗️ Architecture](./ARCHITECTURE.md)

---

### 2. [AI Job Application Assistant](./job_assistant/)
**Status**: MVP Development

An ethical AI-powered job application platform that maximizes interview chances through intelligent automation while maintaining transparency, truthfulness, and user control.

**Key Features**:
- Smart resume parsing (PDF/DOCX)
- Multi-source job aggregation (LinkedIn, Indeed, Greenhouse, Workday)
- AI-powered fit scoring (embeddings + LLM analysis)
- Resume tailoring & cover letter generation (ethics-validated)
- Automated form filling with mandatory user approval
- Application tracking & analytics dashboard
- Interview preparation tools

**Core Principles**:
- Truthfulness First (no fabrication)
- Human-in-the-Loop (explicit approval required)
- Privacy-Focused (encrypted data, no credential storage)
- TOS Compliance (respect platform policies)

**Tech Stack**: FastAPI, Playwright, Claude, Sentence-Transformers, FAISS, Streamlit

[📖 Read More](./job_assistant/README.md) | [🏗️ Architecture](./job_assistant/ARCHITECTURE.md)

---

### 3. Trading Assistant
**Status**: Planned

AI-powered trading analysis and automation platform (details TBD).

---

## Repository Structure

```
.
├── app/                    # Holographic Astrology Platform
│   ├── api/
│   ├── core/
│   ├── predictions/
│   ├── llm/
│   ├── ml/
│   └── models/
│
├── job_assistant/          # AI Job Application Assistant
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── ml/
│   │   ├── automation/
│   │   └── models/
│   ├── frontend/
│   ├── data/
│   └── scripts/
│
├── trading_assistant/      # Trading Assistant (planned)
│   └── (TBD)
│
├── scripts/                # Shared utility scripts
├── .env.example            # Environment variable template
└── README.md               # This file
```

## Quick Start

Each project is self-contained with its own dependencies and documentation.

### Holographic Astrology Platform

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize database
python scripts/init_db.py

# Run server
uvicorn app.main:app --reload
```

See [app/README.md](./app/README.md) for detailed instructions.

### AI Job Application Assistant

```bash
# Navigate to project
cd job_assistant

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Set up environment
cp .env.example .env
# Edit .env with API keys

# Initialize database
python scripts/init_db.py

# Run backend
uvicorn app.main:app --reload --port 8000

# Run frontend (separate terminal)
streamlit run frontend/streamlit_app.py
```

See [job_assistant/README.md](./job_assistant/README.md) for detailed instructions.

## Shared Principles

All projects in this suite adhere to:

### Ethics & Responsibility
- **Transparency**: Users understand what AI is doing
- **Control**: Humans make final decisions
- **Privacy**: Data encryption and minimal storage
- **Truthfulness**: No fabrication or deception
- **Compliance**: Respect laws and platform terms of service

### Technical Excellence
- **Modular Architecture**: Independent, testable components
- **Feedback Loops**: Continuous improvement via user data
- **Type Safety**: Pydantic schemas and type hints
- **Testing**: Unit, integration, and E2E tests
- **Documentation**: Comprehensive READMEs and architecture docs

### User-Centric Design
- **Simplicity**: Easy setup and intuitive interfaces
- **Performance**: Fast response times and reliable operation
- **Observability**: Clear logging and error messages
- **Flexibility**: Configurable to user preferences

## Development

### Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install project-specific dependencies
cd <project_directory>
pip install -r requirements.txt
```

### Environment Variables

Each project has its own `.env` file. Copy `.env.example` as a template:

```bash
cd <project_directory>
cp .env.example .env
# Edit .env with your API keys and configuration
```

Common variables:
- `ANTHROPIC_API_KEY`: For Claude LLM
- `OPENAI_API_KEY`: For GPT/embeddings (optional)
- `DATABASE_URL`: Database connection string
- `REDIS_URL`: Redis for task queues (if applicable)

### Testing

Each project includes comprehensive tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=app --cov-report=html

# Run specific test suite
pytest tests/unit
pytest tests/integration
pytest tests/e2e
```

### Code Quality

We maintain high code quality standards:

```bash
# Linting
ruff check .

# Formatting
black .

# Type checking
mypy .

# Security scanning
bandit -r .
```

## Contributing

Contributions are welcome! Please ensure:

1. **Ethics First**: No features that enable deception or harm
2. **Tests Required**: All new features must include tests
3. **Documentation**: Update relevant README/architecture docs
4. **Code Quality**: Pass linting, formatting, and type checks
5. **User Consent**: Features requiring user data need explicit consent

### Contribution Workflow

```bash
# 1. Fork repository
# 2. Create feature branch
git checkout -b feature/your-feature-name

# 3. Make changes with tests
# 4. Run quality checks
black . && ruff check . && mypy . && pytest

# 5. Commit with descriptive message
git commit -m "feat: add user consent modal for data collection"

# 6. Push and create pull request
git push origin feature/your-feature-name
```

## Tech Stack (Shared)

**Languages & Frameworks**:
- Python 3.11+
- FastAPI (API backend)
- SQLAlchemy (ORM)
- Pydantic (validation)

**AI/ML**:
- Anthropic Claude (LLM reasoning)
- OpenAI GPT (alternative LLM)
- Sentence-Transformers (embeddings)
- FAISS/Qdrant (vector search)
- LangGraph (agent orchestration)

**Databases**:
- PostgreSQL (production)
- SQLite (development)
- Redis (caching & queues)

**Frontend**:
- Streamlit (rapid prototyping)
- Next.js (production apps)
- Tailwind CSS

**DevOps**:
- Docker & Docker Compose
- GitHub Actions (CI/CD)
- Prometheus & Grafana (monitoring)
- Sentry (error tracking)

## License

MIT License - See [LICENSE](./LICENSE) for details.

Each project may have additional licensing requirements specified in their respective directories.

## Disclaimers

**Astrology Platform**: For entertainment and self-reflection purposes. Not a substitute for professional advice.

**Job Application Assistant**: Users are responsible for the accuracy of all submitted information. This tool assists with organization and optimization but cannot guarantee job offers. Comply with all platform terms of service.

**Trading Assistant**: (TBD) Not financial advice. Trading involves risk.

## Support

- **Documentation**: Each project has detailed README and ARCHITECTURE docs
- **Issues**: Submit via [GitHub Issues](../../issues)
- **Discussions**: [GitHub Discussions](../../discussions)

## Roadmap

### Q1 2025
- [x] Astrology Platform: Initial implementation
- [x] Job Assistant: Architecture design
- [ ] Job Assistant: MVP completion
- [ ] Job Assistant: Beta testing program

### Q2 2025
- [ ] Astrology Platform: Frontend development
- [ ] Job Assistant: Production release
- [ ] Trading Assistant: Design phase
- [ ] Shared authentication system

### Q3 2025
- [ ] Trading Assistant: MVP
- [ ] Multi-project dashboard
- [ ] Advanced analytics across projects
- [ ] Mobile app exploration

## Acknowledgments

Built with powerful tools from:
- **Anthropic** (Claude AI)
- **OpenAI** (GPT & embeddings)
- **Playwright** (browser automation)
- **FastAPI** (modern Python web framework)
- **Swiss Ephemeris** (astronomical calculations)
- Open-source community

---

**Repository Status**: Active Development
**Last Updated**: 2025-10-28
**Maintainer**: AI Application Suite Team

For project-specific information, see individual project READMEs.
