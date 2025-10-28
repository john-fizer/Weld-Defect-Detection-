# AI Job Application Assistant

An ethical AI-powered platform that helps job seekers maximize interview chances through intelligent automation, while maintaining complete transparency and truthfulness.

## Core Principles

- **Truthfulness First**: Never fabricate skills, experience, or credentials
- **Human-in-the-Loop**: All submissions require explicit user approval
- **Privacy-Focused**: End-to-end encryption, no credential storage
- **Ethical Automation**: Respect platform terms of service
- **Data-Driven**: Continuous improvement through feedback loops

## Features

### 1. Smart Resume Parsing
- Extract structured data from PDF/DOCX resumes
- Intelligent parsing of work experience, education, skills
- Validation layer to ensure accuracy

### 2. Multi-Source Job Aggregation
- LinkedIn Jobs API
- Indeed Publisher API
- Greenhouse company integrations
- Workday RSS feeds
- Custom scrapers (with ethical rate limiting)

### 3. AI-Powered Fit Scoring
- Semantic similarity matching (embeddings)
- Keyword analysis and skill alignment
- Experience level matching
- LLM-enhanced nuanced evaluation
- Ranked job recommendations (0-100% match score)

### 4. Intelligent Application Customization
- Resume tailoring per job (ATS-optimized)
- AI-generated cover letters (with ethics validation)
- Keyword optimization without fabrication
- Achievement quantification
- Multiple tailoring modes: Conservative, Balanced, Aggressive

### 5. Automated Form Filling (with Approval Gates)
- Playwright-based browser automation
- Support for major ATS platforms (Greenhouse, Lever, Workday, Taleo, iCIMS)
- **Required user approval before submission**
- Screenshot capture and error handling
- Privacy-first design (no password storage)

### 6. Application Tracking Dashboard
- Real-time application status monitoring
- Interview scheduling and reminders
- Response rate analytics
- Resume version A/B testing
- Company performance insights

### 7. Interview Preparation (Coming Soon)
- AI-generated interview questions based on job description
- STAR method answer templates
- Mock interview practice
- Company research summaries

## Architecture

```
job_assistant/
├── app/
│   ├── api/              # FastAPI routes
│   │   ├── profiles.py   # Resume upload & parsing
│   │   ├── jobs.py       # Job search & aggregation
│   │   ├── scoring.py    # Fit score calculation
│   │   ├── applications.py # Application management
│   │   └── analytics.py  # Dashboard metrics
│   ├── core/             # Business logic
│   │   ├── resume_parser.py
│   │   ├── job_scraper.py
│   │   ├── fit_scorer.py
│   │   ├── customizer.py
│   │   └── ethics.py     # Ethics validation layer
│   ├── ml/               # Machine learning
│   │   ├── embeddings.py
│   │   ├── scorer_trainer.py
│   │   └── feedback_loop.py
│   ├── automation/       # Browser automation
│   │   ├── playwright_runner.py
│   │   ├── ats_handlers/
│   │   └── approval_gate.py
│   ├── models/           # Database models
│   │   ├── user.py
│   │   ├── job.py
│   │   ├── application.py
│   │   └── feedback.py
│   └── schemas/          # Pydantic schemas
├── frontend/             # Dashboard UI
│   └── streamlit_app.py  # MVP Streamlit dashboard
├── data/
│   ├── resumes/          # Encrypted resume storage
│   ├── embeddings/       # Vector store
│   └── models/           # ML model weights
├── scripts/
│   ├── init_db.py
│   ├── run_scraper.py
│   └── train_scorer.py
├── tests/
├── requirements.txt
├── .env.example
├── docker-compose.yml
└── README.md
```

See [ARCHITECTURE.md](./ARCHITECTURE.md) for detailed system design.

## Tech Stack

**Backend**:
- FastAPI (Python 3.11+)
- SQLite (dev) / PostgreSQL (prod)
- SQLAlchemy ORM
- Celery + Redis (task queue)

**AI/ML**:
- Anthropic Claude (Sonnet 3.5)
- Sentence-Transformers (embeddings)
- FAISS (vector similarity)
- LangGraph (agent orchestration)

**Automation**:
- Playwright (browser automation)
- BeautifulSoup4 (scraping)
- pdfplumber, PyMuPDF (PDF parsing)
- python-docx (DOCX parsing)

**Frontend**:
- Streamlit (MVP)
- Next.js (production - future)
- Recharts (analytics visualizations)

## Installation

### Prerequisites
- Python 3.11+
- Node.js 18+ (for Playwright)
- Redis (for task queue)

### Setup

```bash
# Navigate to project directory
cd job_assistant

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (Anthropic, OpenAI, etc.)

# Initialize database
python scripts/init_db.py

# Run migrations (if needed)
alembic upgrade head
```

## Usage

### 1. Start the Backend API

```bash
uvicorn app.main:app --reload --port 8000
```

### 2. Start the Dashboard

```bash
streamlit run frontend/streamlit_app.py
```

### 3. Start Background Workers (Optional)

```bash
# Terminal 1: Redis
redis-server

# Terminal 2: Celery worker
celery -A app.tasks worker --loglevel=info
```

## Quick Start Guide

### Step 1: Upload Your Resume
1. Navigate to dashboard: `http://localhost:8501`
2. Upload PDF or DOCX resume
3. Review and verify parsed information
4. Set job search preferences (roles, locations, salary)

### Step 2: Configure Job Search
- Target roles (e.g., "Software Engineer", "Data Scientist")
- Preferred locations or "Remote"
- Salary range
- Excluded companies (optional)

### Step 3: Browse Matched Jobs
- System automatically aggregates jobs from multiple sources
- View fit scores (0-100%) with detailed breakdowns
- Filter by match threshold (e.g., show only >70% matches)

### Step 4: Review Tailored Applications
- Select a job to apply
- Review AI-tailored resume
- Edit AI-generated cover letter
- **Approve or reject** before submission

### Step 5: Auto-Apply with Approval
- System navigates to application page
- Auto-fills form fields
- Pauses for your review
- **You click final "Submit"**

### Step 6: Track Applications
- Monitor status (pending, interview, rejected, offer)
- Update manually or via email parsing
- View analytics (response rates, best-performing resumes)

## Ethics & Compliance

### What This System WILL Do:
- Parse your existing resume accurately
- Reorder bullet points to highlight relevant experience
- Generate cover letters based on YOUR actual achievements
- Auto-fill repetitive form fields
- Track application progress
- Analyze which strategies work best

### What This System WILL NOT Do:
- Fabricate skills, degrees, or experience
- Alter employment dates
- Submit applications without your approval
- Store your passwords
- Violate platform terms of service
- Bypass security measures

### Ethics Validation Layers:
1. **Pre-Generation**: Verify all content exists in original resume
2. **Post-Generation**: LLM self-critique to detect embellishments
3. **User Review**: All AI changes highlighted for approval
4. **Audit Trail**: Version control for all modifications

## Configuration

### Environment Variables (`.env`)

```bash
# API Keys
ANTHROPIC_API_KEY=your_claude_api_key
OPENAI_API_KEY=your_openai_key  # For embeddings (optional)

# Database
DATABASE_URL=sqlite:///./data/job_assistant.db  # Dev
# DATABASE_URL=postgresql://user:pass@localhost/job_assistant  # Prod

# Redis (for Celery)
REDIS_URL=redis://localhost:6379/0

# Job Sources
LINKEDIN_API_KEY=your_linkedin_key
INDEED_PUBLISHER_ID=your_indeed_id

# Security
SECRET_KEY=your_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here  # For resume storage

# Application Settings
MAX_APPLICATIONS_PER_HOUR=10
MIN_FIT_SCORE_THRESHOLD=60
REQUIRE_USER_APPROVAL=true  # NEVER set to false
```

## API Documentation

Once the backend is running, visit:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

### Key Endpoints

```
POST   /api/profiles/upload          # Upload resume
GET    /api/profiles/me              # Get parsed profile
PUT    /api/profiles/me              # Update profile

GET    /api/jobs/search              # Search jobs
GET    /api/jobs/{job_id}/score      # Get fit score
POST   /api/jobs/{job_id}/tailor     # Generate tailored resume

POST   /api/applications/create      # Create application (not submit!)
POST   /api/applications/{id}/approve # User approval to submit
GET    /api/applications              # List all applications
PATCH  /api/applications/{id}        # Update application status

GET    /api/analytics/dashboard      # Dashboard metrics
GET    /api/analytics/performance    # Resume version performance
```

## Development

### Running Tests

```bash
# Unit tests
pytest tests/unit -v

# Integration tests
pytest tests/integration -v

# E2E tests (requires browser)
pytest tests/e2e -v

# Coverage report
pytest --cov=app --cov-report=html
```

### Code Quality

```bash
# Linting
ruff check app/

# Formatting
black app/

# Type checking
mypy app/
```

### Database Migrations

```bash
# Create migration
alembic revision --autogenerate -m "description"

# Apply migration
alembic upgrade head

# Rollback
alembic downgrade -1
```

## Roadmap

### Phase 1: MVP (Current)
- [x] Architecture design
- [ ] Resume parser (PDF/DOCX)
- [ ] Basic job scraper (LinkedIn + Indeed)
- [ ] Fit scoring (embeddings + LLM)
- [ ] Cover letter generation
- [ ] Application tracking database
- [ ] Streamlit dashboard

### Phase 2: Automation
- [ ] Playwright browser automation
- [ ] ATS system detection
- [ ] Form auto-fill with approval gates
- [ ] Multi-source job aggregation
- [ ] Email notification system

### Phase 3: Intelligence
- [ ] Feedback loop (interview rate → model refinement)
- [ ] A/B testing framework
- [ ] Resume version optimization
- [ ] Company insights & recommendations
- [ ] Interview prep module

### Phase 4: Production
- [ ] Next.js production frontend
- [ ] Multi-user authentication
- [ ] Subscription/pricing tiers
- [ ] Security audit
- [ ] Beta user program

## Troubleshooting

### Common Issues

**Resume parsing errors:**
- Ensure PDF is text-based (not scanned image)
- Try DOCX format instead
- Manually verify parsed data

**Job scraper failures:**
- Check API keys in `.env`
- Verify network connectivity
- Review rate limiting settings
- Check scraper logs: `logs/scraper.log`

**Playwright automation fails:**
- Ensure Playwright browsers installed: `playwright install`
- Check website hasn't changed structure
- Enable headless=False to debug visually
- Review screenshots in `data/screenshots/`

**Low fit scores:**
- Ensure resume has detailed experience/skills
- Check if job descriptions are complete
- Adjust scoring weights in config
- Review LLM prompts for scoring

## Contributing

This is an ethical automation tool. Contributions should:
- Maintain ethics guardrails (no fabrication features)
- Improve accuracy and user control
- Respect platform terms of service
- Include tests for new features
- Update documentation

## License

MIT License - See [LICENSE](../LICENSE) for details

## Disclaimer

This tool is designed for **legitimate job search assistance only**. Users are responsible for:
- Ensuring accuracy of all submitted information
- Complying with platform terms of service
- Verifying AI-generated content before submission
- Respecting company application processes

By using this tool, you agree that:
- All resume information is truthful and verifiable
- You will review all AI-generated content
- You will not use this for mass-spamming applications
- You will comply with all applicable laws and regulations

## Support

- **Documentation**: See [ARCHITECTURE.md](./ARCHITECTURE.md)
- **Issues**: Submit via GitHub Issues
- **Discussions**: GitHub Discussions

## Acknowledgments

Built with:
- Anthropic Claude for intelligent content generation
- Playwright for reliable browser automation
- FastAPI for modern Python web framework
- Community feedback for ethical design

---

**Version**: 1.0.0-alpha
**Status**: MVP Development
**Last Updated**: 2025-10-28
