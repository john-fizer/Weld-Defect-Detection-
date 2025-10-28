# AI Job Application Assistant - System Architecture

## Executive Summary

An ethical AI-powered job application platform that maximizes interview chances while maintaining transparency, truthfulness, and user control. The system automates job discovery, application customization, and submission while requiring explicit human approval before any action.

## Core Philosophy: Ethical Automation

- **Truthfulness First**: Never fabricate skills, experience, or credentials
- **Human-in-the-Loop**: All submissions require explicit user approval
- **Privacy-Focused**: Encrypt sensitive data, never store credentials
- **Transparent Operations**: Tag AI-assisted applications when required
- **TOS Compliance**: Respect platform policies and opt-out when necessary

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Frontend Dashboard Layer                      │
│  • Resume upload & profile management                            │
│  • Job search criteria configuration                             │
│  • Application review & approval interface                       │
│  • Analytics dashboard (response rates, interview tracking)      │
│  • Interview prep tools                                          │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    API Layer (FastAPI)                           │
│  ┌──────────────┐ ┌────────────┐ ┌─────────────────────────┐   │
│  │ Profile      │ │ Job        │ │ Customization Engine    │   │
│  │ Ingestion    │ │ Sourcing   │ │ (Resume/Cover Letter)   │   │
│  └──────────────┘ └────────────┘ └─────────────────────────┘   │
│  ┌──────────────┐ ┌────────────┐ ┌─────────────────────────┐   │
│  │ Fit Scoring  │ │ Application│ │ Tracking & Analytics    │   │
│  │ Engine       │ │ Automation │ │ Dashboard               │   │
│  └──────────────┘ └────────────┘ └─────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    Data & ML Layer                               │
│  • PostgreSQL/SQLite: User profiles, jobs, applications          │
│  • Vector Store (Qdrant/FAISS): Resume & job embeddings         │
│  • LLM Integration: Claude for tailoring & generation           │
│  • Feedback Loop: Interview rates → model refinement            │
└─────────────────────────────────────────────────────────────────┘
                              ↕
┌─────────────────────────────────────────────────────────────────┐
│                    External Integration Layer                    │
│  • LinkedIn Jobs API       • Indeed API                          │
│  • Greenhouse API          • Workday Feeds                       │
│  • Playwright/Selenium     • Email notifications                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Module Specifications

### Module 1: Profile Ingestion

**Purpose**: Parse and structure user career data from multiple formats.

**Components**:
- PDF/DOCX parser (PyPDF2, python-docx, pdfplumber)
- Text extraction & NLP preprocessing
- Structured data extraction:
  - Contact information
  - Work experience (company, title, dates, achievements)
  - Education (institution, degree, dates, GPA)
  - Skills (technical, soft, certifications)
  - Projects & publications

**Output Schema**:
```json
{
  "personal_info": {
    "name": "string",
    "email": "string",
    "phone": "string",
    "location": "string",
    "linkedin": "string"
  },
  "work_experience": [
    {
      "company": "string",
      "title": "string",
      "start_date": "YYYY-MM",
      "end_date": "YYYY-MM",
      "achievements": ["string"],
      "skills_used": ["string"]
    }
  ],
  "education": [...],
  "skills": {
    "technical": ["string"],
    "soft": ["string"],
    "certifications": ["string"]
  },
  "preferences": {
    "target_roles": ["string"],
    "locations": ["string"],
    "salary_min": "integer",
    "remote_preference": "remote|hybrid|onsite",
    "excluded_companies": ["string"]
  }
}
```

**Ethics Layer**:
- Verify all extracted information with user before storage
- Flag any inconsistencies or potential parsing errors
- Allow manual corrections and additions

---

### Module 2: Job Sourcing

**Purpose**: Aggregate job postings from multiple sources and filter by criteria.

**Data Sources**:
1. **LinkedIn Jobs API** (Official API for enterprise)
2. **Indeed API** (Publisher API)
3. **Greenhouse API** (Direct company integrations)
4. **Workday Feeds** (RSS/API access)
5. **Custom scrapers** (Playwright-based, with rate limiting)

**Filtering Pipeline**:
```python
Job Input → Deduplication (URL/Title/Company hash)
          → Criteria Matching (location, role, salary)
          → TOS Compliance Check (exclude auto-apply banned sites)
          → Quality Score (completeness, clarity)
          → Output to Scoring Engine
```

**Storage Schema**:
```json
{
  "job_id": "uuid",
  "title": "string",
  "company": "string",
  "location": "string",
  "remote_type": "string",
  "salary_range": {"min": int, "max": int},
  "description": "string",
  "requirements": ["string"],
  "benefits": ["string"],
  "application_url": "string",
  "posting_date": "datetime",
  "source": "string",
  "metadata": {
    "ats_system": "greenhouse|workday|lever|custom",
    "auto_apply_allowed": "boolean"
  }
}
```

**Rate Limiting & Ethics**:
- Respect robots.txt and rate limits
- Rotate user agents responsibly
- Mark sources that prohibit automation
- Implement exponential backoff on errors

---

### Module 3: Fit Scoring Engine

**Purpose**: Rank job opportunities by match quality using semantic similarity.

**Scoring Algorithm**:
```
1. Generate Embeddings:
   - Resume embedding (all-MiniLM-L6-v2 or text-embedding-3-small)
   - Job description embedding

2. Calculate Similarity Scores:
   - Cosine similarity (resume ↔ job description)
   - Keyword overlap (hard skills match)
   - Experience level alignment (years required vs years held)
   - Location/remote preference match

3. LLM Enhancement:
   - Claude API analyzes fit for nuanced criteria
   - Identifies transferable skills
   - Flags potential red flags (unrealistic requirements)

4. Final Score Composition:
   - Embedding similarity: 40%
   - Keyword match: 30%
   - Experience alignment: 20%
   - LLM qualitative score: 10%
```

**Output**:
- Match percentage (0-100%)
- Detailed breakdown (strengths, gaps, suggestions)
- Ranked list of top N jobs
- Automated filtering threshold (e.g., only show >60% matches)

**Feedback Loop**:
- Track which match scores correlate with interviews
- Retrain weighting algorithm quarterly
- A/B test scoring variations

---

### Module 4: Customization Engine

**Purpose**: Dynamically tailor resumes and generate cover letters per job.

**Resume Tailoring**:
1. **Keyword Optimization**:
   - Extract keywords from job description (NER + TF-IDF)
   - Map to user's actual skills/experience
   - Re-order bullet points to highlight relevant experience
   - Adjust skill section prominence

2. **ATS Optimization**:
   - Ensure plain text compatibility
   - Use standard section headers
   - Avoid images/tables in ATS-critical sections
   - Include exact keyword matches from job posting

3. **Achievement Quantification**:
   - LLM rephrases generic bullets with metrics (when data exists)
   - Example: "Led team" → "Led 5-person team, delivered project 3 weeks ahead of schedule"

**Cover Letter Generation**:
```python
Prompt Template:
"""
You are a professional career coach. Write a compelling,
personalized cover letter for this candidate.

Candidate Profile:
{resume_summary}

Job Description:
{job_description}

Requirements:
- 3 paragraphs maximum
- Highlight 2-3 most relevant achievements
- Show enthusiasm without exaggeration
- Address company mission/values
- Professional but warm tone
- NEVER fabricate skills or experience
- Stay truthful and verifiable

Output format: Plain text, ready to paste
"""
```

**Ethics Guardrails**:
- **Pre-generation Check**: Verify all claims are present in original resume
- **Post-generation Validation**: LLM self-critique pass to detect embellishments
- **User Review Required**: Highlight all AI-generated sections for approval
- **Version Control**: Save all tailored versions with audit trail

**Customization Modes**:
- Conservative (minimal changes, high accuracy)
- Balanced (moderate tailoring, recommended)
- Aggressive (maximum keyword optimization, requires extra review)

---

### Module 5: Application Automation

**Purpose**: Auto-fill application forms with human approval checkpoints.

**Technology Stack**:
- **Playwright** (primary - better stability)
- **Selenium** (fallback for legacy sites)
- **Puppeteer** (if Node.js integration needed)

**Workflow**:
```
1. Pre-Application Checklist
   ├─ Verify TOS allows automation
   ├─ Check ATS system compatibility
   └─ Prepare tailored materials

2. Form Detection & Mapping
   ├─ AI identifies field types (name, email, experience)
   ├─ Map user profile data to form fields
   └─ Handle dropdowns, radio buttons, file uploads

3. Auto-Fill Execution
   ├─ Fill standard fields automatically
   ├─ Pause at open-ended questions
   └─ Present to user for review

4. User Approval Gate
   ├─ Show complete form preview
   ├─ Highlight all auto-filled fields
   ├─ Allow edits before submission
   └─ Require explicit "Submit" click

5. Submission & Confirmation
   ├─ Submit form if approved
   ├─ Capture confirmation page/email
   └─ Log to tracking database
```

**Supported ATS Systems**:
- Greenhouse
- Lever
- Workday
- Taleo
- iCIMS
- Custom forms (with fallback logic)

**Error Handling**:
- Screenshot capture on failure
- Retry logic (3 attempts with backoff)
- Fallback to manual application link
- Alert user to review manually

**Privacy Protections**:
- Never store passwords
- Use session tokens only (expire after use)
- No credential caching
- Option to use temporary email forwarding

**Rate Limiting**:
- Max 10 applications/hour per platform
- Randomized delays between actions (2-10 seconds)
- Human-like mouse movements and typing patterns
- Avoid detection as bot traffic

---

### Module 6: Tracking & Analytics Dashboard

**Purpose**: Monitor application status, response rates, and optimize strategy.

**Database Schema**:
```sql
-- Applications table
CREATE TABLE applications (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL,
    job_id UUID NOT NULL,
    resume_version VARCHAR,
    cover_letter_version VARCHAR,
    fit_score FLOAT,
    applied_date TIMESTAMP,
    status VARCHAR, -- pending, rejected, interview, offer
    response_date TIMESTAMP,
    notes TEXT
);

-- Interviews table
CREATE TABLE interviews (
    id UUID PRIMARY KEY,
    application_id UUID REFERENCES applications(id),
    scheduled_date TIMESTAMP,
    interview_type VARCHAR, -- phone, video, onsite
    outcome VARCHAR, -- passed, rejected, offer
    feedback TEXT
);

-- Feedback loop
CREATE TABLE feedback_events (
    id UUID PRIMARY KEY,
    application_id UUID REFERENCES applications(id),
    event_type VARCHAR, -- interview_secured, offer_received
    fit_score_at_time FLOAT,
    resume_version VARCHAR,
    timestamp TIMESTAMP
);
```

**Analytics Metrics**:
1. **Response Rate**: % of applications → interviews
2. **Success by Fit Score**: Correlation between match % and interview rate
3. **Time-to-Response**: Average days from application to reply
4. **Best-Performing Resume Versions**: Which tailoring strategies work
5. **Platform Performance**: Which job boards yield best results
6. **Company Insights**: Blacklist non-responsive companies

**Dashboard Views**:
- **Overview**: Total applied, pending, interviews, offers
- **Timeline**: Visual calendar of applications and interviews
- **Performance**: Charts showing trends over time
- **Job Pipeline**: Kanban board (Applied → Phone Screen → Final → Offer)
- **Insights**: AI-generated suggestions (e.g., "Applications with 80%+ fit score have 3x interview rate")

**Notifications**:
- Email alerts on status changes
- Interview reminders (24 hours before)
- Weekly summary reports
- Suggested follow-up actions

---

### Module 7: Interview Prep (Future Enhancement)

**Purpose**: Generate practice questions and talking points from job descriptions.

**Features**:
1. **Question Generation**:
   - Extract key skills/requirements from job description
   - Generate 10-15 likely interview questions
   - Categorize: Behavioral, Technical, Cultural Fit

2. **Answer Frameworks**:
   - STAR method templates
   - Pre-filled with user's actual achievements
   - Customizable per question

3. **Mock Interview Mode**:
   - Voice-based Q&A with AI interviewer
   - Real-time feedback on answers
   - Confidence scoring

4. **Company Research**:
   - Aggregate public info (Glassdoor, LinkedIn, news)
   - Summarize culture, challenges, recent news
   - Generate smart questions to ask interviewer

---

## Security & Compliance

### Data Security
- **Encryption at Rest**: AES-256 for stored resumes
- **Encryption in Transit**: TLS 1.3 for all API calls
- **Access Control**: JWT-based auth, role-based permissions
- **Audit Logging**: All actions logged with timestamps

### Compliance
- **GDPR**: Right to deletion, data portability, consent management
- **CCPA**: California privacy rights support
- **Terms of Service**: Clear user agreement on automation scope
- **Platform TOS**: Respect LinkedIn, Indeed, etc. automation policies

### Ethical Boundaries
```python
class EthicsGuardrail:
    def validate_resume_changes(self, original, modified):
        """Ensure no fabricated information"""
        if self.detect_new_skills(original, modified):
            raise EthicsViolation("Cannot add skills not in original resume")
        if self.detect_date_manipulation(original, modified):
            raise EthicsViolation("Cannot alter employment dates")
        if self.detect_degree_fabrication(original, modified):
            raise EthicsViolation("Cannot fabricate education credentials")
        return True

    def check_application_consent(self, user_id, job_id):
        """Verify explicit user approval exists"""
        consent = db.get_approval_record(user_id, job_id)
        if not consent or consent.expired():
            raise ConsentError("User approval required before submission")
        return True
```

---

## Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL (production) / SQLite (development)
- **Vector Store**: Qdrant (embeddings) or FAISS (lightweight)
- **Task Queue**: Celery + Redis (async job processing)
- **Caching**: Redis

### AI/ML
- **LLM**: Anthropic Claude (Sonnet 3.5 for tailoring)
- **Embeddings**: OpenAI text-embedding-3-small or Sentence-Transformers
- **Orchestration**: LangGraph (multi-agent workflows)
- **Monitoring**: Langfuse (LLM observability)

### Automation
- **Browser Automation**: Playwright (primary)
- **Scraping**: BeautifulSoup4, Scrapy
- **PDF Processing**: PyMuPDF, pdfplumber
- **DOCX Processing**: python-docx

### Frontend
- **Framework**: Next.js 14 (React) OR Streamlit (rapid prototyping)
- **Styling**: Tailwind CSS
- **Charts**: Recharts / Chart.js
- **State Management**: Zustand or React Query

### DevOps
- **Containerization**: Docker + Docker Compose
- **CI/CD**: GitHub Actions
- **Monitoring**: Prometheus + Grafana
- **Error Tracking**: Sentry

---

## Development Roadmap

### Phase 1: MVP (Weeks 1-4)
- [x] Architecture design
- [ ] Resume parser (PDF/DOCX)
- [ ] Basic job scraper (1-2 sources)
- [ ] Fit scoring (embedding-based)
- [ ] Simple cover letter generation
- [ ] Manual application tracking (no automation yet)
- [ ] Basic Streamlit dashboard

### Phase 2: Automation (Weeks 5-8)
- [ ] Playwright form auto-fill
- [ ] User approval workflow
- [ ] Multi-source job aggregation
- [ ] Enhanced fit scoring with LLM
- [ ] Application tracking database
- [ ] Email notifications

### Phase 3: Intelligence (Weeks 9-12)
- [ ] Feedback loop implementation
- [ ] A/B testing framework
- [ ] Interview rate analytics
- [ ] Resume version optimization
- [ ] Company insights engine
- [ ] Interview prep module

### Phase 4: Scale & Polish (Weeks 13-16)
- [ ] Next.js production frontend
- [ ] Multi-user support
- [ ] Subscription/pricing model
- [ ] API rate optimization
- [ ] Security audit
- [ ] Beta user testing

---

## Success Metrics

### Product Metrics
- **Application Volume**: 50-100 applications/user/month
- **Response Rate**: 10-15% (3x industry average)
- **Time Savings**: 20 hours/user/month vs manual application
- **User Satisfaction**: NPS > 50

### Technical Metrics
- **API Latency**: <200ms for scoring, <5s for customization
- **Uptime**: 99.5%
- **Scraper Success Rate**: >95%
- **Auto-fill Accuracy**: >90% field detection

### Business Metrics
- **User Retention**: 70% month-over-month
- **Conversion to Paid**: 20% of free users
- **Interview Secured**: 15% of applied jobs
- **Offer Rate**: 5% of applied jobs

---

## Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| Platform bans automation | High | Respect TOS, rate limiting, human patterns |
| Resume fabrication by users | Critical | Multi-layer validation, ethics checks |
| Data breach | Critical | Encryption, regular audits, minimal storage |
| LLM hallucination in letters | Medium | Self-critique pass, user review required |
| Scraper breakage | Medium | Multi-source redundancy, graceful degradation |
| Legal liability | High | Clear ToS, user consent, "assisted by AI" tags |

---

## Competitive Differentiation

**vs. Traditional Job Boards**:
- Personalized, not generic
- Proactive, not passive browsing
- Data-driven optimization

**vs. Auto-Apply Tools (Simplify, LazyApply)**:
- Ethical by design (no fabrication)
- Quality over quantity (fit scoring)
- Human-in-the-loop (approval required)
- Feedback-driven improvement

**vs. Manual Application**:
- 10x faster
- Better optimization
- Data-driven insights
- Never forget to follow up

---

## Next Steps

1. **Validate Assumptions**: User interviews with job seekers
2. **Build MVP**: Focus on resume parser + job aggregation + fit scoring
3. **Test Ethics Guardrails**: Red team testing for fabrication detection
4. **Pilot Program**: 20 beta users, track interview rates
5. **Iterate**: Refine based on feedback and success metrics

---

## License & Usage

This system is designed for **defensive career assistance only**.

**Acceptable Use**:
- Helping users organize job search
- Tailoring resumes to highlight relevant experience
- Automating repetitive form-filling tasks
- Tracking application progress

**Prohibited Use**:
- Fabricating credentials, skills, or experience
- Bypassing platform security measures
- Violating terms of service
- Mass-spamming applications without fit checking
- Misrepresenting AI involvement when disclosure required

---

**Architecture Version**: 1.0
**Last Updated**: 2025-10-28
**Maintainer**: AI Job Application Assistant Team
