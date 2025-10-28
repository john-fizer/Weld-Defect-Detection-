"""
Application configuration using Pydantic Settings.
Loads from environment variables and .env file.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import List, Literal
from pathlib import Path


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True
    DEBUG: bool = True

    # Security
    SECRET_KEY: str = Field(..., min_length=32)
    ENCRYPTION_KEY: str = Field(..., min_length=32)

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./data/job_assistant.db"

    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"

    # AI/ML APIs
    ANTHROPIC_API_KEY: str | None = None
    OPENAI_API_KEY: str | None = None

    # LLM Configuration
    LLM_PROVIDER: Literal["anthropic", "openai"] = "anthropic"
    LLM_MODEL: str = "claude-sonnet-3-5-20240620"
    LLM_TEMPERATURE: float = Field(0.7, ge=0.0, le=2.0)
    LLM_MAX_TOKENS: int = Field(4096, ge=1, le=100000)

    # Embeddings Configuration
    EMBEDDING_PROVIDER: Literal["sentence-transformers", "openai"] = "sentence-transformers"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIMENSION: int = 384

    # Job Source APIs
    LINKEDIN_API_KEY: str | None = None
    LINKEDIN_API_SECRET: str | None = None
    INDEED_PUBLISHER_ID: str | None = None
    GREENHOUSE_API_KEY: str | None = None

    # Web Scraping
    USER_AGENT: str = "Mozilla/5.0 (compatible; JobAssistantBot/1.0)"
    SCRAPER_RATE_LIMIT_DELAY: int = Field(2, ge=1)
    SCRAPER_MAX_RETRIES: int = Field(3, ge=1, le=10)
    SCRAPER_TIMEOUT: int = Field(30, ge=5)

    # Browser Automation
    PLAYWRIGHT_HEADLESS: bool = True
    PLAYWRIGHT_TIMEOUT: int = Field(30000, ge=5000)
    PLAYWRIGHT_SLOW_MO: int = Field(100, ge=0)

    # Application Limits & Ethics
    MAX_APPLICATIONS_PER_HOUR: int = Field(10, ge=1, le=50)
    MAX_APPLICATIONS_PER_DAY: int = Field(50, ge=1, le=200)
    MIN_FIT_SCORE_THRESHOLD: float = Field(60.0, ge=0.0, le=100.0)
    REQUIRE_USER_APPROVAL: bool = True  # NEVER False in production
    AUTO_APPLY_ENABLED: bool = False

    # Resume Customization
    CUSTOMIZATION_MODE: Literal["conservative", "balanced", "aggressive"] = "balanced"
    ENABLE_ETHICS_VALIDATION: bool = True  # Always True
    ENABLE_COVER_LETTER_GENERATION: bool = True
    MAX_RESUME_VERSIONS: int = Field(10, ge=1)

    # Notifications
    ENABLE_EMAIL_NOTIFICATIONS: bool = False
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str | None = None
    SMTP_PASSWORD: str | None = None
    EMAIL_FROM: str = "noreply@jobassistant.app"

    # Monitoring & Logging
    LOG_LEVEL: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    LOG_FILE: str = "logs/job_assistant.log"
    SENTRY_DSN: str | None = None
    PROMETHEUS_PORT: int = 9090

    # Vector Store
    VECTOR_STORE_PATH: str = "./data/embeddings"
    VECTOR_INDEX_TYPE: str = "IndexFlatL2"

    # Data Retention
    RESUME_RETENTION_DAYS: int = Field(365, ge=30)
    APPLICATION_RETENTION_DAYS: int = Field(730, ge=90)
    LOG_RETENTION_DAYS: int = Field(90, ge=30)

    # Feature Flags
    ENABLE_JOB_SCRAPING: bool = True
    ENABLE_FIT_SCORING: bool = True
    ENABLE_AUTO_FILL: bool = False
    ENABLE_INTERVIEW_PREP: bool = False
    ENABLE_ANALYTICS_DASHBOARD: bool = True

    # CORS
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]
    CORS_ALLOW_CREDENTIALS: bool = True

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = Field(60, ge=10)
    RATE_LIMIT_PER_HOUR: int = Field(1000, ge=100)

    # Testing
    TESTING: bool = False
    MOCK_LLM_RESPONSES: bool = False
    MOCK_JOB_SOURCES: bool = False

    @field_validator("REQUIRE_USER_APPROVAL")
    @classmethod
    def validate_user_approval(cls, v: bool) -> bool:
        """Ensure user approval is always required in production."""
        if not v:
            raise ValueError(
                "REQUIRE_USER_APPROVAL must be True. "
                "Never disable user approval for ethical compliance."
            )
        return v

    @field_validator("ENABLE_ETHICS_VALIDATION")
    @classmethod
    def validate_ethics_validation(cls, v: bool) -> bool:
        """Ensure ethics validation is always enabled."""
        if not v:
            raise ValueError(
                "ENABLE_ETHICS_VALIDATION must be True. "
                "Ethics validation is a core safety feature."
            )
        return v

    @property
    def data_dir(self) -> Path:
        """Get data directory path."""
        path = Path("./data")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs_dir(self) -> Path:
        """Get logs directory path."""
        path = Path("./logs")
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.DEBUG


# Global settings instance
settings = Settings()
