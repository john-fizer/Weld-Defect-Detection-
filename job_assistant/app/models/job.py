"""
Job posting database models.
"""

from sqlalchemy import Column, String, Integer, JSON, DateTime, Boolean, Text, Float, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import uuid
import enum


def generate_uuid():
    return str(uuid.uuid4())


class JobSource(str, enum.Enum):
    """Enum for job sources."""

    LINKEDIN = "linkedin"
    INDEED = "indeed"
    GREENHOUSE = "greenhouse"
    WORKDAY = "workday"
    LEVER = "lever"
    CUSTOM = "custom"


class Job(Base):
    """Job posting model."""

    __tablename__ = "jobs"

    id = Column(String, primary_key=True, default=generate_uuid)

    # Basic Information
    title = Column(String, nullable=False, index=True)
    company = Column(String, nullable=False, index=True)
    location = Column(String)
    remote_type = Column(String)  # remote, hybrid, onsite
    application_url = Column(String, nullable=False)

    # Salary (if available)
    salary_min = Column(Integer)
    salary_max = Column(Integer)
    salary_currency = Column(String, default="USD")

    # Job Details
    description = Column(Text, nullable=False)
    requirements = Column(JSON, default=[])  # List of requirements
    responsibilities = Column(JSON, default=[])  # List of responsibilities
    benefits = Column(JSON, default=[])  # List of benefits
    qualifications = Column(JSON, default=[])  # Required qualifications

    # Experience Level
    experience_level = Column(String)  # entry, mid, senior, lead, executive
    years_experience_min = Column(Integer)
    years_experience_max = Column(Integer)

    # Source Information
    source = Column(Enum(JobSource), nullable=False, index=True)
    source_job_id = Column(String, unique=True, index=True)  # Original job ID from source
    posting_date = Column(DateTime(timezone=True))
    expiration_date = Column(DateTime(timezone=True))

    # Metadata
    ats_system = Column(String)  # greenhouse, workday, lever, etc.
    auto_apply_allowed = Column(Boolean, default=False)
    company_size = Column(String)  # startup, small, medium, large, enterprise
    industry = Column(String)

    # Embeddings for semantic search
    description_embedding = Column(JSON)  # Stored as list of floats

    # Status
    is_active = Column(Boolean, default=True)
    last_scraped_at = Column(DateTime(timezone=True), server_default=func.now())

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    applications = relationship("Application", back_populates="job")
