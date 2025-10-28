"""
User and UserProfile database models.
"""

from sqlalchemy import Column, String, Integer, JSON, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import uuid


def generate_uuid():
    return str(uuid.uuid4())


class User(Base):
    """User account model."""

    __tablename__ = "users"

    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False, index=True)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    applications = relationship("Application", back_populates="user")


class UserProfile(Base):
    """User profile with resume data and preferences."""

    __tablename__ = "user_profiles"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), unique=True, nullable=False)

    # Personal Information
    full_name = Column(String, nullable=False)
    phone = Column(String)
    location = Column(String)
    linkedin_url = Column(String)
    portfolio_url = Column(String)
    github_url = Column(String)

    # Resume Data (structured JSON)
    work_experience = Column(JSON, default=[])  # List of work experiences
    education = Column(JSON, default=[])  # List of education entries
    skills = Column(JSON, default={})  # Dict with technical, soft, certifications
    projects = Column(JSON, default=[])  # List of projects
    certifications = Column(JSON, default=[])  # List of certifications
    languages = Column(JSON, default=[])  # List of spoken languages

    # Resume Files (encrypted paths)
    original_resume_path = Column(String)  # Original uploaded resume
    parsed_resume_data = Column(JSON)  # Full parsed data

    # Job Search Preferences
    target_roles = Column(JSON, default=[])  # List of target job titles
    target_locations = Column(JSON, default=[])  # Preferred locations
    remote_preference = Column(String, default="remote")  # remote, hybrid, onsite, any
    salary_min = Column(Integer)
    salary_max = Column(Integer)
    excluded_companies = Column(JSON, default=[])  # Companies to avoid

    # Settings
    customization_mode = Column(String, default="balanced")  # conservative, balanced, aggressive
    auto_apply_enabled = Column(Boolean, default=False)
    max_applications_per_day = Column(Integer, default=10)

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    last_resume_update = Column(DateTime(timezone=True))

    # Relationships
    user = relationship("User", back_populates="profile")
