"""
Job application database models.
"""

from sqlalchemy import Column, String, Integer, JSON, DateTime, Boolean, Text, Float, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import uuid
import enum


def generate_uuid():
    return str(uuid.uuid4())


class ApplicationStatus(str, enum.Enum):
    """Enum for application status."""

    DRAFT = "draft"  # Created but not submitted
    PENDING_APPROVAL = "pending_approval"  # Awaiting user approval
    APPROVED = "approved"  # User approved, ready to submit
    SUBMITTED = "submitted"  # Application submitted
    UNDER_REVIEW = "under_review"  # Company is reviewing
    PHONE_SCREEN = "phone_screen"  # Phone screen scheduled/completed
    INTERVIEW = "interview"  # Interview scheduled/in progress
    FINAL_ROUND = "final_round"  # Final interview round
    OFFER = "offer"  # Offer received
    REJECTED = "rejected"  # Application rejected
    WITHDRAWN = "withdrawn"  # User withdrew application
    ACCEPTED = "accepted"  # User accepted offer
    DECLINED = "declined"  # User declined offer


class Application(Base):
    """Job application tracking model."""

    __tablename__ = "applications"

    id = Column(String, primary_key=True, default=generate_uuid)
    user_id = Column(String, ForeignKey("users.id"), nullable=False, index=True)
    job_id = Column(String, ForeignKey("jobs.id"), nullable=False, index=True)

    # Application Materials
    tailored_resume_path = Column(String)  # Path to tailored resume
    cover_letter_text = Column(Text)  # Generated cover letter
    resume_version = Column(String)  # Version identifier
    cover_letter_version = Column(String)

    # Fit Score
    fit_score = Column(Float)  # 0-100 match score
    fit_breakdown = Column(JSON)  # Detailed scoring breakdown
    llm_analysis = Column(JSON)  # LLM's fit analysis

    # Application Status
    status = Column(Enum(ApplicationStatus), default=ApplicationStatus.DRAFT, index=True)
    status_history = Column(JSON, default=[])  # Track status changes

    # Dates
    applied_date = Column(DateTime(timezone=True))
    response_date = Column(DateTime(timezone=True))
    last_status_update = Column(DateTime(timezone=True))

    # User Approval
    approved_by_user = Column(Boolean, default=False)
    approval_date = Column(DateTime(timezone=True))
    user_modifications = Column(JSON)  # User's edits to AI-generated content

    # Application Details
    application_method = Column(String)  # web_form, email, direct_upload, api
    confirmation_number = Column(String)
    confirmation_email = Column(String)

    # Notes and Feedback
    notes = Column(Text)  # User's private notes
    company_feedback = Column(Text)  # Feedback from company (if any)

    # Auto-fill metadata
    form_fields_filled = Column(JSON)  # Fields that were auto-filled
    auto_fill_success = Column(Boolean)
    auto_fill_errors = Column(JSON)  # Errors encountered during auto-fill

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="applications")
    job = relationship("Job", back_populates="applications")
    interviews = relationship("Interview", back_populates="application")
