"""
Interview tracking database models.
"""

from sqlalchemy import Column, String, JSON, DateTime, Boolean, Text, Enum, ForeignKey
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import uuid
import enum


def generate_uuid():
    return str(uuid.uuid4())


class InterviewType(str, enum.Enum):
    """Enum for interview types."""

    PHONE_SCREEN = "phone_screen"
    VIDEO = "video"
    ONSITE = "onsite"
    TECHNICAL = "technical"
    BEHAVIORAL = "behavioral"
    PANEL = "panel"
    PRESENTATION = "presentation"
    ASSESSMENT = "assessment"


class InterviewOutcome(str, enum.Enum):
    """Enum for interview outcomes."""

    SCHEDULED = "scheduled"
    COMPLETED = "completed"
    PASSED = "passed"
    REJECTED = "rejected"
    OFFER = "offer"
    RESCHEDULED = "rescheduled"
    CANCELLED = "cancelled"


class Interview(Base):
    """Interview tracking model."""

    __tablename__ = "interviews"

    id = Column(String, primary_key=True, default=generate_uuid)
    application_id = Column(String, ForeignKey("applications.id"), nullable=False, index=True)

    # Interview Details
    interview_type = Column(Enum(InterviewType), nullable=False)
    round_number = Column(Integer, default=1)  # Interview round (1st, 2nd, final, etc.)

    # Scheduling
    scheduled_date = Column(DateTime(timezone=True))
    duration_minutes = Column(Integer)
    timezone = Column(String)
    location = Column(String)  # Physical location or video call link
    meeting_link = Column(String)

    # Interviewers
    interviewers = Column(JSON, default=[])  # List of interviewer names/titles

    # Preparation
    prep_questions_generated = Column(JSON)  # AI-generated practice questions
    prep_notes = Column(Text)  # User's preparation notes
    company_research = Column(JSON)  # AI-generated company research

    # Outcome
    outcome = Column(Enum(InterviewOutcome), default=InterviewOutcome.SCHEDULED)
    completed_date = Column(DateTime(timezone=True))
    user_feedback = Column(Text)  # User's notes after interview
    company_feedback = Column(Text)  # Feedback from company

    # Follow-up
    thank_you_sent = Column(Boolean, default=False)
    thank_you_date = Column(DateTime(timezone=True))
    follow_up_sent = Column(Boolean, default=False)
    follow_up_date = Column(DateTime(timezone=True))

    # Reminders
    reminder_sent = Column(Boolean, default=False)
    reminder_date = Column(DateTime(timezone=True))

    # Timestamps
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    application = relationship("Application", back_populates="interviews")
