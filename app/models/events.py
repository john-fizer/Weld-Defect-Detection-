"""
User event tracking for correlation with predictions
"""
from sqlalchemy import Column, Integer, String, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
from app.models import Base
from app.models.predictions import EventCategoryEnum


class UserEvent(Base):
    """Actual life events reported by users"""
    __tablename__ = "user_events"

    id = Column(Integer, primary_key=True, index=True)
    natal_chart_id = Column(Integer, ForeignKey("natal_charts.id"), nullable=False)

    # Event details
    event_category = Column(Enum(EventCategoryEnum), nullable=False, index=True)
    event_date = Column(DateTime, nullable=False, index=True)
    event_description = Column(String)

    # Event metadata
    intensity_rating = Column(Integer)  # 1-5, how significant was this event
    emotional_impact = Column(Integer)  # 1-5, emotional significance

    # User notes
    user_notes = Column(String)
    tags = Column(JSON)  # User-defined tags

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    natal_chart = relationship("NatalChart", back_populates="events")
    correlations = relationship("EventPredictionCorrelation", back_populates="event", cascade="all, delete-orphan")


class EventPredictionCorrelation(Base):
    """
    Correlation table between user events and predictions
    Allows ML to learn which techniques predicted which events accurately
    """
    __tablename__ = "event_prediction_correlations"

    id = Column(Integer, primary_key=True, index=True)
    user_event_id = Column(Integer, ForeignKey("user_events.id"), nullable=False)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False)

    # Correlation metrics
    time_delta_days = Column(Integer)  # Days between prediction and event
    category_match = Column(Boolean)  # Did category match?
    accuracy_score = Column(Float)  # Overall accuracy score

    # Metadata
    correlated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    event = relationship("UserEvent", back_populates="correlations")
