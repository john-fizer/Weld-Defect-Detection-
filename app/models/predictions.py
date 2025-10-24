"""
Prediction models with modular feedback architecture
Each prediction technique has isolated feedback collection
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Enum, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.models import Base


class PredictionTypeEnum(str, enum.Enum):
    """Types of predictive techniques"""
    PROGRESSION = "progression"
    TRANSIT = "transit"
    ZODIACAL_RELEASING = "zodiacal_releasing"
    LOOSENING_BONDS = "loosening_bonds"


class EventCategoryEnum(str, enum.Enum):
    """Categories of life events for prediction"""
    CAREER_NEW_JOB = "career_new_job"
    CAREER_PROMOTION = "career_promotion"
    CAREER_TERMINATION = "career_termination"
    CAREER_BUSINESS_START = "career_business_start"

    RELATIONSHIP_START = "relationship_start"
    RELATIONSHIP_MARRIAGE = "relationship_marriage"
    RELATIONSHIP_BREAKUP = "relationship_breakup"
    RELATIONSHIP_DIVORCE = "relationship_divorce"

    FAMILY_BIRTH = "family_birth"
    FAMILY_DEATH = "family_death"
    FAMILY_HEALTH_CRISIS = "family_health_crisis"

    FINANCIAL_WINDFALL = "financial_windfall"
    FINANCIAL_LOSS = "financial_loss"
    FINANCIAL_INVESTMENT = "financial_investment"

    RELOCATION_MOVE = "relocation_move"
    RELOCATION_TRAVEL = "relocation_travel"

    PERSONAL_EDUCATION = "personal_education"
    PERSONAL_SPIRITUAL = "personal_spiritual"
    PERSONAL_HEALTH = "personal_health"

    OTHER = "other"


class Prediction(Base):
    """Base prediction record"""
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    natal_chart_id = Column(Integer, ForeignKey("natal_charts.id"), nullable=False)

    # Prediction metadata
    prediction_type = Column(Enum(PredictionTypeEnum), nullable=False, index=True)
    predicted_date_start = Column(DateTime, nullable=False, index=True)
    predicted_date_end = Column(DateTime)  # For date ranges
    confidence_score = Column(Float)  # Algorithm confidence

    # Event prediction
    predicted_event_category = Column(Enum(EventCategoryEnum))
    predicted_event_description = Column(String)

    # Astrological data (technique-specific)
    technique_data = Column(JSON)  # Stores technique-specific details

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    natal_chart = relationship("NatalChart", back_populates="predictions")
    progression_details = relationship("ProgressionPrediction", back_populates="prediction", uselist=False)
    transit_details = relationship("TransitPrediction", back_populates="prediction", uselist=False)
    zr_details = relationship("ZodiacalReleasingPrediction", back_populates="prediction", uselist=False)
    lb_details = relationship("looseningBondsPrediction", back_populates="prediction", uselist=False)


class ProgressionPrediction(Base):
    """Secondary progression specific prediction data"""
    __tablename__ = "progression_predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, unique=True)

    # Progression specifics
    progressed_planet = Column(String, nullable=False)
    natal_planet = Column(String)
    aspect_type = Column(String)  # conjunction, trine, square, etc.
    orb = Column(Float)

    # Additional data
    progression_data = Column(JSON)  # Full progression chart snapshot

    # Relationships
    prediction = relationship("Prediction", back_populates="progression_details")
    feedback = relationship("ProgressionFeedback", back_populates="progression_prediction", cascade="all, delete-orphan")


class TransitPrediction(Base):
    """Transit specific prediction data"""
    __tablename__ = "transit_predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, unique=True)

    # Transit specifics
    transiting_planet = Column(String, nullable=False)
    natal_planet = Column(String)
    natal_house_cusp = Column(Integer)
    aspect_type = Column(String)
    orb = Column(Float)

    # Transit metadata
    is_retrograde = Column(Boolean, default=False)
    exact_date = Column(DateTime)  # Exact aspect date

    # Relationships
    prediction = relationship("Prediction", back_populates="transit_details")
    feedback = relationship("TransitFeedback", back_populates="transit_prediction", cascade="all, delete-orphan")


class ZodiacalReleasingPrediction(Base):
    """Zodiacal Releasing (ZR) specific prediction data"""
    __tablename__ = "zodiacal_releasing_predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, unique=True)

    # ZR specifics
    period_lord = Column(String, nullable=False)  # Planet ruling the period
    sub_period_lord = Column(String)
    level = Column(Integer)  # ZR level depth

    # Period dates
    period_start = Column(DateTime, nullable=False)
    period_end = Column(DateTime, nullable=False)

    # ZR flags
    is_peak_period = Column(Boolean, default=False)
    is_loosening_period = Column(Boolean, default=False)

    # Additional ZR data
    zr_data = Column(JSON)  # Full ZR calculation details

    # Relationships
    prediction = relationship("Prediction", back_populates="zr_details")
    feedback = relationship("ZRFeedback", back_populates="zr_prediction", cascade="all, delete-orphan")


class LooseningBondsPrediction(Base):
    """Loosening of Bonds (LB) specific prediction data"""
    __tablename__ = "loosening_bonds_predictions"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, ForeignKey("predictions.id"), nullable=False, unique=True)

    # LB specifics
    bond_planet = Column(String, nullable=False)
    loosening_date = Column(DateTime, nullable=False)
    bond_type = Column(String)  # Type of bond being loosened

    # LB metadata
    intensity_score = Column(Float)  # Strength of the loosening

    # Additional LB data
    lb_data = Column(JSON)

    # Relationships
    prediction = relationship("Prediction", back_populates="lb_details")
    feedback = relationship("LBFeedback", back_populates="lb_prediction", cascade="all, delete-orphan")


# MODULAR FEEDBACK MODELS - One per technique

class ProgressionFeedback(Base):
    """Isolated feedback for progression predictions"""
    __tablename__ = "progression_feedback"

    id = Column(Integer, primary_key=True, index=True)
    progression_prediction_id = Column(Integer, ForeignKey("progression_predictions.id"), nullable=False)

    # Validation
    event_occurred = Column(Boolean)
    event_date = Column(DateTime)  # Actual event date
    date_accuracy_days = Column(Integer)  # Days off from prediction

    # Ratings
    accuracy_rating = Column(Integer)  # 1-5
    relevance_rating = Column(Integer)  # 1-5

    # User notes
    user_notes = Column(String)
    actual_event_category = Column(Enum(EventCategoryEnum))

    # Metadata
    submitted_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    progression_prediction = relationship("ProgressionPrediction", back_populates="feedback")


class TransitFeedback(Base):
    """Isolated feedback for transit predictions"""
    __tablename__ = "transit_feedback"

    id = Column(Integer, primary_key=True, index=True)
    transit_prediction_id = Column(Integer, ForeignKey("transit_predictions.id"), nullable=False)

    # Validation
    event_occurred = Column(Boolean)
    event_date = Column(DateTime)
    date_accuracy_days = Column(Integer)

    # Ratings
    accuracy_rating = Column(Integer)
    intensity_rating = Column(Integer)  # How intense was the event vs prediction

    # User notes
    user_notes = Column(String)
    actual_event_category = Column(Enum(EventCategoryEnum))

    # Metadata
    submitted_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    transit_prediction = relationship("TransitPrediction", back_populates="feedback")


class ZRFeedback(Base):
    """Isolated feedback for Zodiacal Releasing predictions"""
    __tablename__ = "zr_feedback"

    id = Column(Integer, primary_key=True, index=True)
    zr_prediction_id = Column(Integer, ForeignKey("zodiacal_releasing_predictions.id"), nullable=False)

    # Validation
    event_occurred = Column(Boolean)
    event_date = Column(DateTime)
    date_accuracy_days = Column(Integer)

    # Ratings
    accuracy_rating = Column(Integer)
    theme_accuracy_rating = Column(Integer)  # How well did ZR themes match

    # User notes
    user_notes = Column(String)
    actual_event_category = Column(Enum(EventCategoryEnum))

    # Metadata
    submitted_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    zr_prediction = relationship("ZodiacalReleasingPrediction", back_populates="feedback")


class LBFeedback(Base):
    """Isolated feedback for Loosening of Bonds predictions"""
    __tablename__ = "lb_feedback"

    id = Column(Integer, primary_key=True, index=True)
    lb_prediction_id = Column(Integer, ForeignKey("loosening_bonds_predictions.id"), nullable=False)

    # Validation
    event_occurred = Column(Boolean)
    event_date = Column(DateTime)
    date_accuracy_days = Column(Integer)

    # Ratings
    accuracy_rating = Column(Integer)
    refinement_value = Column(Integer)  # Did LB refine ZR timing effectively?

    # User notes
    user_notes = Column(String)
    actual_event_category = Column(Enum(EventCategoryEnum))

    # Metadata
    submitted_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    lb_prediction = relationship("LooseningBondsPrediction", back_populates="feedback")
