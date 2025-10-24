"""
Natal chart and chart-related database models
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, Enum
from sqlalchemy.orm import relationship
from datetime import datetime
import enum
from app.models import Base


class HouseSystemEnum(str, enum.Enum):
    """House system types"""
    WHOLE_SIGN = "whole_sign"
    PLACIDUS = "placidus"
    KOCH = "koch"
    EQUAL = "equal"
    VEDIC_WHOLE = "vedic_whole"


class ZodiacSystemEnum(str, enum.Enum):
    """Zodiac system types"""
    TROPICAL = "tropical"
    SIDEREAL = "sidereal"


class NatalChart(Base):
    """Core natal chart data"""
    __tablename__ = "natal_charts"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)  # External user reference
    name = Column(String)

    # Birth data
    birth_datetime = Column(DateTime, nullable=False)
    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    timezone = Column(String, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    chart_calculations = relationship("ChartCalculation", back_populates="natal_chart", cascade="all, delete-orphan")
    predictions = relationship("Prediction", back_populates="natal_chart", cascade="all, delete-orphan")
    events = relationship("UserEvent", back_populates="natal_chart", cascade="all, delete-orphan")


class ChartCalculation(Base):
    """Calculated chart data for different house/zodiac systems"""
    __tablename__ = "chart_calculations"

    id = Column(Integer, primary_key=True, index=True)
    natal_chart_id = Column(Integer, ForeignKey("natal_charts.id"), nullable=False)

    # System configuration
    house_system = Column(Enum(HouseSystemEnum), nullable=False)
    zodiac_system = Column(Enum(ZodiacSystemEnum), nullable=False)

    # Calculated data (stored as JSON for flexibility)
    planets = Column(JSON)  # {planet: {sign, degree, house, retrograde, ...}}
    houses = Column(JSON)   # {house_number: {sign, degree, ...}}
    aspects = Column(JSON)  # [{planet1, planet2, aspect_type, orb, ...}]
    nakshatras = Column(JSON)  # Vedic specific: {planet: {nakshatra, pada, ruler, ...}}
    decans = Column(JSON)   # {planet: {decan, ruler, ...}}

    # Metadata
    calculated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    natal_chart = relationship("NatalChart", back_populates="chart_calculations")
    interpretations = relationship("ChartInterpretation", back_populates="chart_calculation", cascade="all, delete-orphan")


class ChartInterpretation(Base):
    """LLM-generated interpretations of chart calculations"""
    __tablename__ = "chart_interpretations"

    id = Column(Integer, primary_key=True, index=True)
    chart_calculation_id = Column(Integer, ForeignKey("chart_calculations.id"), nullable=False)

    # Interpretation data
    interpretation_text = Column(String)  # Full LLM narrative
    key_themes = Column(JSON)  # Extracted themes/keywords
    confidence_score = Column(Float)  # LLM confidence

    # Metadata
    llm_model = Column(String)  # Which model generated this
    prompt_version = Column(String)  # For A/B testing prompts
    generated_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    chart_calculation = relationship("ChartCalculation", back_populates="interpretations")
    feedback = relationship("InterpretationFeedback", back_populates="interpretation", cascade="all, delete-orphan")


class InterpretationFeedback(Base):
    """User feedback on chart interpretations"""
    __tablename__ = "interpretation_feedback"

    id = Column(Integer, primary_key=True, index=True)
    interpretation_id = Column(Integer, ForeignKey("chart_interpretations.id"), nullable=False)

    # Feedback data
    accuracy_rating = Column(Integer)  # 1-5 scale
    resonance_rating = Column(Integer)  # 1-5 scale
    user_notes = Column(String)

    # Specific feedback per house system (holographic validation)
    house_system_resonance = Column(JSON)  # {system: rating}

    # Metadata
    submitted_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    interpretation = relationship("ChartInterpretation", back_populates="feedback")
