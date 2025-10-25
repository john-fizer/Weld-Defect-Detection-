"""Database initialization and session management."""

import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from .models import Base


# Database URL from environment or default
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///trades.db")

# Create engine
engine = create_engine(
    DATABASE_URL,
    echo=False,  # Set to True for SQL logging
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)

# Session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)
    print(f"Database initialized at {DATABASE_URL}")


def get_session() -> Session:
    """Get database session.

    Returns:
        SQLAlchemy session
    """
    return SessionLocal()


def drop_all():
    """Drop all tables (use with caution)."""
    Base.metadata.drop_all(bind=engine)
    print("All tables dropped")
