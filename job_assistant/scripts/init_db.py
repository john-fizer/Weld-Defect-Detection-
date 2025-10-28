"""
Database initialization script.
Creates all tables and optionally seeds with sample data.
"""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.core.database import engine, Base
from app.models import User, UserProfile, Job, Application, Interview
from loguru import logger


async def init_database():
    """Initialize database by creating all tables."""
    logger.info("Initializing database...")

    async with engine.begin() as conn:
        # Drop all tables (for development - REMOVE IN PRODUCTION)
        # await conn.run_sync(Base.metadata.drop_all)
        # logger.info("Dropped all existing tables")

        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        logger.info("Created all database tables")

    logger.info("Database initialization complete!")


async def seed_sample_data():
    """
    Seed database with sample data for testing.
    Only use in development!
    """
    logger.info("Seeding sample data...")

    # Add sample data creation here if needed

    logger.info("Sample data seeding complete!")


if __name__ == "__main__":
    # Run initialization
    asyncio.run(init_database())

    # Optionally seed sample data
    # asyncio.run(seed_sample_data())

    logger.info("✓ Database ready!")
