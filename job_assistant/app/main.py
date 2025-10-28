"""
Main FastAPI application for AI Job Application Assistant.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from app.core.config import settings
from app.core.database import engine, Base
# from app.api import profiles, jobs, scoring, applications, analytics


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager for FastAPI application.
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting AI Job Application Assistant API...")

    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    logger.info(f"Database initialized at: {settings.DATABASE_URL}")
    logger.info(f"API running in {'DEBUG' if settings.DEBUG else 'PRODUCTION'} mode")

    yield

    # Shutdown
    logger.info("Shutting down AI Job Application Assistant API...")
    await engine.dispose()


# Create FastAPI app
app = FastAPI(
    title="AI Job Application Assistant",
    description="Ethical AI-powered job search automation with human-in-the-loop approval",
    version="1.0.0-alpha",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=settings.CORS_ALLOW_CREDENTIALS,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """
    Health check endpoint for monitoring.
    """
    return {
        "status": "healthy",
        "version": "1.0.0-alpha",
        "environment": "development" if settings.DEBUG else "production",
    }


# Root endpoint
@app.get("/")
async def root():
    """
    Root endpoint with API information.
    """
    return {
        "message": "AI Job Application Assistant API",
        "version": "1.0.0-alpha",
        "docs": "/docs" if settings.DEBUG else "disabled in production",
        "ethics": {
            "truthfulness_first": True,
            "human_approval_required": settings.REQUIRE_USER_APPROVAL,
            "no_fabrication": True,
            "privacy_focused": True,
        },
    }


# Include routers (commented out until implemented)
# app.include_router(profiles.router, prefix="/api/profiles", tags=["profiles"])
# app.include_router(jobs.router, prefix="/api/jobs", tags=["jobs"])
# app.include_router(scoring.router, prefix="/api/scoring", tags=["scoring"])
# app.include_router(applications.router, prefix="/api/applications", tags=["applications"])
# app.include_router(analytics.router, prefix="/api/analytics", tags=["analytics"])


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
