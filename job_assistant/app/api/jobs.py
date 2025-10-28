"""
Job search and management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from app.core.database import get_db
from app.core.job_scraper import JobAggregator
from app.core.fit_scorer import FitScorer
from app.schemas.job import (
    JobSearch,
    JobResponse,
    JobListResponse,
    FitScoreRequest,
    FitScoreResponse,
)
from loguru import logger

router = APIRouter()
job_aggregator = JobAggregator()
fit_scorer = FitScorer()


@router.post("/search", response_model=JobListResponse)
async def search_jobs(
    search_params: JobSearch,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """
    Search for jobs across multiple sources.

    Aggregates results from LinkedIn, Indeed, and custom scrapers.
    Returns ranked results based on relevance.
    """
    try:
        logger.info(f"Searching jobs with keywords: {search_params.keywords}")

        # Aggregate jobs from all sources
        jobs = await job_aggregator.aggregate_jobs(
            keywords=search_params.keywords,
            location=search_params.location,
            remote_only=search_params.remote_only,
            max_results_per_source=search_params.max_results // 3,
        )

        # Filter by user preferences
        filtered_jobs = job_aggregator.filter_jobs(
            jobs,
            min_salary=search_params.min_salary,
        )

        # Pagination
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paginated_jobs = filtered_jobs[start_idx:end_idx]

        # TODO: Convert to JobResponse objects and save to database
        # For MVP, return placeholder response
        return JobListResponse(
            jobs=[],
            total=len(filtered_jobs),
            page=page,
            page_size=page_size,
        )

    except Exception as e:
        logger.error(f"Job search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Job search failed: {str(e)}")


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(
    job_id: str,
    db: AsyncSession = Depends(get_db),
):
    """
    Get job details by ID.
    """
    # TODO: Implement database query
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.post("/score", response_model=FitScoreResponse)
async def calculate_fit_score(
    request: FitScoreRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Calculate fit score for user-job pair.

    Uses embeddings, keyword matching, and LLM analysis.
    Returns comprehensive fit breakdown.
    """
    try:
        logger.info(f"Calculating fit score for user {request.user_id}, job {request.job_id}")

        # TODO: Get user resume and job data from database
        # For MVP, return placeholder
        raise HTTPException(status_code=501, detail="Fit scoring implementation pending database integration")

    except Exception as e:
        logger.error(f"Fit score calculation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Fit score calculation failed: {str(e)}")
