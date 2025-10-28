"""
Pydantic schemas for jobs.
"""

from pydantic import BaseModel, Field, HttpUrl
from typing import List, Optional, Dict, Any
from datetime import datetime


class JobSearch(BaseModel):
    """Job search request schema."""
    keywords: List[str]
    location: Optional[str] = None
    remote_only: bool = False
    min_salary: Optional[int] = None
    max_results: int = Field(default=100, le=500)


class JobCreate(BaseModel):
    """Create job posting schema."""
    title: str
    company: str
    location: Optional[str] = None
    remote_type: Optional[str] = None
    application_url: str
    description: str
    requirements: List[str] = Field(default_factory=list)
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    source: str = "manual"


class FitScoreRequest(BaseModel):
    """Fit score calculation request."""
    job_id: str
    user_id: str


class FitScoreResponse(BaseModel):
    """Fit score response schema."""
    job_id: str
    overall_score: float = Field(..., ge=0, le=100)
    breakdown: Dict[str, float]
    llm_analysis: Dict[str, Any]
    recommendation: str  # HIGHLY_RECOMMENDED, RECOMMENDED, CONSIDER, NOT_RECOMMENDED


class JobResponse(BaseModel):
    """Job response schema."""
    id: str
    title: str
    company: str
    location: Optional[str]
    remote_type: Optional[str]
    application_url: str
    description: str
    requirements: List[str]
    salary_min: Optional[int]
    salary_max: Optional[int]
    source: str
    is_active: bool
    created_at: datetime
    fit_score: Optional[float] = None  # Populated when user context available

    class Config:
        from_attributes = True


class JobListResponse(BaseModel):
    """Job list response schema."""
    jobs: List[JobResponse]
    total: int
    page: int
    page_size: int
