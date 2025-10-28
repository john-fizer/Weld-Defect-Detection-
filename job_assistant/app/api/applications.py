"""
Application management API endpoints.
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime

from app.core.database import get_db
from app.core.customizer import ResumeCustomizer, CoverLetterGenerator
from app.automation.playwright_runner import PlaywrightAutomation
from loguru import logger

router = APIRouter()
resume_customizer = ResumeCustomizer()
cover_letter_gen = CoverLetterGenerator()
automation = PlaywrightAutomation()


class ApplicationCreate(BaseModel):
    """Create application request."""
    job_id: str
    user_id: str
    customization_mode: str = "balanced"


class ApplicationApproval(BaseModel):
    """User approval for application submission."""
    application_id: str
    approved: bool
    user_modifications: Optional[dict] = None
    rejection_reason: Optional[str] = None


class ApplicationResponse(BaseModel):
    """Application response schema."""
    id: str
    user_id: str
    job_id: str
    status: str
    fit_score: Optional[float]
    applied_date: Optional[datetime]
    created_at: datetime


@router.post("/create", response_model=ApplicationResponse)
async def create_application(
    request: ApplicationCreate,
    db: AsyncSession = Depends(get_db),
):
    """
    Create a new application (draft status).

    Generates tailored resume and cover letter.
    Does NOT submit - requires user approval.
    """
    try:
        logger.info(f"Creating application for user {request.user_id}, job {request.job_id}")

        # TODO: Get user resume and job data from database

        # TODO: Tailor resume
        # tailored = await resume_customizer.tailor_resume(
        #     original_resume=user_resume,
        #     job_description=job_data,
        #     mode=request.customization_mode,
        # )

        # TODO: Generate cover letter
        # cover_letter = await cover_letter_gen.generate_cover_letter(
        #     resume_data=user_resume,
        #     job_data=job_data,
        # )

        # TODO: Save to database with status="draft"

        raise HTTPException(status_code=501, detail="Implementation pending database integration")

    except Exception as e:
        logger.error(f"Application creation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/approve", response_model=ApplicationResponse)
async def approve_application(
    approval: ApplicationApproval,
    db: AsyncSession = Depends(get_db),
):
    """
    User approval endpoint for application submission.

    **CRITICAL**: This is the mandatory approval gate.
    User reviews tailored materials and explicitly approves/rejects.
    """
    try:
        logger.info(f"Processing approval for application {approval.application_id}")

        if approval.approved:
            # Update approval status
            automation.approval_gate.approve_application(
                approval.application_id,
                approval.user_modifications,
            )

            # TODO: Update database status to "approved"
            # TODO: Trigger automated submission workflow

            logger.info(f"Application {approval.application_id} APPROVED by user")

        else:
            # User rejected
            automation.approval_gate.reject_application(
                approval.application_id,
                approval.rejection_reason,
            )

            # TODO: Update database status to "rejected"

            logger.info(f"Application {approval.application_id} REJECTED by user")

        raise HTTPException(status_code=501, detail="Implementation pending database integration")

    except Exception as e:
        logger.error(f"Approval processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/", response_model=List[ApplicationResponse])
async def list_applications(
    user_id: str,  # TODO: Get from auth
    status: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    List all applications for current user.

    Optional filter by status.
    """
    # TODO: Implement database query
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.patch("/{application_id}/status")
async def update_application_status(
    application_id: str,
    new_status: str,
    notes: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """
    Update application status.

    Used for manual tracking (interview scheduled, rejected, offer, etc.)
    """
    # TODO: Implement database update
    raise HTTPException(status_code=501, detail="Not implemented yet")
