"""
Profile management API endpoints.
"""

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List

from app.core.database import get_db
from app.core.resume_parser import ResumeParser
from app.schemas.profile import ProfileResponse, ProfileCreate, ProfileUpdate, ResumeData
from loguru import logger
import json

router = APIRouter()
resume_parser = ResumeParser()


@router.post("/upload-resume", response_model=ResumeData)
async def upload_resume(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload and parse resume file.

    Supports PDF and DOCX formats.
    Returns structured resume data.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Check file extension
    allowed_extensions = [".pdf", ".docx", ".doc"]
    if not any(file.filename.lower().endswith(ext) for ext in allowed_extensions):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Allowed: {', '.join(allowed_extensions)}",
        )

    try:
        # Save uploaded file temporarily
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        # Parse resume
        logger.info(f"Parsing resume: {file.filename}")
        parsed_data = resume_parser.parse(tmp_path)

        # Validate parsed data
        warnings = resume_parser.validate_parsed_data(parsed_data)
        if warnings:
            logger.warning(f"Resume parsing warnings: {warnings}")

        # Clean up temp file
        Path(tmp_path).unlink()

        # Return structured data
        return ResumeData(**parsed_data)

    except Exception as e:
        logger.error(f"Resume parsing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Resume parsing failed: {str(e)}")


@router.get("/me", response_model=ProfileResponse)
async def get_my_profile(
    user_id: str,  # TODO: Get from auth token
    db: AsyncSession = Depends(get_db),
):
    """
    Get current user's profile.
    """
    # TODO: Implement database query
    raise HTTPException(status_code=501, detail="Not implemented yet")


@router.put("/me", response_model=ProfileResponse)
async def update_my_profile(
    profile_update: ProfileUpdate,
    user_id: str,  # TODO: Get from auth token
    db: AsyncSession = Depends(get_db),
):
    """
    Update current user's profile.
    """
    # TODO: Implement database update
    raise HTTPException(status_code=501, detail="Not implemented yet")
