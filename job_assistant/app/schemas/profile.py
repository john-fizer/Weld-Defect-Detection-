"""
Pydantic schemas for user profiles and resumes.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class PersonalInfo(BaseModel):
    """Personal information schema."""
    name: Optional[str] = None
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    portfolio: Optional[str] = None


class WorkExperience(BaseModel):
    """Work experience schema."""
    company: str
    title: str
    start_date: str  # YYYY-MM format
    end_date: Optional[str] = None  # YYYY-MM or "Present"
    achievements: List[str] = Field(default_factory=list)
    skills_used: List[str] = Field(default_factory=list)


class Education(BaseModel):
    """Education schema."""
    institution: str
    degree: str
    field_of_study: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    gpa: Optional[float] = None


class Skills(BaseModel):
    """Skills schema."""
    technical: List[str] = Field(default_factory=list)
    soft: List[str] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)


class JobPreferences(BaseModel):
    """Job search preferences schema."""
    target_roles: List[str] = Field(default_factory=list)
    target_locations: List[str] = Field(default_factory=list)
    remote_preference: str = Field(default="remote")  # remote, hybrid, onsite, any
    salary_min: Optional[int] = None
    salary_max: Optional[int] = None
    excluded_companies: List[str] = Field(default_factory=list)


class ResumeData(BaseModel):
    """Complete resume data schema."""
    personal_info: PersonalInfo
    work_experience: List[WorkExperience] = Field(default_factory=list)
    education: List[Education] = Field(default_factory=list)
    skills: Skills = Field(default_factory=Skills)
    projects: List[Dict[str, Any]] = Field(default_factory=list)
    certifications: List[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    parsed_at: Optional[datetime] = None


class ProfileCreate(BaseModel):
    """Create profile request schema."""
    full_name: str
    email: EmailStr
    phone: Optional[str] = None
    location: Optional[str] = None
    preferences: Optional[JobPreferences] = None


class ProfileUpdate(BaseModel):
    """Update profile request schema."""
    full_name: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    preferences: Optional[JobPreferences] = None


class ProfileResponse(BaseModel):
    """Profile response schema."""
    id: str
    user_id: str
    full_name: str
    email: str
    phone: Optional[str]
    location: Optional[str]
    resume_data: Optional[ResumeData]
    preferences: Optional[JobPreferences]
    created_at: datetime
    updated_at: Optional[datetime]

    class Config:
        from_attributes = True
