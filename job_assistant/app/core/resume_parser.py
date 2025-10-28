"""
Resume parsing module for extracting structured data from PDF/DOCX files.
Supports multiple parsing strategies with fallback mechanisms.
"""

from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re
from loguru import logger

# PDF parsing
import pdfplumber
import fitz  # PyMuPDF
from pypdf import PdfReader

# DOCX parsing
from docx import Document

# NLP and text processing
import spacy
from dateutil import parser as date_parser


class ResumeParser:
    """
    Intelligent resume parser supporting PDF and DOCX formats.
    Uses multiple extraction strategies with validation.
    """

    def __init__(self):
        """Initialize resume parser with NLP model."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning(
                "Spacy model 'en_core_web_sm' not found. "
                "Install with: python -m spacy download en_core_web_sm"
            )
            self.nlp = None

        # Common section headers
        self.section_headers = {
            "experience": [
                "experience",
                "work experience",
                "employment",
                "work history",
                "professional experience",
            ],
            "education": ["education", "academic background", "qualifications"],
            "skills": [
                "skills",
                "technical skills",
                "core competencies",
                "expertise",
            ],
            "projects": ["projects", "personal projects", "portfolio"],
            "certifications": [
                "certifications",
                "certificates",
                "licenses",
                "credentials",
            ],
        }

    def parse(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Main parsing method. Automatically detects file type and extracts data.

        Args:
            file_path: Path to resume file (PDF or DOCX)

        Returns:
            Structured resume data dictionary

        Raises:
            ValueError: If file format is unsupported
            FileNotFoundError: If file doesn't exist
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"Resume file not found: {file_path}")

        # Extract text based on file type
        if file_path.suffix.lower() == ".pdf":
            text = self._extract_pdf_text(file_path)
        elif file_path.suffix.lower() in [".docx", ".doc"]:
            text = self._extract_docx_text(file_path)
        else:
            raise ValueError(
                f"Unsupported file format: {file_path.suffix}. "
                "Supported formats: PDF, DOCX"
            )

        # Parse structured data from text
        resume_data = self._parse_text(text)
        resume_data["raw_text"] = text
        resume_data["file_path"] = str(file_path)
        resume_data["file_name"] = file_path.name
        resume_data["parsed_at"] = datetime.utcnow().isoformat()

        return resume_data

    def _extract_pdf_text(self, file_path: Path) -> str:
        """
        Extract text from PDF using multiple strategies.
        Tries pdfplumber first, falls back to PyMuPDF, then pypdf.
        """
        text = ""

        # Strategy 1: pdfplumber (best for tables and layout)
        try:
            with pdfplumber.open(file_path) as pdf:
                text = "\n\n".join(page.extract_text() or "" for page in pdf.pages)
            if text.strip():
                logger.debug(f"Extracted text using pdfplumber: {len(text)} chars")
                return text
        except Exception as e:
            logger.warning(f"pdfplumber extraction failed: {e}")

        # Strategy 2: PyMuPDF (fast and reliable)
        try:
            doc = fitz.open(file_path)
            text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            if text.strip():
                logger.debug(f"Extracted text using PyMuPDF: {len(text)} chars")
                return text
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed: {e}")

        # Strategy 3: pypdf (fallback)
        try:
            reader = PdfReader(file_path)
            text = "\n\n".join(page.extract_text() or "" for page in reader.pages)
            if text.strip():
                logger.debug(f"Extracted text using pypdf: {len(text)} chars")
                return text
        except Exception as e:
            logger.warning(f"pypdf extraction failed: {e}")

        if not text.strip():
            raise ValueError(
                "Could not extract text from PDF. "
                "The file may be image-based (scanned). "
                "Consider using OCR or converting to searchable PDF."
            )

        return text

    def _extract_docx_text(self, file_path: Path) -> str:
        """Extract text from DOCX file."""
        try:
            doc = Document(file_path)
            text = "\n\n".join(paragraph.text for paragraph in doc.paragraphs)
            logger.debug(f"Extracted text from DOCX: {len(text)} chars")
            return text
        except Exception as e:
            raise ValueError(f"Failed to extract text from DOCX: {e}")

    def _parse_text(self, text: str) -> Dict[str, Any]:
        """
        Parse resume text into structured data.
        Uses NLP and regex patterns to extract information.
        """
        resume_data = {
            "personal_info": self._extract_personal_info(text),
            "work_experience": self._extract_work_experience(text),
            "education": self._extract_education(text),
            "skills": self._extract_skills(text),
            "projects": self._extract_projects(text),
            "certifications": self._extract_certifications(text),
        }

        return resume_data

    def _extract_personal_info(self, text: str) -> Dict[str, Optional[str]]:
        """Extract contact information from resume."""
        info = {
            "name": None,
            "email": None,
            "phone": None,
            "location": None,
            "linkedin": None,
            "github": None,
            "portfolio": None,
        }

        # Extract email
        email_pattern = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
        emails = re.findall(email_pattern, text)
        if emails:
            info["email"] = emails[0]

        # Extract phone
        phone_pattern = r"(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}"
        phones = re.findall(phone_pattern, text)
        if phones:
            info["phone"] = "".join(phones[0]) if isinstance(phones[0], tuple) else phones[0]

        # Extract LinkedIn
        linkedin_pattern = r"linkedin\.com/in/[\w-]+"
        linkedin = re.search(linkedin_pattern, text, re.IGNORECASE)
        if linkedin:
            info["linkedin"] = "https://" + linkedin.group(0)

        # Extract GitHub
        github_pattern = r"github\.com/[\w-]+"
        github = re.search(github_pattern, text, re.IGNORECASE)
        if github:
            info["github"] = "https://" + github.group(0)

        # Extract name (usually first line or before email/phone)
        lines = text.split("\n")
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if (
                line
                and len(line.split()) <= 4
                and not any(char.isdigit() for char in line)
                and "@" not in line
            ):
                # Likely a name
                info["name"] = line
                break

        # Extract location (heuristic: common location patterns)
        location_pattern = r"([A-Z][a-z]+,\s*[A-Z]{2}|[A-Z][a-z]+,\s*[A-Z][a-z]+)"
        locations = re.findall(location_pattern, text)
        if locations:
            info["location"] = locations[0]

        return info

    def _extract_work_experience(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract work experience from resume.
        Returns list of jobs with company, title, dates, achievements.
        """
        experiences = []

        # Find experience section
        section_text = self._extract_section(text, "experience")
        if not section_text:
            return experiences

        # Split into individual job entries (heuristic: entries separated by dates)
        # This is a simplified implementation - production would use ML-based parsing
        date_pattern = r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})"
        date_ranges = re.finditer(date_pattern, section_text, re.IGNORECASE)

        # Basic extraction (placeholder for more sophisticated parsing)
        logger.debug("Work experience extraction is simplified in MVP")

        return experiences

    def _extract_education(self, text: str) -> List[Dict[str, Any]]:
        """Extract education history from resume."""
        education = []

        section_text = self._extract_section(text, "education")
        if not section_text:
            return education

        # Extract degree patterns
        degree_patterns = [
            r"(Bachelor|Master|PhD|Doctorate|Associate|B\.S\.|M\.S\.|B\.A\.|M\.A\.)",
            r"(Computer Science|Engineering|Business|Mathematics|Physics)",
        ]

        # Simplified extraction for MVP
        logger.debug("Education extraction is simplified in MVP")

        return education

    def _extract_skills(self, text: str) -> Dict[str, List[str]]:
        """
        Extract skills from resume.
        Categorizes into technical, soft, and certifications.
        """
        skills = {"technical": [], "soft": [], "certifications": []}

        section_text = self._extract_section(text, "skills")
        if not section_text:
            # Try to find skills throughout document
            section_text = text

        # Common technical skills
        tech_skills_keywords = [
            "Python",
            "JavaScript",
            "Java",
            "C++",
            "React",
            "Node.js",
            "SQL",
            "AWS",
            "Docker",
            "Kubernetes",
            "Machine Learning",
            "Data Science",
            "FastAPI",
            "Django",
            "Flask",
            "PostgreSQL",
            "MongoDB",
            "Git",
        ]

        for skill in tech_skills_keywords:
            if re.search(rf"\b{skill}\b", section_text, re.IGNORECASE):
                skills["technical"].append(skill)

        # Remove duplicates
        skills["technical"] = list(set(skills["technical"]))

        return skills

    def _extract_projects(self, text: str) -> List[Dict[str, Any]]:
        """Extract projects from resume."""
        projects = []
        section_text = self._extract_section(text, "projects")

        # Simplified for MVP
        logger.debug("Projects extraction is simplified in MVP")

        return projects

    def _extract_certifications(self, text: str) -> List[Dict[str, Any]]:
        """Extract certifications from resume."""
        certifications = []
        section_text = self._extract_section(text, "certifications")

        # Simplified for MVP
        logger.debug("Certifications extraction is simplified in MVP")

        return certifications

    def _extract_section(self, text: str, section_name: str) -> Optional[str]:
        """
        Extract a specific section from resume text.

        Args:
            text: Full resume text
            section_name: Section to extract (e.g., 'experience', 'education')

        Returns:
            Section text or None if not found
        """
        headers = self.section_headers.get(section_name, [])

        for header in headers:
            # Look for section header (case-insensitive, whole word)
            pattern = rf"(?i)^{header}$"
            matches = re.finditer(pattern, text, re.MULTILINE)

            for match in matches:
                start_pos = match.end()

                # Find next section header
                end_pos = len(text)
                for next_section in self.section_headers.values():
                    for next_header in next_section:
                        next_pattern = rf"(?i)^{next_header}$"
                        next_match = re.search(next_pattern, text[start_pos:], re.MULTILINE)
                        if next_match:
                            end_pos = min(end_pos, start_pos + next_match.start())

                section_text = text[start_pos:end_pos].strip()
                if section_text:
                    return section_text

        return None

    def validate_parsed_data(self, resume_data: Dict[str, Any]) -> List[str]:
        """
        Validate parsed resume data and return list of warnings/issues.

        Args:
            resume_data: Parsed resume data dictionary

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check for missing critical information
        personal_info = resume_data.get("personal_info", {})
        if not personal_info.get("email"):
            warnings.append("Email not found in resume")
        if not personal_info.get("name"):
            warnings.append("Name not clearly identified")
        if not personal_info.get("phone"):
            warnings.append("Phone number not found")

        # Check for empty sections
        if not resume_data.get("work_experience"):
            warnings.append("No work experience found")
        if not resume_data.get("education"):
            warnings.append("No education information found")
        if not resume_data.get("skills", {}).get("technical"):
            warnings.append("No technical skills identified")

        return warnings


# Convenience function
def parse_resume(file_path: str | Path) -> Dict[str, Any]:
    """
    Convenience function to parse a resume file.

    Args:
        file_path: Path to resume file

    Returns:
        Structured resume data

    Example:
        >>> resume_data = parse_resume("resume.pdf")
        >>> print(resume_data["personal_info"]["email"])
    """
    parser = ResumeParser()
    return parser.parse(file_path)
