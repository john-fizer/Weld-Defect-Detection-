"""
Tests for resume parser module.
"""

import pytest
from pathlib import Path
from app.core.resume_parser import ResumeParser, parse_resume


@pytest.fixture
def parser():
    """Create resume parser instance."""
    return ResumeParser()


def test_parser_initialization(parser):
    """Test parser initializes correctly."""
    assert parser is not None
    assert hasattr(parser, "section_headers")


def test_extract_personal_info(parser):
    """Test personal information extraction."""
    sample_text = """
    John Doe
    john.doe@email.com
    (555) 123-4567
    San Francisco, CA
    linkedin.com/in/johndoe
    """

    info = parser._extract_personal_info(sample_text)

    assert info["email"] == "john.doe@email.com"
    assert "555" in info["phone"]
    assert info["linkedin"] is not None
    assert "johndoe" in info["linkedin"]


def test_extract_skills(parser):
    """Test skills extraction."""
    sample_text = """
    Skills:
    Python, JavaScript, React, Node.js, AWS, Docker
    """

    skills = parser._extract_skills(sample_text)

    assert "technical" in skills
    assert "Python" in skills["technical"]
    assert "JavaScript" in skills["technical"]


def test_validation(parser):
    """Test resume data validation."""
    # Valid resume
    valid_resume = {
        "personal_info": {
            "name": "John Doe",
            "email": "john@example.com",
            "phone": "555-1234",
        },
        "work_experience": [
            {"company": "Test Corp", "title": "Engineer", "start_date": "2020-01"}
        ],
        "education": [{"institution": "University", "degree": "BS"}],
        "skills": {"technical": ["Python", "JavaScript"]},
    }

    warnings = parser.validate_parsed_data(valid_resume)
    assert len(warnings) == 0

    # Missing email
    invalid_resume = {
        "personal_info": {"name": "John Doe"},
        "work_experience": [],
        "education": [],
        "skills": {},
    }

    warnings = parser.validate_parsed_data(invalid_resume)
    assert len(warnings) > 0
    assert any("email" in w.lower() for w in warnings)


def test_unsupported_file_type(parser):
    """Test error handling for unsupported file types."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        parser.parse("test.txt")


def test_file_not_found(parser):
    """Test error handling for missing files."""
    with pytest.raises(FileNotFoundError):
        parser.parse("nonexistent_file.pdf")
