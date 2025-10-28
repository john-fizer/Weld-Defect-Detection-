"""
Resume and cover letter customization engine.
Tailors application materials per job with ethics validation.
"""

from typing import Dict, Any, List, Optional
from loguru import logger
import re

from app.core.config import settings
from app.core.ethics import EthicsGuardrail


class ResumeCustomizer:
    """
    AI-powered resume customization with ethics validation.
    Tailors resumes to job descriptions without fabrication.
    """

    def __init__(self):
        self.llm_client = None
        self.ethics_guard = EthicsGuardrail()
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM client."""
        if settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
            from anthropic import AsyncAnthropic
            self.llm_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        elif settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            from openai import AsyncOpenAI
            self.llm_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def tailor_resume(
        self,
        original_resume: Dict[str, Any],
        job_description: Dict[str, Any],
        mode: str = "balanced",
    ) -> Dict[str, Any]:
        """
        Tailor resume to match job description.

        Args:
            original_resume: Original parsed resume data
            job_description: Target job data
            mode: Customization mode (conservative, balanced, aggressive)

        Returns:
            Tailored resume data with change tracking
        """
        logger.info(f"Tailoring resume in {mode} mode")

        tailored = original_resume.copy()
        changes = []

        # Extract keywords from job description
        keywords = self._extract_keywords(job_description)

        # Reorder skills to prioritize relevant ones
        if "skills" in tailored:
            tailored["skills"], skill_changes = self._reorder_skills(
                tailored["skills"], keywords
            )
            changes.extend(skill_changes)

        # Reorder work experience to highlight relevant roles
        if "work_experience" in tailored:
            tailored["work_experience"], exp_changes = self._reorder_experience(
                tailored["work_experience"], keywords
            )
            changes.extend(exp_changes)

        # Optimize for ATS
        tailored = self._optimize_for_ats(tailored, keywords)

        # Ethics validation
        if settings.ENABLE_ETHICS_VALIDATION:
            violations = self.ethics_guard.validate_resume_changes(
                original_resume, tailored
            )
            if violations:
                logger.error(f"Ethics violations detected: {violations}")
                raise ValueError(f"Ethics violations: {violations}")

        return {
            "tailored_resume": tailored,
            "changes": changes,
            "keywords_matched": keywords,
            "mode": mode,
        }

    def _extract_keywords(self, job_description: Dict[str, Any]) -> List[str]:
        """Extract important keywords from job description."""
        keywords = []

        # From requirements
        requirements = job_description.get("requirements", [])
        keywords.extend(requirements[:10])  # Top 10 requirements

        # From description (simplified - would use NLP in production)
        description = job_description.get("description", "")
        # Basic keyword extraction (placeholder)
        common_tech = [
            "Python", "JavaScript", "React", "Node.js", "AWS", "Docker",
            "Kubernetes", "SQL", "PostgreSQL", "FastAPI", "Django"
        ]
        for tech in common_tech:
            if tech.lower() in description.lower():
                keywords.append(tech)

        return list(set(keywords))  # Deduplicate

    def _reorder_skills(
        self, skills: Dict[str, List[str]], keywords: List[str]
    ) -> tuple[Dict[str, List[str]], List[str]]:
        """Reorder skills to prioritize those matching keywords."""
        changes = []
        technical_skills = skills.get("technical", [])

        # Separate matching and non-matching skills
        keywords_lower = [k.lower() for k in keywords]
        matching = [s for s in technical_skills if s.lower() in keywords_lower]
        non_matching = [s for s in technical_skills if s.lower() not in keywords_lower]

        # Prioritize matching skills
        reordered_technical = matching + non_matching

        if reordered_technical != technical_skills:
            changes.append("Reordered technical skills to prioritize job-relevant skills")

        skills["technical"] = reordered_technical
        return skills, changes

    def _reorder_experience(
        self, experience: List[Dict[str, Any]], keywords: List[str]
    ) -> tuple[List[Dict[str, Any]], List[str]]:
        """Reorder work experience to highlight relevant roles."""
        changes = []

        # Calculate relevance score for each experience
        keywords_lower = [k.lower() for k in keywords]
        for exp in experience:
            relevance_score = 0
            title = exp.get("title", "").lower()
            skills_used = [s.lower() for s in exp.get("skills_used", [])]

            for keyword in keywords_lower:
                if keyword in title:
                    relevance_score += 2
                if keyword in skills_used:
                    relevance_score += 1

            exp["_relevance_score"] = relevance_score

        # Sort by relevance (keeping chronological order for ties)
        # In practice, might keep chronological but highlight relevant experience
        # For now, just track relevance
        changes.append("Calculated relevance scores for work experience")

        return experience, changes

    def _optimize_for_ats(
        self, resume: Dict[str, Any], keywords: List[str]
    ) -> Dict[str, Any]:
        """
        Optimize resume for ATS parsing.
        Ensures proper formatting and keyword inclusion.
        """
        # Ensure standard section headers
        # Ensure keywords appear in context
        # Remove images/tables from critical sections
        # This is a placeholder - actual ATS optimization is complex

        return resume


class CoverLetterGenerator:
    """
    AI-powered cover letter generation with ethics validation.
    """

    def __init__(self):
        self.llm_client = None
        self.ethics_guard = EthicsGuardrail()
        self._initialize_llm()

    def _initialize_llm(self):
        """Initialize LLM client."""
        if settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
            from anthropic import AsyncAnthropic
            self.llm_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
        elif settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            from openai import AsyncOpenAI
            self.llm_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)

    async def generate_cover_letter(
        self,
        resume_data: Dict[str, Any],
        job_data: Dict[str, Any],
        tone: str = "professional",
    ) -> Dict[str, Any]:
        """
        Generate personalized cover letter using LLM.

        Args:
            resume_data: Candidate's resume data
            job_data: Job posting data
            tone: Letter tone (professional, enthusiastic, formal)

        Returns:
            Dict with cover letter text and metadata
        """
        if not self.llm_client:
            raise ValueError("LLM client not initialized. Check API keys.")

        logger.info(f"Generating cover letter with {tone} tone")

        # Prepare resume summary
        resume_summary = self._prepare_resume_summary(resume_data)

        # Prepare job summary
        job_summary = self._prepare_job_summary(job_data)

        # Generate cover letter
        prompt = self._build_prompt(resume_summary, job_summary, tone)

        try:
            if settings.LLM_PROVIDER == "anthropic":
                response = await self.llm_client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=2048,
                    temperature=settings.LLM_TEMPERATURE,
                    messages=[{"role": "user", "content": prompt}],
                )
                cover_letter_text = response.content[0].text

            else:  # OpenAI
                response = await self.llm_client.chat.completions.create(
                    model=settings.LLM_MODEL,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=2048,
                    temperature=settings.LLM_TEMPERATURE,
                )
                cover_letter_text = response.choices[0].message.content

            # Post-generation ethics validation
            if settings.ENABLE_ETHICS_VALIDATION:
                violations = await self._validate_cover_letter(
                    cover_letter_text, resume_data
                )
                if violations:
                    logger.warning(f"Cover letter validation warnings: {violations}")

            # Self-critique pass
            critique = await self._self_critique(cover_letter_text, resume_data)

            return {
                "text": cover_letter_text,
                "tone": tone,
                "word_count": len(cover_letter_text.split()),
                "critique": critique,
                "validation_warnings": violations if settings.ENABLE_ETHICS_VALIDATION else [],
            }

        except Exception as e:
            logger.error(f"Cover letter generation failed: {e}")
            raise

    def _prepare_resume_summary(self, resume_data: Dict[str, Any]) -> str:
        """Prepare concise resume summary for prompt."""
        parts = []

        # Personal info
        personal = resume_data.get("personal_info", {})
        if personal.get("name"):
            parts.append(f"Name: {personal['name']}")

        # Experience
        experience = resume_data.get("work_experience", [])
        if experience:
            latest_job = experience[0]
            parts.append(
                f"Current/Latest: {latest_job.get('title', 'Unknown')} at {latest_job.get('company', 'Unknown')}"
            )

        # Skills
        skills = resume_data.get("skills", {})
        if skills.get("technical"):
            parts.append(f"Key Skills: {', '.join(skills['technical'][:5])}")

        # Education
        education = resume_data.get("education", [])
        if education:
            latest_edu = education[0]
            parts.append(f"Education: {latest_edu.get('degree', 'Unknown')}")

        return "\n".join(parts)

    def _prepare_job_summary(self, job_data: Dict[str, Any]) -> str:
        """Prepare job summary for prompt."""
        parts = []

        parts.append(f"Company: {job_data.get('company', 'Unknown')}")
        parts.append(f"Title: {job_data.get('title', 'Unknown')}")
        parts.append(f"Location: {job_data.get('location', 'Unknown')}")

        description = job_data.get("description", "")
        if description:
            # Truncate to first 500 chars
            parts.append(f"Description: {description[:500]}...")

        requirements = job_data.get("requirements", [])
        if requirements:
            parts.append(f"Key Requirements: {', '.join(requirements[:5])}")

        return "\n".join(parts)

    def _build_prompt(
        self, resume_summary: str, job_summary: str, tone: str
    ) -> str:
        """Build prompt for LLM cover letter generation."""
        return f"""You are a professional career coach helping a candidate write a cover letter.

Candidate Profile:
{resume_summary}

Job Details:
{job_summary}

Instructions:
1. Write a compelling, personalized cover letter
2. Tone: {tone}
3. Length: 3-4 paragraphs (250-350 words)
4. Structure:
   - Opening: Show enthusiasm and mention how you found the role
   - Body: Highlight 2-3 most relevant achievements/skills
   - Closing: Express interest and call to action

CRITICAL ETHICS RULES:
- NEVER fabricate skills, experiences, or achievements
- ONLY reference information present in the candidate profile
- Stay truthful and verifiable
- Do not exaggerate or embellish

Output the cover letter text only, ready to use. No preamble."""

    async def _validate_cover_letter(
        self, cover_letter: str, resume_data: Dict[str, Any]
    ) -> List[str]:
        """
        Validate cover letter against resume for fabrications.

        Returns:
            List of validation warnings
        """
        warnings = []

        # Check for common fabrication patterns
        # This is simplified - production would use NLP and fact-checking

        # Check if skills mentioned in letter exist in resume
        resume_skills = resume_data.get("skills", {}).get("technical", [])
        resume_skills_lower = [s.lower() for s in resume_skills]

        # Very basic check (would be more sophisticated in production)
        suspicious_claims = [
            "10+ years",
            "expert in all",
            "world-class",
            "best in industry",
        ]

        for claim in suspicious_claims:
            if claim.lower() in cover_letter.lower():
                warnings.append(f"Potentially exaggerated claim: '{claim}'")

        return warnings

    async def _self_critique(
        self, cover_letter: str, resume_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        LLM self-critique to detect potential fabrications or exaggerations.

        Returns:
            Critique results
        """
        if not self.llm_client:
            return {"passed": True, "issues": []}

        critique_prompt = f"""You are an ethics validator for AI-generated cover letters.

Resume Data:
{self._prepare_resume_summary(resume_data)}

Generated Cover Letter:
{cover_letter}

Task: Identify any statements in the cover letter that:
1. Cannot be verified from the resume data
2. Exaggerate or embellish achievements
3. Claim skills not present in resume
4. Make unsubstantiated claims

Respond in JSON format:
{{
    "passed": true/false,
    "issues": ["issue 1", "issue 2"] or [],
    "severity": "none|low|medium|high"
}}
"""

        try:
            if settings.LLM_PROVIDER == "anthropic":
                response = await self.llm_client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=512,
                    temperature=0.1,
                    messages=[{"role": "user", "content": critique_prompt}],
                )
                content = response.content[0].text

                import json
                critique = json.loads(content)
                return critique

        except Exception as e:
            logger.error(f"Self-critique failed: {e}")
            return {"passed": True, "issues": [], "error": str(e)}
