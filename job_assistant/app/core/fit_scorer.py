"""
Fit scoring engine using embeddings and LLM analysis.
Ranks job opportunities by match quality.
"""

from typing import Dict, Any, List, Optional
import numpy as np
from loguru import logger

from app.core.config import settings


class FitScorer:
    """
    AI-powered job fit scoring system.
    Combines embedding similarity, keyword matching, and LLM analysis.
    """

    def __init__(self):
        self.embedding_model = None
        self.llm_client = None
        self._initialize_models()

    def _initialize_models(self):
        """Initialize embedding model and LLM client."""
        # Initialize embedding model
        if settings.EMBEDDING_PROVIDER == "sentence-transformers":
            from sentence_transformers import SentenceTransformer

            try:
                self.embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
                logger.info(f"Loaded embedding model: {settings.EMBEDDING_MODEL}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")

        # Initialize LLM client
        if settings.LLM_PROVIDER == "anthropic" and settings.ANTHROPIC_API_KEY:
            from anthropic import AsyncAnthropic

            self.llm_client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
            logger.info(f"Initialized Anthropic client with model: {settings.LLM_MODEL}")
        elif settings.LLM_PROVIDER == "openai" and settings.OPENAI_API_KEY:
            from openai import AsyncOpenAI

            self.llm_client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
            logger.info(f"Initialized OpenAI client with model: {settings.LLM_MODEL}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not self.embedding_model:
            logger.warning("Embedding model not initialized")
            return []

        try:
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First vector
            vec2: Second vector

        Returns:
            Similarity score (0-1)
        """
        if not vec1 or not vec2:
            return 0.0

        vec1_np = np.array(vec1)
        vec2_np = np.array(vec2)

        dot_product = np.dot(vec1_np, vec2_np)
        norm1 = np.linalg.norm(vec1_np)
        norm2 = np.linalg.norm(vec2_np)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))

    def calculate_keyword_match(
        self, resume_skills: List[str], job_requirements: List[str]
    ) -> float:
        """
        Calculate keyword match score.

        Args:
            resume_skills: List of skills from resume
            job_requirements: List of required skills from job

        Returns:
            Match score (0-100)
        """
        if not job_requirements:
            return 0.0

        # Normalize to lowercase for comparison
        resume_skills_lower = set(skill.lower() for skill in resume_skills)
        job_requirements_lower = [req.lower() for req in job_requirements]

        matches = sum(
            1 for req in job_requirements_lower if req in resume_skills_lower
        )
        match_score = (matches / len(job_requirements)) * 100

        return match_score

    def calculate_experience_alignment(
        self, user_years: int, required_years_min: Optional[int], required_years_max: Optional[int]
    ) -> float:
        """
        Calculate experience level alignment.

        Args:
            user_years: User's years of experience
            required_years_min: Minimum required years
            required_years_max: Maximum required years

        Returns:
            Alignment score (0-100)
        """
        if required_years_min is None and required_years_max is None:
            return 50.0  # Neutral score if no requirements specified

        if required_years_min and user_years < required_years_min:
            # Under-qualified
            deficit = required_years_min - user_years
            return max(0, 100 - (deficit * 20))  # Penalty for each year short

        if required_years_max and user_years > required_years_max:
            # Over-qualified
            excess = user_years - required_years_max
            return max(0, 100 - (excess * 10))  # Smaller penalty for over-qualified

        # Within range
        return 100.0

    async def llm_analyze_fit(
        self, resume_summary: str, job_description: str
    ) -> Dict[str, Any]:
        """
        Use LLM to analyze job fit with nuanced reasoning.

        Args:
            resume_summary: Summary of candidate's resume
            job_description: Job description text

        Returns:
            Dictionary with qualitative score and analysis
        """
        if not self.llm_client:
            logger.warning("LLM client not initialized")
            return {"score": 50.0, "analysis": "LLM analysis not available"}

        prompt = f"""You are a professional career coach analyzing job fit.

Candidate Profile:
{resume_summary}

Job Description:
{job_description}

Analyze the fit between this candidate and job on a scale of 0-100. Consider:
- Transferable skills not explicitly listed
- Career trajectory and growth potential
- Cultural fit indicators
- Red flags (unrealistic requirements, mismatch)

Respond in JSON format:
{{
    "score": <0-100>,
    "strengths": ["strength 1", "strength 2"],
    "gaps": ["gap 1", "gap 2"],
    "recommendations": ["suggestion 1", "suggestion 2"],
    "red_flags": ["flag 1"] or [],
    "summary": "Brief overall assessment"
}}
"""

        try:
            if settings.LLM_PROVIDER == "anthropic":
                response = await self.llm_client.messages.create(
                    model=settings.LLM_MODEL,
                    max_tokens=1024,
                    temperature=0.3,
                    messages=[{"role": "user", "content": prompt}],
                )
                content = response.content[0].text

                # Parse JSON response
                import json
                analysis = json.loads(content)
                return analysis

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return {
                "score": 50.0,
                "analysis": f"Analysis failed: {str(e)}",
                "strengths": [],
                "gaps": [],
                "recommendations": [],
                "red_flags": [],
            }

    async def calculate_fit_score(
        self, resume_data: Dict[str, Any], job_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive fit score using multiple methods.

        Args:
            resume_data: Parsed resume data
            job_data: Job posting data

        Returns:
            Dictionary with overall score and breakdown
        """
        # Generate embeddings
        resume_text = self._prepare_resume_text(resume_data)
        job_text = job_data.get("description", "")

        resume_embedding = self.generate_embedding(resume_text)
        job_embedding = self.generate_embedding(job_text)

        # Calculate component scores
        embedding_similarity = self.cosine_similarity(resume_embedding, job_embedding) * 100

        keyword_score = self.calculate_keyword_match(
            resume_data.get("skills", {}).get("technical", []),
            job_data.get("requirements", []),
        )

        # Calculate experience alignment
        user_years = self._calculate_years_experience(resume_data)
        experience_score = self.calculate_experience_alignment(
            user_years,
            job_data.get("years_experience_min"),
            job_data.get("years_experience_max"),
        )

        # LLM qualitative analysis
        resume_summary = self._prepare_resume_summary(resume_data)
        llm_analysis = await self.llm_analyze_fit(resume_summary, job_text)
        llm_score = llm_analysis.get("score", 50.0)

        # Weighted final score
        final_score = (
            embedding_similarity * 0.40
            + keyword_score * 0.30
            + experience_score * 0.20
            + llm_score * 0.10
        )

        return {
            "overall_score": round(final_score, 2),
            "breakdown": {
                "embedding_similarity": round(embedding_similarity, 2),
                "keyword_match": round(keyword_score, 2),
                "experience_alignment": round(experience_score, 2),
                "llm_qualitative": round(llm_score, 2),
            },
            "llm_analysis": llm_analysis,
            "recommendation": self._get_recommendation(final_score),
        }

    def _prepare_resume_text(self, resume_data: Dict[str, Any]) -> str:
        """Prepare resume text for embedding."""
        parts = []

        # Add skills
        skills = resume_data.get("skills", {})
        if skills.get("technical"):
            parts.append("Skills: " + ", ".join(skills["technical"]))

        # Add work experience titles
        experience = resume_data.get("work_experience", [])
        for job in experience:
            if job.get("title"):
                parts.append(job["title"])

        # Add education
        education = resume_data.get("education", [])
        for edu in education:
            if edu.get("degree"):
                parts.append(edu["degree"])

        return " | ".join(parts)

    def _prepare_resume_summary(self, resume_data: Dict[str, Any]) -> str:
        """Prepare resume summary for LLM analysis."""
        summary_parts = []

        # Personal info
        personal = resume_data.get("personal_info", {})
        if personal.get("name"):
            summary_parts.append(f"Name: {personal['name']}")

        # Skills
        skills = resume_data.get("skills", {})
        if skills.get("technical"):
            summary_parts.append(f"Technical Skills: {', '.join(skills['technical'][:10])}")

        # Experience summary
        experience = resume_data.get("work_experience", [])
        years = self._calculate_years_experience(resume_data)
        summary_parts.append(f"Years of Experience: {years}")

        return "\n".join(summary_parts)

    def _calculate_years_experience(self, resume_data: Dict[str, Any]) -> int:
        """Calculate total years of experience from resume."""
        # Simplified calculation for MVP
        experience = resume_data.get("work_experience", [])
        return len(experience)  # Placeholder: count number of jobs

    def _get_recommendation(self, score: float) -> str:
        """Get application recommendation based on score."""
        if score >= 80:
            return "HIGHLY_RECOMMENDED"
        elif score >= 60:
            return "RECOMMENDED"
        elif score >= 40:
            return "CONSIDER"
        else:
            return "NOT_RECOMMENDED"
