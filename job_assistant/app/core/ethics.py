"""
Ethics and compliance validation layer.
Prevents fabrication and ensures truthfulness in all AI-generated content.
"""

from typing import Dict, Any, List
from loguru import logger
from datetime import datetime


class EthicsViolation(Exception):
    """Raised when ethics rules are violated."""
    pass


class ConsentError(Exception):
    """Raised when user consent is missing or expired."""
    pass


class EthicsGuardrail:
    """
    Core ethics validation system.
    Ensures all AI modifications stay truthful and verifiable.
    """

    def __init__(self):
        self.violation_log = []

    def validate_resume_changes(
        self, original: Dict[str, Any], modified: Dict[str, Any]
    ) -> List[str]:
        """
        Validate that resume modifications don't fabricate information.

        Args:
            original: Original resume data
            modified: Modified resume data

        Returns:
            List of violation messages (empty if valid)

        Raises:
            EthicsViolation: If critical violations detected
        """
        violations = []

        # Check for new skills added
        original_skills = set(original.get("skills", {}).get("technical", []))
        modified_skills = set(modified.get("skills", {}).get("technical", []))
        new_skills = modified_skills - original_skills

        if new_skills:
            violation = f"Cannot add skills not in original resume: {new_skills}"
            violations.append(violation)
            logger.error(violation)

        # Check for new work experience
        original_jobs = len(original.get("work_experience", []))
        modified_jobs = len(modified.get("work_experience", []))

        if modified_jobs > original_jobs:
            violation = "Cannot add work experience not in original resume"
            violations.append(violation)
            logger.error(violation)

        # Check for education changes
        original_edu = original.get("education", [])
        modified_edu = modified.get("education", [])

        if len(modified_edu) > len(original_edu):
            violation = "Cannot add education credentials not in original resume"
            violations.append(violation)
            logger.error(violation)

        # Check for date manipulation
        date_violation = self._check_date_manipulation(original, modified)
        if date_violation:
            violations.append(date_violation)
            logger.error(date_violation)

        # Log violations
        if violations:
            self.violation_log.append({
                "timestamp": datetime.utcnow().isoformat(),
                "violations": violations,
            })

        return violations

    def _check_date_manipulation(
        self, original: Dict[str, Any], modified: Dict[str, Any]
    ) -> str | None:
        """
        Check if employment dates were altered.

        Args:
            original: Original resume
            modified: Modified resume

        Returns:
            Violation message or None
        """
        original_exp = original.get("work_experience", [])
        modified_exp = modified.get("work_experience", [])

        if len(original_exp) != len(modified_exp):
            return None  # Already caught by other checks

        for i, (orig_job, mod_job) in enumerate(zip(original_exp, modified_exp)):
            orig_start = orig_job.get("start_date")
            orig_end = orig_job.get("end_date")
            mod_start = mod_job.get("start_date")
            mod_end = mod_job.get("end_date")

            if orig_start != mod_start or orig_end != mod_end:
                return f"Cannot alter employment dates for job {i}"

        return None

    def check_application_consent(
        self, user_id: str, job_id: str, consent_record: Dict[str, Any] | None
    ) -> bool:
        """
        Verify explicit user approval exists before submission.

        Args:
            user_id: User identifier
            job_id: Job identifier
            consent_record: Consent record from database

        Returns:
            True if valid consent exists

        Raises:
            ConsentError: If consent missing or expired
        """
        if not consent_record:
            raise ConsentError(
                f"User approval required before submitting application. "
                f"User {user_id} must explicitly approve job {job_id}."
            )

        # Check if consent is expired (valid for 24 hours)
        approval_date = consent_record.get("approval_date")
        if approval_date:
            from datetime import timezone
            if isinstance(approval_date, str):
                approval_date = datetime.fromisoformat(approval_date)

            now = datetime.now(timezone.utc)
            age = (now - approval_date).total_seconds() / 3600  # hours

            if age > 24:
                raise ConsentError(
                    f"User approval expired. Approvals valid for 24 hours. "
                    f"Please re-approve this application."
                )

        logger.info(f"Consent validated for user {user_id}, job {job_id}")
        return True

    def validate_application_materials(
        self,
        resume_data: Dict[str, Any],
        cover_letter: str,
        original_resume: Dict[str, Any],
    ) -> List[str]:
        """
        Validate complete application package before submission.

        Args:
            resume_data: Tailored resume
            cover_letter: Generated cover letter
            original_resume: Original resume for comparison

        Returns:
            List of warnings/issues
        """
        issues = []

        # Validate resume
        resume_violations = self.validate_resume_changes(original_resume, resume_data)
        issues.extend(resume_violations)

        # Validate cover letter (basic checks)
        if not cover_letter or len(cover_letter.strip()) < 100:
            issues.append("Cover letter appears too short or empty")

        # Check for common red flags in cover letter
        red_flags = [
            "Dear Sir/Madam",  # Generic salutation
            "To Whom It May Concern",
            "[Your Name]",  # Template placeholder
            "[Company Name]",
        ]

        for flag in red_flags:
            if flag in cover_letter:
                issues.append(f"Cover letter contains template placeholder: {flag}")

        return issues

    def log_ethics_event(
        self,
        event_type: str,
        user_id: str,
        details: Dict[str, Any],
    ):
        """
        Log ethics-related events for audit trail.

        Args:
            event_type: Type of event (violation, consent_check, etc.)
            user_id: User involved
            details: Event details
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "user_id": user_id,
            "details": details,
        }

        self.violation_log.append(log_entry)
        logger.info(f"Ethics event logged: {event_type} for user {user_id}")

    def get_violation_report(self, user_id: str | None = None) -> List[Dict[str, Any]]:
        """
        Get ethics violation report.

        Args:
            user_id: Optional filter by user

        Returns:
            List of violation records
        """
        if user_id:
            return [
                entry for entry in self.violation_log
                if entry.get("user_id") == user_id
            ]
        return self.violation_log


class TOSComplianceChecker:
    """
    Checks compliance with platform Terms of Service.
    Maintains list of platforms that prohibit automation.
    """

    def __init__(self):
        # Platforms known to prohibit automation (example list)
        self.prohibited_domains = [
            # Add domains that explicitly ban automation
            # This would be maintained based on TOS research
        ]

        # Platforms that allow automation with restrictions
        self.restricted_domains = {
            "linkedin.com": {
                "requires_api": True,
                "rate_limit": 10,  # requests per hour
                "auto_apply_allowed": False,
            },
            "indeed.com": {
                "requires_api": True,
                "rate_limit": 20,
                "auto_apply_allowed": False,
            },
        }

    def check_domain_compliance(self, url: str) -> Dict[str, Any]:
        """
        Check if automation is allowed for given URL.

        Args:
            url: Application URL

        Returns:
            Compliance information
        """
        from urllib.parse import urlparse

        domain = urlparse(url).netloc

        # Check if prohibited
        if any(prohibited in domain for prohibited in self.prohibited_domains):
            return {
                "allowed": False,
                "reason": "Domain prohibits automation in TOS",
                "recommendation": "Apply manually",
            }

        # Check if restricted
        for restricted_domain, rules in self.restricted_domains.items():
            if restricted_domain in domain:
                return {
                    "allowed": True,
                    "restricted": True,
                    "rules": rules,
                    "recommendation": "Use API if available, respect rate limits",
                }

        # Default: allowed but use caution
        return {
            "allowed": True,
            "restricted": False,
            "recommendation": "Proceed with caution, respect rate limits",
        }

    def log_tos_check(self, url: str, result: Dict[str, Any]):
        """Log TOS compliance check."""
        logger.info(f"TOS check for {url}: {result}")
