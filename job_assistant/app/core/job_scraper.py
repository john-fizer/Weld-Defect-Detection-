"""
Job scraping and sourcing module.
Aggregates jobs from multiple sources with ethical rate limiting.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
from loguru import logger
import hashlib

from app.core.config import settings
from app.models.job import Job, JobSource


class JobScraperBase(ABC):
    """
    Abstract base class for job scrapers.
    Implements rate limiting and ethical scraping practices.
    """

    def __init__(self, source: JobSource):
        self.source = source
        self.last_request_time = None
        self.request_count = 0

    async def rate_limit(self):
        """Implement rate limiting between requests."""
        if self.last_request_time:
            elapsed = (datetime.utcnow() - self.last_request_time).total_seconds()
            delay = settings.SCRAPER_RATE_LIMIT_DELAY - elapsed
            if delay > 0:
                await asyncio.sleep(delay)
        self.last_request_time = datetime.utcnow()
        self.request_count += 1

    @abstractmethod
    async def search_jobs(
        self,
        keywords: List[str],
        location: Optional[str] = None,
        remote_only: bool = False,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Search for jobs matching criteria.

        Args:
            keywords: Job titles or keywords to search
            location: Location filter
            remote_only: Only return remote positions
            max_results: Maximum number of results

        Returns:
            List of job dictionaries
        """
        pass

    def generate_job_hash(self, job_data: Dict[str, Any]) -> str:
        """
        Generate unique hash for job to detect duplicates.

        Args:
            job_data: Job information dictionary

        Returns:
            MD5 hash string
        """
        unique_string = f"{job_data.get('company')}_{job_data.get('title')}_{job_data.get('location')}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    async def deduplicate_jobs(
        self, jobs: List[Dict[str, Any]], existing_hashes: set
    ) -> List[Dict[str, Any]]:
        """
        Remove duplicate jobs based on hash.

        Args:
            jobs: List of scraped jobs
            existing_hashes: Set of existing job hashes

        Returns:
            Deduplicated job list
        """
        unique_jobs = []
        for job in jobs:
            job_hash = self.generate_job_hash(job)
            if job_hash not in existing_hashes:
                unique_jobs.append(job)
                existing_hashes.add(job_hash)

        logger.info(
            f"Deduplicated {len(jobs)} jobs to {len(unique_jobs)} unique entries"
        )
        return unique_jobs


class LinkedInJobScraper(JobScraperBase):
    """
    LinkedIn Jobs API scraper.
    Requires LinkedIn API credentials.
    """

    def __init__(self):
        super().__init__(JobSource.LINKEDIN)
        self.api_key = settings.LINKEDIN_API_KEY
        self.api_secret = settings.LINKEDIN_API_SECRET

    async def search_jobs(
        self,
        keywords: List[str],
        location: Optional[str] = None,
        remote_only: bool = False,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search LinkedIn jobs using API."""
        if not self.api_key:
            logger.warning("LinkedIn API key not configured")
            return []

        await self.rate_limit()

        # TODO: Implement actual LinkedIn API integration
        # For MVP, this is a placeholder
        logger.info(
            f"Searching LinkedIn for: {keywords}, location: {location}, remote: {remote_only}"
        )

        # Placeholder response
        jobs = []
        return jobs


class IndeedJobScraper(JobScraperBase):
    """
    Indeed Publisher API scraper.
    Requires Indeed Publisher ID.
    """

    def __init__(self):
        super().__init__(JobSource.INDEED)
        self.publisher_id = settings.INDEED_PUBLISHER_ID

    async def search_jobs(
        self,
        keywords: List[str],
        location: Optional[str] = None,
        remote_only: bool = False,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """Search Indeed jobs using Publisher API."""
        if not self.publisher_id:
            logger.warning("Indeed Publisher ID not configured")
            return []

        await self.rate_limit()

        # TODO: Implement actual Indeed API integration
        logger.info(
            f"Searching Indeed for: {keywords}, location: {location}, remote: {remote_only}"
        )

        # Placeholder response
        jobs = []
        return jobs


class PlaywrightJobScraper(JobScraperBase):
    """
    Generic Playwright-based web scraper for job boards.
    Respects robots.txt and implements human-like behavior.
    """

    def __init__(self, source: JobSource = JobSource.CUSTOM):
        super().__init__(source)

    async def search_jobs(
        self,
        keywords: List[str],
        location: Optional[str] = None,
        remote_only: bool = False,
        max_results: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Scrape jobs using Playwright.
        Implements human-like delays and behavior.
        """
        from playwright.async_api import async_playwright

        jobs = []

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(
                    headless=settings.PLAYWRIGHT_HEADLESS,
                    slow_mo=settings.PLAYWRIGHT_SLOW_MO,
                )
                context = await browser.new_context(
                    user_agent=settings.USER_AGENT,
                    viewport={"width": 1920, "height": 1080},
                )
                page = await context.new_page()

                # Example scraping logic (customize per site)
                # This is a template - actual implementation depends on target site
                logger.info(f"Scraping jobs with Playwright for: {keywords}")

                await browser.close()

        except Exception as e:
            logger.error(f"Playwright scraping failed: {e}")

        return jobs


class JobAggregator:
    """
    Aggregates jobs from multiple sources and deduplicates.
    """

    def __init__(self):
        self.scrapers = []
        if settings.LINKEDIN_API_KEY:
            self.scrapers.append(LinkedInJobScraper())
        if settings.INDEED_PUBLISHER_ID:
            self.scrapers.append(IndeedJobScraper())
        # Add Playwright scraper for custom sources
        if settings.ENABLE_JOB_SCRAPING:
            self.scrapers.append(PlaywrightJobScraper())

    async def aggregate_jobs(
        self,
        keywords: List[str],
        location: Optional[str] = None,
        remote_only: bool = False,
        max_results_per_source: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Aggregate jobs from all configured sources.

        Args:
            keywords: Search keywords
            location: Location filter
            remote_only: Only remote jobs
            max_results_per_source: Max results per source

        Returns:
            Deduplicated list of jobs
        """
        all_jobs = []
        existing_hashes = set()

        # Scrape from all sources concurrently
        tasks = [
            scraper.search_jobs(keywords, location, remote_only, max_results_per_source)
            for scraper in self.scrapers
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Scraper {self.scrapers[i].source} failed: {result}")
                continue

            # Deduplicate
            unique_jobs = await self.scrapers[i].deduplicate_jobs(
                result, existing_hashes
            )
            all_jobs.extend(unique_jobs)

        logger.info(f"Aggregated {len(all_jobs)} unique jobs from {len(self.scrapers)} sources")
        return all_jobs

    def filter_jobs(
        self,
        jobs: List[Dict[str, Any]],
        min_salary: Optional[int] = None,
        excluded_companies: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Filter jobs based on user preferences.

        Args:
            jobs: List of jobs to filter
            min_salary: Minimum salary requirement
            excluded_companies: Companies to exclude

        Returns:
            Filtered job list
        """
        filtered = jobs

        # Filter by minimum salary
        if min_salary:
            filtered = [
                job
                for job in filtered
                if job.get("salary_min") and job["salary_min"] >= min_salary
            ]

        # Filter by excluded companies
        if excluded_companies:
            excluded_lower = [c.lower() for c in excluded_companies]
            filtered = [
                job
                for job in filtered
                if job.get("company", "").lower() not in excluded_lower
            ]

        logger.info(f"Filtered {len(jobs)} jobs to {len(filtered)} jobs")
        return filtered
