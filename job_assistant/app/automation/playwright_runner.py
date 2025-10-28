"""
Playwright-based browser automation for job applications.
Includes human-like behavior and mandatory approval gates.
"""

from typing import Dict, Any, Optional, List
from playwright.async_api import async_playwright, Page, Browser
from loguru import logger
import asyncio
import random
from pathlib import Path

from app.core.config import settings
from app.core.ethics import EthicsGuardrail, ConsentError


class ApprovalGate:
    """
    Mandatory user approval checkpoint before submission.
    Displays form preview and requires explicit consent.
    """

    def __init__(self):
        self.pending_approvals = {}

    async def request_approval(
        self,
        application_id: str,
        form_preview: Dict[str, Any],
        screenshot_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Request user approval for application submission.

        Args:
            application_id: Application ID
            form_preview: Preview of filled form data
            screenshot_path: Path to form screenshot

        Returns:
            Approval request data
        """
        approval_request = {
            "application_id": application_id,
            "form_preview": form_preview,
            "screenshot_path": screenshot_path,
            "status": "pending",
            "requested_at": asyncio.get_event_loop().time(),
        }

        self.pending_approvals[application_id] = approval_request
        logger.info(f"Approval requested for application {application_id}")

        return approval_request

    async def wait_for_approval(
        self, application_id: str, timeout_seconds: int = 3600
    ) -> bool:
        """
        Wait for user approval (blocking).

        Args:
            application_id: Application ID
            timeout_seconds: Max wait time

        Returns:
            True if approved, False if rejected/timeout

        Raises:
            ConsentError: If not approved
        """
        start_time = asyncio.get_event_loop().time()

        while asyncio.get_event_loop().time() - start_time < timeout_seconds:
            approval = self.pending_approvals.get(application_id)

            if not approval:
                raise ConsentError(f"Approval request not found: {application_id}")

            if approval["status"] == "approved":
                logger.info(f"Application {application_id} approved by user")
                return True

            elif approval["status"] == "rejected":
                logger.info(f"Application {application_id} rejected by user")
                raise ConsentError("User rejected application submission")

            # Wait before checking again
            await asyncio.sleep(1)

        # Timeout
        logger.error(f"Approval timeout for application {application_id}")
        raise ConsentError("Approval timeout - user did not respond")

    def approve_application(self, application_id: str, user_modifications: Optional[Dict] = None):
        """Mark application as approved by user."""
        if application_id in self.pending_approvals:
            self.pending_approvals[application_id]["status"] = "approved"
            self.pending_approvals[application_id]["user_modifications"] = user_modifications
            logger.info(f"Application {application_id} approved")

    def reject_application(self, application_id: str, reason: Optional[str] = None):
        """Mark application as rejected by user."""
        if application_id in self.pending_approvals:
            self.pending_approvals[application_id]["status"] = "rejected"
            self.pending_approvals[application_id]["rejection_reason"] = reason
            logger.info(f"Application {application_id} rejected: {reason}")


class PlaywrightAutomation:
    """
    Playwright-based job application automation.
    Implements human-like behavior and ethics guardrails.
    """

    def __init__(self):
        self.browser: Optional[Browser] = None
        self.approval_gate = ApprovalGate()
        self.ethics_guard = EthicsGuardrail()

    async def initialize(self):
        """Initialize Playwright browser."""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=settings.PLAYWRIGHT_HEADLESS,
            slow_mo=settings.PLAYWRIGHT_SLOW_MO,
        )
        logger.info("Playwright browser initialized")

    async def close(self):
        """Close browser."""
        if self.browser:
            await self.browser.close()
            logger.info("Playwright browser closed")

    async def fill_application_form(
        self,
        application_id: str,
        url: str,
        form_data: Dict[str, Any],
        resume_path: str,
        require_approval: bool = True,
    ) -> Dict[str, Any]:
        """
        Navigate to application page and fill form.

        Args:
            application_id: Application ID
            url: Application URL
            form_data: Data to fill
            resume_path: Path to resume file
            require_approval: Whether to require user approval (always True in production)

        Returns:
            Result dictionary with status and metadata
        """
        if not self.browser:
            await self.initialize()

        # Force approval requirement in production
        if not settings.DEBUG and not require_approval:
            logger.error("Attempting to bypass approval gate in production!")
            raise ConsentError("User approval is mandatory in production")

        context = await self.browser.new_context(
            user_agent=settings.USER_AGENT,
            viewport={"width": 1920, "height": 1080},
        )
        page = await context.new_page()

        try:
            # Navigate to application page
            logger.info(f"Navigating to {url}")
            await page.goto(url, timeout=settings.PLAYWRIGHT_TIMEOUT)
            await self._human_delay()

            # Detect ATS system
            ats_system = await self._detect_ats_system(page)
            logger.info(f"Detected ATS system: {ats_system}")

            # Fill form based on ATS system
            if ats_system == "greenhouse":
                filled_data = await self._fill_greenhouse_form(page, form_data, resume_path)
            elif ats_system == "lever":
                filled_data = await self._fill_lever_form(page, form_data, resume_path)
            elif ats_system == "workday":
                filled_data = await self._fill_workday_form(page, form_data, resume_path)
            else:
                filled_data = await self._fill_generic_form(page, form_data, resume_path)

            # Take screenshot for user review
            screenshot_dir = Path("./data/screenshots")
            screenshot_dir.mkdir(parents=True, exist_ok=True)
            screenshot_path = screenshot_dir / f"{application_id}.png"
            await page.screenshot(path=str(screenshot_path), full_page=True)

            # **MANDATORY APPROVAL GATE**
            if require_approval and settings.REQUIRE_USER_APPROVAL:
                logger.info("Requesting user approval before submission...")

                # Create approval request
                await self.approval_gate.request_approval(
                    application_id=application_id,
                    form_preview=filled_data,
                    screenshot_path=str(screenshot_path),
                )

                # Wait for approval (this blocks until user approves/rejects)
                await self.approval_gate.wait_for_approval(application_id)

                logger.info("User approval received, proceeding with submission")

            # Find and click submit button
            submit_button = await self._find_submit_button(page)

            if submit_button:
                # Final confirmation in logs
                logger.warning(f"SUBMITTING APPLICATION {application_id} - USER APPROVED")

                # Click submit
                await submit_button.click()
                await self._human_delay(2000, 4000)

                # Capture confirmation
                confirmation = await self._capture_confirmation(page)

                return {
                    "status": "submitted",
                    "ats_system": ats_system,
                    "filled_data": filled_data,
                    "confirmation": confirmation,
                    "screenshot_path": str(screenshot_path),
                }
            else:
                logger.error("Submit button not found")
                return {
                    "status": "error",
                    "error": "Submit button not found",
                    "filled_data": filled_data,
                    "screenshot_path": str(screenshot_path),
                }

        except Exception as e:
            logger.error(f"Application automation failed: {e}")

            # Take error screenshot
            screenshot_path = Path("./data/screenshots") / f"{application_id}_error.png"
            await page.screenshot(path=str(screenshot_path))

            return {
                "status": "error",
                "error": str(e),
                "screenshot_path": str(screenshot_path),
            }

        finally:
            await context.close()

    async def _detect_ats_system(self, page: Page) -> str:
        """
        Detect which ATS system is being used.

        Args:
            page: Playwright page

        Returns:
            ATS system name
        """
        url = page.url.lower()

        if "greenhouse" in url or "boards.greenhouse.io" in url:
            return "greenhouse"
        elif "lever.co" in url or "jobs.lever.co" in url:
            return "lever"
        elif "workday" in url or "myworkdayjobs.com" in url:
            return "workday"
        elif "taleo" in url:
            return "taleo"
        elif "icims.com" in url:
            return "icims"
        else:
            # Check for meta tags or page content
            content = await page.content()
            if "greenhouse" in content.lower():
                return "greenhouse"
            elif "lever" in content.lower():
                return "lever"

        return "generic"

    async def _fill_greenhouse_form(
        self, page: Page, form_data: Dict[str, Any], resume_path: str
    ) -> Dict[str, Any]:
        """Fill Greenhouse ATS form."""
        filled = {}

        # Common Greenhouse selectors
        selectors = {
            "first_name": "#first_name",
            "last_name": "#last_name",
            "email": "#email",
            "phone": "#phone",
            "resume": "input[type=file]",
        }

        # Fill text fields
        for field, selector in selectors.items():
            if field == "resume":
                continue
            value = form_data.get(field, "")
            if value:
                try:
                    await page.fill(selector, str(value))
                    filled[field] = value
                    await self._human_delay()
                except Exception as e:
                    logger.warning(f"Could not fill {field}: {e}")

        # Upload resume
        try:
            await page.set_input_files(selectors["resume"], resume_path)
            filled["resume"] = resume_path
            await self._human_delay()
        except Exception as e:
            logger.warning(f"Could not upload resume: {e}")

        return filled

    async def _fill_lever_form(
        self, page: Page, form_data: Dict[str, Any], resume_path: str
    ) -> Dict[str, Any]:
        """Fill Lever ATS form."""
        # Similar implementation for Lever
        logger.info("Filling Lever form (placeholder)")
        return {}

    async def _fill_workday_form(
        self, page: Page, form_data: Dict[str, Any], resume_path: str
    ) -> Dict[str, Any]:
        """Fill Workday ATS form."""
        # Similar implementation for Workday
        logger.info("Filling Workday form (placeholder)")
        return {}

    async def _fill_generic_form(
        self, page: Page, form_data: Dict[str, Any], resume_path: str
    ) -> Dict[str, Any]:
        """Fill generic application form using AI field detection."""
        filled = {}

        # Find all input fields
        inputs = await page.query_selector_all("input[type=text], input[type=email], input[type=tel]")

        for input_elem in inputs:
            # Get field attributes
            name = await input_elem.get_attribute("name") or ""
            placeholder = await input_elem.get_attribute("placeholder") or ""
            id_attr = await input_elem.get_attribute("id") or ""

            # Intelligent field matching
            field_text = f"{name} {placeholder} {id_attr}".lower()

            if any(keyword in field_text for keyword in ["first", "fname", "given"]):
                value = form_data.get("first_name", "")
            elif any(keyword in field_text for keyword in ["last", "lname", "surname"]):
                value = form_data.get("last_name", "")
            elif "email" in field_text:
                value = form_data.get("email", "")
            elif any(keyword in field_text for keyword in ["phone", "mobile", "tel"]):
                value = form_data.get("phone", "")
            else:
                continue

            if value:
                try:
                    await input_elem.fill(str(value))
                    filled[field_text[:20]] = value
                    await self._human_delay()
                except Exception as e:
                    logger.warning(f"Could not fill field: {e}")

        # Try to upload resume
        file_inputs = await page.query_selector_all("input[type=file]")
        if file_inputs:
            try:
                await file_inputs[0].set_input_files(resume_path)
                filled["resume"] = resume_path
                await self._human_delay()
            except Exception as e:
                logger.warning(f"Could not upload resume: {e}")

        return filled

    async def _find_submit_button(self, page: Page) -> Optional[Any]:
        """Find the submit button on the page."""
        # Try common submit button selectors
        submit_selectors = [
            "button[type=submit]",
            "input[type=submit]",
            "button:has-text('Submit')",
            "button:has-text('Apply')",
            "a:has-text('Submit Application')",
        ]

        for selector in submit_selectors:
            try:
                button = await page.query_selector(selector)
                if button:
                    return button
            except Exception:
                continue

        return None

    async def _capture_confirmation(self, page: Page) -> Dict[str, Any]:
        """Capture confirmation information after submission."""
        try:
            # Wait for confirmation page
            await page.wait_for_load_state("networkidle", timeout=10000)

            # Look for confirmation text
            content = await page.content()

            confirmation = {
                "url": page.url,
                "title": await page.title(),
                "success": any(
                    keyword in content.lower()
                    for keyword in ["success", "submitted", "received", "thank you"]
                ),
            }

            return confirmation

        except Exception as e:
            logger.error(f"Could not capture confirmation: {e}")
            return {"error": str(e)}

    async def _human_delay(self, min_ms: int = 500, max_ms: int = 2000):
        """Add human-like random delay."""
        delay = random.randint(min_ms, max_ms) / 1000
        await asyncio.sleep(delay)
