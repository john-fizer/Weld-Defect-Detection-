"""
Streamlit dashboard for AI Job Application Assistant.
MVP version with core functionality.
"""

import streamlit as st
import requests
from pathlib import Path
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Configure page
st.set_page_config(
    page_title="AI Job Application Assistant",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# API configuration
API_BASE_URL = "http://localhost:8000"


# Initialize session state
if "resume_data" not in st.session_state:
    st.session_state.resume_data = None
if "applications" not in st.session_state:
    st.session_state.applications = []
if "jobs" not in st.session_state:
    st.session_state.jobs = []


def main():
    """Main dashboard application."""

    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "📋 Dashboard",
            "👤 Profile & Resume",
            "🔍 Job Search",
            "📝 Applications",
            "📊 Analytics",
            "⚙️ Settings",
        ],
    )

    # Ethics disclaimer
    st.sidebar.markdown("---")
    st.sidebar.info(
        """
        **Ethics First**
        - ✓ Truthfulness Required
        - ✓ User Approval Mandatory
        - ✓ No Fabrication
        - ✓ Privacy Focused
        """
    )

    # Route to pages
    if page == "📋 Dashboard":
        show_dashboard()
    elif page == "👤 Profile & Resume":
        show_profile_page()
    elif page == "🔍 Job Search":
        show_job_search_page()
    elif page == "📝 Applications":
        show_applications_page()
    elif page == "📊 Analytics":
        show_analytics_page()
    elif page == "⚙️ Settings":
        show_settings_page()


def show_dashboard():
    """Main dashboard overview."""
    st.title("🎯 AI Job Application Assistant")
    st.markdown("### Ethical AI-powered job search automation")

    # Key metrics (placeholder data)
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="Applications Submitted",
            value="12",
            delta="3 this week",
        )

    with col2:
        st.metric(
            label="Interview Requests",
            value="4",
            delta="2 new",
            delta_color="normal",
        )

    with col3:
        st.metric(
            label="Response Rate",
            value="33%",
            delta="+8%",
        )

    with col4:
        st.metric(
            label="Avg. Fit Score",
            value="78%",
            delta="+5%",
        )

    # Recent activity
    st.markdown("### 📌 Recent Activity")

    # Placeholder activity data
    activity_data = [
        {"date": "2025-10-27", "action": "Applied", "company": "TechCorp", "role": "Software Engineer"},
        {"date": "2025-10-26", "action": "Interview Scheduled", "company": "StartupXYZ", "role": "ML Engineer"},
        {"date": "2025-10-25", "action": "Applied", "company": "BigTech Inc", "role": "Senior Developer"},
    ]

    for activity in activity_data:
        st.write(f"**{activity['date']}** - {activity['action']}: {activity['role']} at {activity['company']}")

    # Quick actions
    st.markdown("### 🚀 Quick Actions")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔍 Search New Jobs", use_container_width=True):
            st.info("Navigate to 'Job Search' page")

    with col2:
        if st.button("📤 Upload Resume", use_container_width=True):
            st.info("Navigate to 'Profile & Resume' page")

    with col3:
        if st.button("📊 View Analytics", use_container_width=True):
            st.info("Navigate to 'Analytics' page")


def show_profile_page():
    """Profile and resume management page."""
    st.title("👤 Profile & Resume")

    tab1, tab2 = st.tabs(["Resume Upload", "Profile Information"])

    with tab1:
        st.markdown("### 📤 Upload Your Resume")
        st.info("Supported formats: PDF, DOCX")

        uploaded_file = st.file_uploader(
            "Choose your resume file",
            type=["pdf", "docx"],
            help="Upload your resume for parsing and analysis",
        )

        if uploaded_file:
            if st.button("Parse Resume"):
                with st.spinner("Parsing resume..."):
                    try:
                        # Call API to parse resume
                        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                        response = requests.post(
                            f"{API_BASE_URL}/api/profiles/upload-resume",
                            files=files,
                        )

                        if response.status_code == 200:
                            resume_data = response.json()
                            st.session_state.resume_data = resume_data

                            st.success("✅ Resume parsed successfully!")

                            # Display parsed data
                            st.markdown("#### Parsed Information")

                            # Personal info
                            personal = resume_data.get("personal_info", {})
                            st.markdown("**Contact Information:**")
                            st.write(f"- Name: {personal.get('name', 'N/A')}")
                            st.write(f"- Email: {personal.get('email', 'N/A')}")
                            st.write(f"- Phone: {personal.get('phone', 'N/A')}")
                            st.write(f"- Location: {personal.get('location', 'N/A')}")

                            # Skills
                            skills = resume_data.get("skills", {})
                            technical_skills = skills.get("technical", [])
                            if technical_skills:
                                st.markdown("**Technical Skills:**")
                                st.write(", ".join(technical_skills))

                        else:
                            st.error(f"Failed to parse resume: {response.text}")

                    except Exception as e:
                        st.error(f"Error: {str(e)}")

        # Display saved resume data
        if st.session_state.resume_data:
            st.markdown("---")
            st.markdown("### 📋 Current Resume Data")
            st.json(st.session_state.resume_data)

    with tab2:
        st.markdown("### 📝 Profile Information")

        col1, col2 = st.columns(2)

        with col1:
            name = st.text_input("Full Name", value="")
            email = st.text_input("Email", value="")
            phone = st.text_input("Phone", value="")

        with col2:
            location = st.text_input("Location", value="")
            linkedin = st.text_input("LinkedIn URL", value="")
            github = st.text_input("GitHub URL", value="")

        st.markdown("#### Job Preferences")

        target_roles = st.text_area(
            "Target Roles (one per line)",
            value="",
            help="Enter job titles you're interested in",
        )

        col1, col2 = st.columns(2)

        with col1:
            remote_pref = st.selectbox(
                "Remote Preference",
                ["Remote", "Hybrid", "Onsite", "Any"],
            )
            salary_min = st.number_input("Minimum Salary ($)", value=0, step=1000)

        with col2:
            locations = st.text_input("Preferred Locations (comma-separated)", value="")
            salary_max = st.number_input("Maximum Salary ($)", value=0, step=1000)

        if st.button("Save Profile"):
            st.success("Profile saved successfully! (Note: Database integration pending)")


def show_job_search_page():
    """Job search page."""
    st.title("🔍 Job Search")

    # Search filters
    st.markdown("### Search Filters")

    col1, col2 = st.columns(2)

    with col1:
        keywords = st.text_input(
            "Keywords",
            value="",
            placeholder="e.g., Software Engineer, Data Scientist",
        )
        location = st.text_input("Location", value="", placeholder="e.g., San Francisco, Remote")

    with col2:
        remote_only = st.checkbox("Remote Only")
        min_salary = st.number_input("Minimum Salary ($)", value=0, step=10000)

    if st.button("🔍 Search Jobs", type="primary"):
        st.info("Job search integration pending. This will aggregate jobs from LinkedIn, Indeed, and other sources.")

    # Placeholder job results
    st.markdown("### 📋 Job Results")
    st.info("No jobs found. Try searching with different keywords.")

    # Example job card
    with st.expander("Example Job (Demo)"):
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown("#### Senior Software Engineer")
            st.markdown("**TechCorp Inc** • San Francisco, CA (Remote)")
            st.markdown("$150,000 - $200,000")
            st.markdown(
                """
                Looking for an experienced software engineer to join our team...

                **Requirements:**
                - 5+ years experience
                - Python, FastAPI, React
                - AWS cloud experience
                """
            )

        with col2:
            st.metric("Fit Score", "87%", delta="High Match")
            st.button("View Details")
            st.button("Apply Now")


def show_applications_page():
    """Applications tracking page."""
    st.title("📝 Applications")

    # Filter controls
    col1, col2, col3 = st.columns(3)

    with col1:
        status_filter = st.selectbox(
            "Status",
            ["All", "Draft", "Submitted", "Interview", "Offer", "Rejected"],
        )

    with col2:
        sort_by = st.selectbox("Sort By", ["Date (Newest)", "Date (Oldest)", "Fit Score"])

    with col3:
        st.metric("Total Applications", "12")

    # Application cards
    st.markdown("### Applications List")

    # Placeholder data
    applications = [
        {
            "company": "TechCorp",
            "role": "Software Engineer",
            "status": "Interview",
            "fit_score": 87,
            "date": "2025-10-27",
        },
        {
            "company": "StartupXYZ",
            "role": "ML Engineer",
            "status": "Submitted",
            "fit_score": 92,
            "date": "2025-10-26",
        },
    ]

    for app in applications:
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])

            with col1:
                st.markdown(f"**{app['role']}** at {app['company']}")

            with col2:
                st.write(f"Status: {app['status']}")

            with col3:
                st.write(f"Fit: {app['fit_score']}%")

            with col4:
                st.write(app['date'])

            st.markdown("---")


def show_analytics_page():
    """Analytics dashboard page."""
    st.title("📊 Analytics")

    # Time range selector
    time_range = st.selectbox("Time Range", ["Last 7 Days", "Last 30 Days", "Last 3 Months", "All Time"])

    # Metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Applications", "42", delta="12 this month")

    with col2:
        st.metric("Interview Rate", "28%", delta="+5%")

    with col3:
        st.metric("Offer Rate", "7%", delta="+2%")

    # Charts
    st.markdown("### 📈 Application Trends")

    # Placeholder chart data
    dates = pd.date_range(start="2025-10-01", end="2025-10-28", freq="D")
    applications_count = [1, 2, 1, 3, 2, 4, 3, 2, 1, 2, 3, 4, 2, 1, 3, 2, 4, 3, 2, 1, 2, 3, 4, 2, 3, 2, 1, 2]

    df = pd.DataFrame({"Date": dates, "Applications": applications_count})

    fig = px.line(df, x="Date", y="Applications", title="Applications Over Time")
    st.plotly_chart(fig, use_container_width=True)

    # Fit score distribution
    st.markdown("### 🎯 Fit Score Distribution")

    fit_scores = [65, 72, 85, 91, 78, 82, 69, 88, 95, 74, 81, 87]
    fig = px.histogram(fit_scores, nbins=10, title="Fit Score Distribution")
    st.plotly_chart(fig, use_container_width=True)


def show_settings_page():
    """Settings page."""
    st.title("⚙️ Settings")

    tab1, tab2, tab3 = st.tabs(["General", "Automation", "Ethics & Privacy"])

    with tab1:
        st.markdown("### General Settings")

        max_apps_per_day = st.slider("Max Applications Per Day", 1, 50, 10)
        min_fit_score = st.slider("Minimum Fit Score to Show", 0, 100, 60)
        customization_mode = st.radio(
            "Resume Customization Mode",
            ["Conservative", "Balanced", "Aggressive"],
            index=1,
        )

        if st.button("Save General Settings"):
            st.success("Settings saved!")

    with tab2:
        st.markdown("### Automation Settings")

        st.warning(
            """
            ⚠️ **Important**: All automated submissions require explicit user approval.
            This cannot be disabled for ethical compliance.
            """
        )

        enable_auto_apply = st.checkbox(
            "Enable Auto-Apply (after approval)",
            value=False,
            help="Allow automated form filling after you approve each application",
        )

        browser_headless = st.checkbox("Run browser in headless mode", value=True)

        if st.button("Save Automation Settings"):
            st.success("Settings saved!")

    with tab3:
        st.markdown("### Ethics & Privacy")

        st.info(
            """
            **Ethics Guardrails (Always Enabled)**:
            - ✓ User approval required before any submission
            - ✓ Ethics validation prevents fabrication
            - ✓ All claims must be verifiable from your resume
            - ✓ Encrypted resume storage
            - ✓ No password storage
            """
        )

        st.markdown("#### Data Management")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Export My Data"):
                st.info("Data export will be available soon")

        with col2:
            if st.button("Delete My Data", type="secondary"):
                st.warning("This will permanently delete all your data")


if __name__ == "__main__":
    main()
