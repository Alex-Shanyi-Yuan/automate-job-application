import pytest
from unittest.mock import Mock
import os
import json
from pathlib import Path


@pytest.fixture
def test_data_dir():
    """Get the test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture
def sample_job_html():
    """Sample job posting HTML content."""
    return """
    <html>
        <body>
            <h1 class="job-title">Senior Software Engineer</h1>
            <div class="company-name">Tech Corp</div>
            <div class="location">Remote</div>
            <div class="job-description">
                <h2>About the Role</h2>
                <p>We're looking for an experienced software engineer...</p>
                
                <h2>Requirements</h2>
                <ul>
                    <li>5+ years of experience in software development</li>
                    <li>Strong Python programming skills</li>
                    <li>Experience with AWS cloud services</li>
                </ul>
                
                <h2>Responsibilities</h2>
                <ul>
                    <li>Design and implement scalable solutions</li>
                    <li>Lead technical projects</li>
                    <li>Mentor junior developers</li>
                </ul>
            </div>
        </body>
    </html>
    """


@pytest.fixture
def sample_job_data():
    """Sample parsed job data."""
    return {
        "title": "Senior Software Engineer",
        "company": "Tech Corp",
        "location": "Remote",
        "description": "We're looking for an experienced software engineer...",
        "requirements": [
            "5+ years of experience in software development",
            "Strong Python programming skills",
            "Experience with AWS cloud services"
        ],
        "technical_skills": ["Python", "AWS"],
        "soft_skills": ["Leadership", "Mentoring"],
        "experience_level": "5+ years",
        "education": "Bachelor's degree",
        "ats_keywords": ["Python", "AWS", "scalable"],
        "company_values": ["Innovation", "Mentorship"],
        "responsibilities": [
            "Design and implement scalable solutions",
            "Lead technical projects",
            "Mentor junior developers"
        ]
    }


@pytest.fixture
def sample_resume_latex():
    """Sample LaTeX resume template."""
    return r"""
\documentclass[letterpaper,11pt]{article}

\begin{document}
\section{Summary}
{summary}

\section{Experience}
{experience}

\section{Skills}
{skills}

\section{Education}
{education}

\section{Projects}
{projects}
\end{document}
"""


@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables."""
    env_vars = {
        'LLM_PROVIDER': 'nvidia',
        'NVIDIA_API_KEY': 'test-key',
        'LLM_MODEL': 'test-model'
    }
    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)
    return env_vars


@pytest.fixture
def mock_llm_response():
    """Mock LLM API response."""
    return {
        'choices': [{
            'message': {
                'content': json.dumps({
                    "test": "response"
                })
            }
        }]
    }


@pytest.fixture
def mock_file_system(tmp_path):
    """Set up a temporary file system for testing."""
    # Create directories
    (tmp_path / "resume").mkdir()
    (tmp_path / "out").mkdir()

    # Create sample files
    resume_tex = tmp_path / "resume" / "template.tex"
    resume_tex.write_text(
        r"\documentclass{article}\begin{document}Test\end{document}")

    return tmp_path


@pytest.fixture
def mock_subprocess(monkeypatch):
    """Mock subprocess calls."""
    mock_run = Mock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = b"Success"
    monkeypatch.setattr('subprocess.run', mock_run)
    return mock_run


class MockResponse:
    """Mock HTTP response."""

    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise Exception(f"HTTP {self.status_code}")


@pytest.fixture
def mock_http_response():
    """Create mock HTTP response."""
    def _create_response(text="", status_code=200):
        return MockResponse(text, status_code)
    return _create_response


def pytest_collection_modifyitems(items):
    """Add markers to tests based on their location."""
    for item in items:
        # Add unit/integration markers
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        else:
            item.add_marker(pytest.mark.unit)

        # Add component-specific markers
        if "llm" in str(item.fspath):
            item.add_marker(pytest.mark.llm)
        elif "agents" in str(item.fspath):
            item.add_marker(pytest.mark.agent)
