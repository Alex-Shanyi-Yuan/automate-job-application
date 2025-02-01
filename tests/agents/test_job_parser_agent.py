import pytest
from unittest.mock import Mock, patch
from src.agents.job_parser_agent import JobParserAgent, JobData
from bs4 import BeautifulSoup


@pytest.fixture
def mock_llm_client():
    """Create a mock LLM client."""
    mock_client = Mock()
    mock_client.generate.return_value = {
        'choices': [{
            'message': {
                'content': '{"test": "response"}'
            }
        }]
    }
    return mock_client


@pytest.fixture
def mock_job_parser(mock_llm_client):
    """Create a JobParserAgent with mocked dependencies."""
    return JobParserAgent(mock_llm_client)


@pytest.fixture
def sample_job_data():
    """Sample job posting data."""
    return {
        "title": "Software Engineer",
        "company": "Test Corp",
        "location": "Remote",
        "description": "Test job description",
        "requirements": ["Python", "AWS"],
        "technical_skills": ["Python", "AWS"],
        "soft_skills": ["Communication"],
        "experience_level": "5+ years",
        "education": "Bachelor's degree",
        "ats_keywords": ["Python", "AWS"],
        "company_values": ["Innovation"],
        "responsibilities": ["Develop software"]
    }


def test_job_data_model():
    """Test JobData model validation."""
    data = JobData(
        title="Test Job",
        company="Test Corp",
        location="Remote",
        description="Test description",
        requirements=["Req 1"],
        technical_skills=["Skill 1"],
        soft_skills=["Soft 1"],
        experience_level="5 years",
        education="BS",
        ats_keywords=["Keyword"],
        company_values=["Value"],
        responsibilities=["Resp 1"]
    )

    assert data.title == "Test Job"
    assert "Req 1" in data.requirements
    assert "Skill 1" in data.technical_skills


def test_setup_agent(mock_job_parser):
    """Test agent setup and tool registration."""
    assert len(mock_job_parser.tools) >= 3  # Should have at least 3 custom tools
    assert any(tool.name == "fetch_url_content" for tool in mock_job_parser.tools)
    assert any(tool.name == "parse_job_content" for tool in mock_job_parser.tools)
    assert any(
        tool.name == "analyze_ats_requirements" for tool in mock_job_parser.tools)


@patch('requests.get')
def test_fetch_url_content(mock_get, mock_job_parser):
    """Test URL content fetching."""
    mock_get.return_value.text = "<html><body>Test content</body></html>"
    mock_get.return_value.raise_for_status = Mock()

    content = mock_job_parser._fetch_url_content("https://test.com/job")

    assert "Test content" in content
    mock_get.assert_called_once_with(
        "https://test.com/job",
        headers=mock_job_parser.headers
    )


def test_parse_job_content(mock_job_parser, sample_job_data):
    """Test job content parsing."""
    mock_job_parser.llm_client.generate.return_value = {
        'choices': [{
            'message': {
                'content': str(sample_job_data)
            }
        }]
    }

    result = mock_job_parser._parse_job_content("<html>Test job</html>")

    assert isinstance(result, dict)
    assert result["title"] == sample_job_data["title"]
    assert result["company"] == sample_job_data["company"]
    assert result["technical_skills"] == sample_job_data["technical_skills"]


def test_analyze_ats_requirements(mock_job_parser):
    """Test ATS requirements analysis."""
    mock_job_parser.llm_client.analyze_ats_requirements.return_value = {
        "critical_keywords": ["Python", "AWS"],
        "recommended_skills": ["Docker"],
        "formatting_tips": ["Use bullet points"]
    }

    result = mock_job_parser._analyze_ats_requirements({"title": "Test Job"})

    assert "critical_keywords" in result
    assert "Python" in result["critical_keywords"]
    mock_job_parser.llm_client.analyze_ats_requirements.assert_called_once()


@patch('requests.get')
def test_parse_job_posting_integration(mock_get, mock_job_parser, sample_job_data):
    """Test full job posting parsing integration."""
    # Mock HTTP response
    mock_get.return_value.text = "<html><body>Test job posting</body></html>"
    mock_get.return_value.raise_for_status = Mock()

    # Mock LLM responses
    mock_job_parser.llm_client.generate.return_value = {
        'choices': [{
            'message': {
                'content': str(sample_job_data)
            }
        }]
    }

    result = mock_job_parser.parse_job_posting("https://test.com/job")

    assert result["title"] == sample_job_data["title"]
    assert result["company"] == sample_job_data["company"]
    assert "job_id" in result


def test_error_handling(mock_job_parser):
    """Test error handling in job parsing."""
    mock_job_parser.llm_client.generate.side_effect = Exception("API Error")

    with pytest.raises(Exception) as exc_info:
        mock_job_parser.parse_job_posting("https://test.com/job")

    assert "Failed to parse job posting" in str(exc_info.value)


def test_extract_job_id(mock_job_parser):
    """Test job ID extraction from different URL formats."""
    test_cases = [
        ("https://linkedin.com/jobs/123456", "123456"),
        ("https://indeed.com/viewjob?jk=abc123", "abc123"),
        ("https://glassdoor.com/job-listing/se-company-jobListingId=987654", "987654"),
    ]

    for url, expected_id in test_cases:
        job_id = mock_job_parser._extract_job_id(url)
        assert job_id == expected_id


@patch('requests.get')
def test_html_cleaning(mock_get, mock_job_parser):
    """Test HTML content cleaning."""
    html_content = """
    <html>
        <head><script>alert('test')</script></head>
        <body>
            <div style="display:none">Hidden content</div>
            <div class="job-description">Visible content</div>
        </body>
    </html>
    """
    mock_get.return_value.text = html_content
    mock_get.return_value.raise_for_status = Mock()

    content = mock_job_parser._fetch_url_content("https://test.com/job")

    assert "Hidden content" not in content
    assert "Visible content" in content
    assert "alert('test')" not in content


def test_agent_execution(mock_job_parser):
    """Test LangChain agent execution."""
    mock_job_parser.agent.run.return_value = str({
        "title": "Test Job",
        "company": "Test Corp"
    })

    result = mock_job_parser.parse_job_posting("https://test.com/job")

    assert isinstance(result, dict)
    assert "title" in result
    assert "company" in result
    mock_job_parser.agent.run.assert_called_once()
