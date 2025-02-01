import pytest
from unittest.mock import Mock, patch
from src.llm.ollama_llm import OllamaLLMClient


@pytest.fixture
def mock_ollama_response():
    """Mock Ollama response."""
    return "Test response content"


@pytest.fixture
def mock_ollama_client(mock_ollama_response):
    """Create an OllamaLLMClient with mocked Ollama."""
    with patch('src.llm.ollama_llm.Ollama') as MockOllama:
        mock_llm = Mock()
        mock_llm.invoke.return_value = mock_ollama_response
        MockOllama.return_value = mock_llm
        client = OllamaLLMClient(model_name="test-model")
        yield client


def test_generate_text(mock_ollama_client, mock_ollama_response):
    """Test generating text with Ollama client."""
    response = mock_ollama_client.generate(
        prompt="Test prompt",
        system_prompt="Test system prompt"
    )

    assert response['choices'][0]['message']['content'] == mock_ollama_response
    mock_ollama_client.llm.invoke.assert_called_once()


def test_parse_job_posting(mock_ollama_client):
    """Test parsing job posting."""
    mock_ollama_client.llm.invoke.return_value = '''
    {
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
    '''

    result = mock_ollama_client.parse_job_posting(
        "<html>Test job posting</html>")

    assert result["title"] == "Software Engineer"
    assert result["company"] == "Test Corp"
    assert "Python" in result["technical_skills"]
    mock_ollama_client.llm.invoke.assert_called_once()


def test_generate_resume_content(mock_ollama_client):
    """Test generating resume content."""
    mock_ollama_client.llm.invoke.return_value = '''
    {
        "sections": {
            "summary": "Test summary",
            "experience": ["Experience 1", "Experience 2"],
            "skills": ["Skill 1", "Skill 2"],
            "education": ["Education 1"],
            "projects": ["Project 1"]
        },
        "ats_score_estimate": "85%",
        "optimization_notes": ["Note 1"]
    }
    '''

    result = mock_ollama_client.generate_resume_content(
        job_data={"title": "Test Job"},
        relevant_experience=[{"text": "Experience"}],
        template="Test template"
    )

    assert "sections" in result
    assert result["sections"]["summary"] == "Test summary"
    assert len(result["sections"]["experience"]) == 2
    mock_ollama_client.llm.invoke.assert_called_once()


def test_generate_cover_letter(mock_ollama_client):
    """Test generating cover letter."""
    expected_content = "Dear Hiring Manager..."
    mock_ollama_client.llm.invoke.return_value = expected_content

    result = mock_ollama_client.generate_cover_letter(
        job_data={"title": "Test Job", "company": "Test Corp"},
        experience_highlights=["Experience 1", "Experience 2"]
    )

    assert result == expected_content
    mock_ollama_client.llm.invoke.assert_called_once()


def test_analyze_ats_requirements(mock_ollama_client):
    """Test analyzing ATS requirements."""
    mock_ollama_client.llm.invoke.return_value = '''
    {
        "critical_keywords": ["Python", "AWS"],
        "recommended_skills": ["Docker", "Kubernetes"],
        "formatting_tips": ["Use bullet points"],
        "content_suggestions": ["Quantify achievements"],
        "ats_score_factors": ["Keyword match", "Format"]
    }
    '''

    result = mock_ollama_client.analyze_ats_requirements(
        job_data={"title": "Test Job", "requirements": ["Python", "AWS"]}
    )

    assert "critical_keywords" in result
    assert "Python" in result["critical_keywords"]
    assert "formatting_tips" in result
    mock_ollama_client.llm.invoke.assert_called_once()


def test_error_handling(mock_ollama_client):
    """Test error handling in Ollama client."""
    mock_ollama_client.llm.invoke.side_effect = Exception("LLM Error")

    with pytest.raises(Exception) as exc_info:
        mock_ollama_client.generate(prompt="Test prompt")

    assert "LLM Error" in str(exc_info.value)


def test_system_prompt_handling(mock_ollama_client):
    """Test handling of system prompts."""
    mock_ollama_client.generate(
        prompt="Test prompt",
        system_prompt="Test system prompt"
    )

    # Verify that system prompt was combined with user prompt
    call_args = mock_ollama_client.llm.invoke.call_args[0][0]
    assert "System: Test system prompt" in call_args
    assert "User: Test prompt" in call_args


def test_temperature_handling(mock_ollama_client):
    """Test handling of temperature parameter."""
    mock_ollama_client.generate(
        prompt="Test prompt",
        temperature=0.5
    )

    # Verify temperature was passed to Ollama
    _, kwargs = mock_ollama_client.llm.invoke.call_args
    assert kwargs.get('temperature') == 0.5


def test_max_tokens_handling(mock_ollama_client):
    """Test handling of max_tokens parameter."""
    mock_ollama_client.generate(
        prompt="Test prompt",
        max_tokens=100
    )

    # Verify max_tokens was passed to Ollama
    _, kwargs = mock_ollama_client.llm.invoke.call_args
    assert kwargs.get('max_tokens') == 100
