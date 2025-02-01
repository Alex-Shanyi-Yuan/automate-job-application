import pytest
from unittest.mock import Mock, patch
from src.llm.nvidia_llm import NvidiaLLMClient


@pytest.fixture
def mock_nvidia_response():
    """Mock NVIDIA API response."""
    return {
        'choices': [{
            'message': {
                'content': 'Test response content'
            }
        }]
    }


@pytest.fixture
def mock_nvidia_client(mock_nvidia_response):
    """Create a NvidiaLLMClient with mocked ChatNVIDIA."""
    with patch('src.llm.nvidia_llm.ChatNVIDIA') as MockChatNVIDIA:
        mock_chat = Mock()
        mock_chat.invoke.return_value.content = mock_nvidia_response[
            'choices'][0]['message']['content']
        MockChatNVIDIA.return_value = mock_chat
        client = NvidiaLLMClient(api_key="test-key")
        yield client


def test_generate_text(mock_nvidia_client, mock_nvidia_response):
    """Test generating text with NVIDIA client."""
    response = mock_nvidia_client.generate(
        prompt="Test prompt",
        system_prompt="Test system prompt"
    )

    assert response == mock_nvidia_response
    mock_nvidia_client.llm.invoke.assert_called_once()


def test_parse_job_posting(mock_nvidia_client):
    """Test parsing job posting."""
    mock_nvidia_client.llm.invoke.return_value.content = '''
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

    result = mock_nvidia_client.parse_job_posting(
        "<html>Test job posting</html>")

    assert result["title"] == "Software Engineer"
    assert result["company"] == "Test Corp"
    assert "Python" in result["technical_skills"]
    mock_nvidia_client.llm.invoke.assert_called_once()


def test_generate_resume_content(mock_nvidia_client):
    """Test generating resume content."""
    mock_nvidia_client.llm.invoke.return_value.content = '''
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

    result = mock_nvidia_client.generate_resume_content(
        job_data={"title": "Test Job"},
        relevant_experience=[{"text": "Experience"}],
        template="Test template"
    )

    assert "sections" in result
    assert result["sections"]["summary"] == "Test summary"
    assert len(result["sections"]["experience"]) == 2
    mock_nvidia_client.llm.invoke.assert_called_once()


def test_generate_cover_letter(mock_nvidia_client):
    """Test generating cover letter."""
    expected_content = "Dear Hiring Manager..."
    mock_nvidia_client.llm.invoke.return_value.content = expected_content

    result = mock_nvidia_client.generate_cover_letter(
        job_data={"title": "Test Job", "company": "Test Corp"},
        experience_highlights=["Experience 1", "Experience 2"]
    )

    assert result == expected_content
    mock_nvidia_client.llm.invoke.assert_called_once()


def test_analyze_ats_requirements(mock_nvidia_client):
    """Test analyzing ATS requirements."""
    mock_nvidia_client.llm.invoke.return_value.content = '''
    {
        "critical_keywords": ["Python", "AWS"],
        "recommended_skills": ["Docker", "Kubernetes"],
        "formatting_tips": ["Use bullet points"],
        "content_suggestions": ["Quantify achievements"],
        "ats_score_factors": ["Keyword match", "Format"]
    }
    '''

    result = mock_nvidia_client.analyze_ats_requirements(
        job_data={"title": "Test Job", "requirements": ["Python", "AWS"]}
    )

    assert "critical_keywords" in result
    assert "Python" in result["critical_keywords"]
    assert "formatting_tips" in result
    mock_nvidia_client.llm.invoke.assert_called_once()


def test_error_handling(mock_nvidia_client):
    """Test error handling in NVIDIA client."""
    mock_nvidia_client.llm.invoke.side_effect = Exception("API Error")

    with pytest.raises(Exception) as exc_info:
        mock_nvidia_client.generate(prompt="Test prompt")

    assert "API Error" in str(exc_info.value)


def test_system_prompt_handling(mock_nvidia_client):
    """Test handling of system prompts."""
    mock_nvidia_client.generate(
        prompt="Test prompt",
        system_prompt="Test system prompt"
    )

    # Verify that system prompt was included in the messages
    call_args = mock_nvidia_client.llm.invoke.call_args[0][0]
    assert len(call_args) == 2
    assert call_args[0]["role"] == "system"
    assert call_args[0]["content"] == "Test system prompt"
    assert call_args[1]["role"] == "user"
    assert call_args[1]["content"] == "Test prompt"
