import pytest
from unittest.mock import Mock, patch, mock_open
import json
import os
from src.agents.resume_agent import ResumeAgent, ResumeContent


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
def mock_resume_agent(mock_llm_client):
    """Create a ResumeAgent with mocked dependencies."""
    return ResumeAgent(mock_llm_client)


@pytest.fixture
def sample_resume_data():
    """Sample resume content data."""
    return {
        "sections": {
            "summary": "Experienced software engineer",
            "experience": [
                "Senior Developer at Tech Corp",
                "Software Engineer at Startup Inc"
            ],
            "skills": ["Python", "AWS", "Docker"],
            "education": ["BS in Computer Science"],
            "projects": ["Project 1", "Project 2"]
        },
        "ats_score_estimate": "85%",
        "optimization_notes": ["Good keyword match"]
    }


def test_resume_content_model():
    """Test ResumeContent model validation."""
    data = ResumeContent(
        summary="Test summary",
        experience=["Exp 1", "Exp 2"],
        skills=["Skill 1"],
        education=["Education 1"],
        projects=["Project 1"],
        ats_score_estimate="90%",
        optimization_notes=["Note 1"]
    )

    assert data.summary == "Test summary"
    assert len(data.experience) == 2
    assert data.ats_score_estimate == "90%"


def test_setup_agent(mock_resume_agent):
    """Test agent setup and tool registration."""
    assert len(mock_resume_agent.tools) >= 4  # Should have at least 4 custom tools
    assert any(
        tool.name == "generate_resume_content" for tool in mock_resume_agent.tools)
    assert any(tool.name == "generate_latex" for tool in mock_resume_agent.tools)
    assert any(tool.name == "compile_pdf" for tool in mock_resume_agent.tools)
    assert any(
        tool.name == "generate_supporting_docs" for tool in mock_resume_agent.tools)


def test_generate_resume_content(mock_resume_agent, sample_resume_data):
    """Test resume content generation."""
    mock_resume_agent.llm_client.generate_resume_content.return_value = sample_resume_data

    result = mock_resume_agent._generate_resume_content(
        job_data={"title": "Software Engineer"},
        relevant_experience=[{"text": "Experience"}],
        template="Test template"
    )

    assert isinstance(result, dict)
    assert "sections" in result
    assert result["sections"]["summary"] == sample_resume_data["sections"]["summary"]
    assert len(result["sections"]["experience"]) == 2


def test_generate_latex_content(mock_resume_agent):
    """Test LaTeX content generation."""
    expected_latex = r"\documentclass{article}\begin{document}Test content\end{document}"
    mock_resume_agent.llm_client.generate.return_value = {
        'choices': [{
            'message': {
                'content': expected_latex
            }
        }]
    }

    result = mock_resume_agent._generate_latex_content(
        resume_data={"sections": {"summary": "Test"}},
        template=r"\documentclass{article}"
    )

    assert result == expected_latex
    mock_resume_agent.llm_client.generate.assert_called_once()


@patch('subprocess.run')
def test_compile_latex(mock_run, mock_resume_agent, tmp_path):
    """Test LaTeX compilation."""
    latex_content = r"\documentclass{article}\begin{document}Test\end{document}"
    output_dir = str(tmp_path)

    mock_run.return_value.returncode = 0

    result = mock_resume_agent._compile_latex(latex_content, output_dir)

    assert mock_run.call_count == 2  # Should compile twice
    assert os.path.join(output_dir, "resume.pdf") in result


def test_generate_supporting_documents(mock_resume_agent, tmp_path):
    """Test supporting document generation."""
    mock_resume_agent.llm_client.generate_cover_letter.return_value = "Dear Hiring Manager"
    mock_resume_agent.llm_client.generate.return_value = {
        'choices': [{
            'message': {
                'content': "I am interested in your company"
            }
        }]
    }

    result = mock_resume_agent._generate_supporting_documents(
        job_data={"title": "Software Engineer", "company": "Tech Corp"},
        resume_data={"experience": ["Exp 1"]},
        output_dir=str(tmp_path)
    )

    assert "cover_letter" in result
    assert "company_interest" in result
    assert os.path.exists(result["cover_letter"])
    assert os.path.exists(result["company_interest"])


@patch('builtins.open', new_callable=mock_open, read_data="Test template")
def test_generate_application_documents_integration(
    mock_file,
    mock_resume_agent,
    sample_resume_data,
    tmp_path
):
    """Test full application document generation integration."""
    # Mock LLM responses
    mock_resume_agent.llm_client.generate_resume_content.return_value = sample_resume_data
    mock_resume_agent.llm_client.generate.return_value = {
        'choices': [{
            'message': {
                'content': "Generated content"
            }
        }]
    }

    # Mock agent response
    mock_resume_agent.agent.run.return_value = json.dumps({
        "resume_pdf": str(tmp_path / "resume.pdf"),
        "cover_letter": str(tmp_path / "cover_letter.txt"),
        "company_interest": str(tmp_path / "company_interest.txt")
    })

    result = mock_resume_agent.generate_application_documents(
        job_data={"title": "Software Engineer"},
        relevant_experience=[{"text": "Experience"}],
        master_resume_path="test.tex",
        output_dir=str(tmp_path)
    )

    assert isinstance(result, dict)
    assert "resume_pdf" in result
    assert "cover_letter" in result
    assert "company_interest" in result
    mock_resume_agent.agent.run.assert_called_once()


def test_error_handling(mock_resume_agent):
    """Test error handling in resume generation."""
    mock_resume_agent.llm_client.generate.side_effect = Exception("LLM Error")

    with pytest.raises(Exception) as exc_info:
        mock_resume_agent._generate_latex_content(
            resume_data={"sections": {}},
            template=""
        )

    assert "Failed to generate LaTeX content" in str(exc_info.value)


@patch('subprocess.run')
def test_latex_compilation_error(mock_run, mock_resume_agent, tmp_path):
    """Test LaTeX compilation error handling."""
    mock_run.side_effect = Exception("Compilation Error")

    with pytest.raises(Exception) as exc_info:
        mock_resume_agent._compile_latex("Test content", str(tmp_path))

    assert "Failed to compile LaTeX" in str(exc_info.value)


def test_agent_tool_execution(mock_resume_agent):
    """Test execution of agent tools."""
    # Test resume content generation tool
    content_tool = next(
        t for t in mock_resume_agent.tools if t.name == "generate_resume_content")
    result = content_tool.func({"title": "Test"}, [], "template")
    assert isinstance(result, dict)

    # Test LaTeX generation tool
    latex_tool = next(
        t for t in mock_resume_agent.tools if t.name == "generate_latex")
    result = latex_tool.func({"sections": {}}, "template")
    assert isinstance(result, str)


def test_analyze_resume_optimization(mock_resume_agent, tmp_path):
    """Test resume optimization analysis."""
    mock_resume_agent.llm_client.generate.return_value = {
        'choices': [{
            'message': {
                'content': json.dumps({
                    "keyword_matches": ["Python", "AWS"],
                    "missing_keywords": ["Docker"],
                    "strength_areas": ["Experience"],
                    "improvement_areas": ["Add more keywords"],
                    "ats_compatibility": ["Good format"],
                    "overall_match_score": "85%"
                })
            }
        }]
    }

    result = mock_resume_agent.analyze_resume_optimization(
        resume_path=str(tmp_path / "test.pdf"),
        job_data={"requirements": ["Python", "AWS", "Docker"]}
    )

    assert isinstance(result, dict)
    assert "keyword_matches" in result
    assert "overall_match_score" in result
    assert "Python" in result["keyword_matches"]
