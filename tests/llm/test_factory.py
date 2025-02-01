import pytest
from src.llm.factory import LLMFactory
from src.llm.nvidia_llm import NvidiaLLMClient
from src.llm.ollama_llm import OllamaLLMClient


def test_create_nvidia_llm():
    """Test creating NVIDIA LLM client."""
    llm = LLMFactory.create_llm(
        provider="nvidia",
        api_key="test-key",
        model_name="test-model"
    )
    assert isinstance(llm, NvidiaLLMClient)
    assert llm.llm.api_key == "test-key"
    assert llm.llm.model == "test-model"


def test_create_ollama_llm():
    """Test creating Ollama LLM client."""
    llm = LLMFactory.create_llm(
        provider="ollama",
        model_name="test-model"
    )
    assert isinstance(llm, OllamaLLMClient)
    assert llm.llm.model == "test-model"


def test_create_llm_invalid_provider():
    """Test creating LLM with invalid provider."""
    with pytest.raises(ValueError) as exc_info:
        LLMFactory.create_llm(provider="invalid")
    assert "Unsupported LLM provider" in str(exc_info.value)


def test_create_nvidia_llm_without_api_key():
    """Test creating NVIDIA LLM without API key."""
    with pytest.raises(ValueError) as exc_info:
        LLMFactory.create_llm(provider="nvidia")
    assert "API key is required" in str(exc_info.value)


def test_get_default_model():
    """Test getting default model names."""
    assert LLMFactory.get_default_model("nvidia") == "deepseek-ai/deepseek-r1"
    assert LLMFactory.get_default_model("ollama") == "deepseek-coder:latest"
    assert LLMFactory.get_default_model("invalid") == ""


def test_get_supported_providers():
    """Test getting supported provider information."""
    providers = LLMFactory.get_supported_providers()

    assert "nvidia" in providers
    assert "ollama" in providers

    assert providers["nvidia"]["requires_api_key"] is True
    assert providers["ollama"]["requires_api_key"] is False

    assert "default_model" in providers["nvidia"]
    assert "default_model" in providers["ollama"]
