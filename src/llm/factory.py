import os
from typing import Optional, Dict, Any

from src.llm.base import BaseLLM
from src.llm.ollama_llm import OllamaLLM


class LLMFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create_llm(
        provider: Optional[str] = None,
        model: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> BaseLLM:
        """Create an LLM client based on configuration.

        Args:
            provider: LLM provider name (e.g., 'ollama', 'nvidia')
            model: Model name to use
            config: Additional configuration options

        Returns:
            Configured LLM client

        Raises:
            ValueError: If provider is not supported or configuration is invalid
        """
        # Use environment variables if not provided
        provider = provider or os.getenv("LLM_PROVIDER", "ollama")
        model = model or os.getenv("LLM_MODEL", "deepseek")
        config = config or {}

        if provider.lower() == "ollama":
            base_url = config.get("base_url") or os.getenv(
                "OLLAMA_BASE_URL", "http://localhost:11434")
            return OllamaLLM(
                model_name=model,
                base_url=base_url
            )
        else:
            raise ValueError(
                f"Unsupported LLM provider: {provider}. Currently supported: ollama")

    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate LLM configuration.

        Args:
            config: Configuration dictionary to validate

        Returns:
            True if configuration is valid, False otherwise
        """
        required_fields = {
            "ollama": ["base_url"],
            "nvidia": ["api_key", "base_url"]
        }

        provider = config.get("provider", "").lower()
        if provider not in required_fields:
            return False

        return all(field in config for field in required_fields[provider])

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default LLM configuration.

        Returns:
            Dictionary with default configuration values
        """
        return {
            "provider": os.getenv("LLM_PROVIDER", "ollama"),
            "model": os.getenv("LLM_MODEL", "deepseek"),
            "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        }

    @staticmethod
    def create_from_env() -> BaseLLM:
        """Create LLM client from environment variables.

        Returns:
            Configured LLM client based on environment variables
        """
        config = LLMFactory.get_default_config()
        return LLMFactory.create_llm(
            provider=config["provider"],
            model=config["model"],
            config=config
        )
