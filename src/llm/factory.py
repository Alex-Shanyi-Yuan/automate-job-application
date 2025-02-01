from typing import Dict, Optional
from .base import BaseLLMClient
from .nvidia_llm import NvidiaLLMClient
from .ollama_llm import OllamaLLMClient


class LLMFactory:
    """Factory for creating LLM clients."""

    @staticmethod
    def create_llm(
        provider: str,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> BaseLLMClient:
        """Create an LLM client instance.

        Args:
            provider: LLM provider name ('nvidia' or 'ollama')
            api_key: API key for providers that require authentication
            model_name: Name of the model to use
            **kwargs: Additional provider-specific arguments

        Returns:
            An instance of BaseLLMClient

        Raises:
            ValueError: If the provider is not supported or required args are missing
        """
        provider = provider.lower()

        if provider == "nvidia":
            if not api_key:
                raise ValueError("API key is required for NVIDIA AI provider")

            return NvidiaLLMClient(
                api_key=api_key,
                model_name=model_name or "deepseek-ai/deepseek-r1"
            )

        elif provider == "ollama":
            return OllamaLLMClient(
                model_name=model_name or "deepseek-coder:latest"
            )

        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")

    @staticmethod
    def get_default_model(provider: str) -> str:
        """Get the default model name for a provider.

        Args:
            provider: LLM provider name

        Returns:
            Default model name for the provider
        """
        provider = provider.lower()

        defaults = {
            "nvidia": "deepseek-ai/deepseek-r1",
            "ollama": "deepseek-coder:latest"
        }

        return defaults.get(provider, "")

    @staticmethod
    def get_supported_providers() -> Dict[str, Dict]:
        """Get information about supported LLM providers.

        Returns:
            Dict containing provider information
        """
        return {
            "nvidia": {
                "name": "NVIDIA AI",
                "requires_api_key": True,
                "default_model": "deepseek-ai/deepseek-r1",
                "description": "NVIDIA's AI endpoints for high-performance inference"
            },
            "ollama": {
                "name": "Ollama",
                "requires_api_key": False,
                "default_model": "deepseek-r1:8b",
                "description": "Local LLM runtime for various open models"
            }
        }
