from typing import Dict, List, Optional
from abc import ABC, abstractmethod


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None
    ) -> Dict:
        """Generate text using the LLM.

        Args:
            prompt: The prompt to generate from
            system_prompt: Optional system prompt for context
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            stop: Optional list of stop sequences

        Returns:
            Dict containing the generated text and metadata
        """
        pass

    @abstractmethod
    def parse_job_posting(self, html_content: str) -> Dict:
        """Parse job posting HTML using the model.

        Args:
            html_content: Raw HTML content of job posting

        Returns:
            Dict containing structured job posting data
        """
        pass

    @abstractmethod
    def generate_resume_content(
        self,
        job_data: Dict,
        relevant_experience: List[Dict],
        template: str
    ) -> Dict:
        """Generate optimized resume content based on job requirements.

        Args:
            job_data: Parsed job posting data
            relevant_experience: List of relevant experience entries
            template: Resume template to use

        Returns:
            Dict containing generated resume sections
        """
        pass

    @abstractmethod
    def generate_cover_letter(
        self,
        job_data: Dict,
        experience_highlights: List[str]
    ) -> str:
        """Generate a tailored cover letter.

        Args:
            job_data: Parsed job posting data
            experience_highlights: Key experience points to highlight

        Returns:
            Generated cover letter text
        """
        pass

    @abstractmethod
    def analyze_ats_requirements(self, job_data: Dict) -> Dict:
        """Analyze job requirements for ATS optimization.

        Args:
            job_data: Parsed job posting data

        Returns:
            Dict containing ATS optimization recommendations
        """
        pass

    def __call__(self, *args, **kwargs) -> Dict:
        """Make the client callable like a LangChain LLM.

        This allows the client to be used as a drop-in replacement
        for LangChain LLMs in agents and chains.
        """
        return self.generate(*args, **kwargs)
