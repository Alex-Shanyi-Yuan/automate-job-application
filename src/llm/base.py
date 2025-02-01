from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


class BaseLLMClient(ABC):
    """Base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate text completion from prompt.

        Args:
            prompt: Input prompt text

        Returns:
            Response dictionary containing generated text
        """
        pass

    @abstractmethod
    def get_chain(self, prompt_template: PromptTemplate) -> LLMChain:
        """Get LangChain chain for the LLM.

        Args:
            prompt_template: Template for chain prompts

        Returns:
            Configured LLMChain
        """
        pass

    @abstractmethod
    def generate_resume_content(self, prompt: str) -> Dict[str, Any]:
        """Generate optimized resume content.

        Args:
            prompt: Input prompt with job requirements and experience

        Returns:
            Dictionary containing generated resume sections
        """
        pass

    @abstractmethod
    def generate_cover_letter(
        self,
        job_data: Dict[str, Any],
        resume_data: Dict[str, Any]
    ) -> str:
        """Generate a cover letter.

        Args:
            job_data: Parsed job posting data
            resume_data: Generated resume content

        Returns:
            Generated cover letter text
        """
        pass

    @abstractmethod
    def analyze_job_requirements(
        self,
        job_description: str,
        resume_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze job requirements and optional resume match.

        Args:
            job_description: Job posting text
            resume_content: Optional resume content to analyze against

        Returns:
            Dictionary containing requirement analysis
        """
        pass

    @abstractmethod
    def extract_relevant_experience(
        self,
        experience: List[Dict[str, str]],
        job_requirements: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Extract experience entries relevant to job requirements.

        Args:
            experience: List of experience entries
            job_requirements: Parsed job requirements

        Returns:
            List of relevant experience entries
        """
        pass

    @abstractmethod
    def optimize_content(
        self,
        content: str,
        keywords: List[str],
        context: Optional[str] = None
    ) -> str:
        """Optimize content for given keywords.

        Args:
            content: Text content to optimize
            keywords: Target keywords
            context: Optional context for optimization

        Returns:
            Optimized content
        """
        pass

    @abstractmethod
    def format_latex(
        self,
        content: Dict[str, Any],
        template: str
    ) -> str:
        """Format content as LaTeX document.

        Args:
            content: Content to format
            template: LaTeX template

        Returns:
            Formatted LaTeX document
        """
        pass

    @abstractmethod
    def evaluate_ats_score(
        self,
        resume_text: str,
        job_description: str
    ) -> Dict[str, Any]:
        """Evaluate resume ATS score for job.

        Args:
            resume_text: Resume content
            job_description: Job posting text

        Returns:
            Dictionary with ATS evaluation results
        """
        pass

    @abstractmethod
    def suggest_improvements(
        self,
        current_content: str,
        target_score: float,
        context: Optional[str] = None
    ) -> List[str]:
        """Suggest content improvements.

        Args:
            current_content: Current content
            target_score: Target ATS score
            context: Optional context for suggestions

        Returns:
            List of improvement suggestions
        """
        pass

    @abstractmethod
    def generate_company_research(
        self,
        company_name: str,
        job_title: str
    ) -> Dict[str, Any]:
        """Research company for application materials.

        Args:
            company_name: Target company name
            job_title: Applied position title

        Returns:
            Dictionary with company research results
        """
        pass
