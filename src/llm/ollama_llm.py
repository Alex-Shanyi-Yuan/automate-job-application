import json
from typing import Dict, Any, Optional, List
import requests
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Ollama

from src.llm.base import BaseLLM


class OllamaLLM(BaseLLM):
    """Ollama LLM client implementation."""

    def __init__(self, model_name: str = "deepseek", base_url: str = "http://localhost:11434"):
        """Initialize Ollama LLM client.

        Args:
            model_name: Name of the Ollama model to use
            base_url: Base URL for Ollama API
        """
        self.model_name = model_name
        self.base_url = base_url
        self.llm = Ollama(base_url=base_url, model=model_name)

    def generate(self, prompt: str) -> Dict[str, Any]:
        """Generate text completion from prompt.

        Args:
            prompt: Input prompt text

        Returns:
            Response dictionary containing generated text
        """
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return {
                "choices": [{
                    "message": {
                        "content": response.json()["response"]
                    }
                }]
            }
        except Exception as e:
            raise Exception(f"Failed to generate text: {str(e)}")

    def get_chain(self, prompt_template: PromptTemplate) -> LLMChain:
        """Get LangChain chain for the LLM.

        Args:
            prompt_template: Template for chain prompts

        Returns:
            Configured LLMChain
        """
        return LLMChain(llm=self.llm, prompt=prompt_template)

    def generate_resume_content(self, prompt: str) -> Dict[str, Any]:
        """Generate optimized resume content.

        Args:
            prompt: Input prompt with job requirements and experience

        Returns:
            Dictionary containing generated resume sections
        """
        system_prompt = """You are an expert resume writer. Generate optimized resume 
        content that will score well with ATS systems. Focus on:
        1. Using relevant keywords naturally
        2. Quantifying achievements
        3. Highlighting transferable skills
        4. Clear, professional language
        
        Format the response as a JSON object with sections for summary, experience, 
        skills, education, and projects. Include an estimated ATS score and 
        optimization notes."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": f"{system_prompt}\n\n{prompt}",
                    "stream": False
                }
            )
            response.raise_for_status()
            return json.loads(response.json()["response"])
        except Exception as e:
            raise Exception(f"Failed to generate resume content: {str(e)}")

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
        prompt = f"""Write a compelling cover letter for the following job position. 
        Highlight relevant experience and align with company values.
        
        Job Details:
        {json.dumps(job_data, indent=2)}
        
        Resume Content:
        {json.dumps(resume_data, indent=2)}
        
        Guidelines:
        1. Professional and engaging tone
        2. Specific examples of relevant achievements
        3. Clear connection to company values and needs
        4. Standard business letter format"""

        try:
            response = self.generate(prompt)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"Failed to generate cover letter: {str(e)}")

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
        prompt = f"""Analyze the following job description and extract key requirements. 
        If resume content is provided, evaluate the match.
        
        Job Description:
        {job_description}
        
        {f'Resume Content: {resume_content}' if resume_content else ''}
        
        Analyze:
        1. Required technical skills
        2. Required soft skills
        3. Experience level
        4. Education requirements
        5. Key responsibilities
        {f'6. Resume match strength' if resume_content else ''}"""

        try:
            response = self.generate(prompt)
            return json.loads(response["choices"][0]["message"]["content"])
        except Exception as e:
            raise Exception(f"Failed to analyze job requirements: {str(e)}")

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
        prompt = f"""Given the following experience entries and job requirements, 
        identify and rank the most relevant experiences.
        
        Experience:
        {json.dumps(experience, indent=2)}
        
        Job Requirements:
        {json.dumps(job_requirements, indent=2)}
        
        For each experience entry, evaluate:
        1. Direct skill matches
        2. Transferable skills
        3. Achievement relevance
        4. Industry alignment"""

        try:
            response = self.generate(prompt)
            return json.loads(response["choices"][0]["message"]["content"])
        except Exception as e:
            raise Exception(f"Failed to extract relevant experience: {str(e)}")

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
        prompt = f"""Optimize the following content to naturally incorporate target 
        keywords while maintaining professional tone and clarity.
        
        Content:
        {content}
        
        Target Keywords:
        {json.dumps(keywords, indent=2)}
        
        {f'Context: {context}' if context else ''}
        
        Guidelines:
        1. Natural keyword integration
        2. Maintain original meaning
        3. Professional language
        4. Clear and concise"""

        try:
            response = self.generate(prompt)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"Failed to optimize content: {str(e)}")

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
        prompt = f"""Format the following content into a LaTeX document using the 
        provided template. Ensure proper escaping of special characters.
        
        Content:
        {json.dumps(content, indent=2)}
        
        Template:
        {template}
        
        Guidelines:
        1. Proper LaTeX syntax
        2. Clean formatting
        3. Professional layout
        4. Consistent styling"""

        try:
            response = self.generate(prompt)
            return response["choices"][0]["message"]["content"]
        except Exception as e:
            raise Exception(f"Failed to format LaTeX: {str(e)}")

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
        prompt = f"""Evaluate how well the resume matches the job requirements and 
        estimate ATS score. Consider keyword matches, formatting, and content relevance.
        
        Resume:
        {resume_text}
        
        Job Description:
        {job_description}
        
        Analyze:
        1. Keyword match rate
        2. Required skills coverage
        3. Experience alignment
        4. Education match
        5. Overall ATS score
        6. Improvement suggestions"""

        try:
            response = self.generate(prompt)
            return json.loads(response["choices"][0]["message"]["content"])
        except Exception as e:
            raise Exception(f"Failed to evaluate ATS score: {str(e)}")

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
        prompt = f"""Review the current content and suggest improvements to reach 
        the target ATS score. Focus on keyword optimization and content strength.
        
        Current Content:
        {current_content}
        
        Target Score: {target_score}
        
        {f'Context: {context}' if context else ''}
        
        Provide specific suggestions for:
        1. Keyword integration
        2. Content enhancement
        3. Format optimization
        4. Achievement highlighting"""

        try:
            response = self.generate(prompt)
            return json.loads(response["choices"][0]["message"]["content"])
        except Exception as e:
            raise Exception(f"Failed to suggest improvements: {str(e)}")

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
        prompt = f"""Research and provide key information about the company and role 
        that can be used to personalize application materials.
        
        Company: {company_name}
        Position: {job_title}
        
        Gather:
        1. Company values and culture
        2. Recent news or developments
        3. Industry position
        4. Role significance
        5. Growth opportunities"""

        try:
            response = self.generate(prompt)
            return json.loads(response["choices"][0]["message"]["content"])
        except Exception as e:
            raise Exception(f"Failed to generate company research: {str(e)}")
