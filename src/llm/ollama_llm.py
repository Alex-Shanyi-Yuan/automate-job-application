from typing import Dict, List, Optional
import json
from langchain_community.llms import Ollama
from .base import BaseLLMClient


class OllamaLLMClient(BaseLLMClient):
    """Ollama implementation of LLM client."""

    def __init__(self, model_name: str = "deepseek-coder:latest"):
        """Initialize Ollama client.

        Args:
            model_name: Name of the model to use
        """
        self.llm = Ollama(model=model_name)

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None
    ) -> Dict:
        """Generate text using Ollama."""
        # Combine system prompt and user prompt if provided
        full_prompt = ""
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\n"
        full_prompt += f"User: {prompt}"

        response = self.llm.invoke(
            full_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            stop=stop or []
        )

        return {
            "choices": [{
                "message": {
                    "content": response
                }
            }]
        }

    def parse_job_posting(self, html_content: str) -> Dict:
        """Parse job posting HTML using Ollama."""
        system_prompt = """You are an expert job posting analyzer. Your task is to extract key information from job postings and structure it in a way that's optimal for resume tailoring and ATS optimization. Focus on:
1. Core job requirements
2. Technical skills needed
3. Soft skills and qualifications
4. Company culture indicators
5. Keywords for ATS optimization"""

        prompt = f"""Analyze the following job posting HTML and extract relevant information. Format the output as a JSON object with the following structure:
{{
    "title": "Job title",
    "company": "Company name",
    "location": "Job location",
    "description": "Brief job description",
    "requirements": ["List of core requirements"],
    "technical_skills": ["List of technical skills"],
    "soft_skills": ["List of soft skills"],
    "experience_level": "Required years/level of experience",
    "education": "Required education level",
    "ats_keywords": ["Important keywords for ATS"],
    "company_values": ["Company culture indicators"],
    "responsibilities": ["Key job responsibilities"]
}}

HTML Content:
{html_content}"""

        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )

        return json.loads(response['choices'][0]['message']['content'])

    def generate_resume_content(
        self,
        job_data: Dict,
        relevant_experience: List[Dict],
        template: str
    ) -> Dict:
        """Generate optimized resume content using Ollama."""
        system_prompt = """You are an expert resume writer specializing in ATS optimization. Your task is to create highly targeted resume content that:
1. Matches job requirements precisely
2. Uses relevant keywords effectively
3. Quantifies achievements
4. Maintains professional formatting
5. Optimizes for both human readers and ATS systems"""

        prompt = f"""Create optimized resume content based on the following information:

Job Details:
{json.dumps(job_data, indent=2)}

Relevant Experience:
{json.dumps(relevant_experience, indent=2)}

Master Resume Template:
{template}

Generate a JSON response with:
{{
    "sections": {{
        "summary": "Professional summary",
        "experience": ["List of formatted experience entries"],
        "skills": ["Optimized skills list"],
        "education": ["Education entries"],
        "projects": ["Relevant project entries"]
    }},
    "ats_score_estimate": "Estimated ATS match percentage",
    "optimization_notes": ["Notes about content optimization"]
}}"""

        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.5
        )

        return json.loads(response['choices'][0]['message']['content'])

    def generate_cover_letter(
        self,
        job_data: Dict,
        experience_highlights: List[str]
    ) -> str:
        """Generate a cover letter using Ollama."""
        system_prompt = """You are an expert cover letter writer. Create compelling, personalized cover letters that:
1. Connect candidate experiences with job requirements
2. Show genuine interest in the company
3. Maintain professional tone
4. Include relevant keywords
5. Keep content concise and impactful"""

        prompt = f"""Write a cover letter based on:

Job Details:
{json.dumps(job_data, indent=2)}

Experience Highlights:
{json.dumps(experience_highlights, indent=2)}

The cover letter should be professional, engaging, and demonstrate clear alignment between the candidate's experience and the job requirements."""

        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )

        return response['choices'][0]['message']['content']

    def analyze_ats_requirements(self, job_data: Dict) -> Dict:
        """Analyze job requirements for ATS optimization using Ollama."""
        system_prompt = """You are an ATS optimization expert. Analyze the job data and provide specific recommendations for resume optimization."""

        prompt = f"""Based on the following job data, provide ATS optimization recommendations:

Job Details:
{json.dumps(job_data, indent=2)}

Provide recommendations in JSON format:
{{
    "critical_keywords": ["Must-have keywords for ATS"],
    "recommended_skills": ["Skills to emphasize"],
    "formatting_tips": ["Specific formatting recommendations"],
    "content_suggestions": ["Content optimization suggestions"],
    "ats_score_factors": ["Factors that will influence ATS scoring"]
}}"""

        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.3
        )

        return json.loads(response['choices'][0]['message']['content'])
