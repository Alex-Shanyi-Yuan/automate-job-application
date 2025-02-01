import os
import requests
from typing import Dict, List, Optional, Union


class NIMClient:
    def __init__(self, api_key: str):
        """Initialize NIM client with API key."""
        self.api_key = api_key
        self.base_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions"
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json"
        }

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stop: Optional[List[str]] = None
    ) -> Dict:
        """Generate text using the deepseek-r1 model."""
        payload = {
            "model": "deepseek-ai/deepseek-r1",
            "messages": [
                *(
                    [{"role": "system", "content": system_prompt}]
                    if system_prompt
                    else []
                ),
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stop": stop or []
        }

        try:
            response = requests.post(
                f"{self.base_url}/deepseek-r1",
                headers=self.headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"Error calling NIM API: {str(e)}")

    def parse_job_posting(self, html_content: str) -> Dict:
        """Parse job posting HTML using the model."""
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
            temperature=0.3  # Lower temperature for more focused extraction
        )

        try:
            return response['choices'][0]['message']['content']
        except (KeyError, IndexError):
            raise Exception("Failed to parse job posting")

    def generate_resume_content(
        self,
        job_data: Dict,
        relevant_experience: List[Dict],
        master_resume: str
    ) -> Dict:
        """Generate optimized resume content based on job requirements."""
        system_prompt = """You are an expert resume writer specializing in ATS optimization. Your task is to create highly targeted resume content that:
1. Matches job requirements precisely
2. Uses relevant keywords effectively
3. Quantifies achievements
4. Maintains professional formatting
5. Optimizes for both human readers and ATS systems"""

        prompt = f"""Create optimized resume content based on the following information:

Job Details:
{job_data}

Relevant Experience:
{relevant_experience}

Master Resume Template:
{master_resume}

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

        try:
            return response['choices'][0]['message']['content']
        except (KeyError, IndexError):
            raise Exception("Failed to generate resume content")

    def generate_cover_letter(self, job_data: Dict, experience_highlights: List[str]) -> str:
        """Generate a tailored cover letter."""
        system_prompt = """You are an expert cover letter writer. Create compelling, personalized cover letters that:
1. Connect candidate experiences with job requirements
2. Show genuine interest in the company
3. Maintain professional tone
4. Include relevant keywords
5. Keep content concise and impactful"""

        prompt = f"""Write a cover letter based on:

Job Details:
{job_data}

Experience Highlights:
{experience_highlights}

The cover letter should be professional, engaging, and demonstrate clear alignment between the candidate's experience and the job requirements."""

        response = self.generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=0.7
        )

        try:
            return response['choices'][0]['message']['content']
        except (KeyError, IndexError):
            raise Exception("Failed to generate cover letter")
