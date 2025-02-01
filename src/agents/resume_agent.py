from typing import Dict, List, Optional
from langchain.agents import AgentType, initialize_agent
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain_community.document_loaders import TextLoader, PDFMinerLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.tools.python.tool import PythonREPL
from langchain.pydantic_v1 import BaseModel, Field
import json
import os
import subprocess
from ..llm.nim_client import NIMClient


class ResumeContent(BaseModel):
    """Schema for resume content."""
    summary: str = Field(description="Professional summary")
    experience: List[str] = Field(
        description="List of formatted experience entries")
    skills: List[str] = Field(description="Optimized skills list")
    education: List[str] = Field(description="Education entries")
    projects: List[str] = Field(description="Relevant project entries")
    ats_score_estimate: str = Field(
        description="Estimated ATS match percentage")
    optimization_notes: List[str] = Field(
        description="Notes about content optimization")


class ResumeAgent:
    def __init__(self, nim_api_key: str):
        """Initialize resume agent with NIM API key."""
        self.nim_client = NIMClient(nim_api_key)
        self.setup_agent()

    def setup_agent(self):
        """Set up LangChain agent with tools."""
        # Create custom tools
        self.tools = [
            # Resume content generation tool
            Tool(
                name="generate_resume_content",
                description="Generate optimized resume content based on job requirements",
                func=self._generate_resume_content
            ),
            # LaTeX generation tool
            Tool(
                name="generate_latex",
                description="Generate LaTeX content from resume data",
                func=self._generate_latex_content
            ),
            # PDF compilation tool
            Tool(
                name="compile_pdf",
                description="Compile LaTeX content to PDF",
                func=self._compile_latex
            ),
            # Supporting documents tool
            Tool(
                name="generate_supporting_docs",
                description="Generate cover letter and company interest statement",
                func=self._generate_supporting_documents
            )
        ]

        # Add Python REPL tool for LaTeX compilation
        python_repl = PythonREPL()
        self.tools.append(
            Tool(
                name="python_repl",
                description="Execute Python code for file operations and compilation",
                func=python_repl.run
            )
        )

        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.nim_client,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def _generate_resume_content(
        self,
        job_data: Dict,
        relevant_experience: List[Dict],
        template: str
    ) -> Dict:
        """Generate optimized resume content."""
        try:
            # Use NIM client to generate content
            resume_data = self.nim_client.generate_resume_content(
                job_data,
                relevant_experience,
                template
            )

            # Parse and validate content
            if isinstance(resume_data, str):
                resume_data = json.loads(resume_data)

            return ResumeContent(**resume_data).dict()

        except Exception as e:
            raise Exception(f"Failed to generate resume content: {str(e)}")

    def _generate_latex_content(self, resume_data: Dict, template: str) -> str:
        """Generate LaTeX content from resume data."""
        system_prompt = """You are an expert LaTeX resume writer. Generate LaTeX content that:
1. Follows the provided template structure
2. Incorporates the optimized content
3. Maintains proper LaTeX formatting
4. Ensures ATS readability"""

        prompt = f"""Generate LaTeX resume content based on:

Template:
{template}

Resume Data:
{json.dumps(resume_data, indent=2)}

Follow these rules:
1. Maintain all LaTeX commands and structure from the template
2. Replace content while preserving formatting
3. Ensure all special LaTeX characters are properly escaped
4. Keep section ordering from the template
5. Use appropriate LaTeX commands for formatting"""

        try:
            response = self.nim_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            raise Exception(f"Failed to generate LaTeX content: {str(e)}")

    def _compile_latex(self, latex_content: str, output_dir: str) -> str:
        """Compile LaTeX content to PDF."""
        os.makedirs(output_dir, exist_ok=True)

        # Save LaTeX content
        tex_path = os.path.join(output_dir, 'resume.tex')
        with open(tex_path, 'w') as f:
            f.write(latex_content)

        try:
            # Use Python REPL tool for compilation
            compile_code = f"""
import subprocess
import os

def compile_latex(tex_path, output_dir):
    try:
        # First compilation
        subprocess.run(
            ['pdflatex', '-output-directory', output_dir, tex_path],
            check=True,
            capture_output=True
        )
        
        # Second compilation for proper formatting
        subprocess.run(
            ['pdflatex', '-output-directory', output_dir, tex_path],
            check=True,
            capture_output=True
        )
        
        return os.path.join(output_dir, 'resume.pdf')
    except subprocess.CalledProcessError as e:
        raise Exception(f"LaTeX compilation failed: {{e.stderr.decode()}}")

result = compile_latex('{tex_path}', '{output_dir}')
print(result)
"""

            result = self.tools[-1].run(compile_code)  # Use Python REPL tool
            return result.strip()

        except Exception as e:
            raise Exception(f"Failed to compile LaTeX: {str(e)}")

    def _generate_supporting_documents(
        self,
        job_data: Dict,
        resume_data: Dict,
        output_dir: str
    ) -> Dict[str, str]:
        """Generate cover letter and company interest statement."""
        try:
            # Generate cover letter
            cover_letter = self.nim_client.generate_cover_letter(
                job_data,
                resume_data['experience'][:2]  # Use top 2 experiences
            )

            # Save cover letter
            cover_letter_path = os.path.join(output_dir, 'cover_letter.txt')
            with open(cover_letter_path, 'w') as f:
                f.write(cover_letter)

            # Generate company interest statement
            interest_prompt = f"""Generate a concise paragraph expressing interest in {job_data['company']}.
Use these details:
- Position: {job_data['title']}
- Company values: {job_data['context']['company_values']}
- Job description: {job_data['description']}

The statement should:
1. Show genuine interest in the company
2. Connect your experience to their needs
3. Demonstrate cultural fit
4. Be specific to this company/role"""

            interest_response = self.nim_client.generate(
                prompt=interest_prompt,
                temperature=0.7
            )

            interest_statement = interest_response['choices'][0]['message']['content']

            # Save interest statement
            interest_path = os.path.join(output_dir, 'company_interest.txt')
            with open(interest_path, 'w') as f:
                f.write(interest_statement)

            return {
                'cover_letter': cover_letter_path,
                'company_interest': interest_path
            }

        except Exception as e:
            raise Exception(
                f"Failed to generate supporting documents: {str(e)}")

    def generate_application_documents(
        self,
        job_data: Dict,
        relevant_experience: List[Dict],
        master_resume_path: str,
        output_dir: str
    ) -> Dict[str, str]:
        """Generate all application documents using LangChain agent."""
        try:
            # Read master resume template
            with open(master_resume_path, 'r') as f:
                template = f.read()

            # Execute agent to generate documents
            result = self.agent.run({
                "input": "Generate application documents",
                "job_data": job_data,
                "relevant_experience": relevant_experience,
                "template": template,
                "output_dir": output_dir
            })

            # Parse agent result
            if isinstance(result, str):
                result = json.loads(result)

            return result

        except Exception as e:
            raise Exception(
                f"Failed to generate application documents: {str(e)}")

    def analyze_resume_optimization(self, resume_path: str, job_data: Dict) -> Dict:
        """Analyze resume optimization against job requirements."""
        try:
            # Load PDF content
            loader = PDFMinerLoader(resume_path)
            documents = loader.load()
            resume_text = documents[0].page_content

            system_prompt = """You are an expert resume reviewer. Analyze the resume against the job requirements and provide detailed feedback."""

            prompt = f"""Analyze this resume against the job requirements:

Resume Text:
{resume_text}

Job Requirements:
{json.dumps(job_data, indent=2)}

Provide analysis in JSON format:
{{
    "keyword_matches": ["Keywords found in both resume and job posting"],
    "missing_keywords": ["Important keywords missing from resume"],
    "strength_areas": ["Areas where resume matches well"],
    "improvement_areas": ["Suggested improvements"],
    "ats_compatibility": ["ATS compatibility notes"],
    "overall_match_score": "Percentage match estimate"
}}"""

            response = self.nim_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=0.3
            )

            return json.loads(response['choices'][0]['message']['content'])

        except Exception as e:
            raise Exception(f"Failed to analyze resume optimization: {str(e)}")
