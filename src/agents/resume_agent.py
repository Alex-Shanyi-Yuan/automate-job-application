from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import os
import json
import subprocess
from pathlib import Path
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.output_parsers import PydanticOutputParser

from src.llm.base import BaseLLMClient
from src.utils.file_utils import ensure_dir


@dataclass
class ResumeContent:
    """Data model for resume content."""
    summary: str
    experience: List[str]
    skills: List[str]
    education: List[str]
    projects: List[str]
    ats_score_estimate: str
    optimization_notes: List[str]


class ResumeAgent:
    """Agent for generating tailored resumes and supporting documents."""

    def __init__(self, llm_client: BaseLLMClient):
        """Initialize the resume agent.

        Args:
            llm_client: LLM client for text generation
        """
        self.llm_client = llm_client
        self.tools = self._setup_tools()
        self.agent = self._setup_agent()
        self.memory = ConversationBufferMemory()
        self.output_parser = PydanticOutputParser(
            pydantic_object=ResumeContent)

    def _setup_tools(self) -> List[Tool]:
        """Set up the tools available to the agent."""
        return [
            Tool(
                name="generate_resume_content",
                func=lambda x, y, z: self._generate_resume_content(x, y, z),
                description="Generate tailored resume content based on job requirements"
            ),
            Tool(
                name="generate_latex",
                func=lambda x, y: self._generate_latex_content(x, y),
                description="Generate LaTeX document from resume content"
            ),
            Tool(
                name="compile_pdf",
                func=lambda x, y: self._compile_latex(x, y),
                description="Compile LaTeX document to PDF"
            ),
            Tool(
                name="generate_supporting_docs",
                func=lambda x, y, z: self._generate_supporting_documents(
                    x, y, z),
                description="Generate cover letter and company interest statement"
            )
        ]

    def _setup_agent(self) -> AgentExecutor:
        """Set up the LangChain agent."""
        prompt = PromptTemplate(
            template="""You are a professional resume writer. Your task is to create 
            tailored resumes and supporting documents that maximize ATS scores and 
            highlight relevant experience.

            Job Details: {job_data}
            Relevant Experience: {experience}
            Current Task: {task}

            Think through this step-by-step:
            1) What aspects of the experience match the job requirements?
            2) How can we phrase achievements to use relevant keywords?
            3) What supporting details will strengthen the application?

            Available tools: {tools}

            Response should be in the following format:
            Thought: Your reasoning about what needs to be done
            Action: The tool to use
            Action Input: The input to the tool
            Observation: The result of the tool
            ... (repeat Thought/Action/Observation if needed)
            Final Answer: The final result

            Begin!
            {agent_scratchpad}""",
            input_variables=["job_data", "experience",
                             "task", "tools", "agent_scratchpad"]
        )

        return AgentExecutor.from_agent_and_tools(
            agent=LLMSingleActionAgent(
                llm_chain=self.llm_client.get_chain(prompt),
                output_parser=self.output_parser,
                stop=["\nObservation:"],
                allowed_tools=[tool.name for tool in self.tools]
            ),
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )

    def _generate_resume_content(
        self,
        job_data: Dict[str, Any],
        relevant_experience: List[Dict[str, str]],
        template: str
    ) -> Dict[str, Any]:
        """Generate tailored resume content.

        Args:
            job_data: Parsed job posting data
            relevant_experience: List of relevant experience entries
            template: Resume template to use

        Returns:
            Dictionary containing generated resume sections
        """
        try:
            prompt = f"""Based on the following job requirements and experience, 
            generate optimized resume content that will score well with ATS systems.
            
            Job Requirements:
            {json.dumps(job_data, indent=2)}
            
            Relevant Experience:
            {json.dumps(relevant_experience, indent=2)}
            
            Template Format:
            {template}
            
            Generate content that:
            1. Uses relevant keywords from the job posting
            2. Quantifies achievements where possible
            3. Highlights transferable skills
            4. Maintains clear, professional language
            """

            response = self.llm_client.generate_resume_content(prompt)
            return response

        except Exception as e:
            raise Exception(f"Failed to generate resume content: {str(e)}")

    def _generate_latex_content(
        self,
        resume_data: Dict[str, Any],
        template: str
    ) -> str:
        """Generate LaTeX document from resume content.

        Args:
            resume_data: Generated resume content
            template: LaTeX template to use

        Returns:
            Complete LaTeX document as string
        """
        try:
            prompt = f"""Convert the following resume content into a LaTeX document 
            using the provided template. Ensure proper formatting and escaping of 
            special characters.
            
            Content:
            {json.dumps(resume_data, indent=2)}
            
            Template:
            {template}
            """

            response = self.llm_client.generate(prompt)
            return response['choices'][0]['message']['content']

        except Exception as e:
            raise Exception(f"Failed to generate LaTeX content: {str(e)}")

    def _compile_latex(self, latex_content: str, output_dir: str) -> str:
        """Compile LaTeX document to PDF.

        Args:
            latex_content: LaTeX document content
            output_dir: Directory to save the PDF

        Returns:
            Path to generated PDF
        """
        try:
            ensure_dir(output_dir)
            tex_path = os.path.join(output_dir, "resume.tex")

            with open(tex_path, 'w') as f:
                f.write(latex_content)

            # Run pdflatex twice to resolve references
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_path],
                cwd=output_dir,
                check=True
            )
            subprocess.run(
                ['pdflatex', '-interaction=nonstopmode', tex_path],
                cwd=output_dir,
                check=True
            )

            return os.path.join(output_dir, "resume.pdf")

        except Exception as e:
            raise Exception(f"Failed to compile LaTeX: {str(e)}")

    def _generate_supporting_documents(
        self,
        job_data: Dict[str, Any],
        resume_data: Dict[str, Any],
        output_dir: str
    ) -> Dict[str, str]:
        """Generate cover letter and company interest statement.

        Args:
            job_data: Parsed job posting data
            resume_data: Generated resume content
            output_dir: Directory to save the documents

        Returns:
            Dictionary with paths to generated documents
        """
        try:
            ensure_dir(output_dir)

            # Generate cover letter
            cover_letter = self.llm_client.generate_cover_letter(
                job_data,
                resume_data
            )
            cover_letter_path = os.path.join(output_dir, "cover_letter.txt")
            with open(cover_letter_path, 'w') as f:
                f.write(cover_letter)

            # Generate company interest statement
            prompt = f"""Based on the job posting and company information, generate 
            a compelling statement explaining your interest in working for this company.
            
            Job Details:
            {json.dumps(job_data, indent=2)}
            """

            response = self.llm_client.generate(prompt)
            interest_statement = response['choices'][0]['message']['content']
            interest_path = os.path.join(output_dir, "company_interest.txt")
            with open(interest_path, 'w') as f:
                f.write(interest_statement)

            return {
                "cover_letter": cover_letter_path,
                "company_interest": interest_path
            }

        except Exception as e:
            raise Exception(
                f"Failed to generate supporting documents: {str(e)}"
            )

    def generate_application_documents(
        self,
        job_data: Dict[str, Any],
        relevant_experience: List[Dict[str, str]],
        master_resume_path: str,
        output_dir: str
    ) -> Dict[str, str]:
        """Generate all application documents.

        Args:
            job_data: Parsed job posting data
            relevant_experience: List of relevant experience entries
            master_resume_path: Path to master resume template
            output_dir: Directory to save generated documents

        Returns:
            Dictionary with paths to all generated documents
        """
        try:
            # Read master resume template
            with open(master_resume_path) as f:
                template = f.read()

            # Generate resume content
            resume_data = self._generate_resume_content(
                job_data,
                relevant_experience,
                template
            )

            # Generate LaTeX
            latex_content = self._generate_latex_content(resume_data, template)

            # Compile PDF
            pdf_path = self._compile_latex(latex_content, output_dir)

            # Generate supporting documents
            supporting_docs = self._generate_supporting_documents(
                job_data,
                resume_data,
                output_dir
            )

            return {
                "resume_pdf": pdf_path,
                **supporting_docs
            }

        except Exception as e:
            raise Exception(
                f"Failed to generate application documents: {str(e)}")

    def analyze_resume_optimization(
        self,
        resume_path: str,
        job_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Analyze resume optimization for ATS.

        Args:
            resume_path: Path to generated resume PDF
            job_data: Parsed job posting data

        Returns:
            Dictionary containing optimization analysis
        """
        try:
            prompt = f"""Analyze how well the resume matches the job requirements 
            and provide optimization suggestions.
            
            Resume Path: {resume_path}
            Job Requirements:
            {json.dumps(job_data, indent=2)}
            
            Provide analysis of:
            1. Keyword matches and missing keywords
            2. Areas of strength
            3. Areas for improvement
            4. ATS compatibility
            5. Overall match score
            """

            response = self.llm_client.generate(prompt)
            return json.loads(response['choices'][0]['message']['content'])

        except Exception as e:
            raise Exception(f"Failed to analyze resume optimization: {str(e)}")
