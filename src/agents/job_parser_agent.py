from typing import Dict, List, Optional
from langchain.agents import AgentType, initialize_agent
from langchain.tools import BaseTool, StructuredTool, Tool
from langchain.tools.browser import BrowserTools
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate
from langchain.pydantic_v1 import BaseModel, Field
import json
import re
from urllib.parse import urlparse
import time
from ..llm.base import BaseLLMClient


class JobData(BaseModel):
    """Schema for job posting data."""
    title: str = Field(description="Job title")
    company: str = Field(description="Company name")
    location: str = Field(description="Job location")
    description: str = Field(description="Brief job description")
    requirements: List[str] = Field(description="List of core requirements")
    technical_skills: List[str] = Field(description="List of technical skills")
    soft_skills: List[str] = Field(description="List of soft skills")
    experience_level: str = Field(
        description="Required years/level of experience")
    education: str = Field(description="Required education level")
    ats_keywords: List[str] = Field(description="Important keywords for ATS")
    company_values: List[str] = Field(description="Company culture indicators")
    responsibilities: List[str] = Field(description="Key job responsibilities")


class JobParserAgent:
    def __init__(self, llm_client: BaseLLMClient):
        """Initialize job parser agent with LLM client."""
        self.llm_client = llm_client
        self.setup_agent()

    def setup_agent(self):
        """Set up LangChain agent with tools."""
        # Initialize browser tools
        self.browser_tools = BrowserTools()

        # Create custom tools
        self.tools = [
            # URL content extraction tool
            Tool(
                name="fetch_url_content",
                description="Fetch and extract content from a URL",
                func=self._fetch_url_content
            ),
            # Content parsing tool
            Tool(
                name="parse_job_content",
                description="Parse job posting content into structured data",
                func=self._parse_job_content
            ),
            # ATS analysis tool
            Tool(
                name="analyze_ats_requirements",
                description="Analyze job content for ATS optimization",
                func=self._analyze_ats_requirements
            )
        ]

        # Add browser tools
        self.tools.extend(self.browser_tools.get_tools())

        # Initialize agent
        self.agent = initialize_agent(
            tools=self.tools,
            llm=self.llm_client,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

    def _fetch_url_content(self, url: str) -> str:
        """Fetch and extract content from URL."""
        try:
            # Use UnstructuredURLLoader for content extraction
            loader = UnstructuredURLLoader(urls=[url])
            documents = loader.load()

            # Split content into manageable chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=2000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)

            # Combine relevant chunks
            content = "\n".join([chunk.page_content for chunk in chunks])
            return content

        except Exception as e:
            raise Exception(f"Failed to fetch URL content: {str(e)}")

    def _parse_job_content(self, content: str) -> Dict:
        """Parse job content into structured data."""
        # Define output schema
        response_schemas = [
            ResponseSchema(name="title", description="Job title"),
            ResponseSchema(name="company", description="Company name"),
            ResponseSchema(name="location", description="Job location"),
            ResponseSchema(name="description",
                           description="Brief job description"),
            ResponseSchema(name="requirements",
                           description="List of core requirements"),
            ResponseSchema(name="technical_skills",
                           description="List of technical skills"),
            ResponseSchema(name="soft_skills",
                           description="List of soft skills"),
            ResponseSchema(name="experience_level",
                           description="Required years/level of experience"),
            ResponseSchema(name="education",
                           description="Required education level"),
            ResponseSchema(name="ats_keywords",
                           description="Important keywords for ATS"),
            ResponseSchema(name="company_values",
                           description="Company culture indicators"),
            ResponseSchema(name="responsibilities",
                           description="Key job responsibilities")
        ]

        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        format_instructions = parser.get_format_instructions()

        # Create parsing prompt
        template = """Extract key information from the job posting content below.
Follow these guidelines:
1. Identify all required and preferred qualifications
2. Extract technical and soft skills separately
3. Determine experience level and education requirements
4. Identify company culture indicators
5. Extract key responsibilities

Content:
{content}

{format_instructions}"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["content"],
            partial_variables={"format_instructions": format_instructions}
        )

        # Generate and parse response
        try:
            _input = prompt.format_prompt(content=content)
            response = self.llm_client.generate(prompt=_input.to_string())
            parsed_data = parser.parse(
                response['choices'][0]['message']['content'])
            return JobData(**parsed_data).dict()
        except Exception as e:
            raise Exception(f"Failed to parse job content: {str(e)}")

    def _analyze_ats_requirements(self, job_data: Dict) -> Dict:
        """Analyze job requirements for ATS optimization."""
        try:
            return self.llm_client.analyze_ats_requirements(job_data)
        except Exception as e:
            raise Exception(f"Failed to analyze ATS requirements: {str(e)}")

    def _extract_job_id(self, url: str) -> str:
        """Extract job ID from URL."""
        try:
            patterns = [
                r'jobs/(\d+)',  # LinkedIn
                r'jk=([a-zA-Z0-9]+)',  # Indeed
                r'jobListingId=(\d+)',  # Glassdoor
                r'jobs?[-/_](?:view/)?(\d+)',  # Generic
                r'positions?[/_](\d+)',  # Generic
            ]

            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)

            # Fallback: use domain + timestamp
            domain = urlparse(url).netloc.split('.')[0]
            return f"{domain}_{int(time.time())}"

        except Exception:
            return str(int(time.time()))

    def parse_job_posting(self, url: str) -> Dict:
        """Parse job posting from URL using LangChain agent."""
        try:
            # Execute agent to process job posting
            result = self.agent.run({
                "input": f"Parse job posting from URL: {url}",
                "url": url
            })

            # Extract job data from agent result
            if isinstance(result, str):
                result = json.loads(result)

            # Add job ID
            result['job_id'] = self._extract_job_id(url)

            return result

        except Exception as e:
            raise Exception(f"Failed to parse job posting: {str(e)}")
