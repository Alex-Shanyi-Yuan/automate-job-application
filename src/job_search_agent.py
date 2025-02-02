from browser_use import Browser, BrowserConfig, Agent, Controller, ActionResult
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, SecretStr
from typing import Optional
import asyncio
import csv
from pathlib import Path
from PyPDF2 import PdfReader


# Configure the browser to connect to your Chrome instance
browser = Browser(
    config=BrowserConfig(
        chrome_instance_path='C:\Program Files\Google\Chrome\Application\chrome.exe',  # Windows path
        disable_security=True
    )
)

# Define the Job model


class Job(BaseModel):
    title: str
    link: str
    company: str
    fit_score: float
    id: Optional[int] = None
    location: Optional[str] = None
    salary: Optional[str] = None


# Set up the controller
controller = Controller()

# NOTE: This is the path to your CV file
CV = Path.cwd() / 'assets' / 'resume' / 'AlexYuanResume.pdf'

if not CV.exists():
    raise FileNotFoundError(f'CV file not found at {CV}')

# Define actions
@controller.action('Save jobs to file', param_model=Job)
def save_jobs(job: Job):
    with open('jobs.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            job.title, 
            job.company,
            job.link,
            job.salary, 
            job.location,
            job.id,  
        ])
    return 'Saved job to file'

@controller.action('Read my cv for context to fill forms')
def read_cv():
    pdf = PdfReader(CV)
    text = ''
    for page in pdf.pages:
        text += page.extract_text() or ''
    return ActionResult(extracted_content=text, include_in_memory=True)


# Define initial actions to open job search URLs
initial_actions = [
    # {'open_tab': {'url': 'https://amazon.jobs/en/search?offset=0&result_limit=10&sort=relevant&category_type=studentprograms&distanceType=Mi&radius=24km&latitude=&longitude=&loc_group_id=&loc_query=&base_query=&city=&country=&region=&county=&query_options=&'}},
    # {'open_tab': {'url': 'https://www.google.com/about/careers/applications/jobs/results#!t=jo&jid=127025001&'}},
    # {'open_tab': {'url': 'https://jobs.careers.microsoft.com/global/en/search?q=software%20engineer%20Intern&lc=Canada&lc=United%20States&l=en_us&pg=1&pgSz=20&o=Relevance&flt=true'}},
    # {'open_tab': {'url': 'https://www.metacareers.com/jobs?q=software%20engineer%20intern'}},
    {'open_tab': {'url': 'https://nvidia.wd5.myworkdayjobs.com/NVIDIAExternalCareerSite/7/refreshFacet/318c8bb6f553100021d223d9780d30be'}},
    # {'open_tab': {'url': 'https://jobs.apple.com/en-ca/search?search=software%20intern&sort=relevance&team=internships-STDNT-INTRN'}},
    # {'open_tab': {'url': 'https://www.linkedin.com/jobs/search/?currentJobId=4140796161&f_E=1&f_TPR=r86400&geoId=101174742&keywords=software%20engineer%20intern&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true'}},
    # {'open_tab': {'url': 'https://www.linkedin.com/jobs/search/?currentJobId=4096785875&f_E=1&f_TPR=r86400&geoId=103644278&keywords=software%20engineer%20intern&origin=JOB_SEARCH_PAGE_JOB_FILTER&refresh=true'}},
]

# Define the main function
async def main():
    ground_task = (
        'You are a professional job finder. '
        '1. Read my cv with read_cv. '
        '2. Search for software engineer internship job postings on the opened tabs ONLY and ignore empty tabs and you do not need to swtich t'
        '3. Filter job postings located in United States and posted within 1 day. '
        '4. Save the job details to a file.'
        '5. Close current tab after you finished'
    )

    model = ChatOpenAI(
        model='gpt-4o',
	    temperature=0.0,
    )

    agent = Agent(
        task=ground_task,
        llm=model,
        controller=controller,
        browser=browser,
        initial_actions=initial_actions
    )

    await agent.run()

if __name__ == '__main__':
    asyncio.run(main())
