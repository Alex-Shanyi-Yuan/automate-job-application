import requests
from bs4 import BeautifulSoup
from typing import Dict, Optional
import re
import time
from urllib.parse import urlparse


class JobParser:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }

    def _extract_linkedin_data(self, soup: BeautifulSoup) -> Dict:
        """Extract job data from LinkedIn posting."""
        job_data = {
            'title': '',
            'company': '',
            'location': '',
            'description': '',
            'requirements': [],
            'responsibilities': []
        }

        # Job title
        title_elem = soup.find('h1', class_='top-card-layout__title')
        if title_elem:
            job_data['title'] = title_elem.get_text(strip=True)

        # Company
        company_elem = soup.find('a', class_='topcard__org-name-link')
        if company_elem:
            job_data['company'] = company_elem.get_text(strip=True)

        # Location
        location_elem = soup.find('span', class_='topcard__flavor--bullet')
        if location_elem:
            job_data['location'] = location_elem.get_text(strip=True)

        # Description
        desc_elem = soup.find('div', class_='description__text')
        if desc_elem:
            job_data['description'] = desc_elem.get_text(strip=True)

            # Extract requirements and responsibilities
            text_blocks = desc_elem.find_all(['p', 'li'])
            current_section = None

            for block in text_blocks:
                text = block.get_text(strip=True).lower()

                if any(keyword in text for keyword in ['requirements', 'qualifications']):
                    current_section = 'requirements'
                elif any(keyword in text for keyword in ['responsibilities', 'duties']):
                    current_section = 'responsibilities'
                elif current_section and block.name == 'li':
                    job_data[current_section].append(
                        block.get_text(strip=True))

        return job_data

    def _extract_indeed_data(self, soup: BeautifulSoup) -> Dict:
        """Extract job data from Indeed posting."""
        job_data = {
            'title': '',
            'company': '',
            'location': '',
            'description': '',
            'requirements': [],
            'responsibilities': []
        }

        # Job title
        title_elem = soup.find('h1', class_='jobsearch-JobInfoHeader-title')
        if title_elem:
            job_data['title'] = title_elem.get_text(strip=True)

        # Company
        company_elem = soup.find('div', class_='jobsearch-InlineCompanyRating')
        if company_elem:
            job_data['company'] = company_elem.find('a').get_text(
                strip=True) if company_elem.find('a') else ''

        # Description
        desc_elem = soup.find('div', id='jobDescriptionText')
        if desc_elem:
            job_data['description'] = desc_elem.get_text(strip=True)

            # Parse lists for requirements and responsibilities
            lists = desc_elem.find_all(['ul', 'ol'])
            for lst in lists:
                items = lst.find_all('li')
                for item in items:
                    text = item.get_text(strip=True)
                    if any(keyword in text.lower() for keyword in ['years', 'degree', 'experience']):
                        job_data['requirements'].append(text)
                    elif any(keyword in text.lower() for keyword in ['develop', 'create', 'manage', 'lead']):
                        job_data['responsibilities'].append(text)

        return job_data

    def _extract_glassdoor_data(self, soup: BeautifulSoup) -> Dict:
        """Extract job data from Glassdoor posting."""
        job_data = {
            'title': '',
            'company': '',
            'location': '',
            'description': '',
            'requirements': [],
            'responsibilities': []
        }

        # Job title
        title_elem = soup.find('div', class_='e1tk4kwz4')
        if title_elem:
            job_data['title'] = title_elem.get_text(strip=True)

        # Company
        company_elem = soup.find('div', class_='e1tk4kwz1')
        if company_elem:
            job_data['company'] = company_elem.get_text(strip=True)

        # Description
        desc_elem = soup.find('div', class_='jobDescriptionContent')
        if desc_elem:
            job_data['description'] = desc_elem.get_text(strip=True)

            # Extract requirements and responsibilities
            sections = desc_elem.find_all(['p', 'li'])
            current_section = None

            for section in sections:
                text = section.get_text(strip=True).lower()

                if any(keyword in text for keyword in ['requirements', 'qualifications']):
                    current_section = 'requirements'
                elif any(keyword in text for keyword in ['responsibilities', 'duties']):
                    current_section = 'responsibilities'
                elif current_section and section.name == 'li':
                    job_data[current_section].append(
                        section.get_text(strip=True))

        return job_data

    def parse_job_posting(self, url: str) -> Dict:
        """Parse job posting from supported job sites."""
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            # Determine which parser to use based on URL
            domain = urlparse(url).netloc

            if 'linkedin.com' in domain:
                return self._extract_linkedin_data(soup)
            elif 'indeed.com' in domain:
                return self._extract_indeed_data(soup)
            elif 'glassdoor.com' in domain:
                return self._extract_glassdoor_data(soup)
            else:
                raise ValueError(f"Unsupported job site: {domain}")

        except requests.RequestException as e:
            raise Exception(f"Failed to fetch job posting: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to parse job posting: {str(e)}")

    def get_job_id(self, url: str) -> str:
        """Extract job ID from URL."""
        try:
            # Extract job ID from common patterns
            patterns = [
                r'jobs/(\d+)',  # LinkedIn
                r'jk=([a-zA-Z0-9]+)',  # Indeed
                r'jobListingId=(\d+)'  # Glassdoor
            ]

            for pattern in patterns:
                match = re.search(pattern, url)
                if match:
                    return match.group(1)

            # Fallback: use timestamp
            return str(int(time.time()))

        except Exception:
            return str(int(time.time()))
