import os
from typing import Dict, List, Optional
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER
import re
import subprocess
from ..rag.retriever import ResumeRetriever


class ResumeGenerator:
    def __init__(self, master_resume_path: str):
        """Initialize resume generator with master resume path."""
        self.master_resume_path = master_resume_path
        self.retriever = ResumeRetriever()

        # Load master resume into RAG system
        with open(master_resume_path, 'r') as f:
            master_resume = f.read()
        self.retriever.index_resume(master_resume)

    def _extract_skills_from_job(self, job_data: Dict) -> List[str]:
        """Extract relevant skills from job description."""
        skills = set()

        # Common technical skills pattern
        skill_pattern = r'\b(?:Python|Java|C\+\+|JavaScript|TypeScript|React|Angular|Vue|Node\.js|SQL|AWS|Docker|Git|REST|API|ML|AI|DevOps|CI/CD|Agile|Scrum)\b'

        # Search in description and requirements
        text_to_search = ' '.join([
            job_data['description'],
            *job_data['requirements'],
            *job_data['responsibilities']
        ])

        matches = re.finditer(skill_pattern, text_to_search, re.IGNORECASE)
        for match in matches:
            skills.add(match.group(0))

        return list(skills)

    def _generate_latex_resume(self, job_data: Dict, relevant_experience: List[Dict]) -> str:
        """Generate LaTeX content for the resume."""
        # Read the master resume template
        with open(self.master_resume_path, 'r') as f:
            template = f.read()

        # Extract skills from job posting
        required_skills = self._extract_skills_from_job(job_data)

        # Modify sections based on job requirements
        sections_to_update = {
            'Experience': relevant_experience,
            'Technical Skills': required_skills
        }

        # Update each section
        for section, content in sections_to_update.items():
            # Find section in template
            section_pattern = fr'\\section{{{section}}}.*?\\section'
            section_match = re.search(section_pattern, template, re.DOTALL)

            if section_match:
                # Generate new section content
                if section == 'Experience':
                    new_content = self._format_experience_section(content)
                elif section == 'Technical Skills':
                    new_content = self._format_skills_section(content)

                # Replace section in template
                template = template.replace(
                    section_match.group(0),
                    f'\\section{{{section}}}\n{new_content}\\section'
                )

        return template

    def _format_experience_section(self, experiences: List[Dict]) -> str:
        """Format experience entries in LaTeX."""
        formatted = []
        for exp in experiences:
            # Parse the experience text to extract components
            exp_text = exp['text']

            # Extract company and position if available
            company_match = re.search(
                r'\\resumeSubheading{(.+?)}{(.+?)}{(.+?)}{(.+?)}', exp_text)
            if company_match:
                formatted.append(exp_text)

        return '\n'.join(formatted)

    def _format_skills_section(self, skills: List[str]) -> str:
        """Format skills section in LaTeX."""
        return f"""\\begin{{itemize}}[leftmargin=0.15in, label={{}}]
    \\small{{\\item{{
     \\textbf{{Languages}}{{: {', '.join(sorted([s for s in skills if s.lower() in ['python', 'java', 'c++', 'javascript', 'typescript']]))}}} \\\\
     \\textbf{{Frameworks}}{{: {', '.join(sorted([s for s in skills if s.lower() in ['react', 'angular', 'vue', 'node.js']]))}}} \\\\
     \\textbf{{Tools}}{{: {', '.join(sorted([s for s in skills if s.lower() in ['git', 'docker', 'aws', 'jenkins', 'ci/cd']]))}}}
    }}}}
\\end{{itemize}}"""

    def generate_resume(self, job_data: Dict, output_dir: str) -> Dict[str, str]:
        """Generate a tailored resume for the job posting."""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Get relevant experience using RAG
        relevant_experience = self.retriever.get_relevant_experience(
            job_description=' '.join([
                job_data['description'],
                *job_data['requirements'],
                *job_data['responsibilities']
            ])
        )

        # Generate LaTeX content
        latex_content = self._generate_latex_resume(
            job_data, relevant_experience)

        # Save LaTeX file
        latex_path = os.path.join(output_dir, 'resume.tex')
        with open(latex_path, 'w') as f:
            f.write(latex_content)

        # Compile LaTeX to PDF
        pdf_path = os.path.join(output_dir, 'resume.pdf')
        try:
            subprocess.run(['pdflatex', '-output-directory', output_dir, latex_path],
                           check=True, capture_output=True)

            # Run twice for proper formatting
            subprocess.run(['pdflatex', '-output-directory', output_dir, latex_path],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            raise Exception(f"Failed to compile LaTeX: {e.stderr.decode()}")

        # Generate cover letter
        cover_letter = self._generate_cover_letter(
            job_data, relevant_experience)
        cover_letter_path = os.path.join(output_dir, 'cover_letter.txt')
        with open(cover_letter_path, 'w') as f:
            f.write(cover_letter)

        # Generate company interest paragraph
        interest_para = self._generate_company_interest(job_data)
        interest_path = os.path.join(output_dir, 'company_interest.txt')
        with open(interest_path, 'w') as f:
            f.write(interest_para)

        return {
            'resume_pdf': pdf_path,
            'cover_letter': cover_letter_path,
            'company_interest': interest_path
        }

    def _generate_cover_letter(self, job_data: Dict, relevant_experience: List[Dict]) -> str:
        """Generate a cover letter based on job requirements and experience."""
        # Template for cover letter
        template = f"""Dear Hiring Manager,

I am writing to express my strong interest in the {job_data['title']} position at {job_data['company']}. With my background in computer engineering and extensive experience in software development, I am confident in my ability to contribute significantly to your team.

Based on the job requirements, I have particularly relevant experience in:
{self._format_relevant_points(relevant_experience)}

I am excited about the opportunity to bring my technical expertise and problem-solving skills to {job_data['company']} and contribute to your team's success.

Thank you for considering my application. I look forward to discussing how I can contribute to your team.

Best regards,
Alex Yuan"""

        return template

    def _generate_company_interest(self, job_data: Dict) -> str:
        """Generate a paragraph expressing interest in the company."""
        template = f"""I am particularly drawn to {job_data['company']} because of its reputation for innovation and technical excellence. The opportunity to work on {job_data['title']} aligns perfectly with my career goals and technical interests. I am excited about the prospect of contributing to {job_data['company']}'s continued success and growth while working alongside talented professionals in a collaborative environment."""

        return template

    def _format_relevant_points(self, experiences: List[Dict]) -> str:
        """Format relevant experience points for cover letter."""
        points = []
        for exp in experiences:
            # Extract bullet points from experience
            bullets = re.findall(r'\\resumeItem{(.+?)}', exp['text'])
            points.extend(bullets[:2])  # Take top 2 most relevant points

        # Clean and format points
        formatted_points = []
        for point in points:
            # Remove LaTeX formatting
            clean_point = re.sub(r'\\textbf{(.+?)}', r'\1', point)
            formatted_points.append(f"• {clean_point}")

        return '\n'.join(formatted_points)
