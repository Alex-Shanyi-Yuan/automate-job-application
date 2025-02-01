from typing import Dict, List, Tuple
import re
from pdfminer.high_level import extract_text
import os


class ATSScorer:
    def __init__(self):
        """Initialize ATS scorer with common scoring criteria."""
        self.keyword_weights = {
            'education': 0.15,
            'experience': 0.35,
            'skills': 0.30,
            'format': 0.20
        }

    def score_resume(self, pdf_path: str, job_data: Dict) -> Dict:
        """Score a resume against job requirements."""
        # Extract text from PDF
        resume_text = extract_text(pdf_path)

        # Calculate individual scores
        education_score = self._score_education(resume_text)
        experience_score = self._score_experience(resume_text, job_data)
        skills_score = self._score_skills(resume_text, job_data)
        format_score = self._score_format(resume_text)

        # Calculate weighted total
        total_score = (
            education_score * self.keyword_weights['education'] +
            experience_score * self.keyword_weights['experience'] +
            skills_score * self.keyword_weights['skills'] +
            format_score * self.keyword_weights['format']
        )

        # Generate detailed report
        report = self._generate_report(
            education_score,
            experience_score,
            skills_score,
            format_score,
            total_score
        )

        return {
            'total_score': total_score,
            'component_scores': {
                'education': education_score,
                'experience': experience_score,
                'skills': skills_score,
                'format': format_score
            },
            'report': report
        }

    def _score_education(self, text: str) -> float:
        """Score education section."""
        score = 0.0

        # Check for degree keywords
        degree_keywords = [
            'bachelor', 'master', 'phd', 'degree',
            'university', 'college', 'b.s.', 'm.s.',
            'computer science', 'engineering'
        ]

        for keyword in degree_keywords:
            if re.search(rf'\b{keyword}\b', text.lower()):
                score += 0.25

        return min(score, 1.0)

    def _score_experience(self, text: str, job_data: Dict) -> float:
        """Score experience section against job requirements."""
        score = 0.0

        # Extract years of experience requirement
        years_required = self._extract_years_required(job_data)

        # Check for years of experience
        years_pattern = r'(\d+)[\+]?\s*(?:years?|yrs?)'
        years_matches = re.findall(years_pattern, text, re.IGNORECASE)
        if years_matches:
            max_years = max(int(y) for y in years_matches)
            if max_years >= years_required:
                score += 0.5
            else:
                score += 0.3

        # Check for relevant experience keywords
        exp_keywords = self._extract_experience_keywords(job_data)
        for keyword in exp_keywords:
            if re.search(rf'\b{keyword}\b', text.lower()):
                score += 0.1

        return min(score, 1.0)

    def _score_skills(self, text: str, job_data: Dict) -> float:
        """Score skills section against job requirements."""
        score = 0.0

        # Extract required skills from job description
        required_skills = self._extract_required_skills(job_data)

        # Check for each required skill
        for skill in required_skills:
            if re.search(rf'\b{re.escape(skill)}\b', text, re.IGNORECASE):
                score += 1.0 / len(required_skills)

        return min(score, 1.0)

    def _score_format(self, text: str) -> float:
        """Score resume format and structure."""
        score = 0.0

        # Check for clear section headers
        if re.search(r'\b(education|experience|skills|projects)\b', text, re.IGNORECASE):
            score += 0.3

        # Check for consistent formatting
        if re.search(r'\d{4}\s*[-–]\s*(?:\d{4}|present)', text, re.IGNORECASE):
            score += 0.2

        # Check for bullet points
        if re.search(r'[•·]', text):
            score += 0.2

        # Check for contact information
        if re.search(r'(?:email|phone|linkedin)', text, re.IGNORECASE):
            score += 0.3

        return min(score, 1.0)

    def _extract_years_required(self, job_data: Dict) -> int:
        """Extract required years of experience from job description."""
        years_pattern = r'(\d+)[\+]?\s*(?:years?|yrs?)'
        years_matches = []

        # Search in requirements and description
        text_to_search = ' '.join([
            job_data['description'],
            *job_data['requirements']
        ])

        matches = re.findall(years_pattern, text_to_search, re.IGNORECASE)
        if matches:
            return max(int(y) for y in matches)
        return 0

    def _extract_experience_keywords(self, job_data: Dict) -> List[str]:
        """Extract experience-related keywords from job description."""
        keywords = set()

        # Common experience-related terms
        for text in job_data['responsibilities']:
            # Extract verbs and technical terms
            words = re.findall(r'\b\w+\b', text.lower())
            keywords.update([
                word for word in words
                if word in ['developed', 'managed', 'led', 'created', 'implemented',
                            'designed', 'architected', 'built', 'maintained']
            ])

        return list(keywords)

    def _extract_required_skills(self, job_data: Dict) -> List[str]:
        """Extract required skills from job description."""
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

    def _generate_report(
        self,
        education_score: float,
        experience_score: float,
        skills_score: float,
        format_score: float,
        total_score: float
    ) -> str:
        """Generate a detailed scoring report."""
        report = f"""ATS SCORING REPORT
=================

Overall Score: {total_score:.2%}

Component Scores:
----------------
Education:  {education_score:.2%}
Experience: {experience_score:.2%}
Skills:     {skills_score:.2%}
Format:     {format_score:.2%}

Analysis:
--------
"""
        # Add score-based feedback
        if total_score >= 0.85:
            report += "✓ Excellent match! Your resume is well-optimized for ATS.\n"
        elif total_score >= 0.70:
            report += "✓ Good match. Minor improvements could increase your score.\n"
        else:
            report += "⚠ Consider revising your resume to better match the job requirements.\n"

        # Component-specific feedback
        if education_score < 0.7:
            report += "- Education section could be more prominent or detailed.\n"
        if experience_score < 0.7:
            report += "- Experience section could better highlight relevant achievements.\n"
        if skills_score < 0.7:
            report += "- Skills section could better match job requirements.\n"
        if format_score < 0.7:
            report += "- Resume format could be more ATS-friendly.\n"

        return report

    def save_report(self, report: Dict, output_dir: str):
        """Save the ATS scoring report to a file."""
        report_path = os.path.join(output_dir, 'ats_report.txt')
        with open(report_path, 'w') as f:
            f.write(report['report'])
        return report_path
