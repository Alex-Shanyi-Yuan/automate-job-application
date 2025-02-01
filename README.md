# Automated Job Application System

A Python-based system that automates the process of tailoring resumes and generating supporting documents for job applications. The system uses RAG (Retrieval-Augmented Generation) to create highly relevant, ATS-optimized resumes based on job postings.

## Features

- **Job Posting Analysis**: Automatically extracts key information from job postings on LinkedIn, Indeed, and Glassdoor
- **Resume Customization**: Uses RAG to tailor your resume to specific job requirements
- **ATS Optimization**: Scores and optimizes resumes for Applicant Tracking Systems
- **Document Generation**: Creates customized resumes, cover letters, and company interest statements
- **Multiple Job Site Support**: Works with LinkedIn, Indeed, and Glassdoor job postings

## Prerequisites

- Python 3.9+
- LaTeX installation (for PDF generation)
- Ollama with nomic-embed-text model installed

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/automate-job-application.git
cd automate-job-application
```

2. Install required Python packages:
```bash
pip install -r requirements.txt
```

3. Ensure Ollama is running with the nomic-embed-text model:
```bash
ollama run nomic-embed-text
```

## Usage

1. Place your master resume in LaTeX format in the `resume` directory.

2. Run the application with a job posting URL:
```bash
python src/main.py "https://www.linkedin.com/jobs/view/job-id"
```

Optional: Specify a different master resume template:
```bash
python src/main.py "https://www.linkedin.com/jobs/view/job-id" --resume path/to/resume.tex
```

The system will:
1. Parse the job posting
2. Generate a tailored resume using RAG
3. Create a cover letter and company interest statement
4. Score the resume using ATS algorithms
5. Save all documents in `out/<company>_<position>_<jobid>/`

## Output Files

For each job application, the system generates:
- `resume.pdf`: Tailored resume in PDF format
- `cover_letter.txt`: Customized cover letter
- `company_interest.txt`: Statement of interest in the company
- `ats_report.txt`: ATS scoring report with detailed feedback

## Project Structure

```
automate-job-application/
├── src/
│   ├── rag/
│   │   ├── embeddings.py      # Nomic embeddings implementation
│   │   ├── vector_store.py    # Vector store using FAISS
│   │   └── retriever.py       # RAG system implementation
│   ├── scraper/
│   │   ├── job_parser.py      # Job posting extraction
│   │   └── html_extractor.py  # HTML content extraction
│   ├── resume/
│   │   ├── generator.py       # Resume generation logic
│   │   └── ats_scorer.py      # ATS scoring implementation
│   └── main.py               # Main application entry
├── resume/                   # Master resume templates
├── out/                     # Generated documents
└── requirements.txt         # Python dependencies
```

## Customization

- Modify `src/resume/generator.py` to adjust resume formatting and content selection
- Update `src/resume/ats_scorer.py` to customize ATS scoring criteria
- Edit `src/scraper/job_parser.py` to add support for additional job sites

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
