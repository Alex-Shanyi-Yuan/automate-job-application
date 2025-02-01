# Automated Job Application System

A Python-based system that automates the process of tailoring resumes and generating supporting documents for job applications. The system uses RAG (Retrieval-Augmented Generation) and LLMs to create highly relevant, ATS-optimized resumes based on job postings.

## Features

- **Job Posting Analysis**: Automatically extracts key information from any job posting URL
- **Resume Customization**: Uses RAG and LLMs to tailor your resume to specific job requirements
- **ATS Optimization**: Scores and optimizes resumes for Applicant Tracking Systems
- **Document Generation**: Creates customized resumes, cover letters, and company interest statements
- **Flexible LLM Support**: Works with multiple LLM providers (NVIDIA AI, Ollama)
- **Intelligent Parsing**: Uses LLMs to understand job requirements from any source

## Prerequisites

- Python 3.9+
- LaTeX installation (for PDF generation)
- One of the following LLM providers:
  * NVIDIA AI account with API key
  * Ollama installation with deepseek-coder model

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

3. Set up your environment:
```bash
cp .env.example .env
# Edit .env with your configuration
```

4. If using Ollama, ensure it's running with the required model:
```bash
ollama run deepseek-coder:latest
```

## Configuration

The system can be configured through environment variables:

```env
# LLM Provider Selection
LLM_PROVIDER=nvidia  # or 'ollama'
NVIDIA_API_KEY=your-key-here  # Required for NVIDIA AI
LLM_MODEL=deepseek-ai/deepseek-r1  # Optional, provider-specific model name

# Other Configuration
OUTPUT_DIR=out
MASTER_RESUME_PATH=resume/masterResume.tex
```

## Usage

1. Place your master resume in LaTeX format in the `resume` directory.

2. Run the application with a job posting URL:
```bash
python src/main.py "https://any-job-posting-url"
```

Optional: Specify a different master resume template:
```bash
python src/main.py "https://any-job-posting-url" --resume path/to/resume.tex
```

The system will:
1. Parse the job posting using LLM-powered analysis
2. Generate a tailored resume using RAG and LLM
3. Create a cover letter and company interest statement
4. Score the resume using ATS algorithms
5. Save all documents in `out/<company>_<position>_<jobid>/`

## Output Files

For each job application, the system generates:
- `resume.pdf`: Tailored resume in PDF format
- `cover_letter.txt`: Customized cover letter
- `company_interest.txt`: Statement of interest in the company
- `optimization_report.json`: Detailed ATS analysis and optimization report

## Project Structure

```
automate-job-application/
├── src/
│   ├── agents/
│   │   ├── job_parser_agent.py   # LLM-powered job parsing
│   │   └── resume_agent.py       # Resume generation agent
│   ├── llm/
│   │   ├── base.py              # Base LLM interface
│   │   ├── nvidia_llm.py        # NVIDIA AI implementation
│   │   ├── ollama_llm.py        # Ollama implementation
│   │   └── factory.py           # LLM factory
│   ├── rag/
│   │   ├── embeddings.py        # Embeddings implementation
│   │   ├── vector_store.py      # FAISS vector store
│   │   └── retriever.py         # RAG system
│   └── main.py                  # Main application
├── resume/                      # Master resume templates
├── out/                        # Generated documents
└── requirements.txt            # Python dependencies
```

## Customization

- Modify `src/agents/resume_agent.py` to adjust document generation
- Update `src/llm/` to add support for additional LLM providers
- Edit prompts in agents to customize LLM behavior

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
