# Automated Job Application System

A Python-based system for automating the job application process, including resume tailoring, cover letter generation, and ATS optimization.

## Features

- **Resume Generation**: Automatically generate tailored resumes based on job requirements
- **Cover Letter Creation**: Generate customized cover letters matching job and company
- **ATS Optimization**: Analyze and optimize documents for ATS compatibility
- **LaTeX Support**: Professional PDF generation using LaTeX templates
- **Multiple LLM Support**: Flexible architecture supporting different LLM providers

## Prerequisites

- Python 3.9+
- LaTeX installation (for PDF generation)
- Ollama or other supported LLM provider
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/automate-job-application.git
cd automate-job-application
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
# Create .env file
cp .env.example .env

# Edit .env with your configuration
LLM_PROVIDER=ollama
LLM_MODEL=deepseek
OLLAMA_BASE_URL=http://localhost:11434
```

## Usage

### Basic Usage

Generate application documents from a job posting:

```bash
python src/main.py \
    --job-url "https://example.com/job-posting" \
    --company "Example Corp" \
    --position "Senior Software Engineer" \
    --job-id "12345"
```

### Using Local Job Data

If you have job data in a JSON file:

```bash
python src/main.py \
    --job-data "path/to/job-data.json" \
    --company "Example Corp" \
    --position "Senior Software Engineer" \
    --job-id "12345"
```

### Custom Configuration

Use a custom configuration file:

```bash
python src/main.py \
    --job-url "https://example.com/job-posting" \
    --company "Example Corp" \
    --position "Senior Software Engineer" \
    --job-id "12345" \
    --config "path/to/config.json"
```

## Configuration

### Job Data Format

The job data JSON should follow this structure:

```json
{
    "title": "Senior Software Engineer",
    "company": "Example Corp",
    "location": "Remote",
    "description": "We're looking for...",
    "requirements": [
        "5+ years of experience",
        "Strong Python skills",
        "Cloud platform experience"
    ],
    "responsibilities": [
        "Design and implement solutions",
        "Lead technical projects",
        "Mentor junior developers"
    ]
}
```

### Configuration File

The configuration file supports these options:

```json
{
    "llm": {
        "provider": "ollama",
        "model": "deepseek",
        "base_url": "http://localhost:11434"
    },
    "output": {
        "format": "pdf",
        "template": "modern"
    },
    "optimization": {
        "target_ats_score": 85,
        "keyword_threshold": 0.8
    }
}
```

## Output Structure

Generated files are organized in the `out` directory:

```
out/
└── CompanyName_Position_JobID_Timestamp/
    ├── resume.pdf
    ├── cover_letter.txt
    ├── company_interest.txt
    ├── job_data.json
    └── optimization.json
```

## Development

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/agents/test_resume_agent.py

# Run with coverage report
pytest --cov=src
```

### Code Style

The project uses:
- Black for code formatting
- Flake8 for linting
- MyPy for type checking
- isort for import sorting

Run all checks:

```bash
black src tests
flake8 src tests
mypy src tests
isort src tests
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
