import os
import argparse
from agents.job_parser_agent import JobParserAgent
from agents.resume_agent import ResumeAgent
from rag.retriever import ResumeRetriever
from typing import Dict
import json
from dotenv import load_dotenv
from llm.factory import LLMFactory


def load_config() -> Dict:
    """Load configuration from environment variables."""
    load_dotenv()

    config = {
        'llm_provider': os.getenv('LLM_PROVIDER', 'nvidia').lower(),
        'llm_model': os.getenv('LLM_MODEL', ''),
        'nvidia_api_key': os.getenv('NVIDIA_API_KEY', '')
    }

    # Validate configuration
    if config['llm_provider'] == 'nvidia' and not config['nvidia_api_key']:
        raise ValueError(
            "NVIDIA_API_KEY environment variable not set. Please set it with your Nvidia API key."
        )

    # Use default model if not specified
    if not config['llm_model']:
        config['llm_model'] = LLMFactory.get_default_model(
            config['llm_provider'])

    return config


def process_job_application(job_url: str, master_resume_path: str, config: Dict) -> Dict[str, str]:
    """Process a job application by generating tailored resume and supporting documents."""
    try:
        # Create LLM client
        llm = LLMFactory.create_llm(
            provider=config['llm_provider'],
            api_key=config.get('nvidia_api_key'),
            model_name=config['llm_model']
        )

        # Initialize agents
        job_parser = JobParserAgent(llm)
        resume_agent = ResumeAgent(llm)

        # Parse job posting
        print("\nParsing job posting...")
        job_data = job_parser.parse_job_posting(job_url)

        # Initialize RAG system
        print("\nInitializing RAG system...")
        retriever = ResumeRetriever()

        # Load master resume into RAG
        print("Loading master resume...")
        with open(master_resume_path, 'r') as f:
            master_resume = f.read()
        retriever.index_resume(master_resume)

        # Get relevant experience using RAG
        print("Retrieving relevant experience...")
        relevant_experience = retriever.get_relevant_experience(
            job_description=' '.join([
                job_data['description'],
                *job_data['requirements'],
                *job_data['responsibilities']
            ])
        )

        # Create output directory
        company = job_data['company'].replace(' ', '_')
        position = job_data['title'].replace(' ', '_')
        job_id = job_data['job_id']
        output_dir = os.path.join('out', f"{company}_{position}_{job_id}")
        os.makedirs(output_dir, exist_ok=True)

        # Generate application documents
        print("\nGenerating application documents...")
        output_files = resume_agent.generate_application_documents(
            job_data,
            relevant_experience,
            master_resume_path,
            output_dir
        )

        # Analyze resume optimization
        print("\nAnalyzing resume optimization...")
        optimization_results = resume_agent.analyze_resume_optimization(
            output_files['resume_pdf'],
            job_data
        )

        # Save optimization results
        optimization_path = os.path.join(
            output_dir, 'optimization_report.json')
        with open(optimization_path, 'w') as f:
            json.dump(optimization_results, f, indent=2)

        # Add optimization report to output files
        output_files['optimization_report'] = optimization_path

        print(f"\nJob Application Processing Complete!")
        print(f"Output files saved to: {output_dir}")
        print(
            f"\nATS Match Score: {optimization_results['overall_match_score']}")

        return output_files

    except Exception as e:
        print(f"Error processing job application: {str(e)}")
        raise


def main():
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(
        description='Generate tailored resume and supporting documents for job applications'
    )
    parser.add_argument(
        'job_url',
        help='URL of the job posting'
    )
    parser.add_argument(
        '--resume',
        default='resume/masterResume.tex',
        help='Path to master resume template (default: resume/masterResume.tex)'
    )

    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config()

        # Process job application
        output_files = process_job_application(
            args.job_url, args.resume, config)

        # Print file locations
        print("\nGenerated Files:")
        print(f"Resume PDF: {output_files['resume_pdf']}")
        print(f"Cover Letter: {output_files['cover_letter']}")
        print(f"Company Interest: {output_files['company_interest']}")
        print(f"Optimization Report: {output_files['optimization_report']}")

    except Exception as e:
        print(f"\nError: {str(e)}")
        exit(1)


if __name__ == '__main__':
    main()
