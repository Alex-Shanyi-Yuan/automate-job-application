import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

from src.llm.factory import LLMFactory
from src.agents.resume_agent import ResumeAgent
from src.utils.file_utils import (
    create_output_dir,
    read_text_file,
    save_json,
    load_json
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate tailored resumes and application documents"
    )

    parser.add_argument(
        "--job-url",
        type=str,
        help="URL of the job posting"
    )

    parser.add_argument(
        "--job-data",
        type=str,
        help="Path to JSON file containing job data"
    )

    parser.add_argument(
        "--master-resume",
        type=str,
        default="resume/masterResume.tex",
        help="Path to master resume template (default: resume/masterResume.tex)"
    )

    parser.add_argument(
        "--company",
        type=str,
        required=True,
        help="Company name for output directory"
    )

    parser.add_argument(
        "--position",
        type=str,
        required=True,
        help="Position title for output directory"
    )

    parser.add_argument(
        "--job-id",
        type=str,
        required=True,
        help="Job posting ID for output directory"
    )

    parser.add_argument(
        "--llm-provider",
        type=str,
        default=None,
        help="LLM provider to use (default: from environment)"
    )

    parser.add_argument(
        "--llm-model",
        type=str,
        default=None,
        help="LLM model to use (default: from environment)"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration JSON file"
    )

    return parser.parse_args()


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults.

    Args:
        config_path: Optional path to configuration file

    Returns:
        Configuration dictionary
    """
    if config_path:
        return load_json(config_path)
    return {}


def process_job_data(
    job_url: Optional[str],
    job_data_path: Optional[str]
) -> Dict[str, Any]:
    """Process job posting data from URL or file.

    Args:
        job_url: Optional job posting URL
        job_data_path: Optional path to job data JSON

    Returns:
        Parsed job data dictionary

    Raises:
        ValueError: If neither job_url nor job_data_path is provided
    """
    if not job_url and not job_data_path:
        raise ValueError("Must provide either job URL or job data file")

    if job_data_path:
        return load_json(job_data_path)

    # TODO: Implement job parsing from URL
    raise NotImplementedError("Job URL parsing not yet implemented")


def main() -> None:
    """Main entry point for resume generation."""
    args = parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Create LLM client
        llm_client = LLMFactory.create_llm(
            provider=args.llm_provider,
            model=args.llm_model,
            config=config.get("llm", {})
        )

        # Create resume agent
        agent = ResumeAgent(llm_client)

        # Process job data
        job_data = process_job_data(args.job_url, args.job_data)

        # Create output directory
        output_dir = create_output_dir(
            args.company,
            args.position,
            args.job_id
        )

        # Save job data
        save_json(job_data, output_dir / "job_data.json")

        # Load master resume
        master_resume = read_text_file(args.master_resume)

        # Extract relevant experience
        relevant_experience = agent.llm_client.extract_relevant_experience(
            experience=[],  # TODO: Load from master resume
            job_requirements=job_data
        )

        # Generate application documents
        result = agent.generate_application_documents(
            job_data=job_data,
            relevant_experience=relevant_experience,
            master_resume_path=args.master_resume,
            output_dir=str(output_dir)
        )

        # Analyze optimization
        optimization = agent.analyze_resume_optimization(
            resume_path=result["resume_pdf"],
            job_data=job_data
        )

        # Save optimization analysis
        save_json(optimization, output_dir / "optimization.json")

        print(f"\nGenerated application documents in: {output_dir}")
        print("\nFiles generated:")
        for key, path in result.items():
            print(f"- {key}: {path}")
        print(f"\nATS Score: {optimization.get('overall_match_score', 'N/A')}")

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
