import os
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Union
import subprocess
from datetime import datetime


def ensure_dir(path: Union[str, Path]) -> Path:
    """Ensure directory exists, create if necessary.

    Args:
        path: Directory path to ensure exists

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def create_output_dir(company: str, position: str, job_id: str) -> Path:
    """Create output directory for application documents.

    Args:
        company: Company name
        position: Position title
        job_id: Job posting ID

    Returns:
        Path to created directory
    """
    # Sanitize inputs for directory name
    company = "".join(c for c in company if c.isalnum() or c in "- _")
    position = "".join(c for c in position if c.isalnum() or c in "- _")
    job_id = "".join(c for c in job_id if c.isalnum() or c in "- _")

    # Create directory name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"{company}_{position}_{job_id}_{timestamp}"

    # Create directory
    output_dir = Path("out") / dir_name
    return ensure_dir(output_dir)


def save_json(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save data as JSON file.

    Args:
        data: Data to save
        path: Path to save JSON file
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: Union[str, Path]) -> Dict[str, Any]:
    """Load data from JSON file.

    Args:
        path: Path to JSON file

    Returns:
        Loaded data
    """
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def copy_file(src: Union[str, Path], dst: Union[str, Path]) -> None:
    """Copy file from source to destination.

    Args:
        src: Source file path
        dst: Destination file path
    """
    shutil.copy2(src, dst)


def read_text_file(path: Union[str, Path]) -> str:
    """Read text file content.

    Args:
        path: Path to text file

    Returns:
        File content as string
    """
    with open(path, 'r', encoding='utf-8') as f:
        return f.read()


def write_text_file(content: str, path: Union[str, Path]) -> None:
    """Write content to text file.

    Args:
        content: Content to write
        path: Path to text file
    """
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)


def compile_latex(tex_path: Union[str, Path], output_dir: Optional[Union[str, Path]] = None) -> Path:
    """Compile LaTeX document to PDF.

    Args:
        tex_path: Path to LaTeX source file
        output_dir: Optional output directory (defaults to source directory)

    Returns:
        Path to generated PDF

    Raises:
        subprocess.CalledProcessError: If compilation fails
    """
    tex_path = Path(tex_path)
    output_dir = Path(output_dir) if output_dir else tex_path.parent
    ensure_dir(output_dir)

    # Copy tex file to output directory if different
    if output_dir != tex_path.parent:
        copy_file(tex_path, output_dir / tex_path.name)
        tex_path = output_dir / tex_path.name

    # Run pdflatex twice to resolve references
    for _ in range(2):
        subprocess.run(
            ['pdflatex', '-interaction=nonstopmode', tex_path.name],
            cwd=output_dir,
            check=True,
            capture_output=True
        )

    return output_dir / tex_path.with_suffix('.pdf').name


def cleanup_latex_files(directory: Union[str, Path], base_name: str) -> None:
    """Clean up auxiliary LaTeX files.

    Args:
        directory: Directory containing LaTeX files
        base_name: Base name of LaTeX files (without extension)
    """
    extensions = ['.aux', '.log', '.out', '.toc']
    directory = Path(directory)

    for ext in extensions:
        aux_file = directory / f"{base_name}{ext}"
        if aux_file.exists():
            aux_file.unlink()


def get_file_hash(path: Union[str, Path]) -> str:
    """Get SHA-256 hash of file content.

    Args:
        path: Path to file

    Returns:
        Hex string of file hash
    """
    import hashlib

    path = Path(path)
    sha256_hash = hashlib.sha256()

    with open(path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    return sha256_hash.hexdigest()


def is_file_newer(file1: Union[str, Path], file2: Union[str, Path]) -> bool:
    """Check if file1 is newer than file2.

    Args:
        file1: Path to first file
        file2: Path to second file

    Returns:
        True if file1 is newer than file2, False otherwise
    """
    file1, file2 = Path(file1), Path(file2)

    if not file1.exists() or not file2.exists():
        return False

    return file1.stat().st_mtime > file2.stat().st_mtime


def get_project_root() -> Path:
    """Get project root directory.

    Returns:
        Path to project root
    """
    return Path(__file__).parent.parent.parent


def get_resume_template_dir() -> Path:
    """Get resume template directory.

    Returns:
        Path to resume template directory
    """
    return get_project_root() / "resume" / "templates"


def get_output_dir() -> Path:
    """Get output directory.

    Returns:
        Path to output directory
    """
    return get_project_root() / "out"
