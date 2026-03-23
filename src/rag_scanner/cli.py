"""
cli.py

Module 6 — CLI Entry Point for the RAG Injection Scanner.
Wires all modules together into a single runnable command.

Usage:
    rag-scan document.pdf
    rag-scan ./docs/
    rag-scan ./corpus/ --output report.json
    rag-scan ./corpus/ --strict
"""

import sys
import logging
from pathlib import Path
from typing import Any

import click
import pdfplumber
from bs4 import BeautifulSoup
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from rag_scanner.chunker import chunk_text
from rag_scanner.layer1_regex import scan_chunks
from rag_scanner.layer2_heuristic import score_chunks
from rag_scanner.layer3_llm import judge_flagged_chunks
from rag_scanner.classifier import classify_all_chunks
from rag_scanner.reporter import generate_report

logger = logging.getLogger(__name__)
console = Console()

# ── Constants ────────────────────────────────────────────────────────────────

SUPPORTED_EXTENSIONS: set[str] = {".txt", ".md", ".pdf", ".html"}

CHUNK_SIZE: int = 512
CHUNK_OVERLAP: int = 50


# ── File Loaders ─────────────────────────────────────────────────────────────

def load_txt(path: Path) -> str:
    """
    Load plain text or markdown file.

    Args:
        path: Path to .txt or .md file.

    Returns:
        Raw text content as string.

    Raises:
        OSError: If file cannot be read.
    """
    return path.read_text(encoding="utf-8", errors="replace")


def load_pdf(path: Path) -> str:
    """
    Extract text from a PDF file using pdfplumber.

    Args:
        path: Path to .pdf file.

    Returns:
        Concatenated text from all pages.

    Raises:
        Exception: If PDF cannot be parsed.
    """
    pages: list[str] = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
    return "\n".join(pages)


def load_html(path: Path) -> str:
    """
    Extract visible text from an HTML file using BeautifulSoup.
    Strips all tags, scripts, and styles.

    Args:
        path: Path to .html file.

    Returns:
        Visible text content as string.

    Raises:
        Exception: If HTML cannot be parsed.
    """
    raw = path.read_text(encoding="utf-8", errors="replace")
    soup = BeautifulSoup(raw, "html.parser")

    # Remove script and style elements — they contain no readable content
    for element in soup(["script", "style"]):
        element.decompose()

    return soup.get_text(separator="\n", strip=True)


def load_file(path: Path) -> str | None:
    """
    Load a file based on its extension.
    Returns None if the file type is unsupported or loading fails.

    Args:
        path: Path to the file to load.

    Returns:
        Raw text content, or None on failure.
    """
    ext = path.suffix.lower()

    try:
        if ext in {".txt", ".md"}:
            return load_txt(path)
        elif ext == ".pdf":
            return load_pdf(path)
        elif ext == ".html":
            return load_html(path)
        else:
            logger.warning("Unsupported file type: %s", path)
            return None

    except Exception as e:
        logger.error("Failed to load %s: %s", path, e)
        console.print(f"[red]Error loading {path.name}: {e}[/red]")
        return None


# ── Single File Scanner ───────────────────────────────────────────────────────

def scan_file(
    path: Path,
    output_path: Path | None = None,
    strict: bool = False,
) -> int:
    """
    Run the full scanning pipeline on a single file.

    Args:
        path:        Path to the file to scan.
        output_path: Optional path to save JSON report.
        strict:      If True, exit code 1 on SUSPICIOUS as well.

    Returns:
        Exit code: 0 (clean), 1 (suspicious), 2 (dangerous).
    """
    console.print(f"\n[bold cyan]Scanning:[/bold cyan] {path.name}")

    text = load_file(path)
    if text is None:
        console.print(f"[red]Skipping {path.name} — could not load file.[/red]")
        return 1

    if not text.strip():
        console.print(f"[yellow]Skipping {path.name} — file is empty.[/yellow]")
        return 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("Chunking document...", total=None)

        chunks = chunk_text(text, source=path.name,
                           chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP)
        progress.update(task, description=f"Layer 1 — regex scan ({len(chunks)} chunks)...")
        l1_results = scan_chunks(chunks)

        progress.update(task, description="Layer 2 — heuristic scoring...")
        l2_results = score_chunks(chunks)

        progress.update(task, description="Layer 3 — LLM judge (flagged chunks only)...")
        l3_results = judge_flagged_chunks(chunks, l1_results, l2_results)

        progress.update(task, description="Classifying risk levels...")
        classifications = classify_all_chunks(chunks, l1_results, l2_results, l3_results)

    report, exit_code = generate_report(
        source=path.name,
        classifications=classifications,
        output_path=output_path,
        print_terminal=True,
    )

    # Strict mode — treat SUSPICIOUS as a failure too
    if strict and exit_code == 0:
        summary = report["summary"]
        if summary["suspicious"] > 0:
            exit_code = 1

    return exit_code


# ── Directory Scanner ─────────────────────────────────────────────────────────

def scan_directory(
    directory: Path,
    output_dir: Path | None = None,
    strict: bool = False,
) -> int:
    """
    Scan all supported files in a directory.
    Continues scanning even if individual files fail.

    Args:
        directory:  Path to directory containing documents.
        output_dir: Optional directory to save per-file JSON reports.
        strict:     If True, exit code 1 on SUSPICIOUS as well.

    Returns:
        Highest exit code found across all files.
    """
    files = [
        f for f in sorted(directory.rglob("*"))
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not files:
        console.print(
            f"[yellow]No supported files found in {directory}[/yellow]\n"
            f"[dim]Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}[/dim]"
        )
        return 0

    console.print(
        f"[bold]Found {len(files)} file(s) to scan in {directory}[/bold]"
    )

    # Track highest exit code across all files
    max_exit_code: int = 0

    for file_path in files:
        output_path = None
        if output_dir:
            output_path = output_dir / f"{file_path.stem}_report.json"

        exit_code = scan_file(file_path, output_path, strict)
        max_exit_code = max(max_exit_code, exit_code)

    return max_exit_code


# ── CLI Entry Point ───────────────────────────────────────────────────────────

@click.command()
@click.argument(
    "target",
    type=click.Path(exists=True),
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Path to save JSON report. For directories, saves per-file reports.",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Exit with code 1 on SUSPICIOUS chunks (default: only on DANGEROUS).",
)
@click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging output.",
)
def main(
    target: str,
    output: str | None,
    strict: bool,
    verbose: bool,
) -> None:
    """
    RAG Injection Scanner — detect prompt injection payloads before
    they enter your vector store.

    TARGET can be a file or directory.

    Exit codes:
        0 — Clean
        1 — Suspicious (or Dangerous in --strict mode)
        2 — Dangerous
    """
    # ── Logging setup ─────────────────────────────────────────────────────────
    log_level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    # Suppress warning-level logs unless verbose — keeps terminal clean
    if not verbose:
        logging.getLogger("rag_scanner").setLevel(logging.ERROR)
    
    target_path = Path(target)
    output_path = Path(output) if output else None

    # ── Route to file or directory scanner ───────────────────────────────────
    if target_path.is_file():
        exit_code = scan_file(target_path, output_path, strict)

    elif target_path.is_dir():
        output_dir = output_path if output_path else Path("reports")
        exit_code = scan_directory(target_path, output_dir, strict)

    else:
        console.print(f"[red]Target not found: {target}[/red]")
        exit_code = 1

    sys.exit(exit_code)