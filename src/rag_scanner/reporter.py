"""
reporter.py

Module 5 of the RAG injection scanning pipeline.
Generates terminal output, JSON reports, and exit codes
from classification results.

Exit codes:
    0 — All chunks clean
    1 — At least one suspicious chunk, none dangerous
    2 — At least one dangerous chunk

These exit codes enable CI/CD pipeline integration.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

logger = logging.getLogger(__name__)
console = Console()

# ── Constants ────────────────────────────────────────────────────────────────

SCANNER_VERSION: str = "0.1.0"

RISK_CLEAN:      str = "CLEAN"
RISK_SUSPICIOUS: str = "SUSPICIOUS"
RISK_DANGEROUS:  str = "DANGEROUS"

EXIT_CLEAN:      int = 0
EXIT_SUSPICIOUS: int = 1
EXIT_DANGEROUS:  int = 2

# Rich color mapping per risk level
RISK_COLORS: dict[str, str] = {
    RISK_CLEAN:      "green",
    RISK_SUSPICIOUS: "yellow",
    RISK_DANGEROUS:  "red",
}

# Risk level ordering for overall risk calculation
RISK_ORDER: dict[str, int] = {
    RISK_CLEAN:      0,
    RISK_SUSPICIOUS: 1,
    RISK_DANGEROUS:  2,
}


# ── Summary Builder ───────────────────────────────────────────────────────────

def build_summary(classifications: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Build a summary dictionary from all chunk classifications.

    Args:
        classifications: List of classification results from classifier.py.

    Returns:
        Summary dictionary with counts and overall risk level.
    """
    clean_count      = sum(1 for r in classifications if r["risk_level"] == RISK_CLEAN)
    suspicious_count = sum(1 for r in classifications if r["risk_level"] == RISK_SUSPICIOUS)
    dangerous_count  = sum(1 for r in classifications if r["risk_level"] == RISK_DANGEROUS)

    # Overall risk = maximum risk found across all chunks
    if dangerous_count > 0:
        overall_risk = RISK_DANGEROUS
    elif suspicious_count > 0:
        overall_risk = RISK_SUSPICIOUS
    else:
        overall_risk = RISK_CLEAN

    return {
        "clean":        clean_count,
        "suspicious":   suspicious_count,
        "dangerous":    dangerous_count,
        "total":        len(classifications),
        "overall_risk": overall_risk,
    }


# ── Exit Code ─────────────────────────────────────────────────────────────────

def get_exit_code(overall_risk: str) -> int:
    """
    Map overall risk level to a process exit code.

    Args:
        overall_risk: One of CLEAN, SUSPICIOUS, DANGEROUS.

    Returns:
        Exit code integer: 0, 1, or 2.
    """
    mapping = {
        RISK_CLEAN:      EXIT_CLEAN,
        RISK_SUSPICIOUS: EXIT_SUSPICIOUS,
        RISK_DANGEROUS:  EXIT_DANGEROUS,
    }
    return mapping.get(overall_risk, EXIT_SUSPICIOUS)


# ── Terminal Output ───────────────────────────────────────────────────────────

def print_terminal_report(
    source: str,
    summary: dict[str, Any],
    classifications: list[dict[str, Any]],
) -> None:
    """
    Print a rich-formatted report to the terminal.

    Args:
        source:          Document name or path being scanned.
        summary:         Summary dictionary from build_summary().
        classifications: Full list of classification results.
    """
    overall_risk  = summary["overall_risk"]
    overall_color = RISK_COLORS[overall_risk]

    # ── Header panel ─────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold]RAG Injection Scanner[/bold]\n"
        f"[dim]Source:[/dim] {source}\n"
        f"[dim]Scanned:[/dim] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"[dim]Overall Risk:[/dim] [{overall_color}]{overall_risk}[/{overall_color}]",
        title="[bold cyan]Scan Report[/bold cyan]",
        box=box.ROUNDED,
    ))

    # ── Summary table ─────────────────────────────────────────────────────────
    summary_table = Table(box=box.SIMPLE, show_header=False)
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value")

    summary_table.add_row("Total chunks",  str(summary["total"]))
    summary_table.add_row(
        "Clean",
        f"[green]{summary['clean']}[/green]"
    )
    summary_table.add_row(
        "Suspicious",
        f"[yellow]{summary['suspicious']}[/yellow]"
    )
    summary_table.add_row(
        "Dangerous",
        f"[red]{summary['dangerous']}[/red]"
    )

    console.print(summary_table)

    # ── Flagged chunks detail ─────────────────────────────────────────────────
    flagged = [
        r for r in classifications
        if r["risk_level"] != RISK_CLEAN
    ]

    if not flagged:
        console.print("[green]✓ No injection payloads detected.[/green]\n")
        return

    console.print(f"\n[bold]Flagged Chunks ({len(flagged)}):[/bold]")

    detail_table = Table(box=box.SIMPLE)
    detail_table.add_column("Chunk",  style="dim", width=6)
    detail_table.add_column("Risk",   width=12)
    detail_table.add_column("L1",     width=5)
    detail_table.add_column("L2",     width=8)
    detail_table.add_column("L3",     width=12)
    detail_table.add_column("Reason", no_wrap=False)

    for r in flagged:
        color = RISK_COLORS[r["risk_level"]]
        l3_display = (
            f"{r['layer3_classification']} {r['layer3_confidence']:.2f}"
            if r.get("layer3_ran")
            else "skipped"
        )
        detail_table.add_row(
            str(r["chunk_index"]),
            f"[{color}]{r['risk_level']}[/{color}]",
            "✓" if r["layer1_flagged"] else "–",
            f"{r['layer2_score']:.3f}",
            l3_display,
            r["reason"][:120],
        )

    console.print(detail_table)
    console.print()


# ── JSON Report ───────────────────────────────────────────────────────────────

def build_json_report(
    source: str,
    summary: dict[str, Any],
    classifications: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Build a complete JSON report dictionary.

    Args:
        source:          Document name or path being scanned.
        summary:         Summary dictionary from build_summary().
        classifications: Full list of classification results.

    Returns:
        Complete report dictionary ready for JSON serialization.
    """
    return {
        "scan_metadata": {
            "source":           source,
            "timestamp":        datetime.now().isoformat(),
            "scanner_version":  SCANNER_VERSION,
            "total_chunks":     summary["total"],
        },
        "summary": summary,
        "chunks":  classifications,
    }


def save_json_report(
    report: dict[str, Any],
    output_path: Path,
) -> None:
    """
    Save a JSON report to disk.

    Args:
        report:      Complete report dictionary from build_json_report().
        output_path: Path where the JSON file should be written.

    Raises:
        OSError: If the file cannot be written.
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        logger.info("JSON report saved to %s", output_path)
        console.print(f"[dim]Report saved to: {output_path}[/dim]")
    except OSError as e:
        logger.error("Failed to save JSON report: %s", e)
        raise


# ── Full Report Runner ────────────────────────────────────────────────────────

def generate_report(
    source: str,
    classifications: list[dict[str, Any]],
    output_path: Path | None = None,
    print_terminal: bool = True,
) -> tuple[dict[str, Any], int]:
    """
    Generate complete report — terminal output and optional JSON file.

    Args:
        source:          Document name or path being scanned.
        classifications: Full list of classification results.
        output_path:     Optional path to save JSON report.
        print_terminal:  Whether to print rich terminal output.

    Returns:
        Tuple of (json_report dict, exit_code int).
    """
    summary   = build_summary(classifications)
    exit_code = get_exit_code(summary["overall_risk"])
    report    = build_json_report(source, summary, classifications)

    if print_terminal:
        print_terminal_report(source, summary, classifications)

    if output_path:
        save_json_report(report, output_path)

    logger.info(
        "Report generated | source=%s | overall_risk=%s | exit_code=%d",
        source, summary["overall_risk"], exit_code,
    )

    return report, exit_code