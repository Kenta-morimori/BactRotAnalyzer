"""Output locations for current and archived issue #38 diagnostics."""

from pathlib import Path

REPORT_ROOT = Path(__file__).resolve().parents[2] / "docs/analysis/issue-38"
FIGURES = REPORT_ROOT / "figures"
TABLES = REPORT_ROOT / "tables"
LEGACY = REPORT_ROOT / "archive/legacy"


def ensure_output_dirs() -> None:
    """Create current diagnostic output folders when running from a clean checkout."""
    for directory in (FIGURES, TABLES):
        directory.mkdir(parents=True, exist_ok=True)
