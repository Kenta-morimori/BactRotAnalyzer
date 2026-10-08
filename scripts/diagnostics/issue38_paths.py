"""Output locations for current and archived issue #38 diagnostics."""

from pathlib import Path

REPORT_ROOT = Path(__file__).resolve().parents[2] / "docs/analysis/issue-38"
FIGURES = REPORT_ROOT / "figures"
TABLES = REPORT_ROOT / "tables"


METHOD_COLORS = {"legacy": "#0072b2", "ellipse_constrained": "#009e73", "geometric_ellipse": "#d55e00"}

WINDOWS = {
    "initial_invalid": 15.781025,
    "early_displaced": 20.789,
    "rise_crossing": 38.318,
    "later_reference": 64.862,
    "terminal_hold": 82.500132,
}


def ensure_output_dirs() -> None:
    """Create current diagnostic output folders when running from a clean checkout."""
    for directory in (FIGURES, TABLES):
        directory.mkdir(parents=True, exist_ok=True)
