"""Per-run response outputs, independent of canonical input/cache locations."""

from __future__ import annotations

import json
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

from utils import param


@dataclass
class ResponseOutput:
    day: str
    root: Path
    mode: str
    run_id: str
    ranges: list[dict] = field(default_factory=list)
    options: dict = field(default_factory=dict)

    def metadata(self):
        return dict(mode=self.mode, run_id=self.run_id, source_frame_ranges=self.ranges, **self.options)

    def record(self, status, **extra):
        values = dict(day=self.day, output_root=str(self.root.resolve()), status=status, **self.metadata(), **extra)
        (self.root / "run.json").write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n")


_active: ContextVar[ResponseOutput | None] = ContextVar("response_output", default=None)


def current(day=None):
    value = _active.get()
    return value if value is not None and (day is None or value.day == day) else None


def response_root(day):
    value = current(day)
    return value.root if value else Path(param.save_dir_bef) / day / "repellent_response"


def metadata(day=None):
    value = current(day)
    return value.metadata() if value else {}


def set_ranges(day, ranges):
    value = current(day)
    if value:
        value.ranges = ranges
        value.record("running")


@contextmanager
def response_run(day, *, all_frames=False, output_suffix=None, options=None):
    base = Path(param.save_dir_bef)
    if output_suffix:
        suffix = Path(output_suffix)
        if suffix.is_absolute() or ".." in suffix.parts:
            raise ValueError("output-suffix must be a relative path without parent traversal")
        base = base / suffix
    mode = "all_frames" if all_frames else "configured"
    run_id = datetime.now(ZoneInfo("Asia/Tokyo")).strftime("%Y%m%dT%H%M%S%f_JST_") + uuid4().hex[:8]
    root = base / day / "repellent_response" / "runs" / mode / run_id
    root.mkdir(parents=True, exist_ok=False)
    value = ResponseOutput(day, root, mode, run_id, options=options or {})
    token = _active.set(value)
    try:
        value.record("running")
        yield value
    except BaseException as error:
        value.record("failed", error_type=type(error).__name__, error=str(error))
        raise
    else:
        value.record("success")
    finally:
        _active.reset(token)


def annotate_figure(fig, day=None, samples=None):
    """Label the run and source ranges without changing the plotted data."""
    value = current(day)
    if value is None:
        return
    selected = set(samples) if samples is not None else None
    ranges = [r for r in value.ranges if selected is None or r["sample_no"] in selected]
    lines = [f"{value.mode} | {value.run_id} | source frames (1-based, inclusive)"]
    for offset in range(0, len(ranges), 4):
        lines.append(
            "; ".join(
                f"No.{r['sample_no']}: {r['source_start_frame_1based']}-{r['source_end_frame_1based']}"
                for r in ranges[offset : offset + 4]
            )
        )
    fig.text(
        0.01,
        0.002,
        "\n".join(lines),
        va="bottom",
        fontsize=7,
        bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
    )
