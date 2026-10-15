"""Compare two, five, ten cycle windows across known trajectory changes."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.diagnostics.issue39_validation import (  # noqa: E402
    OUT,
    TRUE_CENTER,
    TRUE_RADIUS,
)
from utils.functions.rotation_center_candidates import estimate_center  # noqa: E402

METHODS = ("legacy", "ellipse_constrained", "geometric_ellipse")
SCENARIOS = ("stable", "center_step", "center_drift", "radius_step", "reversal", "stop")
ANCHORS = (300, 480, 570, 600, 630, 720, 900)
CYCLES = (2, 5, 10)
FRAMES_PER_CYCLE = 60


def trajectory(scenario: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = 1200
    phase = np.arange(n) * 2 * np.pi / FRAMES_PER_CYCLE
    center = np.tile(TRUE_CENTER, (n, 1))
    axes = np.tile(np.array([TRUE_RADIUS, 0.18]), (n, 1))
    if scenario == "center_step":
        center[600:, 0] += 0.12
    elif scenario == "center_drift":
        center[:, 0] += np.linspace(-0.12, 0.12, n)
    elif scenario == "radius_step":
        axes[600:] *= 0.65
    elif scenario == "reversal":
        phase[600:] = phase[599] - np.arange(1, n - 599) * 2 * np.pi / FRAMES_PER_CYCLE
    elif scenario == "stop":
        phase[600:] = phase[599]
    elif scenario != "stable":
        raise ValueError(scenario)
    points = center + axes * np.column_stack((np.cos(phase), np.sin(phase)))
    points += rng.normal(0, 0.02, points.shape)
    return points, center


def comparison() -> pd.DataFrame:
    records = []
    for scenario in SCENARIOS:
        for seed in range(4100, 4105):
            points, true_centers = trajectory(scenario, seed)
            for anchor in ANCHORS:
                for cycles in CYCLES:
                    width = cycles * FRAMES_PER_CYCLE
                    for placement in ("forward", "centered"):
                        first = anchor if placement == "forward" else anchor - width // 2
                        last = first + width
                        valid = 0 <= first and last <= len(points)
                        crosses_change = first < 600 < last
                        for method in METHODS:
                            row = {
                                "scenario": scenario,
                                "seed": seed,
                                "anchor_frame_0based": anchor,
                                "cycles": cycles,
                                "placement": placement,
                                "first_frame_0based": first,
                                "last_frame_exclusive_0based": last,
                                "crosses_change": crosses_change,
                                "method": method,
                                "status": "incomplete_window",
                                "center_error_R": np.nan,
                            }
                            if valid:
                                result = estimate_center(points[first:last], method)
                                row["status"] = result.status
                                if result.center is not None:
                                    row["center_error_R"] = (
                                        np.linalg.norm(result.center - true_centers[anchor]) / TRUE_RADIUS
                                    )
                            records.append(row)
    return pd.DataFrame(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    comparison().to_csv(OUT / "window_study.csv", index=False)


if __name__ == "__main__":
    main()
