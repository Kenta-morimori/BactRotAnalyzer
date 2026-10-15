"""Holdout evaluation of the post-#39 within-window center-change check."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from issue40_integration_check import OUT  # noqa: E402

from utils.functions.validated_rotation_center import (  # noqa: E402
    estimate_center_series,
)


def trial(scenario: str, seed: int) -> dict:
    n = 1200
    time = np.arange(n) / 200
    phase = np.arange(n) * 2 * np.pi / 60
    true_center = np.tile([1.3, 1.45], (n, 1))
    noise = 0.08 if scenario == "thick" else 0.02
    if scenario == "reversal":
        phase[n // 2 :] = phase[n // 2 - 1] - np.arange(1, n // 2 + 1) * 2 * np.pi / 60
    elif scenario == "linear_drift":
        true_center[:, 0] += (np.arange(n) - 600) * (0.12 / 360)
    elif scenario == "center_step":
        true_center[600:, 0] += 0.12
    elif scenario not in ("stable", "thick"):
        raise ValueError(scenario)
    rng = np.random.default_rng(seed)
    points = true_center + np.column_stack((0.3 * np.cos(phase), 0.18 * np.sin(phase)))
    points += rng.normal(0, noise, points.shape)
    centers, records = estimate_center_series(time, points)
    valid = np.isfinite(centers).all(axis=1)
    error = np.linalg.norm(centers[valid] - true_center[valid], axis=1) / 0.3
    crossing_error = np.linalg.norm(centers[500:700] - true_center[500:700], axis=1) / 0.3
    crossing_error = crossing_error[np.isfinite(crossing_error)]
    return {
        "scenario": scenario,
        "seed": seed,
        "supported_frames": int(valid.sum()),
        "crossing_supported_frames": int(valid[500:700].sum()),
        "accepted_windows": sum(row["gate_pass"] for row in records),
        "center_error_p95_R": float(np.quantile(error, 0.95)) if len(error) else np.nan,
        "crossing_error_max_R": float(max(crossing_error)) if len(crossing_error) else np.nan,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        trial(scenario, seed)
        for scenario in ("stable", "thick", "reversal", "linear_drift", "center_step")
        for seed in range(4700, 4730)
    ).to_csv(OUT / "change_holdout.csv", index=False)


if __name__ == "__main__":
    main()
