"""Prospective holdout for the exploratory gate in issue-39/protocol.md."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.diagnostics.issue39_validation import (  # noqa: E402
    OUT,
    ROOT,
    TRUE_CENTER,
    TRUE_RADIUS,
    avi_centroids,
    synthetic_points,
)
from utils.functions.rotation_center_candidates import robust_radius  # noqa: E402
from utils.functions.rotation_center_diagnostics import evaluate_center  # noqa: E402

METHODS = ("ellipse_constrained", "geometric_ellipse")
POSITIVE = ("ellipse", "circle", "thin_ellipse", "thick_ellipse", "reversal", "center_drift", "radius_change")
NEGATIVE = ("short_arc", "stopped_cloud", "partial_stop", "line_oscillation", "low_amplitude_periodic")


def holdout_points(scenario: str, seed: int) -> np.ndarray:
    if scenario not in ("partial_stop", "line_oscillation", "low_amplitude_periodic"):
        return synthetic_points(scenario, seed)[0]
    rng = np.random.default_rng(seed)
    n = 360
    phase = np.linspace(0, 12 * np.pi, n, endpoint=False)
    if scenario == "partial_stop":
        points = synthetic_points("ellipse", seed)[0]
        points[n // 2 :] = points[n // 2 - 1] + rng.normal(0, 0.008, (n // 2, 2))
    elif scenario == "line_oscillation":
        points = TRUE_CENTER + np.column_stack((0.3 * np.cos(phase), 0.18 * np.cos(phase)))
        points += rng.normal(0, 0.02, points.shape)
    else:
        # A stopped object with coherent subpixel centroid jitter: an
        # observational counterexample to equating periodicity with rotation.
        points = TRUE_CENTER + 0.008 * np.column_stack((np.cos(phase), np.sin(phase)))
        points += rng.normal(0, 0.001, points.shape)
    return points


def block_extent_ratio(points: np.ndarray) -> float:
    if not np.isfinite(points).all():
        return np.nan
    extents = [robust_radius(block) for block in np.array_split(points, 4)]
    return float(min(extents) / max(extents)) if max(extents) > 0 else np.nan


def gate(row: dict) -> tuple[bool, str]:
    checks = (
        ("missing", row["n_used"] == row["n_input"]),
        ("quadrature", row["rotation_quadrature_score"] >= 0.8),
        ("spectral_energy", row["peak_energy_fraction"] >= 0.15),
        ("block_extent", row["block_extent_ratio"] >= 0.25),
        ("fit_status", row["status"] == "computed"),
        ("block_fits", row.get("block_cv_computationally_ok") == 4),
        ("heldout_residual", row.get("block_cv_p95_R", np.nan) <= 0.55),
        ("center_stability", row.get("center_max_shift_R", np.nan) <= 0.10),
        ("origin_stability", row.get("origin_shift_error_R", np.nan) <= 0.01),
    )
    failures = [name for name, passed in checks if not passed]
    return not failures, ";".join(failures)


def record(points: np.ndarray, method: str, sample: str, scenario: str, seed: int | None) -> dict:
    row = evaluate_center(points, method, sample_rate_hz=200)
    row["sample"] = sample
    row["scenario"] = scenario
    row["seed"] = seed
    row["block_extent_ratio"] = block_extent_ratio(points)
    row["gate_pass"], row["gate_failure"] = gate(row)
    return row


def synthetic_holdout() -> pd.DataFrame:
    rows = []
    for scenario in POSITIVE + NEGATIVE:
        for seed in range(4000, 4030):
            points = holdout_points(scenario, seed)
            true_center = synthetic_points(scenario, seed)[1].mean(axis=0) if scenario in POSITIVE else None
            for method in METHODS:
                row = record(points, method, "synthetic", scenario, seed)
                row["true_center_error_R"] = (
                    np.nan
                    if true_center is None or not np.isfinite([row["center_x_um"], row["center_y_um"]]).all()
                    else np.linalg.norm(np.array([row["center_x_um"], row["center_y_um"]]) - true_center) / TRUE_RADIUS
                )
                rows.append(row)
    return pd.DataFrame(rows)


def real_samples() -> pd.DataFrame:
    rows = []
    for path in sorted((ROOT / "data/test_data").glob("test_*.avi")):
        points, count, fps = avi_centroids(path, px2um=0.127)
        if count != 1200 or not np.isclose(fps, 200):
            raise ValueError(f"Unexpected test AVI geometry: {path}")
        for method in METHODS:
            rows.append(record(points, method, path.name, "test_avi", None))
    source = pd.read_csv(ROOT / "outputs/repellent-response/23/raw_centroid/2026_0802_182430.csv")
    if not np.array_equal(source.source_frame_1based, np.arange(1, 18433)):
        raise ValueError("No.14 frame alignment changed")
    for start in (15.781, 20.789, 38.318, 64.862, 82.500):
        selected = source.loc[(source.time_sec >= start) & (source.time_sec < start + 5.080367)]
        points = selected[["x_um", "y_um"]].to_numpy(float)
        for method in METHODS:
            rows.append(record(points, method, f"No.14 {start:.3f}s", "no14", None))
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic_holdout().to_csv(OUT / "gate_holdout_synthetic.csv", index=False)
    real_samples().to_csv(OUT / "gate_real_samples.csv", index=False)


if __name__ == "__main__":
    main()
