"""Reproducible issue #39 comparison; see docs/analysis/issue-39/protocol.md."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

from utils.functions.get_centroid_coordinate import contours  # noqa: E402
from utils.functions.rotation_center_candidates import estimate_center  # noqa: E402
from utils.functions.rotation_center_diagnostics import evaluate_center  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/analysis/issue-39/tables"
METHODS = ("legacy", "ellipse_constrained", "geometric_ellipse", "robust_circle")
TRUE_CENTER = np.array([1.3, 1.45])
TRUE_RADIUS = 0.3


def synthetic_points(scenario: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return 360 frame-aligned samples and the actual (possibly moving) centers."""
    rng = np.random.default_rng(seed)
    n = 360
    phase = np.linspace(0, 12 * np.pi, n, endpoint=False)
    centers = np.tile(TRUE_CENTER, (n, 1))
    axes = np.array([TRUE_RADIUS, 0.18])
    noise = 0.02
    if scenario == "circle":
        axes[1] = TRUE_RADIUS
    elif scenario == "thin_ellipse":
        axes[1] = 0.075
    elif scenario == "thick_ellipse":
        noise = 0.08
    elif scenario == "short_arc":
        phase = np.linspace(-0.3, 0.3, n)
    elif scenario == "stopped_cloud":
        phase = np.zeros(n)
        noise = 0.008
    elif scenario == "reversal":
        phase[n // 2 :] = phase[n // 2 - 1] - np.linspace(0, 6 * np.pi, n // 2)
    elif scenario == "center_drift":
        centers[:, 0] += np.linspace(-0.06, 0.06, n)
    elif scenario == "radius_change":
        pass
    elif scenario == "missing_rows":
        pass
    elif scenario != "ellipse":
        raise ValueError(scenario)
    sizes = np.ones(n)
    if scenario == "radius_change":
        sizes[n // 2 :] = 0.65
    points = centers + np.column_stack((np.cos(phase), np.sin(phase))) * axes * sizes[:, None]
    points += rng.normal(0, noise, points.shape)
    if scenario == "missing_rows":
        points[90:110] = np.nan
    return points, centers


def synthetic_comparison() -> pd.DataFrame:
    records = []
    scenarios = (
        "ellipse",
        "circle",
        "thin_ellipse",
        "thick_ellipse",
        "short_arc",
        "stopped_cloud",
        "reversal",
        "center_drift",
        "radius_change",
        "missing_rows",
    )
    for scenario in scenarios:
        for trial in range(8):
            points, centers = synthetic_points(scenario, seed=3900 + trial)
            for method in METHODS:
                fit = estimate_center(points, method)
                row = {
                    "scenario": scenario,
                    "trial": trial,
                    "method": method,
                    "status": fit.status,
                    "n_input": fit.n_input,
                    "n_used": fit.n_used,
                    "true_center_error_R": np.nan,
                    "origin_shift_error_R": np.nan,
                    "reversal_sign_errors": np.nan,
                    "reversal_extra_switches": np.nan,
                }
                if fit.center is not None:
                    row["true_center_error_R"] = np.linalg.norm(fit.center - centers.mean(axis=0)) / TRUE_RADIUS
                    shifted = estimate_center(points + [2, -1], method)
                    if shifted.center is not None:
                        row["origin_shift_error_R"] = (
                            np.linalg.norm(shifted.center - [2, -1] - fit.center) / TRUE_RADIUS
                        )
                    if scenario == "reversal":
                        observed_phase = np.unwrap(
                            np.arctan2(points[:, 1] - fit.center[1], points[:, 0] - fit.center[0])
                        )
                        signed_speed = np.gradient(savgol_filter(observed_phase, 11, 2))
                        expected = np.ones(len(points), dtype=int)
                        expected[len(points) // 2 :] = -1
                        keep = np.ones(len(points), dtype=bool)
                        keep[len(points) // 2 - 10 : len(points) // 2 + 10] = False
                        row["reversal_sign_errors"] = int(np.sum(np.sign(signed_speed[keep]) != expected[keep]))
                        reliable_sign = np.sign(signed_speed[np.abs(signed_speed) >= 0.015])
                        row["reversal_extra_switches"] = max(0, int(np.sum(np.diff(reliable_sign) != 0)) - 1)
                records.append(row)
    return pd.DataFrame(records)


def avi_centroids(path: Path, px2um: float) -> tuple[np.ndarray, int, float]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(path)
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    points = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            x, y, _ = contours(frame)
            points.append((x * px2um, y * px2um))
    finally:
        capture.release()
    return np.asarray(points, dtype=float), len(points), fps


def test_avi_comparison() -> pd.DataFrame:
    records = []
    for path in sorted((ROOT / "data/test_data").glob("test_*.avi")):
        points, count, fps = avi_centroids(path, px2um=0.127)
        if count != 1200 or not np.isclose(fps, 200.0):
            raise ValueError(f"Unexpected frames or FPS for {path.name}: {count}, {fps}")
        for method in METHODS:
            row = evaluate_center(points, method, sample_rate_hz=200)
            row.update(avi=path.name, frame_count=count, missing_frames=int((~np.isfinite(points).all(axis=1)).sum()))
            records.append(row)
    return pd.DataFrame(records)


def no14_comparison() -> pd.DataFrame:
    """Evaluate the original selected frame ranges in a second real AVI."""
    path = ROOT / "outputs/repellent-response/23/raw_centroid/2026_0802_182430.csv"
    if not path.exists():
        raise FileNotFoundError(f"No.14 raw centroid is needed: {path}")
    source = pd.read_csv(path)
    if not np.array_equal(source.source_frame_1based, np.arange(1, 18433)):
        raise ValueError("No.14 source frame alignment changed")
    records = []
    for start_sec in (15.781, 20.789, 38.318, 64.862, 82.500):
        selected = source.loc[(source.time_sec >= start_sec) & (source.time_sec < start_sec + 5.080367)]
        points = selected[["x_um", "y_um"]].to_numpy(float)
        for method in METHODS:
            row = evaluate_center(points, method, sample_rate_hz=200)
            row.update(
                start_sec=start_sec,
                first_frame=int(selected.source_frame_1based.iloc[0]),
                last_frame=int(selected.source_frame_1based.iloc[-1]),
            )
            records.append(row)
    return pd.DataFrame(records)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic_comparison().to_csv(OUT / "synthetic.csv", index=False)
    test_avi_comparison().to_csv(OUT / "test_avi.csv", index=False)
    no14_comparison().to_csv(OUT / "no14_windows.csv", index=False)


if __name__ == "__main__":
    main()
