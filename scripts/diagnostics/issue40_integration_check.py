"""Reproduce frame coverage and known-center errors for issue #40."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from issue39_validation import ROOT, avi_centroids  # noqa: E402

from utils.functions.validated_rotation_center import (  # noqa: E402
    estimate_center_series,
)

OUT = ROOT / "docs/analysis/issue-40/tables"


def summarize(
    name: str,
    time: np.ndarray,
    points: np.ndarray,
    *,
    truth: np.ndarray | None = None,
    stop_frame: int | None = None,
    bands: tuple[tuple[float, float], ...] | None = None,
) -> list[dict]:
    limit = len(time) if stop_frame is None else stop_frame
    centers = np.full_like(points, np.nan)
    centers[:limit], records = estimate_center_series(time[:limit], points[:limit])
    if bands is None:
        bands = ((float(time[0]), float(time[-1]) + 1e-9),)
    rows = []
    for start, end in bands:
        selected = (time >= start) & (time < end)
        valid = selected & np.isfinite(centers).all(axis=1)
        relevant = [r for r in records if start <= time[r["anchor_frame_0based"]] < end]
        errors = np.linalg.norm(centers[valid] - truth[valid], axis=1) / 0.3 if truth is not None else np.array([])
        rows.append(
            {
                "sample": name,
                "band_start_sec": start,
                "band_end_sec": end,
                "source_frames": int(selected.sum()),
                "supported_center_frames": int(valid.sum()),
                "supported_fraction": float(valid.sum() / selected.sum()) if selected.any() else np.nan,
                "accepted_windows": sum(r["gate_pass"] for r in relevant),
                "attempted_windows": len(relevant),
                "known_center_error_p95_R": np.quantile(errors, 0.95) if len(errors) else np.nan,
            }
        )
    return rows


def synthetic_cases() -> list[dict]:
    rows = []
    n = 1200
    time = np.arange(n) / 200
    center = np.tile([1.3, 1.45], (n, 1))
    for scenario in ("stable", "reversal", "short_arc", "stopped"):
        phase = np.arange(n) * 2 * np.pi / 60
        if scenario == "reversal":
            phase[n // 2 :] = phase[n // 2 - 1] - np.arange(1, n // 2 + 1) * 2 * np.pi / 60
        elif scenario == "short_arc":
            phase = np.linspace(-0.3, 0.3, n)
        elif scenario == "stopped":
            phase = np.zeros(n)
        rng = np.random.default_rng(4300)
        noise = 0.008 if scenario == "stopped" else 0.02
        points = center + np.column_stack((0.3 * np.cos(phase), 0.18 * np.sin(phase)))
        points += rng.normal(0, noise, points.shape)
        rows.extend(summarize(f"synthetic_{scenario}", time, points, truth=center))
    return rows


def real_cases() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "data/test_data").glob("test_*.avi")):
        points, count, fps = avi_centroids(path, px2um=0.127)
        if count != 1200 or not np.isclose(fps, 200):
            raise ValueError(f"Unexpected test AVI geometry: {path}")
        rows.extend(summarize(path.name, np.arange(count) / fps, points))
    source = pd.read_csv(ROOT / "outputs/repellent-response/23/raw_centroid/2026_0802_182430.csv")
    if not np.array_equal(source.source_frame_1based, np.arange(1, 18433)):
        raise ValueError("No.14 source frame alignment changed")
    # Existing day 23 configuration analyzes source frames 2049–16384 for No.14.
    source = source.iloc[2048:16384]
    if (int(source.source_frame_1based.iloc[0]), int(source.source_frame_1based.iloc[-1])) != (2049, 16384):
        raise ValueError("No.14 configured frame range changed")
    rows.extend(
        summarize(
            "No.14 configured range",
            source.time_sec.to_numpy(float),
            source[["x_um", "y_um"]].to_numpy(float),
            bands=((15.781, 40), (40, 65), (65, 82.5), (82.5, 87.579)),
        )
    )
    return rows


def signed_velocity(points: np.ndarray, centers: np.ndarray, time: np.ndarray) -> np.ndarray:
    """Use the production frame-difference sign convention without gap filling."""
    angle = np.arctan2(points[:, 1] - centers[:, 1], points[:, 0] - centers[:, 0])
    dt = np.diff(time)
    delta = np.angle(np.exp(1j * np.diff(angle)))
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    nominal_dt = float(np.median(valid_dt)) if len(valid_dt) else np.nan
    valid = np.isfinite(delta) & np.isfinite(dt) & (dt > 0) & (dt <= 1.5 * nominal_dt)
    return np.divide(-delta, dt, out=np.full_like(delta, np.nan), where=valid)


def no14_angular_velocity_comparison() -> pd.DataFrame:
    """Compare signed AV on identical supported intervals, without a truth claim."""
    source = pd.read_csv(ROOT / "outputs/repellent-response/23/raw_centroid/2026_0802_182430.csv")
    source = source.iloc[2048:16384].reset_index(drop=True)
    saved = pd.read_csv(ROOT / "docs/analysis/issue-38/tables/no14_centroid_center_timeseries.csv")
    if not np.array_equal(source.source_frame_1based, saved.source_frame_1based) or not np.allclose(
        source[["x_um", "y_um"]].to_numpy(float), saved[["x_um", "y_um"]].to_numpy(float)
    ):
        raise ValueError("No.14 saved center does not match the source centroid frames")
    time = source.time_sec.to_numpy(float)
    points = source[["x_um", "y_um"]].to_numpy(float)
    new_centers, _ = estimate_center_series(time, points)
    old_centers = saved[["center_x_um", "center_y_um"]].to_numpy(float)
    new_av = signed_velocity(points, new_centers, time)
    old_av = signed_velocity(points, old_centers, time)
    rows = []
    for start, end in ((15.781, 40), (40, 65), (65, 82.5), (82.5, 87.579)):
        selected = (time[:-1] >= start) & (time[:-1] < end)
        valid = selected & np.isfinite(new_av) & np.isfinite(old_av)
        n = int(valid.sum())
        new_switches = int(np.sum(valid[:-1] & valid[1:] & (np.sign(new_av[:-1]) != np.sign(new_av[1:]))))
        old_switches = int(np.sum(valid[:-1] & valid[1:] & (np.sign(old_av[:-1]) != np.sign(old_av[1:]))))
        rows.append(
            {
                "band_start_sec": start,
                "band_end_sec": end,
                "common_valid_intervals": n,
                "new_abs_speed_median_rad_s": float(np.median(np.abs(new_av[valid]))) if n else np.nan,
                "old_abs_speed_median_rad_s": float(np.median(np.abs(old_av[valid]))) if n else np.nan,
                "new_abs_speed_p95_rad_s": float(np.quantile(np.abs(new_av[valid]), 0.95)) if n else np.nan,
                "old_abs_speed_p95_rad_s": float(np.quantile(np.abs(old_av[valid]), 0.95)) if n else np.nan,
                "sign_disagreement_fraction": (
                    float(np.mean(np.sign(new_av[valid]) != np.sign(old_av[valid]))) if n else np.nan
                ),
                "new_sign_switches": new_switches,
                "old_sign_switches_on_same_intervals": old_switches,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(synthetic_cases() + real_cases()).to_csv(OUT / "integration_summary.csv", index=False)
    no14_angular_velocity_comparison().to_csv(OUT / "no14_angular_velocity_comparison.csv", index=False)


if __name__ == "__main__":
    main()
