"""Separate center error from frame-difference noise in angular velocity."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import savgol_filter  # noqa: E402

from scripts.diagnostics.issue39_validation import (  # noqa: E402
    METHODS,
    OUT,
    ROOT,
    synthetic_points,
)
from utils.functions.rotation_center_candidates import estimate_center  # noqa: E402


def signed_velocity(points: np.ndarray, centers: np.ndarray, time: np.ndarray, *, smooth: bool) -> np.ndarray:
    phase = np.unwrap(np.arctan2(points[:, 1] - centers[:, 1], points[:, 0] - centers[:, 0]))
    if smooth:
        phase = savgol_filter(phase, 11, 2)
    dt = np.diff(time)
    delta = np.angle(np.exp(1j * np.diff(phase)))
    return np.divide(-delta, dt, out=np.full_like(delta, np.nan), where=np.isfinite(dt) & (dt > 0))


def extra_switches(velocity: np.ndarray, expected: int) -> int:
    signs = np.sign(velocity[np.isfinite(velocity)])
    signs = signs[signs != 0]
    return max(0, int(np.sum(np.diff(signs) != 0)) - expected)


def synthetic_angles() -> pd.DataFrame:
    rows = []
    n = 360
    dt = 1 / 200
    time = np.arange(n) * dt
    for scenario in ("ellipse", "thick_ellipse", "reversal", "center_drift"):
        for seed in range(3900, 3908):
            points, true_centers = synthetic_points(scenario, seed)
            phase = np.arange(n) * 12 * np.pi / n
            phase_rate = np.full(n - 1, 12 * np.pi / (n * dt))
            expected_switches = 0
            keep = np.ones(n - 1, dtype=bool)
            if scenario == "reversal":
                phase[n // 2 :] = phase[n // 2 - 1] - np.linspace(0, 6 * np.pi, n // 2)
                phase_rate[n // 2 :] *= -1
                expected_switches = 1
                keep[n // 2 - 11 : n // 2 + 10] = False
            a, b = 0.3, 0.18
            true_omega = -a * b * phase_rate / (a * a * np.cos(phase[:-1]) ** 2 + b * b * np.sin(phase[:-1]) ** 2)
            for method in ("true_center",) + METHODS:
                fit = None if method == "true_center" else estimate_center(points, method)
                center = (
                    true_centers if fit is None else np.tile(fit.center, (n, 1)) if fit.center is not None else None
                )
                for smooth in (False, True):
                    if center is None:
                        continue
                    velocity = signed_velocity(points, center, time, smooth=smooth)
                    expected_sign = np.sign(true_omega[keep])
                    rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "method": method,
                            "fit_status": "known" if fit is None else fit.status,
                            "smooth_11_frames": smooth,
                            "sign_errors_excluding_reversal": int(np.sum(np.sign(velocity[keep]) != expected_sign)),
                            "extra_sign_switches": extra_switches(velocity, expected_switches),
                            "absolute_speed_error_median_rad_s": float(
                                np.median(np.abs(velocity[keep] - true_omega[keep]))
                            ),
                            "absolute_speed_error_p95_rad_s": float(
                                np.quantile(np.abs(velocity[keep] - true_omega[keep]), 0.95)
                            ),
                        }
                    )
    return pd.DataFrame(rows)


def no14_angles() -> pd.DataFrame:
    source = pd.read_csv(ROOT / "outputs/repellent-response/23/raw_centroid/2026_0802_182430.csv")
    saved = pd.read_csv(ROOT / "docs/analysis/issue-38/tables/no14_centroid_center_timeseries.csv")
    if not np.array_equal(source.source_frame_1based, np.arange(1, 18433)):
        raise ValueError("No.14 source frame alignment changed")
    rows = []
    for start in (15.781, 20.789, 38.318, 64.862, 82.500):
        selected = source.loc[(source.time_sec >= start) & (source.time_sec < start + 5.080367)]
        saved_selected = saved.loc[(saved.time_sec >= start) & (saved.time_sec < start + 5.080367)]
        if not np.array_equal(selected.source_frame_1based, saved_selected.source_frame_1based):
            raise ValueError("Saved center frame alignment changed")
        if not np.allclose(
            selected[["x_um", "y_um"]].to_numpy(float), saved_selected[["x_um", "y_um"]].to_numpy(float)
        ):
            raise ValueError("Saved center coordinates use a different centroid series")
        points = selected[["x_um", "y_um"]].to_numpy(float)
        time = selected.time_sec.to_numpy(float)
        for method in ("saved_center",) + METHODS:
            if method == "saved_center":
                centers = saved_selected[["center_x_um", "center_y_um"]].to_numpy(float)
                status = str(saved_selected.center_source.mode().iloc[0])
            else:
                fit = estimate_center(points, method)
                if fit.center is None:
                    continue
                centers = np.tile(fit.center, (len(points), 1))
                status = fit.status
            for smooth in (False, True):
                velocity = signed_velocity(points, centers, time, smooth=smooth)
                rows.append(
                    {
                        "start_sec": start,
                        "first_frame_1based": int(selected.source_frame_1based.iloc[0]),
                        "last_frame_1based": int(selected.source_frame_1based.iloc[-1]),
                        "method": method,
                        "fit_status": status,
                        "smooth_11_frames": smooth,
                        "sign_switches": extra_switches(velocity, 0),
                        "abs_velocity_median_rad_s": float(np.nanmedian(np.abs(velocity))),
                        "abs_velocity_p95_rad_s": float(np.nanquantile(np.abs(velocity), 0.95)),
                    }
                )
    return pd.DataFrame(rows)


def point_smoothing_sensitivity() -> pd.DataFrame:
    """Measure the denoising versus reversal-timing tradeoff on held-out seeds."""
    rows = []
    n = 360
    time = np.arange(n) / 200
    true_reversal_sec = time[n // 2]
    for scenario in ("ellipse", "thick_ellipse", "reversal"):
        for seed in range(4100, 4130):
            points, centers = synthetic_points(scenario, seed)
            for method in ("true_center", "ellipse_constrained"):
                if method == "true_center":
                    used_centers = centers
                else:
                    fit = estimate_center(points, method)
                    if fit.center is None:
                        continue
                    used_centers = np.tile(fit.center, (n, 1))
                for width in (1, 5, 11, 21, 41):
                    smoothed = points if width == 1 else savgol_filter(points, width, 2, axis=0)
                    velocity = signed_velocity(smoothed, used_centers, time, smooth=False)
                    signs = np.sign(velocity)
                    switches = np.flatnonzero(np.diff(signs) != 0)
                    expected = int(scenario == "reversal")
                    nearest_error_ms = np.nan
                    if expected and len(switches):
                        switch_times = time[switches + 1]
                        nearest_error_ms = float(
                            1000 * (switch_times[np.argmin(abs(switch_times - true_reversal_sec))] - true_reversal_sec)
                        )
                    rows.append(
                        {
                            "scenario": scenario,
                            "seed": seed,
                            "method": method,
                            "point_smoothing_frames": width,
                            "extra_sign_switches": max(0, len(switches) - expected),
                            "nearest_reversal_error_ms": nearest_error_ms,
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    synthetic_angles().to_csv(OUT / "angle_synthetic.csv", index=False)
    no14_angles().to_csv(OUT / "angle_no14.csv", index=False)
    point_smoothing_sensitivity().to_csv(OUT / "angle_smoothing_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
