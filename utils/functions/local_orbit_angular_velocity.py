"""Center-free, local approximation of centroid-orbit angular velocity.

The window mean is a nuisance offset, not an estimate of the physical orbit
center. Consequently the returned magnitude is approximate. For image-space
coordinates (y increases downward), visually clockwise motion is positive.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def _continuous_runs(time: np.ndarray, points: np.ndarray) -> list[tuple[int, int]]:
    dt = np.diff(time)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    if len(positive) == 0:
        return []
    nominal = float(np.median(positive))
    finite = np.isfinite(time) & np.isfinite(points).all(axis=1)
    edge = (~finite[:-1]) | (~finite[1:]) | (~np.isfinite(dt)) | (dt <= 0) | (dt > 1.5 * nominal)
    starts = np.r_[0, np.flatnonzero(edge) + 1]
    ends = np.r_[np.flatnonzero(edge) + 1, len(time)]
    return [(int(a), int(b)) for a, b in zip(starts, ends) if finite[a] and b - a >= 3]


def estimate_local_orbit_velocity(
    time: np.ndarray,
    points: np.ndarray,
    *,
    window_seconds: float = 0.2,
    step_seconds: float = 0.025,
    smoothing_frames: int = 11,
) -> pd.DataFrame:
    """Return signed area/radius²/dt in centered windows of continuous data.

    A row is a window centered on ``frame_0based``. No window crosses missing
    coordinates or a timestamp gap. ``direction_coherence`` is the absolute
    net signed area divided by the sum of absolute step areas (0..1), and
    ``motion_noise_ratio`` compares the window motion RMS with the residual
    from smoothing. Neither diagnostic is a calibrated probability.
    """
    time = np.asarray(time, dtype=float)
    points = np.asarray(points, dtype=float)
    if time.ndim != 1 or points.shape != (len(time), 2):
        raise ValueError("time and points must have shapes (n,) and (n, 2)")
    if window_seconds <= 0 or step_seconds <= 0 or smoothing_frames < 3:
        raise ValueError("window, step, and smoothing must be positive")
    if smoothing_frames % 2 != 1:
        raise ValueError("smoothing_frames must be odd")

    rows: list[dict] = []
    for segment_id, (start, end) in enumerate(_continuous_runs(time, points)):
        t = time[start:end]
        raw = points[start:end]
        dt_nominal = float(np.median(np.diff(t)))
        smooth_width = min(smoothing_frames, len(t) if len(t) % 2 else len(t) - 1)
        if smooth_width < 3:
            continue
        smooth = savgol_filter(raw, smooth_width, 2, axis=0, mode="interp")
        width = max(3, int(round(window_seconds / dt_nominal)) + 1)
        if width % 2 == 0:
            width += 1
        half = width // 2
        step = max(1, int(round(step_seconds / dt_nominal)))
        if len(t) < width:
            continue
        for anchor in range(half, len(t) - half, step):
            window = smooth[anchor - half : anchor + half + 1]
            raw_window = raw[anchor - half : anchor + half + 1]
            window_dt = np.diff(t[anchor - half : anchor + half + 1])
            centered = window - window.mean(axis=0)
            cross = centered[:-1, 0] * centered[1:, 1] - centered[:-1, 1] * centered[1:, 0]
            radius_squared = np.sum(centered[:-1] ** 2, axis=1)
            denominator = float(np.sum(radius_squared * window_dt))
            omega = float(np.sum(cross)) / denominator if denominator > 0 else np.nan
            sum_abs_cross = float(np.sum(np.abs(cross)))
            coherence = abs(float(np.sum(cross))) / sum_abs_cross if sum_abs_cross > 0 else 0.0
            motion_rms = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))
            residual = raw_window - window
            noise_rms = float(np.sqrt(np.mean(np.sum(residual**2, axis=1))))
            snr = motion_rms / noise_rms if noise_rms > 0 else np.inf
            rows.append(
                {
                    "segment_id": segment_id,
                    "frame_0based": start + anchor,
                    "time_sec": float(t[anchor]),
                    "local_reference_x_um": float(window[:, 0].mean()),
                    "local_reference_y_um": float(window[:, 1].mean()),
                    "window_seconds_requested": window_seconds,
                    "window_seconds_actual": float(t[anchor + half] - t[anchor - half]),
                    "window_frames": width,
                    "signed_angular_velocity_rad_s": omega,
                    "direction_coherence": coherence,
                    "motion_rms_um": motion_rms,
                    "smoothing_residual_rms_um": noise_rms,
                    "motion_noise_ratio": snr,
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "segment_id",
            "frame_0based",
            "time_sec",
            "local_reference_x_um",
            "local_reference_y_um",
            "window_seconds_requested",
            "window_seconds_actual",
            "window_frames",
            "signed_angular_velocity_rad_s",
            "direction_coherence",
            "motion_rms_um",
            "smoothing_residual_rms_um",
            "motion_noise_ratio",
        ],
    )
