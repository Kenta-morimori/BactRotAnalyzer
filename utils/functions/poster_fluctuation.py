"""Frame-preserving poster fluctuation measurements and synthetic controls."""

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from utils.functions.fluctuation_analysis import (
    get_weighted_stats,
    get_weights_gaussian,
)
from utils.functions.frequency_analysis import (
    CONTINUITY_GAP_FACTOR,
    fft,
    longest_continuous_segment,
)

WINDOWS_S = (0.1, 0.2, 0.5, 1.0, 1.5, 2.0)


def duplicate_source_map(files_by_day: dict[str, list[Path]]) -> dict[tuple[str, int], str | None]:
    """Mark byte-identical AVIs as one source across all supplied conditions."""
    seen: dict[str, str] = {}
    result: dict[tuple[str, int], str | None] = {}
    for day, files in files_by_day.items():
        for sample_no, path in enumerate(files, 1):
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            result[(day, sample_no)] = seen.get(digest)
            seen.setdefault(digest, f"{day}/No.{sample_no:02d}")
    return result


@dataclass
class SeriesResult:
    frame: pd.DataFrame
    spectra: pd.DataFrame
    metadata: list[dict]


def major_axis_deg(ellipse: Optional[tuple]) -> float:
    """OpenCV ellipse angle converted to its long, unoriented axis [0, 180)."""
    if ellipse is None:
        return float("nan")
    axes = ellipse[1]
    if min(axes) <= 0:
        return float("nan")
    return float((ellipse[2] + (90.0 if axes[1] > axes[0] else 0.0)) % 180.0)


def axis_angular_velocity(time: np.ndarray, axis_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """CCW-positive increments from the unoriented long axis, without bridging gaps.

    The 180-degree ambiguity remains: true changes of 90 degrees or more per
    observation cannot be identified from this measurement alone.
    """
    time = np.asarray(time, dtype=float)
    axis_deg = np.asarray(axis_deg, dtype=float)
    if time.shape != axis_deg.shape:
        raise ValueError("time and axis_deg shape differ")
    n = len(time)
    angle = np.full(n, np.nan)
    omega = np.full(n, np.nan)
    dt = np.diff(time)
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    nominal_dt = float(np.median(valid_dt)) if len(valid_dt) else np.nan
    for j in range(n):
        if not np.isfinite(time[j]) or not np.isfinite(axis_deg[j]):
            continue
        if j == 0 or not np.isfinite(angle[j - 1]):
            angle[j] = 0.0
            continue
        delta_t = dt[j - 1]
        if (
            not np.isfinite(delta_t)
            or delta_t <= 0
            or (np.isfinite(nominal_dt) and delta_t > CONTINUITY_GAP_FACTOR * nominal_dt)
        ):
            continue
        # Image-space positive angle is visually clockwise; exported CCW is positive.
        delta_deg = (axis_deg[j] - axis_deg[j - 1] + 90.0) % 180.0 - 90.0
        delta = -np.deg2rad(delta_deg)
        angle[j] = angle[j - 1] + delta
        omega[j] = delta / delta_t
    return angle, omega


def circle_center(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """A single comparison center; return missing for degenerate trajectories."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(x) & np.isfinite(y)
    if good.sum() < 10:
        return float("nan"), float("nan")
    xx, yy = x[good], y[good]
    matrix = np.column_stack((2 * xx, 2 * yy, np.ones(len(xx))))
    if np.linalg.matrix_rank(matrix) < 3:
        return float("nan"), float("nan")
    fit, *_ = np.linalg.lstsq(matrix, xx * xx + yy * yy, rcond=None)
    return float(fit[0]), float(fit[1])


def orbit_angular_velocity(
    time: np.ndarray, x: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray, tuple[float, float]]:
    center = circle_center(x, y)
    n = len(time)
    angle = np.full(n, np.nan)
    omega = np.full(n, np.nan)
    if not np.all(np.isfinite(center)):
        return angle, omega, center
    raw = np.arctan2(np.asarray(y) - center[1], np.asarray(x) - center[0])
    dt = np.diff(time)
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    nominal_dt = float(np.median(valid_dt)) if len(valid_dt) else np.nan
    for j in range(n):
        if not np.isfinite(time[j]) or not np.isfinite(raw[j]):
            continue
        if j == 0 or not np.isfinite(angle[j - 1]):
            angle[j] = 0.0
            continue
        delta_t = dt[j - 1]
        if (
            not np.isfinite(delta_t)
            or delta_t <= 0
            or (np.isfinite(nominal_dt) and delta_t > CONTINUITY_GAP_FACTOR * nominal_dt)
        ):
            continue
        delta = (raw[j] - raw[j - 1] + np.pi) % (2 * np.pi) - np.pi
        delta = -delta  # image y increases down; report CCW positive
        angle[j] = angle[j - 1] + delta
        omega[j] = delta / delta_t
    return angle, omega, center


def analyze(time: np.ndarray, omega: np.ndarray, windows: tuple[float, ...] = WINDOWS_S) -> SeriesResult:
    """Existing Gaussian-window SD(abs(omega))/Mean and z-scored FFT.

    Each output row retains the original time index. A window containing a
    missing observation or timestamp gap remains unavailable.
    """
    time = np.asarray(time, dtype=float)
    omega = np.asarray(omega, dtype=float)
    if time.shape != omega.shape:
        raise ValueError("time and omega shape differ")
    frame = pd.DataFrame({"time_sec": time, "angular_velocity_rad_s": omega})
    spectra: list[pd.DataFrame] = []
    metadata = []
    nominal_diffs = np.diff(time)
    positive = nominal_diffs[np.isfinite(nominal_diffs) & (nominal_diffs > 0)]
    nominal_dt = float(np.median(positive)) if len(positive) else np.nan
    valid_time_indices = np.flatnonzero(np.isfinite(time))
    valid_times = time[valid_time_indices]
    if len(valid_times) > 1 and np.any(np.diff(valid_times) <= 0):
        raise ValueError("finite times must be strictly increasing")
    speed = np.abs(omega)
    for width in windows:
        ratio = np.full(len(time), np.nan)
        count = np.zeros(len(time), dtype=int)
        for j, t in enumerate(time):
            if not np.isfinite(t) or not np.isfinite(nominal_dt):
                continue
            if t - width / 2 < valid_times[0] or t + width / 2 > valid_times[-1] + nominal_dt:
                continue
            lo = int(np.searchsorted(valid_times, t - width / 2, side="left"))
            hi = int(np.searchsorted(valid_times, t + width / 2, side="left"))
            indices = valid_time_indices[lo:hi]
            if len(indices) < 3:
                continue
            values = speed[indices]
            times = time[indices]
            count[j] = int(np.isfinite(values).sum())
            diffs = np.diff(times)
            if (
                not np.isfinite(values).all()
                or not np.isfinite(times).all()
                or np.any(np.diff(indices) > 1)
                or np.any(diffs <= 0)
                or (np.isfinite(nominal_dt) and np.any(diffs > CONTINUITY_GAP_FACTOR * nominal_dt))
            ):
                continue
            weights = get_weights_gaussian(float(t - width / 2), float(width), times)
            sd, mean = get_weighted_stats(values, weights)
            if np.isfinite(mean) and mean > 0:
                ratio[j] = sd / mean
        key = f"sd_per_mean_{width:g}s"
        zkey = f"{key}_z"
        frame[key] = ratio
        frame[f"{key}_n"] = count
        valid = ratio[np.isfinite(ratio)]
        std = float(np.std(valid)) if len(valid) >= 2 else np.nan
        z = (ratio - float(np.mean(valid))) / std if np.isfinite(std) and std > 0 else np.full(len(time), np.nan)
        frame[zkey] = z
        (segment_time, segment_z), meta = longest_continuous_segment(time, z)
        segment_dt = float(np.median(np.diff(segment_time))) if len(segment_time) > 1 else np.nan
        freq, amp = fft(segment_z, segment_dt) if np.isfinite(segment_dt) else (np.array([]), np.array([]))
        spectra.append(pd.DataFrame({"window_s": width, "frequency_hz": freq, "amplitude": amp}))
        metadata.append(
            {
                "window_s": width,
                **meta,
                "peak_frequency_hz": float(freq[np.argmax(amp)]) if len(amp) else np.nan,
                "valid_ratio_count": len(valid),
            }
        )
    return SeriesResult(frame, pd.concat(spectra, ignore_index=True), metadata)


def synthetic_trajectory(
    model: str,
    rate_hz: float,
    duration_s: float,
    seed: int,
    base_hz: float = 8.0,
    modulation_hz: float = 2.0,
    modulation_rad_s: float = 8.0,
    noise_rad_s: float = 3.0,
    radius_um: float = 0.5,
    coordinate_noise_sd_um: float = 0.01,
) -> pd.DataFrame:
    """Known angular velocity -> integrated angle -> image-space orbit."""
    if model not in ("periodic_plus_noise", "random_only"):
        raise ValueError(model)
    time = np.arange(int(rate_hz * duration_s), dtype=float) / rate_hz
    rng = np.random.default_rng(seed)
    periodic = (
        modulation_rad_s * np.sin(2 * np.pi * modulation_hz * time)
        if model == "periodic_plus_noise"
        else np.zeros_like(time)
    )
    omega_true = 2 * np.pi * base_hz + periodic + rng.normal(0, noise_rad_s, len(time))
    theta = np.r_[0.0, np.cumsum(omega_true[1:] / rate_hz)]
    # Image y down; positive CCW rotates toward decreasing image y.
    true_x = radius_um * np.cos(theta)
    true_y = -radius_um * np.sin(theta)
    x = true_x + rng.normal(0, coordinate_noise_sd_um, len(time))
    y = true_y + rng.normal(0, coordinate_noise_sd_um, len(time))
    return pd.DataFrame(
        {
            "time_sec": time,
            "x_um": x,
            "y_um": y,
            "true_x_um": true_x,
            "true_y_um": true_y,
            "true_angular_velocity_rad_s": omega_true,
            "true_angle_rad": theta,
        }
    )
