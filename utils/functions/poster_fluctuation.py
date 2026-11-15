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
LOW_BAND_HZ = (0.0, 1.0)
HIGH_BAND_HZ = (80.0, 100.0)


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


def band_amplitude_ratios(spectra: pd.DataFrame, windows: tuple[float, ...] = WINDOWS_S) -> pd.DataFrame:
    """Mean FFT amplitude in fixed low/high bands for each SD/Mean window.

    Both bands exclude their upper bound. FFT frequencies are rounded only for
    band assignment so a nominal 1 Hz bin cannot enter the open low band due
    to floating-point error. DC is excluded explicitly. A band
    with no usable FFT bins, or a zero high-band mean, has an unavailable ratio.
    """
    rows = []
    for width in windows:
        group = spectra[spectra["window_s"] == width]
        freq = group["frequency_hz"].to_numpy(float)
        band_freq = np.round(freq, decimals=9)
        amp = group["amplitude"].to_numpy(float)
        usable = np.isfinite(freq) & np.isfinite(amp) & (amp >= 0)
        low = amp[usable & (band_freq > LOW_BAND_HZ[0]) & (band_freq < LOW_BAND_HZ[1])]
        high = amp[usable & (band_freq >= HIGH_BAND_HZ[0]) & (band_freq < HIGH_BAND_HZ[1])]
        low_mean = float(np.mean(low)) if len(low) else np.nan
        high_mean = float(np.mean(high)) if len(high) else np.nan
        rows.append(
            {
                "window_s": float(width),
                "low_band_start_hz_exclusive": LOW_BAND_HZ[0],
                "low_band_end_hz_exclusive": LOW_BAND_HZ[1],
                "high_band_start_hz_inclusive": HIGH_BAND_HZ[0],
                "high_band_end_hz_exclusive": HIGH_BAND_HZ[1],
                "low_bin_count": len(low),
                "high_bin_count": len(high),
                "low_mean_amplitude": low_mean,
                "high_mean_amplitude": high_mean,
                "low_minus_high_amplitude": low_mean - high_mean,
                "low_to_high_amplitude_ratio": (
                    low_mean / high_mean if np.isfinite(high_mean) and high_mean > 0 else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


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


def synthetic_control(
    model: str,
    rate_hz: float,
    duration_s: float,
    seed: int,
    *,
    mean_angular_speed_rad_s: float,
    angular_speed_sd_rad_s: float,
) -> pd.DataFrame:
    """Generate an angle-error control or a direct independent-speed control."""
    if model not in ("constant_angle_step_plus_noise", "random_speed_noise"):
        raise ValueError(model)
    if (
        not np.isfinite(rate_hz)
        or rate_hz <= 0
        or not np.isfinite(duration_s)
        or duration_s <= 0
        or not np.isfinite(mean_angular_speed_rad_s)
        or mean_angular_speed_rad_s <= 0
        or not np.isfinite(angular_speed_sd_rad_s)
        or angular_speed_sd_rad_s < 0
    ):
        raise ValueError("Finite positive rate, duration, and mean speed; nonnegative speed SD required")
    count = int(rate_hz * duration_s)
    if count < 3:
        raise ValueError("At least three observations are required")
    time = np.arange(count, dtype=float) / rate_hz
    rng = np.random.default_rng(seed)
    common = {"source_frame_1based": np.arange(1, count + 1), "time_sec": time}
    if model == "constant_angle_step_plus_noise":
        baseline_angle = mean_angular_speed_rad_s * time
        angle_noise_sd = angular_speed_sd_rad_s / (np.sqrt(2.0) * rate_hz)
        angle_noise = rng.normal(0.0, angle_noise_sd, count)
        observed_angle = baseline_angle + angle_noise
        return pd.DataFrame(
            {
                **common,
                "long_axis_deg": (-np.rad2deg(observed_angle)) % 180.0,
                "axis_ratio": np.full(count, np.nan),
                "baseline_angle_rad": baseline_angle,
                "angle_noise_rad": angle_noise,
            }
        )
    speed_noise = np.full(count, np.nan)
    speed_noise[1:] = rng.normal(0.0, angular_speed_sd_rad_s, count - 1)
    return pd.DataFrame(
        {
            **common,
            "direct_omega_rad_s": mean_angular_speed_rad_s + speed_noise,
            "speed_noise_rad_s": speed_noise,
        }
    )
