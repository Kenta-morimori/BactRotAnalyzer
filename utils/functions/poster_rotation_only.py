"""Rotation-only cell selection and descriptive poster band statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.functions.poster_fluctuation import analyze
from utils.functions.sign_reference_center_v2 import estimate_sign_reference_v2
from utils.functions.stopped_orbit_reference import (
    StopReferenceConfig,
    _compact_rotation_evidence,
    label_candidates,
)

WINDOWS_S = (0.1, 0.2, 0.5, 1.0)
LOW_BAND_HZ = (0.0, 1.0)
HIGH_BANDS_HZ = ((80.0, 100.0), (60.0, 80.0))


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[list[float]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return [(start, end) for start, end in merged]


def detect_stops(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Detect compact V2 stops and corroborated long-axis plateaus.

    V2 unknown by itself is not a stop. The 0.2 s secant is used only for
    stop screening; the fluctuation analysis retains the raw angular velocity.
    """
    time = frame.time_sec.to_numpy(float)
    xy = frame[["x_um", "y_um"]].to_numpy(float)
    angle = frame.long_axis_unwrapped_rad.to_numpy(float)
    v2 = estimate_sign_reference_v2(time, xy)
    if v2.empty:
        return v2, pd.DataFrame(columns=["start_s", "end_s", "method"])
    _, events = label_candidates(v2, StopReferenceConfig(), time, xy)
    intervals = [(float(e["start_s"]), float(e["end_s"]), "v2_compact") for e in events]
    valid_dt = np.diff(time)
    valid_dt = valid_dt[np.isfinite(valid_dt) & (valid_dt > 0)]
    if not len(valid_dt):
        return v2, pd.DataFrame(intervals, columns=["start_s", "end_s", "method"])
    dt = float(np.median(valid_dt))
    half = max(1, round(0.2 / (2 * dt)))
    local_hz = np.full(len(time), np.nan)
    for i in range(half, len(time) - half):
        left, right = i - half, i + half
        if (
            np.isfinite(angle[left : right + 1]).all()
            and np.isfinite(time[left : right + 1]).all()
            and np.all((np.diff(time[left : right + 1]) > 0) & (np.diff(time[left : right + 1]) <= 1.5 * dt))
        ):
            local_hz[i] = (angle[right] - angle[left]) / (time[right] - time[left]) / (2 * np.pi)
    row_frame = v2.frame_0based.to_numpy(int)
    rotating = v2.rotation_state.eq("rotating").to_numpy() & np.isfinite(v2.motion_rms_um.to_numpy(float))
    reference = float(np.median(v2.motion_rms_um.to_numpy(float)[rotating])) if rotating.any() else np.nan
    weak_motion = v2.motion_rms_um.to_numpy(float) <= 0.35 * reference
    candidate = (np.abs(local_hz[row_frame]) < 1.0) & weak_motion
    row_time = v2.time_sec.to_numpy(float)
    row_dt = np.diff(row_time)
    row_dt = row_dt[np.isfinite(row_dt) & (row_dt > 0)]
    step = float(np.median(row_dt)) if len(row_dt) else np.nan
    if np.isfinite(step):
        for start, end in _runs(candidate):
            # The centered secant loses up to 0.1 s at each end of a plateau.
            support_start = max(float(time[0]), float(row_time[start]) - 0.1)
            support_end = min(float(time[-1]), float(row_time[end - 1]) + 0.1)
            if end - start < 2 or support_end - support_start + step < 0.3:
                continue
            # A timestamp gap cannot create a continuous low-motion episode.
            if np.any(np.diff(row_time[start:end]) > 1.5 * step):
                continue
            first, last = row_frame[start], row_frame[end - 1] + 1
            if _compact_rotation_evidence(time[first:last], xy[first:last]):
                continue
            intervals.append((support_start, support_end, "axis_low_motion_secant_support"))
    return v2, pd.DataFrame(intervals, columns=["start_s", "end_s", "method"])


def longest_rotation_interval(frame: pd.DataFrame, stops: pd.DataFrame, margin_s: float = 0.1) -> pd.DataFrame:
    """Return one observed run, preserving original frame and time values."""
    if margin_s < 0:
        raise ValueError("margin_s must be nonnegative")
    time = frame.time_sec.to_numpy(float)
    source = frame.source_frame_1based.to_numpy(int)
    valid = np.isfinite(frame[["time_sec", "x_um", "y_um", "long_axis_omega_rad_s"]]).all(axis=1).to_numpy()
    excluded = [(float(r.start_s) - margin_s, float(r.end_s) + margin_s) for r in stops.itertuples()]
    for start, end in _merge_intervals(excluded):
        valid &= ~((time >= start) & (time <= end))
    dt = np.diff(time)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    if not len(positive):
        return frame.iloc[0:0].copy()
    nominal = float(np.median(positive))
    breaks = np.r_[True, (np.diff(source) != 1) | (~np.isfinite(dt)) | (dt <= 0) | (dt > 1.5 * nominal)]
    # Force a break after each missing observation or timestamp discontinuity.
    runs: list[tuple[int, int]] = []
    run_start: int | None = None
    for i, ok in enumerate(valid):
        if run_start is not None and (not ok or breaks[i]):
            runs.append((run_start, i))
            run_start = None
        if ok and run_start is None:
            run_start = i
    if run_start is not None:
        runs.append((run_start, len(valid)))
    if not runs:
        return frame.iloc[0:0].copy()
    first, last = max(runs, key=lambda bounds: (time[bounds[1] - 1] - time[bounds[0]], bounds[1] - bounds[0]))
    return frame.iloc[first:last].copy()


def band_metrics(spectra: pd.DataFrame, high_band: tuple[float, float]) -> pd.DataFrame:
    rows = []
    for width in WINDOWS_S:
        part = spectra[spectra.window_s.eq(width)]
        freq = np.round(part.frequency_hz.to_numpy(float), 9)
        amp = part.amplitude.to_numpy(float)
        good = np.isfinite(freq) & np.isfinite(amp) & (amp >= 0)
        low = amp[good & (freq > LOW_BAND_HZ[0]) & (freq < LOW_BAND_HZ[1])]
        high = amp[good & (freq >= high_band[0]) & (freq < high_band[1])]
        low_mean = float(np.mean(low)) if len(low) else np.nan
        high_mean = float(np.mean(high)) if len(high) else np.nan
        rows.append(
            {
                "window_s": width,
                "high_band_start_hz": high_band[0],
                "high_band_end_hz": high_band[1],
                "low_bin_count": len(low),
                "high_bin_count": len(high),
                "low_mean_amplitude": low_mean,
                "high_mean_amplitude": high_mean,
                "low_to_high_amplitude_ratio": (
                    low_mean / high_mean if np.isfinite(high_mean) and high_mean > 0 else np.nan
                ),
                "frequency_resolution_hz": float(np.median(np.diff(freq))) if len(freq) > 1 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def evaluate_interval(interval: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str]:
    if len(interval) < 3:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), "no_continuous_rotation"
    result = analyze(interval.time_sec.to_numpy(float), interval.long_axis_omega_rad_s.to_numpy(float), WINDOWS_S)
    primary = band_metrics(result.spectra, HIGH_BANDS_HZ[0])
    sensitivity = band_metrics(result.spectra, HIGH_BANDS_HZ[1])
    if (primary.low_bin_count < 3).any():
        reason = "fewer_than_3_low_bins"
    elif (primary.high_bin_count == 0).any() or (~np.isfinite(primary.low_to_high_amplitude_ratio)).any():
        reason = "invalid_primary_high_band"
    elif (sensitivity.high_bin_count == 0).any() or (~np.isfinite(sensitivity.low_to_high_amplitude_ratio)).any():
        reason = "invalid_sensitivity_high_band"
    else:
        reason = "included"
    return (
        result.frame,
        result.spectra,
        primary.merge(
            sensitivity[["window_s", "high_bin_count", "high_mean_amplitude", "low_to_high_amplitude_ratio"]],
            on="window_s",
            suffixes=("", "_60_80hz"),
        ),
        reason,
    )


def summarize_groups(metrics: pd.DataFrame) -> pd.DataFrame:
    """Equal-cell geometric ratio summaries and arithmetic normalized shapes."""
    if metrics.empty:
        return pd.DataFrame()
    accepted = metrics[metrics.selection_reason.eq("included")].copy()
    maximum = accepted.groupby(["day", "sample_no"]).low_to_high_amplitude_ratio.transform("max")
    accepted["normalized_ratio"] = accepted.low_to_high_amplitude_ratio / maximum
    rows = []
    for (day, width), group in accepted.groupby(["day", "window_s"], sort=True):
        log_ratio = np.log(group.low_to_high_amplitude_ratio.to_numpy(float))
        rows.append(
            {
                "day": day,
                "window_s": width,
                "cell_count": len(group),
                "ratio_geometric_mean": float(np.exp(np.mean(log_ratio))),
                "ratio_geometric_sd": float(np.exp(np.std(log_ratio, ddof=1))) if len(group) > 1 else np.nan,
                "log_ratio_sd": float(np.std(log_ratio, ddof=1)) if len(group) > 1 else np.nan,
                "normalized_ratio_mean": float(group.normalized_ratio.mean()),
                "normalized_ratio_sd": float(group.normalized_ratio.std(ddof=1)) if len(group) > 1 else np.nan,
                "low_mean_amplitude_mean": float(group.low_mean_amplitude.mean()),
                "high_mean_amplitude_mean": float(group.high_mean_amplitude.mean()),
            }
        )
    return pd.DataFrame(rows)
