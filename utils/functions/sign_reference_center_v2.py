"""V2 sign reference with gap isolation and rotation-only active support.

Five-frame smoothing preserves fast motion; missing signs remain missing.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

from utils.functions.local_orbit_angular_velocity import (
    _continuous_runs,
    estimate_local_orbit_velocity,
)


def _area_sign(window: np.ndarray, center: np.ndarray) -> float:
    relative = window - center
    cross = relative[:-1, 0] * relative[1:, 1] - relative[:-1, 1] * relative[1:, 0]
    value = float(np.sum(cross))  # clockwise positive in image coordinates
    return float(np.sign(value)) if np.isfinite(value) and value != 0 else np.nan


def _phase(window: np.ndarray, time: np.ndarray, center: np.ndarray) -> tuple[float, float, float]:
    relative = window - center
    angle = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
    offsets = time - time.mean()
    centered = angle - angle.mean()
    slope = float(np.dot(offsets, centered) / np.dot(offsets, offsets))
    residual = centered - slope * offsets
    variance = float(np.dot(centered, centered))
    linearity = max(0.0, 1 - float(np.dot(residual, residual)) / variance) if variance > 0 else 0.0
    return float(np.sign(slope)), linearity, abs(slope) * (time[-1] - time[0])


def estimate_sign_reference_v2(
    time: np.ndarray,
    points: np.ndarray,
    *,
    coherence_threshold: float = 0.5,
    motion_noise_threshold: float = 2.0,
) -> pd.DataFrame:
    time = np.asarray(time, dtype=float)
    points = np.asarray(points, dtype=float)
    if time.ndim != 1 or points.shape != (len(time), 2):
        raise ValueError("time and points must have shapes (n,) and (n, 2)")
    if not 0 < coherence_threshold <= 1 or motion_noise_threshold <= 0:
        raise ValueError("invalid activity threshold")
    frame = estimate_local_orbit_velocity(time, points, smoothing_frames=5)
    columns = (
        "reference_x_um",
        "reference_y_um",
        "candidate_median_x_um",
        "candidate_median_y_um",
        "candidate_mean_x_um",
        "candidate_mean_y_um",
        "rotation_sign",
        "sign_area",
        "sign_phase",
        "phase_linearity",
        "phase_travel_rad",
        "active_support_s",
        "active_support_left_s",
        "active_support_right_s",
        "short_sign",
        "sign_local",
    )
    if frame.empty:
        return pd.DataFrame(
            columns=list(dict.fromkeys([*frame.columns, "rotation_state", "decision_reason", *columns]))
        )
    short = estimate_local_orbit_velocity(time, points, window_seconds=0.1, smoothing_frames=5)
    short_lookup = short.set_index("frame_0based").signed_angular_velocity_rad_s
    local_sign = np.sign(frame.signed_angular_velocity_rad_s.to_numpy(float))
    short_sign = np.sign(frame.frame_0based.map(short_lookup).to_numpy(float))
    coherence = frame.direction_coherence.to_numpy(float) >= coherence_threshold
    amplitude = frame.motion_noise_ratio.to_numpy(float) >= motion_noise_threshold
    agreement = np.isfinite(short_sign) & (short_sign == local_sign) & (local_sign != 0)
    active_anchor = coherence & amplitude & agreement
    state = np.where(active_anchor, "rotating", np.where(~coherence & ~amplitude, "nonrotating", "uncertain"))
    frame["rotation_state"] = state
    frame["decision_reason"] = np.where(active_anchor, "insufficient_active_reference", state)
    for column in columns:
        frame[column] = np.nan
    frame["short_sign"] = short_sign
    frame["sign_local"] = local_sign
    values = {column: np.full(len(frame), np.nan) for column in columns if column not in {"short_sign", "sign_local"}}
    reasons = frame.decision_reason.to_numpy(copy=True)
    for start, end in _continuous_runs(time, points):
        selected = frame.loc[frame.frame_0based.between(start, end - 1)]
        if selected.empty:
            continue
        dt = float(np.median(np.diff(time[start:end])))
        step = max(1, int(round(0.025 / dt)))
        active = np.zeros(end - start, dtype=bool)
        for row in cast(Any, selected.itertuples()):
            if row.rotation_state == "rotating":
                index = row.frame_0based - start
                active[max(0, index - step // 2) : min(end - start, index + (step + 1) // 2)] = True
        raw = points[start:end]
        masked = raw.copy()
        masked[~active] = np.nan
        size = int(round(5.0 / dt)) | 1
        minimum = int(round(1.0 / dt))
        if len(raw) < minimum:
            continue
        median = pd.DataFrame(masked).rolling(size, center=True, min_periods=minimum).median().to_numpy()
        mean = pd.DataFrame(masked).rolling(size, center=True, min_periods=minimum).mean().to_numpy()
        smooth_width = min(5, len(raw) if len(raw) % 2 else len(raw) - 1)
        smooth = savgol_filter(raw, smooth_width, 2, axis=0)
        prefix = np.r_[0, np.cumsum(active.astype(int))]
        half_sign = int(round(0.1 / dt))
        for row in cast(Any, selected.itertuples()):
            if row.rotation_state != "rotating":
                continue
            index = row.frame_0based - start
            left, right = max(0, index - size // 2), min(len(raw), index + size // 2 + 1)
            left_support = (prefix[index] - prefix[left]) * dt
            right_support = (prefix[right] - prefix[index]) * dt
            support = left_support + right_support
            for column, value in zip(
                ("active_support_s", "active_support_left_s", "active_support_right_s"),
                (support, left_support, right_support),
            ):
                values[column][row.Index] = value
            if (
                left_support < 0.2
                or right_support < 0.2
                or support < 1.0
                or not np.isfinite(median[index]).all()
                or not np.isfinite(mean[index]).all()
            ):
                continue
            window = smooth[index - half_sign : index + half_sign + 1]
            clock = time[start + index - half_sign : start + index + half_sign + 1]
            median_area = _area_sign(window, median[index])
            mean_area = _area_sign(window, mean[index])
            median_phase, linearity, travel = _phase(window, clock, median[index])
            mean_phase, mean_linearity, mean_travel = _phase(window, clock, mean[index])
            for column, value in zip(
                ("candidate_median_x_um", "candidate_median_y_um", "candidate_mean_x_um", "candidate_mean_y_um"),
                (*median[index], *mean[index]),
            ):
                values[column][row.Index] = value
            for column, value in zip(
                ("sign_area", "sign_phase", "phase_linearity", "phase_travel_rad"),
                (median_area, median_phase, min(linearity, mean_linearity), min(travel, mean_travel)),
            ):
                values[column][row.Index] = value
            if min(linearity, mean_linearity) < 0.7 or min(travel, mean_travel) < 0.3:
                reasons[row.Index] = "weak_phase_motion"
            elif not median_area == mean_area == median_phase == mean_phase == row.sign_local:
                reasons[row.Index] = "sign_disagreement"
            else:
                for column, value in zip(
                    ("reference_x_um", "reference_y_um", "rotation_sign"), (*median[index], median_phase)
                ):
                    values[column][row.Index] = value
                reasons[row.Index] = "accepted"
    for column, data in values.items():
        frame[column] = data
    frame["decision_reason"] = reasons
    return frame
