"""Private poster comparison of repellent candidates on original frame clocks.

The post-rise sign changes are diagnostic adjacent-frame transitions, not
validated biological motor reversals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from utils.functions.poster_fluctuation import analyze, band_amplitude_ratios

WINDOWS_S = (0.1, 0.2, 0.5, 1.0, 1.5, 2.0)


def validate_series(series: pd.DataFrame) -> pd.DataFrame:
    """Return an ordered, one-row-per-source-frame copy without compacting gaps."""
    required = (
        "source_frame_1based",
        "time_sec",
        "centroid_x",
        "centroid_y",
        "center_x",
        "center_y",
        "angular_velocity_rad_s",
        "background_intensity",
    )
    missing = set(required) - set(series)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    out = series.loc[:, required].copy()
    frame = out.source_frame_1based.to_numpy(int)
    time = out.time_sec.to_numpy(float)
    if len(out) < 3 or not np.all(np.diff(frame) > 0):
        raise ValueError("source frames must be unique and increasing")
    if not np.isfinite(time).all() or not np.all(np.diff(time) > 0):
        raise ValueError("times must be finite and increasing")
    return out.reset_index(drop=True)


def radius_excursion_candidates(series: pd.DataFrame, *, width_frames: int = 401) -> np.ndarray:
    """Conservative review flags; these are not automatically excluded."""
    radius = pd.Series(np.hypot(series.centroid_x - series.center_x, series.centroid_y - series.center_y))
    local_median = radius.rolling(width_frames, center=True, min_periods=50).median()
    local_mad = (radius - local_median).abs().rolling(width_frames, center=True, min_periods=50).median()
    return ((radius > 1.8 * local_median) & (radius > local_median + 6 * local_mad)).fillna(False).to_numpy(bool)


def apply_masks(series: pd.DataFrame, invalid_ranges: list[list[int]]) -> pd.DataFrame:
    """Mask reviewed centroid failures and both velocities using an invalid endpoint."""
    out = validate_series(series)
    frames = out.source_frame_1based.to_numpy(int)
    invalid = np.zeros(len(out), bool)
    for first, last in invalid_ranges:
        if first > last:
            raise ValueError("invalid frame range is reversed")
        invalid |= (frames >= first) & (frames <= last)
    for name in ("centroid_x", "centroid_y"):
        out.loc[invalid, name] = np.nan
    time = out.time_sec.to_numpy(float)
    dt = np.diff(time)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    nominal = float(np.median(positive)) if len(positive) else np.nan
    gap = np.r_[True, (np.diff(frames) != 1) | (dt > 1.5 * nominal)]
    velocity_invalid = invalid | np.r_[True, invalid[:-1]] | gap
    out.loc[velocity_invalid, "angular_velocity_rad_s"] = np.nan
    out["centroid_invalid"] = invalid
    out["velocity_invalid"] = velocity_invalid
    return out


def contiguous_eligible_pairs(series: pd.DataFrame) -> np.ndarray:
    """Original adjacent signed velocity pairs, excluding NaNs/zero/gaps."""
    t = series.time_sec.to_numpy(float)
    frame = series.source_frame_1based.to_numpy(int)
    v = series.angular_velocity_rad_s.to_numpy(float)
    dt = np.diff(t)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    nominal = float(np.median(positive)) if len(positive) else np.nan
    return (
        (np.diff(frame) == 1)
        & (dt > 0)
        & (dt <= 1.5 * nominal)
        & np.isfinite(v[:-1])
        & np.isfinite(v[1:])
        & (v[:-1] != 0)
        & (v[1:] != 0)
    )


def sign_change_summary(series: pd.DataFrame, start: float, end: float) -> tuple[dict, pd.DataFrame]:
    """Count each adjacent sign change once; divide by eligible observed time."""
    if not start < end:
        raise ValueError("post-rise interval must be positive")
    post = series.loc[(series.time_sec >= start) & (series.time_sec < end)].reset_index(drop=True)
    pairs = contiguous_eligible_pairs(post)
    t = post.time_sec.to_numpy(float)
    v = post.angular_velocity_rad_s.to_numpy(float)
    dt = np.diff(t)
    changed = pairs & (np.sign(v[:-1]) != np.sign(v[1:]))
    exposure = float(np.sum(dt[pairs]))
    count = int(changed.sum())
    summary = {
        "post_start_s": start,
        "post_end_s": end,
        "post_frame_count": len(post),
        "post_valid_velocity_count": int(np.isfinite(v).sum()),
        "eligible_pair_count": int(pairs.sum()),
        "eligible_seconds": exposure,
        "sign_change_count": count,
        "sign_changes_per_observed_s": count / exposure if exposure > 0 else np.nan,
    }
    # Disjoint one-second bins are diagnostic time courses, not independent replicates.
    rows = []
    for index in range(int(np.ceil(end - start))):
        lo, hi = start + index, min(start + index + 1, end)
        # Assign each eligible pair to the bin containing its later endpoint,
        # so bin counts and exposure add up exactly to the full post interval.
        in_bin = (t[1:] >= lo) & (t[1:] < hi)
        selected = pairs & in_bin
        seconds = float(np.sum(dt[selected]))
        n = int(np.sum(changed & selected))
        rows.append(
            {
                "window_start_s": lo,
                "window_end_s": hi,
                "eligible_seconds": seconds,
                "sign_change_count": n,
                "sign_changes_per_observed_s": n / seconds if seconds > 0 else np.nan,
            }
        )
    return summary, pd.DataFrame(rows)


def sliding_sign_change_series(series: pd.DataFrame, start: float, end: float, *, width_s: float = 1.0) -> pd.DataFrame:
    """Centered full-width windows at every original frame, with valid-time rates."""
    if not start < end or width_s <= 0:
        raise ValueError("analysis interval and sliding width must be positive")
    shown = series.loc[(series.time_sec >= start) & (series.time_sec < end)].reset_index(drop=True)
    t = shown.time_sec.to_numpy(float)
    v = shown.angular_velocity_rad_s.to_numpy(float)
    pairs = contiguous_eligible_pairs(shown)
    changed = pairs & (np.sign(v[:-1]) != np.sign(v[1:]))
    pair_seconds = np.where(pairs, np.diff(t), 0.0)
    count_prefix = np.r_[0, np.cumsum(np.r_[0, changed.astype(int)])]
    seconds_prefix = np.r_[0.0, np.cumsum(np.r_[0.0, pair_seconds])]
    lo = t - width_s / 2
    hi = t + width_s / 2
    left = np.searchsorted(t, lo, side="left")
    right = np.searchsorted(t, hi, side="left")
    counts = count_prefix[right] - count_prefix[left]
    seconds = seconds_prefix[right] - seconds_prefix[left]
    full = (lo >= start) & (hi <= end) & np.isfinite(v)
    rate = np.divide(counts, seconds, out=np.full(len(t), np.nan), where=seconds > 0)
    return pd.DataFrame(
        {
            "source_frame_1based": shown.source_frame_1based.to_numpy(int),
            "time_sec": t,
            "window_start_s": lo,
            "window_end_s": hi,
            "sign_change_count": np.where(full, counts, np.nan),
            "eligible_seconds": np.where(full, seconds, np.nan),
            "sign_changes_per_observed_s": np.where(full, rate, np.nan),
        }
    )


def pre_rise_fluctuation(
    series: pd.DataFrame, start: float, rise: float
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Poster SD/Mean and 0–1 Hz / 80–100 Hz FFT ratio before the rise."""
    if not start < rise:
        raise ValueError("pre-rise interval must be positive")
    pre = series.loc[(series.time_sec >= start) & (series.time_sec < rise)].reset_index(drop=True)
    result = analyze(pre.time_sec.to_numpy(float), pre.angular_velocity_rad_s.to_numpy(float), WINDOWS_S)
    ratios = band_amplitude_ratios(result.spectra, WINDOWS_S)
    frame = pd.concat([pre[["source_frame_1based", "time_sec"]], result.frame.drop(columns="time_sec")], axis=1)
    return frame, result.spectra, ratios, result.metadata
