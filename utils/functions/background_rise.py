"""Robust local change-point candidates for sustained background Mean rises.

Only detector fits are robustified. Original values, rows, frames and timestamps
are never smoothed, interpolated, compacted or written back by this module.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from utils.functions.observation_timebase import continuous_edges

METHOD = "robust-piecewise-sustained-v1"


def json_diagnostics(value):
    """Retain nested diagnostics while expressing unavailable numbers as null."""
    if isinstance(value, dict):
        return {name: json_diagnostics(item) for name, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_diagnostics(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


@dataclass(frozen=True)
class RiseSettings:
    baseline_sec: float = 5.0
    rise_sec: float = 5.0
    persistence_sec: float = 5.0
    search_step_sec: float = 0.25
    sigma_threshold: float = 6.0
    min_points: int = 3
    min_loss_improvement: float = 0.30
    persistence_fraction: float = 0.80
    persistence_sigma: float = 3.0
    persistence_effect_fraction: float = 0.20
    huber_delta: float = 1.345

    def __post_init__(self):
        for name in (
            "baseline_sec",
            "rise_sec",
            "persistence_sec",
            "search_step_sec",
            "sigma_threshold",
            "huber_delta",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(self.min_points) is not int or self.min_points < 3:
            raise ValueError("min_points must be an integer >= 3")
        for name in ("min_loss_improvement", "persistence_fraction", "persistence_effect_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not np.isfinite(self.persistence_sigma) or self.persistence_sigma <= 0:
            raise ValueError("persistence_sigma must be finite and positive")


def _mad(values):
    return float(1.4826 * np.median(np.abs(values - np.median(values))))


def _fit(time, values, scale, delta):
    design = np.column_stack([np.ones(len(time)), time])
    beta = np.linalg.lstsq(design, values, rcond=None)[0]
    for _ in range(20):
        residual = values - design @ beta
        weight = np.minimum(1.0, delta * scale / np.maximum(np.abs(residual), np.finfo(float).tiny))
        root = np.sqrt(weight)
        updated = np.linalg.lstsq(design * root[:, None], values * root, rcond=None)[0]
        if np.allclose(updated, beta, rtol=1e-8, atol=scale * 1e-8):
            beta = updated
            break
        beta = updated
    return beta, values - design @ beta


def _cost(residual, scale, delta):
    z = np.abs(residual) / scale
    return float(np.sum(np.where(z <= delta, 0.5 * z**2, delta * (z - 0.5 * delta))))


def _candidate(t, y, index, scale, settings):
    c = settings
    tau = t[index]
    # A candidate needs actual, continuous observations throughout all windows.
    left = np.searchsorted(t, tau - c.baseline_sec, side="left")
    middle = np.searchsorted(t, tau + c.rise_sec, side="left")
    right = np.searchsorted(t, tau + c.rise_sec + c.persistence_sec, side="right")
    if min(index - left, middle - index, right - middle) < c.min_points:
        return None
    if tau - t[0] < c.baseline_sec or t[-1] - tau < c.rise_sec + c.persistence_sec:
        return None
    before, after = slice(left, index), slice(index, middle)
    baseline, residual_before = _fit(t[before] - tau, y[before], scale, c.huber_delta)
    noise = max(scale, _mad(residual_before))
    rising, residual_after = _fit(t[after] - tau, y[after], noise, c.huber_delta)
    jump, slope = rising - baseline
    effect = jump + c.rise_sec * slope
    # A recovery from a falling baseline must also be an observed level rise.
    if (
        effect < c.sigma_threshold * noise
        or np.median(y[after]) - np.median(y[before]) < c.sigma_threshold * noise
        or jump < -c.persistence_sigma * noise
        or (slope < 0 and jump < c.sigma_threshold * noise)
    ):
        return None
    hold_t, hold_y = t[middle:right], y[middle:right]
    excess = hold_y - (baseline[0] + baseline[1] * (hold_t - tau))
    threshold = max(c.persistence_sigma * noise, c.persistence_effect_fraction * effect)
    above = excess > threshold
    dt = np.diff(hold_t)
    fraction = float(np.sum(dt * (above[:-1] & above[1:])) / np.sum(dt))
    if fraction < c.persistence_fraction:
        return None
    _, residual_null = _fit(t[left:middle] - tau, y[left:middle], noise, c.huber_delta)
    null_cost = _cost(residual_null, noise, c.huber_delta)
    if null_cost <= 0:
        return None
    split_cost = _cost(residual_before, noise, c.huber_delta) + _cost(residual_after, noise, c.huber_delta)
    improvement = 1 - split_cost / null_cost
    if improvement < c.min_loss_improvement:
        return None
    return dict(
        rise_index=int(index),
        rise_time_sec=float(tau),
        baseline_mean=float(baseline[0]),
        baseline_std=float(noise),
        baseline_slope=float(baseline[1]),
        post_slope=float(rising[1]),
        onset_jump=float(jump),
        rise_effect=float(effect),
        threshold=float(baseline[0] + c.persistence_sigma * noise),
        loss_improvement=float(improvement),
        persistence_fraction=fraction,
        support_start_time_sec=float(t[left]),
        support_end_time_sec=float(hold_t[-1]),
    )


def detect_sustained_rise(values, times, *, source_frames=None, segment_ids=None, max_dt=None, settings=None):
    """Return an original-row boundary or an explicit unavailable reason.

    A local baseline and rising line allow both a level jump and a slope change.
    Competing separated candidate groups are retained as ambiguous. Candidate
    ranges describe accepted search locations, never statistical confidence.
    """
    c = RiseSettings() if settings is None else settings
    y, t = np.asarray(values, float), np.asarray(times, float)
    if y.ndim != 1 or t.ndim != 1 or len(y) != len(t):
        raise ValueError("background values and original times must be aligned one-dimensional arrays")
    frames = np.arange(1, len(y) + 1) if source_frames is None else np.asarray(source_frames, float)
    if frames.ndim != 1 or len(frames) != len(y) or not np.isfinite(frames).all():
        raise ValueError("original source frames must match values")
    if np.any(frames < 1) or np.any(frames != np.floor(frames)) or np.any(np.diff(frames) <= 0):
        raise ValueError("original source frames must be increasing positive integers")
    if max_dt is not None and (not np.isfinite(max_dt) or max_dt <= 0):
        raise ValueError("max_dt must be finite and positive")
    result: dict[str, Any] = dict(
        rise_index=np.nan,
        rise_time_sec=None,
        rise_source_frame=None,
        baseline_mean=np.nan,
        baseline_std=np.nan,
        threshold=np.nan,
        method=METHOD,
        status="not_detected",
        reason="no_sustained_rise",
        parameters=asdict(c),
        candidates=[],
        timebase="original_timestamps",
    )
    valid = np.isfinite(y) & np.isfinite(t)
    edges = continuous_edges(t, source_frames=frames, segment_ids=segment_ids, max_dt=max_dt)
    edges &= valid[:-1] & valid[1:]
    starts = np.flatnonzero(valid & np.r_[True, ~edges])
    ends = np.flatnonzero(valid & np.r_[~edges, True]) + 1
    result["continuous_runs"] = len(starts)
    result["unavailable_observations"] = int(np.count_nonzero(~valid))
    groups: list[list[dict[str, Any]]] = []
    supported = False
    for start, end in zip(starts, ends):
        rt, ry = t[start:end], y[start:end]
        if len(rt) < c.min_points * 3 or rt[-1] - rt[0] < c.baseline_sec + c.rise_sec + c.persistence_sec:
            continue
        supported = True
        scale = max(_mad(np.diff(ry)) / np.sqrt(2), np.finfo(float).eps * max(1.0, float(np.max(np.abs(ry)))) * 100)
        targets = np.arange(rt[0] + c.baseline_sec, rt[-1] - c.rise_sec - c.persistence_sec, c.search_step_sec)
        last_time = None
        for index in np.unique(np.searchsorted(rt, targets)):
            candidate = _candidate(rt, ry, index, scale, c)
            if candidate is None:
                continue
            candidate["rise_index"] += int(start)
            candidate["rise_source_frame"] = int(frames[candidate["rise_index"]])
            tau = candidate["rise_time_sec"]
            if last_time is None or tau - last_time > 2 * c.search_step_sec:
                groups.append([])
            groups[-1].append(candidate)
            last_time = tau
    candidates = []
    for group in groups:
        best = dict(max(group, key=lambda item: item["loss_improvement"]))
        best["candidate_time_range_sec"] = [group[0]["rise_time_sec"], group[-1]["rise_time_sec"]]
        candidates.append(best)
    result["candidates"] = candidates
    if len(candidates) == 1:
        result.update(candidates[0], status="detected", reason="sustained_rise")
    elif candidates:
        result.update(status="ambiguous", reason="multiple_candidates")
    elif not valid.any():
        result["reason"] = "missing_background_or_time"
    elif not supported:
        result["reason"] = "insufficient_continuous_baseline_and_followup"
    elif len(starts) > 1:
        result["reason"] = "gap_or_missing_observations"
    return result
