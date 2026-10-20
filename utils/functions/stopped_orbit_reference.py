"""Diagnostic measurement references during compact-motion candidates.

The frozen V2 sign model remains authoritative for rotating intervals. References
here are assumptions from adjacent rotation, not observed centers during stops.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import csd, savgol_filter, welch

from utils.functions.local_orbit_angular_velocity import _continuous_runs
from utils.functions.sign_reference_center_v2 import estimate_sign_reference_v2


@dataclass(frozen=True)
class StopReferenceConfig:
    compact_ratio: float = 0.35
    minimum_duration_s: float = 0.30
    anchor_span_s: float = 1.0
    maximum_anchor_distance_s: float = 4.0
    maximum_stop_s: float = 8.0
    minimum_anchor_points: int = 12
    radius_noise_multiplier: float = 5.0
    maximum_radial_change_R: float = 0.20
    minimum_lever_fraction: float = 0.50


def _runs(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))


def _compact_rotation_evidence(time, points):
    if len(points) < 48:
        return False
    rate = 1 / np.median(np.diff(time))
    # Longer records need finer spectral resolution to see a small orbit
    # above localization noise; a fixed 64-point spectrum can hide it.
    width = min(256, max(16, (len(points) // 3) // 2 * 2))
    frequency, xx = welch(points[:, 0], fs=rate, nperseg=width)
    _, yy = welch(points[:, 1], fs=rate, nperseg=width)
    _, xy = csd(points[:, 0], points[:, 1], fs=rate, nperseg=width)
    band = (frequency >= 2) & (frequency <= min(80, 0.45 * rate))
    peak = np.flatnonzero(band)[np.argmax((xx + yy)[band])]
    denominator = xx[peak] * yy[peak]
    coherence = abs(xy[peak]) ** 2 / denominator if denominator > 0 else 0
    quadrature = abs(xy[peak].imag) / np.sqrt(denominator) if denominator > 0 else 0
    energy = (xx + yy)[max(1, peak - 1) : peak + 2].sum() / max((xx + yy)[band].sum(), 1e-30)
    return coherence >= 0.80 and quadrature >= 0.65 and energy >= 0.15


def label_candidates(frame, config, time=None, points=None):
    """Label compact, weak-rotation episodes; NaN signs alone are not stops."""
    labels = np.full(len(frame), "unknown", dtype=object)
    labels[frame.rotation_sign.notna().to_numpy()] = "rotation_supported"
    episodes: list[dict] = []
    for segment, part in frame.groupby("segment_id", sort=False):
        index = part.index.to_numpy(int)
        active = part.rotation_sign.notna()
        if active.sum() < config.minimum_anchor_points:
            continue
        scale = float(part.loc[active, "motion_rms_um"].median())
        candidate = (part.motion_rms_um.to_numpy(float) <= config.compact_ratio * scale) & part.rotation_state.ne(
            "rotating"
        ).to_numpy()
        dt = float(np.median(np.diff(part.time_sec))) if len(part) > 1 else 0
        periodic = np.zeros(len(part), dtype=bool)
        clusters: list[list[int]] = []
        for lower, upper in _runs(candidate):
            if clusters and part.time_sec.iloc[lower] - part.time_sec.iloc[clusters[-1][1] - 1] <= 0.5:
                clusters[-1][1] = upper
            else:
                clusters.append([lower, upper])
        for lower, upper in clusters:
            first = int(part.frame_0based.iloc[lower])
            last = int(part.frame_0based.iloc[upper - 1]) + 1
            if points is not None and _compact_rotation_evidence(time[first:last], points[first:last]):
                periodic[lower:upper] = True
        for lower, upper in _runs(candidate):
            duration = float(part.time_sec.iloc[upper - 1] - part.time_sec.iloc[lower] + dt)
            if duration < config.minimum_duration_s:
                continue
            selected = index[lower:upper]
            first = int(part.frame_0based.iloc[lower])
            last = int(part.frame_0based.iloc[upper - 1]) + 1
            if periodic[lower:upper].any():
                labels[selected] = "compact_periodic_motion"
                continue
            labels[selected] = "stop_candidate"
            episodes.append(
                {
                    "episode_id": len(episodes),
                    "segment_id": int(segment),
                    "start_s": float(part.time_sec.iloc[lower]),
                    "end_s": float(part.time_sec.iloc[upper - 1]),
                    "duration_s": duration,
                    "rows": selected,
                    "rotation_motion_scale_um": scale,
                }
            )
    return labels, episodes


def _anchor(time, points, frame, event, side, source, config):
    segment = frame[frame.segment_id.eq(event["segment_id"]) & frame.rotation_sign.notna()]
    boundary = event["start_s"] if side == "left" else event["end_s"]
    # Exclude mixed .2 s windows adjacent to the candidate boundary.
    if side == "left":
        pool = segment[
            (segment.time_sec < boundary - 0.15) & (segment.time_sec >= boundary - config.maximum_anchor_distance_s)
        ]

    else:
        pool = segment[
            (segment.time_sec > boundary + 0.15) & (segment.time_sec <= boundary + config.maximum_anchor_distance_s)
        ]

    # A few isolated accepted points must not hide a nearby usable orbit.
    # Search from the boundary outwards for the first sufficiently supported
    # window, rather than anchoring irrevocably to the first/last single point.
    chosen = None
    clocks = pool.time_sec.to_numpy(float)
    for edge in clocks[::-1] if side == "left" else clocks:
        block = (
            pool[pool.time_sec.between(edge - config.anchor_span_s, edge)]
            if side == "left"
            else pool[pool.time_sec.between(edge, edge + config.anchor_span_s)]
        )
        if len(block) >= config.minimum_anchor_points:
            chosen = block
            break
    if chosen is None:
        return None
    pool = chosen
    clock = float(pool.time_sec.median())
    values = pool[["reference_x_um", "reference_y_um"]].to_numpy(float)
    center = np.median(values, axis=0)
    spread = float(np.quantile(np.linalg.norm(values - center, axis=1), 0.90))
    return {"time": clock, "center": center, "spread": spread, "points": len(pool)}


def estimate_stopped_reference(
    time, points, *, frame=None, source="v2", strategy="interpolate", config=StopReferenceConfig(), enabled=True
):
    """Return per-anchor diagnostic measurements and per-episode decisions.

    Stop boundaries come from data, never from supplied synthetic truth. A .2 s
    measurement must lie fully inside one candidate episode and one recording
    run. Center sensitivity is diagnostic, not a calibrated confidence interval.
    """
    if source != "v2" or strategy != "interpolate":
        raise ValueError("Unknown center source or strategy")
    time, points = np.asarray(time, float), np.asarray(points, float)
    frame = estimate_sign_reference_v2(time, points) if frame is None else frame.copy().reset_index(drop=True)
    labels, events = (
        label_candidates(frame, config, time, points)
        if enabled and not frame.empty
        else (np.full(len(frame), "unknown", dtype=object), [])
    )
    result = frame.reindex(columns=["segment_id", "frame_0based", "time_sec", "rotation_sign", "rotation_state"]).copy()
    if not enabled:
        labels[frame.rotation_sign.notna().to_numpy()] = "rotation_supported"
    result["motion_label"] = labels
    result["measurement_reason"] = "not_stop_candidate" if enabled else "disabled"
    for name in (
        "episode_id",
        "measurement_center_x_um",
        "measurement_center_y_um",
        "stop_angular_velocity_rad_s",
        "position_noise_um",
        "lever_arm_um",
        "center_sensitivity_um",
        "omega_noise_scale_rad_s",
    ):
        result[name] = np.nan
    result["center_source"] = source
    result["center_strategy"] = strategy
    runs = {
        segment: (start, end)
        for segment, (start, end) in enumerate(_continuous_runs(time, points))
        if result.segment_id.eq(segment).any()
    }
    # Segment IDs from V2 enumerate all continuous runs, including short ones.
    records = []
    for event in events:
        indices = event["rows"]
        result.loc[indices, "episode_id"] = event["episode_id"]
        result.loc[indices, "measurement_reason"] = "candidate_boundary"
        record = {key: value for key, value in event.items() if key != "rows"}
        left = _anchor(time, points, frame, event, "left", source, config)
        right = _anchor(time, points, frame, event, "right", source, config)
        reason = None
        if event["duration_s"] > config.maximum_stop_s:
            reason = "stop_too_long"
        elif left is None:
            reason = "missing_left_reference"
        elif strategy == "interpolate" and right is None:
            reason = "missing_right_reference"
        if reason:
            result.loc[indices, "measurement_reason"] = reason
            records.append({**record, "reference_status": reason, "measured_points": 0})
            continue
        for side, anchor in [("left", left), ("right", right)]:
            if anchor is not None:
                record.update(
                    {
                        f"{side}_time_s": anchor["time"],
                        f"{side}_x_um": anchor["center"][0],
                        f"{side}_y_um": anchor["center"][1],
                        f"{side}_spread_um": anchor["spread"],
                    }
                )
        start, end = runs[event["segment_id"]]
        width = min(5, (end - start - 1) // 2 * 2 + 1)
        smooth = savgol_filter(points[start:end], width, 2, axis=0)
        residual = points[start:end] - smooth

        def reference(clock):
            weight = (clock - left["time"]) / (right["time"] - left["time"])
            return left["center"] + weight[:, None] * (right["center"] - left["center"])

        sensitivity = max(left["spread"], right["spread"])
        measured = 0
        for index in indices:
            row = frame.loc[index]
            anchor = int(row.frame_0based)
            half = int(row.window_frames) // 2
            lower, upper = anchor - half, anchor + half + 1
            if time[lower] < event["start_s"] or time[upper - 1] > event["end_s"]:
                continue
            clock = time[lower:upper]
            centers = reference(clock)
            relative = smooth[lower - start : upper - start] - centers
            radii = np.linalg.norm(relative, axis=1)
            noise = float(np.sqrt(np.mean(residual[lower - start : upper - start] ** 2)))
            lever = float(np.quantile(radii, 0.10))
            result.loc[index, ["position_noise_um", "lever_arm_um", "center_sensitivity_um"]] = (
                noise,
                lever,
                sensitivity,
            )
            if lever <= max(
                config.radius_noise_multiplier * noise + sensitivity,
                config.minimum_lever_fraction * event["rotation_motion_scale_um"],
            ):
                result.loc[index, "measurement_reason"] = "unresolved_lever_arm"
                continue
            if np.ptp(radii) > config.maximum_radial_change_R * event["rotation_motion_scale_um"] + 6 * noise:
                result.loc[index, "measurement_reason"] = "inconsistent_radial_motion"
                continue
            angle = np.unwrap(np.arctan2(relative[:, 1], relative[:, 0]))
            offsets = clock - clock.mean()
            slope = float(np.dot(offsets, angle - angle.mean()) / np.dot(offsets, offsets))
            noise_scale = noise / lever / np.sqrt(np.dot(offsets, offsets))
            center = reference(np.array([time[anchor]]))[0]
            result.loc[
                index,
                [
                    "measurement_center_x_um",
                    "measurement_center_y_um",
                    "stop_angular_velocity_rad_s",
                    "omega_noise_scale_rad_s",
                ],
            ] = (*center, slope, noise_scale)
            result.loc[index, "measurement_reason"] = "measured"
            measured += 1
        records.append({**record, "reference_status": "supported", "measured_points": measured})
    columns = [
        "episode_id",
        "segment_id",
        "start_s",
        "end_s",
        "duration_s",
        "rotation_motion_scale_um",
        "left_time_s",
        "left_x_um",
        "left_y_um",
        "left_spread_um",
        "right_time_s",
        "right_x_um",
        "right_y_um",
        "right_spread_um",
        "reference_status",
        "measured_points",
    ]
    return result, pd.DataFrame(records, columns=columns)
