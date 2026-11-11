"""Original-frame SD/FFT and diagnostic sign changes; no physical reversal decisions."""

from typing import Any, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from utils import param
from utils.functions import frequency_analysis
from utils.functions.observation_timebase import continuous_edges


def get_weights_gaussian(
    start_time: float,
    window_width: float,
    time_list: Union[Sequence[float], np.ndarray],
) -> np.ndarray:
    edge_peak_divisor = param.edge_peak_divisor
    t = np.asarray(time_list, dtype=float)

    if t.size == 0:
        return np.array([], dtype=float)
    if window_width <= 0:
        raise ValueError("window_width must be > 0")
    if edge_peak_divisor <= 1.0:
        raise ValueError("edge_peak_divisor must be > 1")

    center_time_sec = start_time + window_width / 2.0
    sigma_sec = window_width / (2.0 * np.sqrt(2.0 * np.log(edge_peak_divisor)))

    weights = np.exp(-0.5 * ((t - center_time_sec) / sigma_sec) ** 2)

    weight_sum = weights.sum()
    return (weights / weight_sum) if weight_sum > 0 else np.zeros_like(weights)


def get_weighted_stats(
    data: Union[Sequence[float], np.ndarray],
    weights: Union[Sequence[float], np.ndarray],
) -> Tuple[float, float]:
    values_arr = np.asarray(data, dtype=float)
    weights_arr = np.asarray(weights, dtype=float)

    if values_arr.size == 0 or weights_arr.size == 0:
        return np.nan, np.nan
    if values_arr.shape != weights_arr.shape:
        raise ValueError("`data` and `weights` must have the same shape.")

    total_weight = float(np.sum(weights_arr))
    if total_weight <= 0.0:
        return np.nan, np.nan

    # Normalize weights: Σ w_i = 1
    normalized_weights = weights_arr / total_weight
    # Weighted mean: μ = Σ (w_i * x_i) / Σ w_i
    weighted_mean = float(np.sum(normalized_weights * values_arr))
    # Weighted variance: σ² = Σ (w_i * (x_i - μ)²) / Σ w_i
    weighted_variance = float(np.sum(normalized_weights * ((values_arr - weighted_mean) ** 2)))
    # Weighted standard deviation: σ = sqrt(σ²)
    weighted_std = float(np.sqrt(weighted_variance))

    return weighted_std, weighted_mean


def sd_at_observations(primary, width, *, comparison_range=None):
    """One centered SD/mean value per original row; never regenerate time."""
    from utils.functions.observation_timebase import continuous_edges

    t = primary.time_sec.to_numpy(float)
    v = np.abs(primary.angular_velocity_rad_s.to_numpy(float))
    edges = continuous_edges(t, source_frames=primary.source_frame_1based, segment_ids=primary.segment_id)
    out = []
    resets = np.flatnonzero((~np.isfinite(np.diff(t))) | (np.diff(t) <= 0)) + 1
    cuts = np.r_[0, resets, len(t)]
    bounds = np.zeros((len(t), 2), dtype=int)
    for a, b in zip(cuts[:-1], cuts[1:]):
        bounds[a:b] = [a, b]
    for index, clock in enumerate(t):
        value, mean, count, reason = np.nan, np.nan, 0, "missing_time"
        if np.isfinite(clock):
            start, end = clock - width / 2, clock + width / 2
            block_start, block_end = bounds[index]
            block = t[block_start:block_end]
            selected = np.arange(
                block_start + np.searchsorted(block, start - 1e-12), block_start + np.searchsorted(block, end - 1e-12)
            )
            if comparison_range is not None and (
                start < comparison_range[0] - 1e-12 or end > comparison_range[1] + 1e-12
            ):
                reason = "comparison_boundary"
            elif not len(selected):
                reason = "insufficient_samples"
            else:
                lower, upper = int(selected[0]), int(selected[-1])
                count = int(np.isfinite(v[selected]).sum())
                edge_lower, edge_upper = max(block_start, lower - 1), min(block_end - 1, upper + 1)
                overlapping = (t[edge_lower:edge_upper] < end - 1e-12) & (
                    t[edge_lower + 1 : edge_upper + 1] > start + 1e-12
                )
                if np.any(overlapping & ~edges[edge_lower:edge_upper]):
                    reason = "recording_gap"
                elif not np.array_equal(selected, np.arange(lower, upper + 1)):
                    reason = "recording_gap"
                elif not edges[lower:upper].all():
                    reason = "recording_gap"
                elif not np.isfinite(v[selected]).all():
                    reason = "missing_velocity"
                elif block[0] > start + 1e-10 or block[-1] < end - 1e-10:
                    reason = "window_boundary"
                elif len(selected) < 3:
                    reason = "insufficient_samples"
                else:
                    if param.flag_apply_gaussian_window:
                        weights = get_weights_gaussian(start, width, t[selected])
                        value, mean = get_weighted_stats(v[selected], weights)
                    else:
                        value, mean = float(np.std(v[selected], ddof=1)), float(np.mean(v[selected]))
                    reason = "available"
                    if param.mode_evaluate_SD_fluctuation == 1:
                        if mean == 0 or not np.isfinite(mean):
                            value, reason = np.nan, "zero_mean"
                        else:
                            value /= mean
        out.append((value, mean, count, reason))
    return pd.DataFrame(out, columns=["sd", "mean", "data_num", "reason"])


def primary_fluctuation_features(primary, *, comparison_range=None, comparison_source=""):
    """Traditional FFT feature definition, explicit range and reasons.

    No sampling density, frequency band, normalization or smoothing is changed.
    The longest finite continuous SD run is used, with its true clock spacing.
    """
    from utils.functions.observation_timebase import continuous_edges

    if comparison_range is not None:
        if (
            len(comparison_range) != 2
            or not np.isfinite(comparison_range).all()
            or comparison_range[1] <= comparison_range[0]
        ):
            raise ValueError("comparison range must be finite increasing seconds")
        if not comparison_source:
            raise ValueError("comparison source is required")
    feature_rows, sd_rows, spectrum_rows = [], [], []
    time = primary.time_sec.to_numpy(float)
    for width in param.SD_window_width_list:
        base = dict(
            sd_window_sec=width,
            comparison_start_sec=np.nan,
            comparison_end_sec=np.nan,
            comparison_source=comparison_source,
            source_start_frame_1based=np.nan,
            source_end_frame_1based=np.nan,
            fft_start_sec=np.nan,
            fft_end_sec=np.nan,
            fft_sample_count=0,
            effective_interval_sec=np.nan,
            low_bin_count=0,
            high_bin_count=0,
            low_amplitude=np.nan,
            high_amplitude=np.nan,
            amplitude_difference=np.nan,
            relative_intensity=np.nan,
            reciprocal_intensity=np.nan,
            reason="comparison_unspecified",
        )
        if comparison_range is None:
            feature_rows.append(base)
            continue
        base.update(comparison_start_sec=comparison_range[0], comparison_end_sec=comparison_range[1])
        stats = sd_at_observations(primary, width, comparison_range=comparison_range)
        values = stats.sd.to_numpy(float)
        finite = np.isfinite(values)
        normalized = np.full(len(values), np.nan)
        if finite.sum() >= 2 and np.std(values[finite]) > 0:
            normalized[finite] = (values[finite] - values[finite].mean()) / values[finite].std()
        else:
            base["reason"] = "zero_sd_variance" if finite.any() else "no_valid_sd"
            feature_rows.append(base)
            for j, row in enumerate(stats.itertuples()):
                sd_rows.append(
                    dict(
                        source_frame_1based=primary.source_frame_1based.iloc[j],
                        time_sec=time[j],
                        segment_id=primary.segment_id.iloc[j],
                        sd_window_sec=width,
                        sd=row.sd,
                        mean=row.mean,
                        data_num=row.data_num,
                        standardized_sd=normalized[j],
                        reason=row.reason,
                    )
                )
            continue
        for j, row in enumerate(stats.itertuples()):
            sd_rows.append(
                dict(
                    source_frame_1based=primary.source_frame_1based.iloc[j],
                    time_sec=time[j],
                    segment_id=primary.segment_id.iloc[j],
                    sd_window_sec=width,
                    sd=row.sd,
                    mean=row.mean,
                    data_num=row.data_num,
                    standardized_sd=normalized[j],
                    reason=row.reason,
                )
            )
        edges = continuous_edges(time, source_frames=primary.source_frame_1based, segment_ids=primary.segment_id)
        eligible = finite & np.isfinite(time)
        edge = edges & eligible[:-1] & eligible[1:]
        cuts = np.r_[0, np.flatnonzero(~edge) + 1, len(time)]
        runs = [(a, b) for a, b in zip(cuts[:-1], cuts[1:]) if b > a and eligible[a]]
        if not runs:
            base["reason"] = "no_continuous_sd"
            feature_rows.append(base)
            continue
        a, b = max(runs, key=lambda pair: pair[1] - pair[0])
        dt = float(np.median(np.diff(time[a:b]))) if b - a > 1 else np.nan
        base.update(
            source_start_frame_1based=int(primary.source_frame_1based.iloc[a]),
            source_end_frame_1based=int(primary.source_frame_1based.iloc[b - 1]),
            fft_start_sec=time[a],
            fft_end_sec=time[b - 1],
            fft_sample_count=b - a,
            effective_interval_sec=dt,
        )
        if not np.isfinite(dt):
            base["reason"] = "insufficient_fft_samples"
            feature_rows.append(base)
            continue
        freq, amp = frequency_analysis.fft(normalized[a:b], dt)
        for f, value in zip(freq, amp):
            spectrum_rows.append(dict(sd_window_sec=width, frequency_hz=f, amplitude=value))
        high = amp[freq >= 80]
        low = amp[:3]
        base.update(
            low_bin_count=len(low),
            high_bin_count=len(high),
            low_amplitude=float(low.mean()) if len(low) else np.nan,
            high_amplitude=float(high.mean()) if len(high) else np.nan,
        )
        if not len(low):
            base["reason"] = "insufficient_fft_samples"
        elif not len(high):
            base["reason"] = "reference_band_unavailable"
        elif high.mean() == 0:
            base["reason"] = "zero_reference_amplitude"
        else:
            lo, hi = float(low.mean()), float(high.mean())
            base.update(
                relative_intensity=lo / hi,
                reciprocal_intensity=hi / lo if lo != 0 else np.nan,
                amplitude_difference=lo - hi,
                reason="available",
            )
        feature_rows.append(base)
    return (
        pd.DataFrame(feature_rows),
        pd.DataFrame(
            sd_rows,
            columns=[
                "source_frame_1based",
                "time_sec",
                "segment_id",
                "sd_window_sec",
                "sd",
                "mean",
                "data_num",
                "standardized_sd",
                "reason",
            ],
        ),
        pd.DataFrame(spectrum_rows, columns=["sd_window_sec", "frequency_hz", "amplitude"]),
    )


WINDOW_COLUMNS = [
    "source_frame_1based",
    "segment_id",
    "window_start_sec",
    "window_end_sec",
    "window_center_sec",
    "window_width_sec",
    "recorded_seconds",
    "finite_velocity_seconds",
    "direction_seconds",
    "recorded_fraction",
    "finite_velocity_fraction",
    "direction_fraction",
    "finite_count",
    "zero_count",
    "missing_count",
    "eligible_pairs",
    "zero_fraction",
    "has_gap",
    "has_inference_boundary",
    "edge_incomplete",
    "switch_count",
    "switch_frequency_hz",
    "cw_fraction",
    "status",
    "reason",
]


def switching_windows(primary, *, width=1.0):
    """Start one half-open window per original row, including unknown times.

    Durations sum clipped original adjacent edges (no extrapolation); a switch
    belongs to a window only when both endpoints lie in its half-open bounds.
    A NaN/zero/boundary breaks the sign sequence. No minimum coverage threshold.
    """
    if not np.isfinite(width) or width <= 0:
        raise ValueError("window width must be positive")
    t = primary.time_sec.to_numpy(float)
    v = primary.angular_velocity_rad_s.to_numpy(float)
    frames = primary.source_frame_1based.to_numpy(int)
    segments = primary.segment_id.to_numpy()
    source_segments = primary.source_segment_id.to_numpy() if "source_segment_id" in primary else segments
    recorded_edges = continuous_edges(t, source_frames=frames, segment_ids=source_segments)
    edges = recorded_edges & continuous_edges(t, source_frames=frames, segment_ids=segments)
    finite = np.isfinite(v)
    signed = finite & (v != 0)
    finite_edges = edges & finite[:-1] & finite[1:]
    direction_edges = edges & signed[:-1] & signed[1:]
    changes = np.sign(v[:-1]) != np.sign(v[1:])
    rows = []
    resets = np.flatnonzero((~np.isfinite(np.diff(t))) | (np.diff(t) <= 0)) + 1
    cuts = np.r_[0, resets, len(t)]
    bounds = np.zeros((len(t), 2), dtype=int)
    for a, b in zip(cuts[:-1], cuts[1:]):
        bounds[a:b] = [a, b]
    for index, start in enumerate(t):
        row: dict[str, Any] = dict.fromkeys(WINDOW_COLUMNS, np.nan)
        row.update(source_frame_1based=int(frames[index]), segment_id=segments[index], window_width_sec=width)
        if not np.isfinite(start):
            row.update(
                status="unavailable",
                reason="missing_time",
                has_gap=False,
                has_inference_boundary=False,
                edge_incomplete=True,
            )
            rows.append(row)
            continue
        end = start + width
        inside = (t >= start) & (t < end)
        # Preserve row order: times must not be sorted or joined across resets.
        # Windows are limited to this original monotone recording clock block.
        lower, upper = bounds[index]
        inside[:lower] = False
        inside[upper:] = False
        spans = np.maximum(0.0, np.minimum(t[1:], end) - np.maximum(t[:-1], start))
        spans[~np.isfinite(spans)] = 0
        spans[:lower] = 0
        spans[max(lower, upper - 1) :] = 0
        recorded = float(spans[recorded_edges].sum())
        observed = float(spans[finite_edges].sum())
        directional = float(spans[direction_edges].sum())
        pairs = direction_edges & inside[:-1] & inside[1:]
        count = int(np.sum(changes & pairs)) if pairs.any() else np.nan
        values = v[inside]
        finite_values = values[np.isfinite(values)]
        missing = int(np.count_nonzero(~np.isfinite(values)))
        zeros = int(np.count_nonzero(finite_values == 0))
        gap = bool(np.any((spans > 0) & ~recorded_edges))
        inference_boundary = bool(np.any((spans > 0) & recorded_edges & ~edges))
        partial = bool(t[upper - 1] < end - 1e-12)
        reasons = []
        if not len(finite_values):
            reasons.append("no_finite_velocity")
        if not pairs.any():
            reasons.append("no_direction_pairs")
        if missing:
            reasons.append("missing_velocity")
        if zeros:
            reasons.append("zero_velocity")
        if gap:
            reasons.append("recording_gap")
        if inference_boundary:
            reasons.append("inference_boundary")
        if partial:
            reasons.append("recording_edge")
        row.update(
            window_start_sec=start,
            window_end_sec=end,
            window_center_sec=start + width / 2,
            recorded_seconds=recorded,
            finite_velocity_seconds=observed,
            direction_seconds=directional,
            recorded_fraction=recorded / width,
            finite_velocity_fraction=observed / width,
            direction_fraction=directional / width,
            finite_count=len(finite_values),
            zero_count=zeros,
            missing_count=missing,
            eligible_pairs=int(pairs.sum()),
            zero_fraction=zeros / len(finite_values) if len(finite_values) else np.nan,
            has_gap=gap,
            has_inference_boundary=inference_boundary,
            edge_incomplete=partial,
            switch_count=count,
            switch_frequency_hz=count / directional if directional > 0 and np.isfinite(count) else np.nan,
            cw_fraction=float(np.mean(finite_values < 0)) if len(finite_values) else np.nan,
            status="unavailable" if not len(finite_values) else "partial" if reasons else "available",
            reason=";".join(reasons) if reasons else "available",
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=WINDOW_COLUMNS)
