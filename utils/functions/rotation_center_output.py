"""Local tables and plots for rotation references and stopped-orbit measurements."""

import json
from dataclasses import asdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.functions.repellent_output import annotate_figure, metadata, response_root
from utils.functions.repellent_plotting import (
    candidate_bands,
    coordinate_component,
    plot_series,
    recording_gaps,
    reference_series,
    velocities,
)
from utils.functions.stopped_orbit_reference import StopReferenceConfig

PRIMARY = "#286b9c"
STOP = "#d38128"
LABELS = {
    "rotation_supported": (0, PRIMARY),
    "stop_candidate": (1, STOP),
    "compact_periodic_motion": (2, "#8866a5"),
    "unknown": (3, "#999999"),
}


def _gaps(axes, time, points):
    recording_gaps(axes, time, points)


def _series(axis, frame, column, **kwargs):
    plot_series(axis, frame.time_sec, frame[column], segments=frame.segment_id, **kwargs)


def _states(axis, frame):
    for name, (row, color) in LABELS.items():
        part = frame[frame.motion_label.eq(name)]
        axis.scatter(part.time_sec, np.full(len(part), row), s=5, color=color)
    axis.set(yticks=range(len(LABELS)), yticklabels=list(LABELS), ylim=(3.5, -0.5))


def _coordinates(axes, time, points, frame):
    for component, axis in enumerate(axes):
        plot_series(axis, time, coordinate_component(points, component), color="#bfc6cb", lw=0.35)
        reference_series(axis, frame, component=component, components=True)
        axis.set_ylabel(["x (µm)", "y (µm)"][component])


def _velocities(axis, frame):
    velocities(axis, frame)


def _finish(fig, path):
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    annotate_figure(fig)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_sample(time, points, frame, episodes, sample, out):
    """Plot raw observations, provenance, abstention, and decision diagnostics."""
    fig = plt.figure(figsize=(14, 11))
    grid = fig.add_gridspec(5, 2)
    axes = [fig.add_subplot(grid[row, :]) for row in range(4)]
    xy = fig.add_subplot(grid[4, 0])
    text = fig.add_subplot(grid[4, 1])
    _coordinates(axes[:2], time, points, frame)
    _velocities(axes[2], frame)
    _states(axes[3], frame)
    _gaps(axes, time, points)
    axes[3].set_xlabel("Time (s); grey bands = recording gaps")
    plot_series(xy, time, points, xy=True, color="#bfc6cb", lw=0.3)
    reference_series(xy, frame, components=True)
    candidate_bands(axes, episodes)
    xy.set(xlabel="x (µm)", ylabel="y (µm)", aspect="equal")
    text.axis("off")
    text.text(
        0,
        0.9,
        "Blue: accepted V2 reference\nOrange: inferred stop measurement reference\n"
        "Blank: unavailable; orange x: low confidence\nStop candidate is not verified physical arrest\n"
        "Stop ω is excluded from primary CW/switching statistics",
        va="top",
        fontsize=9,
    )
    fig.suptitle(f"No.{sample}: centroid and reference centers")
    _finish(fig, out / "overview.png")

    fig, axes = plt.subplots(7, 1, figsize=(14, 12), sharex=True)
    groups = [
        (["motion_rms_um", "smoothing_residual_rms_um"], "Motion / residual (µm)"),
        (["direction_coherence", "phase_linearity"], "Coherence / phase R²"),
        (["motion_noise_ratio"], "Motion / noise"),
        (["active_support_s", "active_support_left_s", "active_support_right_s"], "Active support (s)"),
        (["sign_local", "short_sign", "sign_area", "sign_phase", "rotation_sign"], "CW+ signs"),
    ]
    for axis, (columns, ylabel) in zip(axes, groups):
        for column in columns:
            if column in frame:
                _series(axis, frame, column, lw=0.6, marker=".", markersize=1.5, label=column)
        axis.set_ylabel(ylabel)
        axis.legend(loc="upper right", fontsize=6, ncol=3)
    reasons = list(dict.fromkeys(frame.decision_reason.tolist() + frame.stop_velocity_reasons.tolist()))
    for row, reason in enumerate(reasons):
        part = frame[frame.decision_reason.eq(reason) | frame.stop_velocity_reasons.eq(reason)]
        axes[5].scatter(part.time_sec, np.full(len(part), row), s=3)
    axes[5].set(yticks=range(len(reasons)), yticklabels=reasons, ylabel="Decisions")
    axes[5].tick_params(axis="y", labelsize=6)
    _states(axes[6], frame)
    _gaps(axes, time, points)
    candidate_bands(axes, episodes)
    axes[6].set_xlabel("Time (s)")
    fig.suptitle(f"No.{sample}: rotation support and measurement decisions")
    _finish(fig, out / "diagnostics.png")

    for event in episodes.itertuples():
        lower = max(time[0], event.start_s - 4.5)
        upper = min(time[-1], event.end_s + 4.5)
        selected = (time >= lower) & (time <= upper)
        nearby = frame[frame.time_sec.between(lower, upper)]
        fig, axes = plt.subplots(5, 1, figsize=(13, 10), sharex=True)
        _coordinates(axes[:2], time[selected], points[selected], nearby)
        _velocities(axes[2], nearby)
        _velocities(axes[3], nearby)
        axes[3].set_ylabel("Stop ω (rad/s)")
        _states(axes[4], nearby)
        for axis in axes:
            axis.axvspan(event.start_s, event.end_s, color=STOP, alpha=0.12)
            axis.set_xlim(lower, upper)
        _gaps(axes, time[selected], points[selected])
        axes[4].set_xlabel("Time (s)")
        fig.suptitle(
            f"No.{sample}, stop candidate {event.episode_id+1}: {event.reference_status}; "
            f"{event.measured_points} measurements (not calibrated confidence)"
        )
        folder = out / "episodes"
        folder.mkdir(exist_ok=True)
        _finish(fig, folder / f"stop_{event.episode_id+1:03d}.png")


def reference_counts(frame):
    """Counts use evaluation points for centers and candidate points for stop quality."""
    count = len(frame)
    candidates = frame.motion_label.eq("stop_candidate")
    denominator = int(candidates.sum())
    values: dict[str, float] = {"final_reference_denominator": count, "stop_quality_denominator": denominator}
    for source in ("v2", "stop_interpolated", "missing"):
        number = int(frame.final_reference_source.eq(source).sum())
        values[f"final_reference_{source}_count"] = number
        values[f"final_reference_{source}_fraction"] = number / count if count else np.nan
    for quality in ("qualified", "low_confidence", "unavailable"):
        number = int((candidates & frame.stop_velocity_quality.eq(quality)).sum())
        values[f"stop_quality_{quality}_count"] = number
        values[f"stop_quality_{quality}_fraction"] = number / denominator if denominator else np.nan
    return values


def save_results(day, samples, *, stop_measurement_enabled):
    """Write only generated local outputs; never upload observation tables."""
    root = response_root(day) / "rotation_center"
    root.mkdir(parents=True, exist_ok=True)
    settings = dict(
        output_schema_version=2,
        final_reference="V2 accepted anchors, otherwise bracketed stop interpolation; no added evaluation points",
        stop_velocity_raw="mathematically computable values, including low confidence; separate from primary",
        stop_quality="qualified / low_confidence / unavailable; reasons are semicolon-separated in stable order",
        summary_denominators="final_reference: evaluation_points; stop_quality: stop_candidates",
        center_estimator="v2",
        stop_measurement_enabled=bool(stop_measurement_enabled),
        stop_center_source="v2",
        stop_center_strategy="interpolate",
        stop_config=asdict(StopReferenceConfig()),
        sign_convention="CW +1, CCW -1; unknown NaN",
        sampling="point at center of 0.2 s window, approximately 0.025 s step",
        units=dict(position="µm", time="s", angular_velocity="rad/s"),
        primary="angular_velocity_rad_s; stop values excluded from CW/switching statistics",
        stop_uncertainty="omega_noise_scale_rad_s is approximate, not a calibrated confidence interval",
        v2_config=dict(
            coherence_threshold=0.5,
            motion_noise_threshold=2.0,
            smoothing_frames=5,
            reference_window_s=5.0,
            minimum_support_s=1.0,
            minimum_side_support_s=0.2,
            minimum_phase_linearity=0.7,
            minimum_phase_travel_rad=0.3,
        ),
    )
    settings.update(metadata(day))
    (root / "settings.json").write_text(json.dumps(settings, ensure_ascii=False, indent=2) + "\n")
    rows = []
    for sample, (time, points, frame, episodes) in enumerate(samples, 1):
        out = root / f"no{sample:02d}"
        out.mkdir(exist_ok=True)
        frame.to_csv(out / "timeseries.csv", index=False)
        episodes.to_csv(out / "stop_episodes.csv", index=False)
        count = len(frame)
        accepted = int(frame.rotation_sign.notna().sum())
        rows.append(
            dict(
                **reference_counts(frame),
                sample_no=sample,
                source_frames=len(time),
                evaluation_points=count,
                primary_reported=accepted,
                primary_coverage=accepted / count if count else np.nan,
                stop_candidates=int(frame.motion_label.eq("stop_candidate").sum()),
                stop_measurements=int(frame.stop_angular_velocity_rad_s.notna().sum()),
                **{f"primary_reason_{key}": int(value) for key, value in frame.decision_reason.value_counts().items()},
                **{f"stop_reason_{key}": int(value) for key, value in frame.measurement_reason.value_counts().items()},
            )
        )
        plot_sample(time, points, frame, episodes, sample, out)
    summary = pd.DataFrame(rows)
    counts = [column for column in summary if str(column).startswith(("primary_reason_", "stop_reason_"))]
    summary[counts] = summary[counts].fillna(0)
    summary.to_csv(root / "summary.csv", index=False)
