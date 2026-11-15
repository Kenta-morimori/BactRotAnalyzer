"""Build private poster screening figures from existing repellent-response exports."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.run_context import analysis_run  # noqa: E402

plt.rcParams["font.family"] = "Arial"

REQUIRED = (
    "00_all_rotational_analysis/sample_index_map.csv",
    "00_all_rotational_analysis/time_list.csv",
    "00_all_rotational_analysis/centroid_coordinate/centroid_time_series.csv",
    "00_all_rotational_analysis/center_coordinate/center_coordinate.csv",
    "00_all_rotational_analysis/angular_velocity/angle_time-series.csv",
    "00_all_rotational_analysis/angular_velocity/angular-velocity_time-series.csv",
    "01_brightness_change/background_intensity_time_series.csv",
    "01_brightness_change/rise_summary.csv",
)
# These are alternative exports of the same named samples, not additional cells.
KNOWN_VARIANTS = {
    "20260312_mode=1": "20260312",
    "20260312_mode=2": "20260312",
    "20260416_23_1": "20260416_23",
    "20260416_23_2": "20260416_23",
    "23_bef": "23",
}
PROXY_RATIOS = (0.1, 0.2, 0.3)


def review_priority(metrics: dict) -> str:
    """Mark a conservative visual-review shortlist, not an acceptance decision."""
    if (
        metrics["background_median_post_over_pre"] >= 1.5
        and metrics["center_step_over_radius_max"] <= 0.1
        and metrics["speed_above_0.2x_pre_median_time_fraction_of_record"] > 0.5
    ):
        return "review_first"
    return "review_remaining"


def complete_bundle(folder: Path) -> bool:
    return all((folder / name).is_file() for name in REQUIRED)


def source_inventory(source_root: Path, raw_root: Path) -> pd.DataFrame:
    names = {p.name for p in source_root.iterdir() if p.is_dir()}
    raw_names = {p.name for p in raw_root.iterdir() if p.is_dir()} if raw_root.is_dir() else set()
    rows = []
    for name in sorted(names | raw_names):
        exported = source_root / name / "repellent_response"
        if name in KNOWN_VARIANTS:
            status = "alternate_export_not_independent"
        elif complete_bundle(exported):
            status = "screened"
        elif name in names:
            status = "incomplete_export"
        else:
            status = "raw_only_not_screened"
        rows.append(
            {
                "dataset": name,
                "status": status,
                "possible_same_samples_as": KNOWN_VARIANTS.get(name, ""),
                "raw_folder_with_same_name": name in raw_names,
                "source_export_folder": str(exported) if name in names else "",
            }
        )
    return pd.DataFrame(rows)


def _as_float(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame:
        raise ValueError(f"Missing source column: {column}")
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def velocity_time_from_export(time: np.ndarray, angle: np.ndarray, omega: np.ndarray) -> tuple[np.ndarray, str]:
    if len(angle) != len(time):
        raise ValueError("Exported angle and time rows differ")
    if len(omega) == len(time):
        return time, "point"
    if len(omega) == len(time) - 1:
        return time[:-1], "interval_start"
    raise ValueError("Exported angular velocity has an unsupported row count")


def _paired_steps(time: np.ndarray, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    good = np.isfinite(time) & np.isfinite(x) & np.isfinite(y)
    indices = np.flatnonzero(good)
    if len(indices) < 2:
        return np.array([]), np.array([])
    dt = np.diff(time[indices])
    positive = dt[np.isfinite(dt) & (dt > 0)]
    if not len(positive):
        return np.array([]), np.array([])
    nominal = float(np.median(positive))
    adjacent = (dt > 0) & (dt <= 1.5 * nominal)
    distance = np.hypot(np.diff(x[indices]), np.diff(y[indices]))
    return time[indices[1:]][adjacent], distance[adjacent]


def _line_with_gaps(time: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    good = np.isfinite(time) & np.isfinite(values)
    t, y = time[good], values[good]
    if len(t) < 2:
        return t, y
    dt = np.diff(t)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    if not len(positive):
        return t, y
    gap = np.flatnonzero((dt <= 0) | (dt > 1.5 * np.median(positive))) + 1
    return np.insert(t, gap, np.nan), np.insert(y, gap, np.nan)


def summarize_cell(
    position_time: np.ndarray,
    relative_x: np.ndarray,
    relative_y: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    velocity_time: np.ndarray,
    omega: np.ndarray,
    velocity_interval_end: np.ndarray,
    velocity_alignment: str,
    background_time: np.ndarray,
    background: np.ndarray,
    rise_time: float,
) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    radius = np.hypot(relative_x, relative_y)
    finite_radius = radius[np.isfinite(radius) & (radius > 0)]
    median_radius = float(np.median(finite_radius)) if len(finite_radius) else np.nan
    step_time, center_step = _paired_steps(position_time, center_x, center_y)
    step_ratio = center_step / median_radius if np.isfinite(median_radius) else np.full(len(center_step), np.nan)

    before = np.isfinite(background_time) & np.isfinite(background) & (background_time < rise_time)
    after = np.isfinite(background_time) & np.isfinite(background) & (background_time >= rise_time)
    baseline_brightness = float(np.median(background[before])) if np.any(before) else np.nan
    post_brightness = float(np.median(background[after])) if np.any(after) else np.nan
    brightness_ratio = (
        post_brightness / baseline_brightness
        if np.isfinite(baseline_brightness) and baseline_brightness > 0
        else np.nan
    )

    time_slots = np.isfinite(velocity_time)
    valid_velocity = time_slots & np.isfinite(omega)
    interval_dt = velocity_interval_end - velocity_time
    positive_dt = interval_dt[np.isfinite(interval_dt) & (interval_dt > 0)]
    nominal_dt = float(np.median(positive_dt)) if len(positive_dt) else np.nan
    valid_interval = valid_velocity & np.isfinite(interval_dt) & (interval_dt > 0) & (interval_dt <= 1.5 * nominal_dt)
    if velocity_alignment == "point":
        valid_interval[:-1] &= np.isfinite(omega[1:])
    elif velocity_alignment != "interval_start":
        raise ValueError(f"Unsupported velocity alignment: {velocity_alignment}")
    record_duration = (
        float(np.max(velocity_interval_end[np.isfinite(velocity_interval_end)]) - np.min(velocity_time[time_slots]))
        if np.any(np.isfinite(velocity_interval_end)) and np.any(time_slots)
        else np.nan
    )
    valid_interval_duration = float(np.sum(interval_dt[valid_interval]))
    baseline_speed = valid_velocity & (velocity_time < rise_time)
    reference_speed = float(np.median(np.abs(omega[baseline_speed]))) if np.any(baseline_speed) else np.nan
    active = np.full(len(omega), np.nan)
    metrics = {
        "rise_time_s": rise_time,
        "pre_rise_duration_s": (
            rise_time - float(np.min(background_time[np.isfinite(background_time)]))
            if np.isfinite(rise_time) and np.any(np.isfinite(background_time))
            else np.nan
        ),
        "post_rise_duration_s": (
            float(np.max(background_time[np.isfinite(background_time)])) - rise_time
            if np.isfinite(rise_time) and np.any(np.isfinite(background_time))
            else np.nan
        ),
        "background_median_post_over_pre": brightness_ratio,
        "background_pre_points": int(np.count_nonzero(before)),
        "background_post_points": int(np.count_nonzero(after)),
        "median_orbit_radius_um": median_radius,
        "center_step_points": len(center_step),
        "center_step_over_radius_max": float(np.max(step_ratio)) if len(step_ratio) else np.nan,
        "center_step_over_radius_p99": float(np.percentile(step_ratio, 99)) if len(step_ratio) else np.nan,
        "velocity_time_slots": int(np.count_nonzero(time_slots)),
        "velocity_valid_points": int(np.count_nonzero(valid_velocity)),
        "velocity_valid_fraction": (
            float(np.count_nonzero(valid_velocity) / np.count_nonzero(time_slots)) if np.any(time_slots) else np.nan
        ),
        "velocity_valid_time_fraction_of_record": (
            valid_interval_duration / record_duration
            if np.isfinite(record_duration) and record_duration > 0
            else np.nan
        ),
        "median_pre_rise_abs_omega_rad_s": reference_speed,
    }
    for ratio in PROXY_RATIOS:
        key = f"speed_above_{ratio:g}x_pre_median_fraction"
        if np.isfinite(reference_speed) and reference_speed > 0 and np.any(time_slots):
            moving = valid_velocity & (np.abs(omega) >= ratio * reference_speed)
            metrics[key + "_of_record"] = float(np.count_nonzero(moving) / np.count_nonzero(time_slots))
            metrics[key + "_of_valid"] = float(np.count_nonzero(moving) / np.count_nonzero(valid_velocity))
            moving_duration = float(np.sum(interval_dt[valid_interval & moving]))
            metrics[f"speed_above_{ratio:g}x_pre_median_time_fraction_of_record"] = (
                moving_duration / record_duration if np.isfinite(record_duration) and record_duration > 0 else np.nan
            )
            metrics[f"speed_above_{ratio:g}x_pre_median_time_fraction_of_valid"] = (
                moving_duration / valid_interval_duration if valid_interval_duration > 0 else np.nan
            )
            if ratio == 0.2:
                active[valid_velocity] = moving[valid_velocity].astype(float)
        else:
            metrics[key + "_of_record"] = np.nan
            metrics[key + "_of_valid"] = np.nan
            metrics[f"speed_above_{ratio:g}x_pre_median_time_fraction_of_record"] = np.nan
            metrics[f"speed_above_{ratio:g}x_pre_median_time_fraction_of_valid"] = np.nan
    return metrics, step_time, step_ratio, active


def plot_cell(
    target: Path,
    dataset: str,
    sample_no: int,
    position_time: np.ndarray,
    relative_x: np.ndarray,
    relative_y: np.ndarray,
    center_x: np.ndarray,
    center_y: np.ndarray,
    angle_time: np.ndarray,
    velocity_time: np.ndarray,
    angle: np.ndarray,
    omega: np.ndarray,
    background_time: np.ndarray,
    background: np.ndarray,
    metrics: dict,
    step_time: np.ndarray,
    step_ratio: np.ndarray,
    active: np.ndarray,
) -> None:
    fig, axes = plt.subplots(3, 3, figsize=(27, 21), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.25, h_pad=0.22, wspace=0.08, hspace=0.10)
    raw_x, raw_y = relative_x + center_x, relative_y + center_y
    rise = metrics["rise_time_s"]
    for ax, raw, center, label in ((axes[0, 0], raw_x, center_x, "x"), (axes[0, 1], raw_y, center_y, "y")):
        for values, name, color in ((raw, "Centroid", "#2563a6"), (center, "Center", "#bd4b6a")):
            t, y = _line_with_gaps(position_time, values)
            ax.plot(t, y, color=color, lw=1.4, label=name)
        ax.set(xlabel="Time [s]", ylabel=f"{label} [µm]", title=f"Centroid and center: {label}")
        ax.legend(loc="upper right")
    good_xy = np.isfinite(raw_x) & np.isfinite(raw_y)
    axes[0, 2].scatter(raw_x[good_xy], raw_y[good_xy], s=4, alpha=0.3, color="#2563a6", label="Centroid")
    good_center = np.isfinite(center_x) & np.isfinite(center_y)
    axes[0, 2].scatter(center_x[good_center], center_y[good_center], s=5, alpha=0.5, color="#bd4b6a", label="Center")
    axes[0, 2].set(xlabel="x [µm]", ylabel="y [µm] (image down)", title="x-y trajectory")
    axes[0, 2].set_aspect("equal", adjustable="box")
    axes[0, 2].legend(loc="upper right")

    for values, name, color in ((center_x, "Center x", "#2563a6"), (center_y, "Center y", "#bd4b6a")):
        t, y = _line_with_gaps(position_time, values)
        axes[1, 0].plot(t, y, color=color, lw=1.2, label=name)
    axes[1, 0].set(xlabel="Time [s]", ylabel="Position [µm]", title="Center continuity")
    axes[1, 0].legend(loc="upper right")
    wrapped = np.arctan2(np.sin(angle), np.cos(angle))
    t, y = _line_with_gaps(angle_time, wrapped)
    axes[1, 1].plot(t, y, lw=1.2, color="#2563a6")
    axes[1, 1].set(xlabel="Time [s]", ylabel="Angle [rad]", title="Exported orbit angle (wrapped)")
    for values, name, color, alpha in ((omega, "Signed", "#2563a6", 0.7), (np.abs(omega), "Absolute", "#df822c", 0.65)):
        t, y = _line_with_gaps(velocity_time, values)
        axes[1, 2].plot(t, y, lw=1, alpha=alpha, color=color, label=name)
    axes[1, 2].axhline(0, color="black", lw=0.7)
    axes[1, 2].set(xlabel="Time [s]", ylabel="Angular velocity [rad/s]", title="Rotation speed")
    axes[1, 2].legend(loc="upper right")

    t, y = _line_with_gaps(background_time, background)
    axes[2, 0].plot(t, y, color="#6c4b9c", lw=1.5)
    axes[2, 0].set(xlabel="Time [s]", ylabel="Brightness [a.u.]", title="Background brightness")
    axes[2, 1].plot(step_time, step_ratio, ".", ms=2.5, color="#bd4b6a")
    axes[2, 1].set(xlabel="Time [s]", ylabel="Center step / radius", title="Center movement per evaluation step")
    t, y = _line_with_gaps(velocity_time, active)
    axes[2, 2].plot(t, y, color="#2563a6", lw=0.9)
    axes[2, 2].set(
        xlabel="Time [s]",
        ylabel="Above threshold [0/1]",
        title="Provisional speed activity (0.2 × pre-rise median)",
    )
    axes[2, 2].set_ylim(-0.1, 1.1)

    times = np.concatenate((position_time[np.isfinite(position_time)], background_time[np.isfinite(background_time)]))
    if len(times):
        limits = (float(np.min(times)), float(np.max(times)))
        for ax in axes.flat:
            if ax is not axes[0, 2]:
                ax.set_xlim(*limits)
    if np.isfinite(rise):
        for ax in axes.flat:
            if ax is not axes[0, 2]:
                ax.axvline(rise, color="black", ls="--", lw=1.4, alpha=0.7)
    for ax in axes.flat:
        ax.title.set_fontsize(22)
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        ax.tick_params(labelsize=17)
        ax.grid(alpha=0.18)
        legend = ax.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(16)
    fig.suptitle(f"{dataset}  No.{sample_no:02d}  |  provisional screening", fontsize=29)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def screen_dataset(dataset: str, exported: Path, target: Path) -> list[dict]:
    files = {name: pd.read_csv(exported / name) for name in REQUIRED}
    mapping = files[REQUIRED[0]]
    velocity_time = files[REQUIRED[1]]
    centroid = files[REQUIRED[2]]
    centers = files[REQUIRED[3]]
    angle = files[REQUIRED[4]]
    omega = files[REQUIRED[5]]
    background = files[REQUIRED[6]]
    rises = files[REQUIRED[7]]
    settings_path = exported / "settings.json"
    settings = json.loads(settings_path.read_text()) if settings_path.is_file() else {}
    method = settings.get("center_estimator", settings.get("primary_method", "legacy_export_unknown"))
    target.mkdir(parents=True, exist_ok=True)
    results = []
    for row in mapping.itertuples(index=False):
        segment_no, original_no = int(row.segment_no), int(row.original_sample_no)
        prefix = f"No.{segment_no}"
        rise_row = rises.loc[rises.sample_no == original_no]
        if len(rise_row) != 1:
            raise ValueError(f"{dataset} No.{original_no}: rise summary mapping is ambiguous")
        rise = float(rise_row.iloc[0].rise_time)
        pt = _as_float(centroid, f"{prefix}_time")
        rx, ry = _as_float(centroid, f"{prefix}_x"), _as_float(centroid, f"{prefix}_y")
        cx, cy = _as_float(centers, f"{prefix}_x"), _as_float(centers, f"{prefix}_y")
        vt, theta, w = _as_float(velocity_time, prefix), _as_float(angle, prefix), _as_float(omega, prefix)
        bt, bv = _as_float(background, f"{prefix}_time"), _as_float(background, f"{prefix}_background")
        if not (len(pt) == len(rx) == len(ry) == len(cx) == len(cy)):
            raise ValueError(f"{dataset} No.{original_no}: position table rows differ")
        wt, velocity_alignment = velocity_time_from_export(vt, theta, w)
        velocity_interval_end = vt[1:] if velocity_alignment == "interval_start" else np.append(vt[1:], np.nan)
        if len(bt) != len(bv):
            raise ValueError(f"{dataset} No.{original_no}: background table rows differ")
        metrics, step_time, step_ratio, active = summarize_cell(
            pt, rx, ry, cx, cy, wt, w, velocity_interval_end, velocity_alignment, bt, bv, rise
        )
        plot_file = f"{dataset}/No.{original_no:02d}.png"
        plot_cell(
            target / f"No.{original_no:02d}.png",
            dataset,
            original_no,
            pt,
            rx,
            ry,
            cx,
            cy,
            vt,
            wt,
            theta,
            w,
            bt,
            bv,
            metrics,
            step_time,
            step_ratio,
            active,
        )
        results.append(
            {
                "dataset": dataset,
                "sample_no": original_no,
                "segment_no": segment_no,
                "source_export": str(exported),
                "center_method": method,
                "velocity_time_alignment": velocity_alignment,
                "plot_file": plot_file,
                "review_status": "unreviewed",
                "review_priority": review_priority(metrics),
                **metrics,
            }
        )
    return results


def run(source_root: Path, raw_root: Path, output_root: Path, selected: set[str] | None = None) -> None:
    inventory = source_inventory(source_root, raw_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if selected is not None:
        inventory.loc[(inventory.status == "screened") & ~inventory.dataset.isin(selected), "status"] = (
            "not_selected_this_run"
        )
    rows = []
    for record in inventory.itertuples(index=False):
        if record.status != "screened":
            continue
        print(f"Screening {record.dataset}", flush=True)
        rows.extend(
            screen_dataset(
                record.dataset, source_root / record.dataset / "repellent_response", output_root / record.dataset
            )
        )
    inventory.to_csv(output_root / "source_inventory.csv", index=False)
    index = pd.DataFrame(rows)
    index.to_csv(output_root / "index.csv", index=False)
    index.loc[index.review_priority == "review_first"].to_csv(output_root / "review_first.csv", index=False)
    template = index[["dataset", "sample_no", "plot_file"]].copy()
    template["decision"] = "unreviewed"
    template["center_notes"] = ""
    template["brightness_notes"] = ""
    template["rotation_notes"] = ""
    template.to_csv(output_root / "selection_template.csv", index=False)
    (output_root / "README.txt").write_text(
        "Private poster repellent-response candidate screening. index.csv links each original sample number to one PNG. "
        "source_inventory.csv separates raw-only data, incomplete exports and known alternative exports; these are not counted as independent cells. "
        "The figures use existing exported center-relative centroids, centers, orbit angles, angular velocities and background brightness. "
        "Only the x-y panel combines centroid and center on the same source rows; rotation and brightness keep their own time arrays. "
        "No gaps are interpolated or filled. A dashed line marks the exported brightness-rise time, not proven stimulus arrival. "
        "Center-step/radius is the displacement between adjacent finite center evaluations divided by the median finite orbit radius; "
        "pairs spanning a time gap are excluded. The provisional speed-activity fractions compare absolute exported angular velocity "
        "with 0.1, 0.2 and 0.3 times the pre-rise median. Point fractions are retained as diagnostics. Time fractions sum "
        "adjacent valid intervals, never bridge timestamp gaps, and count missing duration as unproven in the full-record denominator. "
        "These are screening proxies, not a physical stop classification. "
        "review_first.csv is a conservative visual-review queue: post/pre brightness median at least 1.5, maximum adjacent "
        "center step at most 0.1 times median orbit radius, and over half of recorded duration above 0.2 times the pre-rise "
        "median speed. It is not an acceptance list. All other screened cells remain in index.csv. "
        "Older exports without settings.json have an unknown center method; unmatched raw folders require provenance review. "
        "selection_template.csv is regenerated; any hand-edited selection.csv is preserved. Pre-rise fluctuation and post-rise switching "
        "are reserved for the next analysis step. Measured figures and tables must remain local.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "outputs" / "repellent-response")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data" / "repellent-response")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "poster_fluctuation" / "4_repellent_response"
    )
    parser.add_argument("--datasets", nargs="+", default=None, help="Screen only these complete export datasets")
    args = parser.parse_args()
    source_root, raw_root, output_root = args.source_root.resolve(), args.raw_root.resolve(), args.output_root.resolve()
    inventory = source_inventory(source_root, raw_root)
    available = set(inventory.loc[inventory.status == "screened", "dataset"])
    selected = set(args.datasets) if args.datasets else None
    if selected is not None and not selected <= available:
        parser.error(f"Datasets without complete canonical exports: {sorted(selected - available)}")
    inputs = [
        source_root / dataset / "repellent_response" / name for dataset in (selected or available) for name in REQUIRED
    ]
    inputs += [
        source_root / dataset / "repellent_response" / "settings.json"
        for dataset in (selected or available)
        if (source_root / dataset / "repellent_response" / "settings.json").is_file()
    ]
    with analysis_run(parameters=vars(args), inputs=inputs, outputs=[output_root]) as run_root:
        run(source_root, raw_root, output_root, selected)
    archive = output_root / "_run_records" / run_root.parent.name / run_root.name
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run_root / "manifest.json", archive / "manifest.json")
    shutil.copy2(run_root / "run.log", archive / "run.log")


if __name__ == "__main__":
    main()
