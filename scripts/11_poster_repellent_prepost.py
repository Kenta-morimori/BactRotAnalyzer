"""Compare pre-rise fluctuation and post-rise diagnostic sign changes.

This CLI reads private, source-aligned repellent results through a private JSON
manifest. It never edits those inputs or publishes measured data to Git.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.functions.poster_repellent_prepost import (  # noqa: E402
    apply_masks,
    pre_rise_fluctuation,
    radius_excursion_candidates,
    sign_change_summary,
    validate_series,
)
from utils.run_context import analysis_run  # noqa: E402

plt.rcParams["font.family"] = "Arial"


def _join_bundle_table(base: Path, relative: str, sample: int, name: str) -> pd.DataFrame:
    path = base / relative
    table = pd.read_csv(path)
    table = table.loc[table.sample_no == sample].copy()
    if table.empty or table.source_frame_1based.duplicated().any():
        raise ValueError(f"{name}: missing sample or duplicate source frame: {path}")
    table = table.set_index("source_frame_1based")
    table = table.rename(columns={"time_sec": f"time_{name}"})
    return table


def load_series(source_root: Path, case: dict) -> tuple[pd.DataFrame, list[Path]]:
    path = source_root / case["source"]
    if case["input_kind"] == "source_aligned_csv":
        raw = pd.read_csv(path)
        series = pd.DataFrame(
            {
                "source_frame_1based": raw.source_frame_1based,
                "time_sec": raw.time_sec,
                "centroid_x": raw.raw_x_px,
                "centroid_y": raw.raw_y_px,
                "center_x": raw.center_x_px,
                "center_y": raw.center_y_px,
                "angular_velocity_rad_s": raw.angular_velocity_rad_s,
                "background_intensity": raw.background_intensity,
            }
        )
        return validate_series(series), [path]
    if case["input_kind"] != "repellent_bundle":
        raise ValueError(f"unknown input kind: {case['input_kind']}")
    sample = int(case["sample_no"])
    names = {
        "centroid": "00_all_rotational_analysis/centroid_coordinate/centroid_time_series.csv",
        "center": "00_all_rotational_analysis/center_coordinate/center_coordinate.csv",
        "velocity": "00_all_rotational_analysis/angular_velocity/angular_velocity_time_series.csv",
        "brightness": "01_brightness_change/background_intensity_time_series.csv",
    }
    paths = [path / rel for rel in names.values()]
    tables = {name: _join_bundle_table(path, rel, sample, name) for name, rel in names.items()}
    common = tables["centroid"].index
    for name, table in tables.items():
        if not common.equals(table.index):
            raise ValueError(f"source-frame identity mismatch: {case['id']} {name}")
    clock = tables["centroid"].time_centroid.to_numpy(float)
    for name, table in tables.items():
        if not np.allclose(clock, table[f"time_{name}"].to_numpy(float), rtol=0, atol=1e-5):
            raise ValueError(f"time identity mismatch: {case['id']} {name}")
    series = pd.DataFrame(
        {
            "source_frame_1based": common.to_numpy(int),
            "time_sec": clock,
            "centroid_x": tables["centroid"].raw_x_um.to_numpy(float),
            "centroid_y": tables["centroid"].raw_y_um.to_numpy(float),
            "center_x": tables["center"].center_x_um.to_numpy(float),
            "center_y": tables["center"].center_y_um.to_numpy(float),
            "angular_velocity_rad_s": tables["velocity"].angular_velocity_rad_s.to_numpy(float),
            "background_intensity": tables["brightness"].background_intensity.to_numpy(float),
        }
    )
    return validate_series(series), paths


def _masked_time_spans(time: np.ndarray, invalid: np.ndarray) -> list[tuple[float, float]]:
    """Return inclusive runs as plot spans without joining separate gaps."""
    indices = np.flatnonzero(invalid)
    if not len(indices):
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1) + 1
    runs = np.split(indices, breaks)
    step = float(np.median(np.diff(time))) if len(time) > 1 else 0.0
    return [(float(time[run[0]]), float(time[run[-1]] + step)) for run in runs]


def plot_overview(cases: list[tuple[dict, pd.DataFrame, pd.DataFrame]], path: Path) -> None:
    fig = plt.figure(figsize=(18, 5.8 * len(cases)))
    outer = fig.add_gridspec(len(cases), 1, left=0.11, right=0.98, top=0.97, bottom=0.07, hspace=0.43)
    for row, (case, _raw, series) in enumerate(cases):
        inner = outer[row].subgridspec(2, 1, height_ratios=(1.4, 1), hspace=0.04)
        speed_ax = fig.add_subplot(inner[0])
        brightness_ax = fig.add_subplot(inner[1], sharex=speed_ax)
        start, rise, end = case["analysis_start_s"], case["rise_time_s"], case["analysis_end_s"]
        shown = series.loc[(series.time_sec >= start) & (series.time_sec < end)]
        t = shown.time_sec.to_numpy(float)
        omega = shown.angular_velocity_rad_s.to_numpy(float)
        speed_ax.plot(t, np.where(omega > 0, omega, np.nan), lw=0.9, color="#2774ae", alpha=0.8)
        speed_ax.plot(t, np.where(omega < 0, omega, np.nan), lw=0.9, color="#d95f02", alpha=0.85)
        speed_ax.fill_between(t, 0, omega, where=omega < 0, color="#d95f02", alpha=0.17)
        speed_ax.axhline(0, color="black", lw=0.8)
        speed_ax.set_ylabel("Angular velocity [rad/s]", fontsize=23)
        speed_ax.set_title(case["label"], fontsize=26)
        brightness_ax.plot(t, shown.background_intensity, lw=1.8, color="#6a4c93")
        brightness_ax.set_ylabel("Brightness [a.u.]", fontsize=23)
        brightness_ax.set_xlabel("Time [s]", fontsize=23)
        speed_ax.tick_params(labelbottom=False)
        for ax in (speed_ax, brightness_ax):
            ax.axvline(rise, color="#d81b60", lw=3.0, ls="--")
            for lo, hi in _masked_time_spans(t, shown.centroid_invalid.to_numpy(bool)):
                ax.axvspan(lo, hi, color="#c62828", alpha=0.16, lw=0)
            ax.set_xlim(start, end)
            ax.tick_params(labelsize=20)
            ax.grid(alpha=0.18)
    fig.legend(
        handles=[
            Line2D([], [], color="#2774ae", label="CCW (+)"),
            Line2D([], [], color="#d95f02", label="CW (−)"),
            Line2D([], [], color="#d81b60", ls="--", label="Brightness rise"),
            Patch(facecolor="#c62828", alpha=0.16, label="Excluded centroid frames"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=4,
        frameon=False,
        fontsize=19,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_centroid_mask_qc(cases: list[tuple[dict, pd.DataFrame, pd.DataFrame]], path: Path) -> None:
    """Show original excluded coordinates alongside the retained centroid and center."""
    fig, axes = plt.subplots(len(cases), 2, figsize=(21, 4.7 * len(cases)))
    fig.subplots_adjust(left=0.09, right=0.98, top=0.97, bottom=0.09, hspace=0.67, wspace=0.18)
    if len(cases) == 1:
        axes = axes[None, :]
    for row, (case, raw, series) in enumerate(cases):
        start, end = case["analysis_start_s"], case["analysis_end_s"]
        included = (raw.time_sec >= start) & (raw.time_sec < end)
        original = raw.loc[included]
        clean = series.loc[included]
        t = original.time_sec.to_numpy(float)
        invalid = clean.centroid_invalid.to_numpy(bool)
        unit = "px" if case["input_kind"] == "source_aligned_csv" else "µm"
        for col, axis in enumerate(("x", "y")):
            ax = axes[row, col]
            raw_coord = original[f"centroid_{axis}"].to_numpy(float)
            clean_coord = clean[f"centroid_{axis}"].to_numpy(float)
            center = clean[f"center_{axis}"].to_numpy(float)
            ax.plot(t, raw_coord, color="#9ecae1", lw=0.55, alpha=0.6)
            ax.plot(t, clean_coord, color="#2774ae", lw=0.65, alpha=0.8)
            ax.plot(t, center, color="#e6550d", lw=1.1, alpha=0.9)
            ax.scatter(t[invalid], raw_coord[invalid], color="#c62828", marker="x", s=26, zorder=4)
            for lo, hi in _masked_time_spans(t, invalid):
                ax.axvspan(lo, hi, color="#c62828", alpha=0.12, lw=0)
            ax.axvline(case["rise_time_s"], color="#d81b60", lw=1.8, ls="--")
            ax.set_xlim(start, end)
            ax.set_title(f"{case['label']}: {axis}(t) · excluded {invalid.sum()} frames", fontsize=24)
            ax.set_xlabel("Time [s]", fontsize=22)
            ax.set_ylabel(f"Position [{unit}]", fontsize=22)
            ax.tick_params(labelsize=20)
            ax.grid(alpha=0.18)
    fig.legend(
        handles=[
            Line2D([], [], color="#9ecae1", label="Original centroid"),
            Line2D([], [], color="#2774ae", label="Retained centroid"),
            Line2D([], [], color="#e6550d", label="Rotation center"),
            Line2D([], [], color="#c62828", marker="x", ls="None", label="Excluded original centroid"),
            Line2D([], [], color="#d81b60", ls="--", label="Brightness rise"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=5,
        frameon=False,
        fontsize=18,
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_comparison(summary: pd.DataFrame, ratios: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.25, wspace=0.3)
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary)))
    widths = sorted(ratios.window_s.unique())
    width_positions = {value: index for index, value in enumerate(widths)}
    for color, row in zip(colors, summary.itertuples()):
        r = ratios.loc[ratios.case_id == row.case_id]
        axes[0].plot(
            [width_positions[value] for value in r.window_s],
            r.low_to_high_amplitude_ratio,
            "o-",
            lw=3,
            ms=10,
            color=color,
            label=row.label,
        )
    axes[0].set_xlabel("Window width [s]")
    axes[0].set_ylabel("Pre-rise Low/High ratio")
    axes[0].set_xticks(range(len(widths)), [f"{value:g}" for value in widths])
    for color, row in zip(colors, summary.itertuples()):
        axes[1].scatter(row.low_high_ratio_0p5s, row.sign_changes_per_observed_s, color=color, s=230)
    axes[1].set_xlabel("Pre-rise Low/High ratio (0.5 s window)")
    axes[1].set_ylabel("Post-rise sign changes [1/s]")
    axes[1].set_xlim(left=0)
    axes[1].set_ylim(bottom=0)
    for ax in axes:
        ax.tick_params(labelsize=21)
        ax.xaxis.label.set_fontsize(23)
        ax.yaxis.label.set_fontsize(23)
        ax.grid(alpha=0.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.04), ncol=5, frameon=False, fontsize=19)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_pre_low_high_vs_post_sign_changes(summary: pd.DataFrame, path: Path) -> None:
    """Poster-sized single-panel view of the already computed pre/post metrics."""
    fig, ax = plt.subplots(figsize=(10.5, 8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(summary)))
    for color, row in zip(colors, summary.itertuples()):
        ax.scatter(
            row.low_high_ratio_0p5s,
            row.sign_changes_per_observed_s,
            color=color,
            s=290,
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
    ax.set_xlabel("Pre-rise Low/High ratio (0.5 s window)", fontsize=26)
    ax.set_ylabel("Post-rise sign changes [1/s]", fontsize=26)
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=24)
    ax.grid(alpha=0.2)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not manifest.get("cases"):
        raise ValueError("manifest version 1 with nonempty cases is required")
    cases = manifest["cases"]
    if len({c["id"] for c in cases}) != len(cases):
        raise ValueError("case ids must be unique")
    source_paths = []
    for case in cases:
        if case["input_kind"] == "source_aligned_csv":
            source_paths.append(args.source_root / case["source"])
        else:
            bundle = args.source_root / case["source"]
            source_paths.extend(
                bundle / rel
                for rel in (
                    "00_all_rotational_analysis/centroid_coordinate/centroid_time_series.csv",
                    "00_all_rotational_analysis/center_coordinate/center_coordinate.csv",
                    "00_all_rotational_analysis/angular_velocity/angular_velocity_time_series.csv",
                    "01_brightness_change/background_intensity_time_series.csv",
                )
            )
    with analysis_run(
        parameters={"manifest": manifest, "source_root": str(args.source_root.resolve())},
        inputs=[args.manifest, *source_paths],
        outputs=[args.output_dir],
    ):
        args.output_dir.mkdir(parents=True, exist_ok=True)
        tables = args.output_dir / "tables"
        tables.mkdir(exist_ok=True)
        summaries, ratio_tables, overviews = [], [], []
        for case in cases:
            raw, _ = load_series(args.source_root, case)
            suggested = radius_excursion_candidates(raw)
            included = (raw.time_sec >= case["analysis_start_s"]) & (raw.time_sec < case["analysis_end_s"])
            reviewed_ranges = case.get("invalid_source_frame_ranges", [])
            series = apply_masks(raw, reviewed_ranges)
            qc = series[["source_frame_1based", "time_sec", "centroid_invalid", "velocity_invalid"]].copy()
            qc["radius_excursion_review_flag"] = suggested
            qc.to_csv(tables / f"{case['id']}_frame_qc.csv", index=False)
            pre, fft, ratios, metadata = pre_rise_fluctuation(series, case["analysis_start_s"], case["rise_time_s"])
            post, bins = sign_change_summary(series, case["rise_time_s"], case["analysis_end_s"])
            pre.to_csv(tables / f"{case['id']}_pre_sd_time_series.csv", index=False)
            fft.to_csv(tables / f"{case['id']}_pre_sd_fft.csv", index=False)
            bins.to_csv(tables / f"{case['id']}_post_1s_bins.csv", index=False)
            ratios.insert(0, "case_id", case["id"])
            ratios.insert(1, "label", case["label"])
            ratio_tables.append(ratios)
            r05 = ratios.loc[ratios.window_s == 0.5, "low_to_high_amplitude_ratio"]
            summaries.append(
                {
                    "case_id": case["id"],
                    "label": case["label"],
                    "rise_time_s": case["rise_time_s"],
                    "analysis_start_s": case["analysis_start_s"],
                    "analysis_end_s": case["analysis_end_s"],
                    "total_source_frames": int(included.sum()),
                    "masked_centroid_frames": int((series.centroid_invalid & included).sum()),
                    "unreviewed_radius_excursion_frames": int((suggested & included & ~series.centroid_invalid).sum()),
                    "low_high_ratio_0p5s": float(r05.iloc[0]) if len(r05) else np.nan,
                    "pre_fft_0p5s_samples": int(next(m["sample_count"] for m in metadata if m["window_s"] == 0.5)),
                    **post,
                }
            )
            overviews.append((case, raw, series))
            print(
                f"{case['id']}: pre 0.5s Low/High={summaries[-1]['low_high_ratio_0p5s']:.4g}, "
                f"post sign changes/s={post['sign_changes_per_observed_s']:.4g}, "
                f"masked={summaries[-1]['masked_centroid_frames']}"
            )
        summary = pd.DataFrame(summaries)
        all_ratios = pd.concat(ratio_tables, ignore_index=True)
        summary.to_csv(args.output_dir / "summary.csv", index=False)
        all_ratios.to_csv(args.output_dir / "pre_low_high_by_window.csv", index=False)
        plot_overview(overviews, args.output_dir / "candidate_prepost_overview.png")
        plot_centroid_mask_qc(overviews, args.output_dir / "centroid_mask_qc.png")
        plot_comparison(summary, all_ratios, args.output_dir / "compare_pre_low_high_and_post_switching.png")
        plot_pre_low_high_vs_post_sign_changes(summary, args.output_dir / "pre_low_high_vs_post_sign_changes.png")
        (args.output_dir / "README.txt").write_text(
            "Private five-cell poster comparison. Pre-rise: Gaussian SD(|primary angular velocity|)/Mean, "
            "z-score, longest contiguous FFT, mean amplitude 0<f<1 Hz divided by 80<=f<100 Hz. "
            "Post-rise: each adjacent nonzero signed velocity change once, divided by eligible observed seconds. "
            "This is a diagnostic sign-change rate, not confirmed biological reversals. "
            "Reviewed centroid failures are NaN; no interpolation across them. "
            "centroid_mask_qc.png overlays excluded original coordinates on retained centroid and center; "
            "candidate_prepost_overview.png stacks aligned angular velocity and brightness; "
            "negative CW velocity is orange and positive CCW is blue. "
            "compare_pre_low_high_and_post_switching.png has width-wise ratio and unlabeled scatter panels. "
            "pre_low_high_vs_post_sign_changes.png is the legend-free standalone scatter. "
            "Radius excursion flags are review suggestions only and do not alter the analysis.\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
