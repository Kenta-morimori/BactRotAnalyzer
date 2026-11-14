"""Local poster figures for the existing SJW46 temperature measurements."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from contextlib import nullcontext
from itertools import repeat
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.functions.frequency_analysis import CONTINUITY_GAP_FACTOR  # noqa: E402
from utils.functions.poster_fluctuation import WINDOWS_S  # noqa: E402
from utils.functions.poster_temperature import (  # noqa: E402
    analyze_cell,
    cell_coordinates,
    cell_summary,
    local_conic_coordinates,
    read_sources,
    source_paths,
)
from utils.run_context import analysis_run  # noqa: E402

TEMPERATURES_C = (10, 23, 30)
COLORS = {10: "#2563a6", 23: "#df822c", 30: "#bd4b6a"}
SPEED = "mean_absolute_angular_velocity_rad_s"
RADIUS = "mean_rotation_radius_um"
RATIO = "low_to_high_amplitude_ratio"


def _style(axes) -> None:
    for ax in np.ravel(axes):
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.2)
        legend = ax.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(11)


def _with_gap_breaks(time: np.ndarray, values: np.ndarray, *, wrap: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Insert plot-only breaks while retaining every source observation."""
    dt = np.diff(time)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    nominal_dt = float(np.median(positive)) if len(positive) else np.nan
    breaks = (
        np.flatnonzero(~np.isfinite(dt) | (dt <= 0) | (dt > CONTINUITY_GAP_FACTOR * nominal_dt)) + 1
        if np.isfinite(nominal_dt)
        else np.array([], dtype=int)
    )
    if wrap:
        jumps = (
            np.flatnonzero(np.isfinite(values[1:]) & np.isfinite(values[:-1]) & (np.abs(np.diff(values)) > np.pi)) + 1
        )
        breaks = np.union1d(breaks, jumps)
    return np.insert(time, breaks, np.nan), np.insert(values, breaks, np.nan)


def _wrapped_plot(ax, time: np.ndarray, angle: np.ndarray) -> None:
    plot_time, plot_angle = _with_gap_breaks(time, angle, wrap=True)
    ax.plot(plot_time, plot_angle, lw=0.8)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0", "π/2", "π"])


def cell_figure(
    frame: pd.DataFrame,
    spectra: pd.DataFrame,
    ratios: pd.DataFrame,
    summary: dict,
    target: Path,
    nominal_rate_hz: float,
) -> None:
    """Show geometry, orbit kinematics, and the six established windows."""
    t = frame["time_sec"].to_numpy(float)
    fig, axes = plt.subplots(4, 3, figsize=(21, 24), constrained_layout=True)
    for ax, suffix, ylabel in (
        (axes[0, 0], "x", "x (µm)"),
        (axes[0, 1], "y", "y (µm; image down)"),
    ):
        tx, centroid = _with_gap_breaks(t, frame[f"centroid_{suffix}_um"].to_numpy(float))
        _, center = _with_gap_breaks(t, frame[f"center_{suffix}_um"].to_numpy(float))
        ax.plot(tx, centroid, lw=0.8, label="Centroid")
        ax.plot(tx, center, lw=1.0, label="Rotation center")
        ax.set(xlabel="Time (s)", ylabel=ylabel, title=f"Centroid and center: {suffix}(t)")
        ax.legend()
    _, xy_x = _with_gap_breaks(t, frame["centroid_x_um"].to_numpy(float))
    _, xy_y = _with_gap_breaks(t, frame["centroid_y_um"].to_numpy(float))
    axes[0, 2].plot(xy_x, xy_y, lw=0.7, label="Centroid")
    axes[0, 2].scatter(frame["center_x_um"], frame["center_y_um"], s=6, alpha=0.3, label="Center")
    axes[0, 2].set(xlabel="x (µm)", ylabel="y (µm; image down)", title="Centroid trajectory and center")
    axes[0, 2].set_aspect("equal", adjustable="box")
    axes[0, 2].invert_yaxis()
    axes[0, 2].legend()

    radius_t, radius = _with_gap_breaks(t, frame["radius_um"].to_numpy(float))
    axes[1, 0].plot(radius_t, radius, lw=0.7)
    axes[1, 0].set(xlabel="Time (s)", ylabel="Radius (µm)", title="Center–centroid distance")
    _wrapped_plot(axes[1, 1], t, frame["orbit_angle_rad"].to_numpy(float))
    axes[1, 1].set(xlabel="Time (s)", ylabel="Angle (rad; CCW +)", title="Orbit angle [−π, π)")
    axes[1, 2].plot(t, frame["orbit_omega_rad_s"], lw=0.7)
    axes[1, 2].set(xlabel="Time (s)", ylabel="Angular velocity (rad/s)", title="Signed orbit angular velocity")

    axes[2, 0].plot(t, frame["orbit_abs_omega_rad_s"], lw=0.7)
    axes[2, 0].set(xlabel="Time (s)", ylabel="Angular speed (rad/s)", title="Absolute orbit angular velocity")
    for width in WINDOWS_S:
        key = f"sd_per_mean_{width:g}s"
        label = f"{width:g} s"
        axes[2, 1].plot(t, frame[key], lw=0.7, label=label)
        axes[2, 2].plot(t, frame[f"{key}_z"], lw=0.7, label=label)
        part = spectra[(spectra.window_s == width) & (spectra.amplitude > 0)]
        axes[3, 0].plot(part.frequency_hz, part.amplitude, lw=0.8, label=label)
    axes[2, 1].set(xlabel="Time (s)", ylabel="SD / Mean", title="Gaussian-window fluctuation")
    axes[2, 2].set(xlabel="Time (s)", ylabel="z-score", title="Standardized fluctuation")
    axes[3, 0].set(xlabel="Frequency (Hz)", ylabel="FFT amplitude (log)", title="FFT of standardized fluctuation")
    axes[3, 0].set_xlim(0, nominal_rate_hz / 2)
    axes[3, 0].set_yscale("log")
    for ax in (axes[2, 1], axes[2, 2], axes[3, 0]):
        ax.legend(ncol=2)

    good = np.isfinite(ratios[RATIO]) & (ratios[RATIO] > 0)
    axes[3, 1].plot(ratios.loc[good, "window_s"], ratios.loc[good, RATIO], "-o")
    axes[3, 1].set(
        xlabel="Gaussian window width (s)",
        ylabel="Mean amplitude ratio (log)",
        title="Low / High: 0–1 Hz / 80–100 Hz",
    )
    axes[3, 1].set_yscale("log")
    axes[3, 1].set_xticks(WINDOWS_S)
    axes[3, 1].tick_params(axis="x", labelrotation=45)
    axes[3, 2].axis("off")
    center_detail = (
        f"Center-adjusted frames: {summary['corrected_center_frame_count']}"
        if "local_conic_center_valid_count" not in summary
        else (
            f"Stored-center-adjusted frames: {summary['corrected_center_frame_count']}\n"
            f"Local-conic center valid: {summary['local_conic_center_valid_count']}"
        )
    )
    axes[3, 2].text(
        0.03,
        0.95,
        "Per-cell summary\n"
        f"Mean |ω|: {summary[SPEED]:.3g} rad/s\n"
        f"Mean radius: {summary[RADIUS]:.3g} µm\n"
        f"0.5 s Low/High: {summary['low_to_high_amplitude_ratio_0.5s']:.3g}\n"
        f"Valid velocity: {summary['valid_velocity_count']}/{summary['source_frame_count']} frames\n"
        f"Timestamp gaps: {summary['timestamp_gap_count']}\n"
        f"{center_detail}",
        transform=axes[3, 2].transAxes,
        va="top",
        fontsize=15,
        linespacing=1.7,
    )
    _style(axes)
    fig.suptitle(f"SJW46 {summary['temperature_c']}°C — No.{summary['sample_no']:02d}", fontsize=22)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def _condition_points(ax, summary: pd.DataFrame, column: str, ylabel: str, title: str) -> None:
    for temp in TEMPERATURES_C:
        group = summary[summary.temperature_c == temp]
        values = group[column].to_numpy(float)
        ax.scatter(np.full(len(group), temp), values, s=42, color=COLORS[temp], alpha=0.72)
        finite = values[np.isfinite(values)]
        if len(finite):
            ax.scatter([temp], [np.mean(finite)], marker="D", s=110, edgecolor="black", color=COLORS[temp], zorder=4)
    ax.set(xlabel="Temperature (°C)", ylabel=ylabel, title=title)
    ax.set_xticks(TEMPERATURES_C)


def comparison_figures(summary: pd.DataFrame, all_ratios: pd.DataFrame, root: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    _condition_points(ax, summary, SPEED, "Mean |ω| (rad/s)", "Mean orbit angular speed per cell")
    _style([ax])
    fig.savefig(root / "compare_mean_angular_speed.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    _condition_points(ax, summary, RADIUS, "Mean radius (µm)", "Mean rotation radius per cell")
    _style([ax])
    fig.savefig(root / "compare_mean_rotation_radius.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8), constrained_layout=True)
    for temp in TEMPERATURES_C:
        group = all_ratios[all_ratios.temperature_c == temp]
        for no, cell in group.groupby("sample_no"):
            points = cell.sort_values("window_s")
            good = np.isfinite(points[RATIO]) & (points[RATIO] > 0)
            ax.plot(points.loc[good, "window_s"], points.loc[good, RATIO], color=COLORS[temp], alpha=0.2, lw=0.8)
        center = group.groupby("window_s", as_index=False)[RATIO].mean()
        good = np.isfinite(center[RATIO]) & (center[RATIO] > 0)
        ax.plot(
            center.loc[good, "window_s"],
            center.loc[good, RATIO],
            "-o",
            color=COLORS[temp],
            lw=2.5,
            label=f"{temp}°C (n={group.sample_no.nunique()})",
        )
    ax.set(
        xlabel="Gaussian window width (s)",
        ylabel="Mean FFT amplitude ratio (log)",
        title="Low / High by window width (thin: cells; bold: cell mean)",
    )
    ax.set_xticks(WINDOWS_S)
    ax.set_yscale("log")
    ax.legend()
    _style([ax])
    fig.savefig(root / "compare_low_frequency_relative_strength.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 8), constrained_layout=True)
    for temp in TEMPERATURES_C:
        group = summary[summary.temperature_c == temp]
        ax.scatter(group[SPEED], group["low_to_high_amplitude_ratio_0.5s"], s=58, color=COLORS[temp], label=f"{temp}°C")
    ax.set(
        xlabel="Mean |ω| (rad/s)",
        ylabel="0.5 s Low / High mean amplitude ratio (log)",
        title="Angular speed vs low-frequency relative strength",
    )
    ax.set_yscale("log")
    ax.legend()
    _style([ax])
    fig.savefig(root / "compare_speed_vs_low_frequency_strength.png", dpi=180)
    plt.close(fig)


def compare_center_methods(summary: pd.DataFrame, baseline_path: Path, destination: Path) -> None:
    baseline = pd.read_csv(baseline_path)
    keys = ["temperature_c", "sample_no"]
    if baseline.duplicated(keys).any() or summary.duplicated(keys).any():
        raise ValueError("temperature and cell identities must be unique")
    joined = baseline[keys + [SPEED, RADIUS, "low_to_high_amplitude_ratio_0.5s"]].merge(
        summary,
        on=keys,
        how="outer",
        suffixes=("_stored", "_local_conic"),
        validate="one_to_one",
        indicator=True,
    )
    if not (joined["_merge"] == "both").all():
        raise ValueError("stored and local_conic cell identities differ")
    joined = joined.drop(columns="_merge")
    metrics = [SPEED, RADIUS, "low_to_high_amplitude_ratio_0.5s"]
    for metric in metrics:
        joined[f"delta_{metric}"] = joined[f"{metric}_local_conic"] - joined[f"{metric}_stored"]
    joined["log10_low_high_fold_change"] = np.log10(
        joined["low_to_high_amplitude_ratio_0.5s_local_conic"] / joined["low_to_high_amplitude_ratio_0.5s_stored"]
    )
    joined.to_csv(destination / "compare_center_methods.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(19, 6), constrained_layout=True)
    for ax, column, label in (
        (axes[0], f"delta_{SPEED}", "Δ mean |ω| (rad/s)"),
        (axes[1], f"delta_{RADIUS}", "Δ mean radius (µm)"),
        (axes[2], "log10_low_high_fold_change", "log10(Low/High fold change)"),
    ):
        for temp in TEMPERATURES_C:
            group = joined[joined.temperature_c == temp]
            ax.scatter(np.full(len(group), temp), group[column], color=COLORS[temp], alpha=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set(xlabel="Temperature (°C)", ylabel=label)
        ax.set_xticks(TEMPERATURES_C)
    _style(axes)
    fig.suptitle("Local conic minus stored-center analysis: cell-level changes", fontsize=19)
    fig.savefig(destination / "compare_center_method_effect.png", dpi=180)
    plt.close(fig)


def run(
    source_root: Path, config_root: Path, destination: Path, *, center_mode: str = "stored", workers: int = 4
) -> None:
    if center_mode not in {"stored", "local_conic"}:
        raise ValueError(center_mode)
    if workers < 1:
        raise ValueError("workers must be positive")
    destination.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_ratios = []
    pool_context = ProcessPoolExecutor(max_workers=workers) if center_mode == "local_conic" else nullcontext(None)
    with pool_context as pool:
        for temp in TEMPERATURES_C:
            sources = read_sources(source_root, config_root, temp)
            folder = destination / f"{temp}C"
            tables = folder / "tables"
            tables.mkdir(parents=True, exist_ok=True)
            print(f"{temp}°C: {sources.sample_count} cells", flush=True)
            coordinates = (
                pool.map(local_conic_coordinates, repeat(sources), range(1, sources.sample_count + 1))
                if pool is not None
                else ((cell_coordinates(sources, no), None) for no in range(1, sources.sample_count + 1))
            )
            for no, (raw, method) in enumerate(coordinates, start=1):
                stem = f"No.{no:02d}"
                frame, spectra, ratios, fft_metadata = analyze_cell(raw)
                summary = cell_summary(frame, ratios, temp, no)
                if center_mode == "local_conic":
                    status = frame["local_conic_center_status"]
                    summary["local_conic_center_valid_count"] = int(
                        frame[["center_x_um", "center_y_um"]].notna().all(axis=1).sum()
                    )
                    summary["local_conic_direct_count"] = int((status == "direct").sum())
                    summary["local_conic_short_interpolation_count"] = int((status == "short_interpolation").sum())
                    shift = np.hypot(
                        frame["center_x_um"] - frame["stored_center_x_um"],
                        frame["center_y_um"] - frame["stored_center_y_um"],
                    )
                    summary["mean_center_shift_um"] = float(shift.mean())
                summary["plot_file"] = f"{temp}C/{stem}.png"
                summary["series_file"] = f"{temp}C/tables/{stem}_time_series.csv"
                summary["fft_file"] = f"{temp}C/tables/{stem}_fft.csv"
                summary["ratio_file"] = f"{temp}C/tables/{stem}_band_amplitude_ratio.csv"
                summaries.append(summary)
                ratios.insert(0, "sample_no", no)
                ratios.insert(0, "temperature_c", temp)
                all_ratios.append(ratios)
                frame.to_csv(tables / f"{stem}_time_series.csv", index=False)
                spectra.to_csv(tables / f"{stem}_fft.csv", index=False)
                ratios.to_csv(tables / f"{stem}_band_amplitude_ratio.csv", index=False)
                (tables / f"{stem}_metadata.json").write_text(
                    json.dumps(
                        {
                            "temperature_c": temp,
                            "sample_no": no,
                            "configured_frame_rate_hz": sources.frame_rate_hz,
                            "fft_timebase": fft_metadata,
                            "centroid_source": "source centroid_coordinate.csv stores centroid minus corrected rotation center in um",
                            "angle_definition": "atan2(-centered_y, centered_x), wrapped to [-pi,pi), CCW positive",
                            "velocity_definition": "shortest wrapped phase increment / actual timestamp difference; abs for speed",
                            "fluctuation_definition": "Gaussian-window SD(abs(omega))/Mean(abs(omega)), one-frame slide",
                            "relative_strength_definition": "mean FFT amplitude at 0<f<1 Hz / mean FFT amplitude at 80<=f<100 Hz",
                            "center_method": center_mode,
                            "local_conic_method": method,
                        },
                        indent=2,
                        allow_nan=True,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                cell_figure(frame, spectra, ratios, summary, folder / f"{stem}.png", sources.frame_rate_hz)
                print(f"  No.{no:02d} done", flush=True)
    summary_frame = pd.DataFrame(summaries)
    ratio_frame = pd.concat(all_ratios, ignore_index=True)
    summary_frame.to_csv(destination / "cell_summary.csv", index=False)
    ratio_frame.to_csv(destination / "window_band_amplitude_ratio.csv", index=False)
    comparison_figures(summary_frame, ratio_frame, destination)
    if center_mode == "local_conic":
        compare_center_methods(
            summary_frame,
            source_root / "poster_fluctuation" / "3_Temperature_Control" / "cell_summary.csv",
            destination,
        )
    (destination / "README.txt").write_text(
        "Private local SJW46 temperature-control analysis. The source tables are the existing "
        "SJW46_temp=10/23/30 outputs; they contain corrected-center-relative centroids in µm, "
        "corrected absolute centers, and source timestamps. Each 10C/23C/30C folder contains "
        "one PNG per cell and matching time-series, FFT, fixed-band ratios, and metadata in tables/. "
        "The orbit angle is atan2(-centered y, centered x), CCW-positive and wrapped to [-pi,pi). "
        "Signed velocity uses adjacent valid frames and actual dt; gaps are not bridged. "
        "Speed is abs(velocity); mean speed and radius average valid within-cell frames. "
        "The six Gaussian windows slide by one source frame. The standardized SD/Mean FFT "
        "uses the longest continuous valid interval. Relative low-frequency strength is mean "
        "amplitude over 0<f<1 Hz divided by mean amplitude over 80<=f<100 Hz. "
        "The comparison scatter uses the 0.5 s window. Group lines are cell means, with "
        "individual cells shown; these cells are not assumed to be independent experimental replicates. "
        "Previous center corrections and source timestamp gaps are counted in cell_summary.csv. "
        + (
            "For local_conic, the same stored raw centroids are refit by the repellent default local_conic "
            "estimator and completion. Center failures remain missing; compare_center_methods.csv and "
            "compare_center_method_effect.png compare cell-level results with stored-center analysis.\n"
            if center_mode == "local_conic"
            else "\n"
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "outputs")
    parser.add_argument("--config-root", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--output-root", type=Path, default=ROOT / "outputs" / "poster_fluctuation" / "3_Temperature_Control"
    )
    parser.add_argument("--center-mode", choices=("stored", "local_conic"), default="stored")
    parser.add_argument("--workers", type=int, default=4, help="parallel local_conic cell fits")
    args = parser.parse_args()
    source_root, config_root, output_root = (
        args.source_root.resolve(),
        args.config_root.resolve(),
        args.output_root.resolve(),
    )
    if args.center_mode == "local_conic":
        output_root /= "local_conic"
    inputs = [path for temp in TEMPERATURES_C for path in source_paths(source_root, config_root, temp)]
    if args.center_mode == "local_conic":
        inputs.append(source_root / "poster_fluctuation" / "3_Temperature_Control" / "cell_summary.csv")
    with analysis_run(parameters=vars(args), inputs=inputs, outputs=[output_root]) as run_root:
        run(source_root, config_root, output_root, center_mode=args.center_mode, workers=args.workers)
    archive = output_root / "_run_records" / run_root.parent.name / run_root.name
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run_root / "manifest.json", archive / "manifest.json")
    shutil.copy2(run_root / "run.log", archive / "run.log")


if __name__ == "__main__":
    main()
