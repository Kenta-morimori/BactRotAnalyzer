"""Local poster figures for long-axis fluctuation and synthetic null controls."""

import argparse
import configparser
import json
import re
import shutil
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.functions.get_centroid_coordinate import contours  # noqa: E402
from utils.functions.poster_fluctuation import (  # noqa: E402
    WINDOWS_S,
    analyze,
    axis_angular_velocity,
    band_amplitude_ratios,
    duplicate_source_map,
    major_axis_deg,
    synthetic_long_axis,
)
from utils.run_context import analysis_run  # noqa: E402

DATASETS = {"20231005": "MotAB overexpression", "20230607": "Fixed rotational direction", "20231107": "Brake protein"}
POSTER_CANDIDATES = (("20230607", 6), ("20230607", 8), ("20231005", 1), ("20231107", 6))
BAND_PANELS = (
    ("low_to_high_amplitude_ratio", "Low / High", "Mean FFT amplitude ratio", "log"),
    ("low_mean_amplitude", "Low only", "Mean FFT amplitude", "linear"),
    ("low_minus_high_amplitude", "Low − High", "Difference of mean FFT amplitudes", "linear"),
)


def dataset_files(data_root: Path, day: str) -> tuple[list[Path], dict]:
    folder = data_root / day
    config = configparser.ConfigParser()
    if not config.read(folder / "config.ini"):
        raise FileNotFoundError(folder / "config.ini")
    section = config["Settings"]
    files = sorted(folder.glob("*.avi"), key=lambda p: int(re.findall(r"\d+", p.stem)[-1]))
    expected = section.getint("sample_num")
    if len(files) != expected:
        raise ValueError(f"{day}: expected {expected} AVI files, found {len(files)}")
    settings = {
        "rate_hz": section.getfloat("FrameRate"),
        "duration_s": section.getfloat("total_time"),
        "px2um_x": section.getfloat("px2um_x"),
        "px2um_y": section.getfloat("px2um_y"),
    }
    return files, settings


def read_avi(path: Path, settings: dict) -> pd.DataFrame:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise OSError(f"Cannot open {path}")
    rate = settings["rate_hz"]
    rows: list[tuple[int, float, float, float, float, float]] = []
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            x, y, ellipse = contours(image)
            rows.append(
                (
                    len(rows) + 1,
                    len(rows) / rate,
                    x * settings["px2um_x"],
                    y * settings["px2um_y"],
                    major_axis_deg(ellipse),
                    float(max(ellipse[1]) / min(ellipse[1])) if ellipse is not None and min(ellipse[1]) > 0 else np.nan,
                )
            )
    finally:
        capture.release()
    if len(rows) != round(rate * settings["duration_s"]):
        raise ValueError(f"{path}: frame count disagrees with config")
    return pd.DataFrame(
        rows, columns=["source_frame_1based", "time_sec", "x_um", "y_um", "long_axis_deg", "axis_ratio"]
    )


def analyze_frame(source: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict]:
    time = source["time_sec"].to_numpy(float)
    axis = source["long_axis_deg"].to_numpy(float)
    long_angle, long_omega = axis_angular_velocity(time, axis)
    long_result = analyze(time, long_omega)
    frame = source.copy()
    frame["long_axis_unwrapped_rad"] = long_angle
    frame["long_axis_phase_rad"] = (long_angle + np.pi) % (2 * np.pi) - np.pi
    frame["long_axis_omega_rad_s"] = long_omega
    frame["long_axis_abs_omega_rad_s"] = np.abs(long_omega)
    for column in long_result.frame.columns:
        if column != "time_sec" and column != "angular_velocity_rad_s":
            frame[f"long_axis_{column}"] = long_result.frame[column]
    spectra = {"long_axis": long_result.spectra}
    valid_axis = int(np.isfinite(long_omega).sum())
    ratios = source["axis_ratio"].to_numpy(float)
    finite_ratios = ratios[np.isfinite(ratios)]
    raw_delta = (np.diff(axis) + 90.0) % 180.0 - 90.0
    finite_delta = np.abs(raw_delta[np.isfinite(raw_delta)])
    info = {
        "frame_count": len(frame),
        "detected_count": (
            int(np.isfinite(source["x_um"].to_numpy(float)).sum()) if "x_um" in source else int(np.isfinite(axis).sum())
        ),
        "long_axis_omega_count": valid_axis,
        "long_axis_omega_fraction": valid_axis / max(1, len(frame) - 1),
        "axis_ratio_p10": float(np.percentile(finite_ratios, 10)) if len(finite_ratios) else np.nan,
        "axis_step_abs_deg_p95": float(np.percentile(finite_delta, 95)) if len(finite_delta) else np.nan,
        "long_axis_fft": long_result.metadata,
        "angle_warning": "The unoriented long axis has a 180-degree ambiguity; angular changes of >=90 degrees per sample are not identifiable.",
    }
    return frame, spectra, info


def wrapped_phase_plot(time: np.ndarray, phase: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep every observed phase while breaking the line at a 2π wrap."""
    jumps = np.flatnonzero(np.isfinite(phase[1:]) & np.isfinite(phase[:-1]) & (np.abs(np.diff(phase)) > np.pi)) + 1
    return np.insert(time, jumps, np.nan), np.insert(phase, jumps, np.nan)


def style_axes(axes: np.ndarray) -> None:
    for ax in axes.flat:
        ax.title.set_fontsize(16)
        ax.xaxis.label.set_fontsize(14)
        ax.yaxis.label.set_fontsize(14)
        ax.tick_params(labelsize=12)
        ax.grid(alpha=0.2)
        legend = ax.get_legend()
        if legend is not None:
            for label in legend.get_texts():
                label.set_fontsize(11)


def figure(
    frame: pd.DataFrame,
    spectra: dict[str, pd.DataFrame],
    title: str,
    target: Path,
    rate_hz: float,
    band_ratios: pd.DataFrame,
    *,
    includes_stops: bool = False,
) -> None:
    """One complete long-axis poster sheet per observed cell or synthetic case."""
    t = frame["time_sec"].to_numpy(float)
    measured = "x_um" in frame and "y_um" in frame
    rows = 4 if measured else 3
    fig, axes = plt.subplots(rows, 3, figsize=(18, 20 if measured else 16), constrained_layout=True)
    if measured:
        x = frame["x_um"].to_numpy(float)
        y = frame["y_um"].to_numpy(float)
        axes[0, 0].plot(t, x, linewidth=0.7)
        axes[0, 0].set(xlabel="Time (s)", ylabel="x (um)", title="Centroid x vs time")
        axes[0, 1].plot(t, y, linewidth=0.7)
        axes[0, 1].set(xlabel="Time (s)", ylabel="y (um, image down)", title="Centroid y vs time")
        axes[0, 2].plot(x, y, linewidth=0.7)
        axes[0, 2].set(xlabel="x (um)", ylabel="y (um, image down)", title="Centroid x-y trajectory")
        axes[0, 2].set_aspect("equal", adjustable="box")
        axes[0, 2].invert_yaxis()
    phase_ax, signed_ax, absolute_ax = axes[1 if measured else 0]
    stats_axes = axes[2 if measured else 1]
    band_axes = axes[3 if measured else 2]
    phase_t, phase = wrapped_phase_plot(t, frame["long_axis_phase_rad"].to_numpy(float))
    phase_ax.plot(phase_t, phase, linewidth=0.7)
    phase_ax.set(
        xlabel="Time (s)",
        ylabel="Relative phase (rad, CCW +)",
        title="Inferred phase (180° axis ambiguity)",
    )
    if not measured and frame["baseline_angle_rad"].eq(0).all():
        phase_limit = max(0.01, 1.1 * np.nanmax(np.abs(phase)))
        phase_ax.set_ylim(-phase_limit, phase_limit)
    else:
        phase_ax.set_ylim(-np.pi, np.pi)
        phase_ax.set_yticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], ["−π", "−π/2", "0", "π/2", "π"])
    signed_ax.plot(t, frame["long_axis_omega_rad_s"], linewidth=0.7, label="long axis (CCW +)")
    absolute_ax.plot(t, frame["long_axis_abs_omega_rad_s"], linewidth=0.7, label="|long axis|")
    signed_ax.set(xlabel="Time (s)", ylabel="rad/s", title="Signed angular velocity")
    absolute_ax.set(xlabel="Time (s)", ylabel="rad/s", title="Absolute angular velocity")
    signed_ax.legend()
    absolute_ax.legend()
    for width in WINDOWS_S:
        label = f"{width:g} s"
        prefix = f"long_axis_sd_per_mean_{width:g}s"
        stats_axes[0].plot(t, frame[prefix], linewidth=0.8, label=label)
        stats_axes[1].plot(t, frame[f"{prefix}_z"], linewidth=0.8, label=label)
        spectrum = spectra["long_axis"]
        part = spectrum[(spectrum.window_s == width) & np.isfinite(spectrum.amplitude) & (spectrum.amplitude > 0)]
        stats_axes[2].plot(part.frequency_hz, part.amplitude, linewidth=0.8, label=label)
    stats_axes[0].set(xlabel="Time (s)", ylabel="SD / Mean", title="Gaussian-window SD / Mean")
    stats_axes[1].set(xlabel="Time (s)", ylabel="z-score", title="Standardized SD / Mean")
    stats_axes[2].set(xlabel="Frequency (Hz)", ylabel="FFT amplitude (log)", title="FFT of standardized SD / Mean")
    stats_axes[2].set_xlim(0, rate_hz / 2)
    stats_axes[2].set_xticks(np.linspace(0, rate_hz / 2, 6))
    stats_axes[2].set_yscale("log")
    for ax in stats_axes:
        ax.legend(ncol=2)
    for ax, (column, label, ylabel, scale) in zip(band_axes, BAND_PANELS):
        values = band_ratios[column].to_numpy(float)
        good = np.isfinite(values) & ((values > 0) if scale == "log" else True)
        ax.plot(band_ratios.loc[good, "window_s"], values[good], "-o")
        ax.set(xlabel="Gaussian window width (s)", ylabel=ylabel, title=label)
        ax.set_xticks(WINDOWS_S)
        ax.set_yscale(scale)
        ax.tick_params(axis="x", labelrotation=45)
    band_axes[0].set_title("Low / High (full record; stops retained)" if includes_stops else "Low / High (full record)")
    style_axes(axes)
    fig.suptitle(title, fontsize=20)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def candidate_band_comparison(ratios: pd.DataFrame, target: Path) -> None:
    """Compare the user-selected cells without averaging distinct cells."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 7), constrained_layout=True)
    for ax, (column, title, ylabel, scale) in zip(axes, BAND_PANELS):
        for day, sample_no in POSTER_CANDIDATES:
            group = ratios[(ratios.day == day) & (ratios.sample_no == sample_no)].sort_values("window_s")
            values = group[column].to_numpy(float)
            good = np.isfinite(values) & ((values > 0) if scale == "log" else True)
            label = f"{day} No.{sample_no:02d}"
            if (day, sample_no) == ("20231005", 1):
                label += " (stops included)"
            ax.plot(group.loc[good, "window_s"], values[good], "-o", label=label)
        ax.set(xlabel="Gaussian window width (s)", ylabel=ylabel, title=title)
        ax.set_xticks(WINDOWS_S)
        ax.set_yscale(scale)
        ax.tick_params(axis="x", labelrotation=45)
    axes[0].legend()
    fig.suptitle("Candidate cells: Low = 0–1 Hz, High = 80–100 Hz", fontsize=18)
    style_axes(axes)
    fig.savefig(target, dpi=180)
    plt.close(fig)


def save_case(source: pd.DataFrame, destination: Path, stem: str, title: str, metadata: dict) -> dict:
    rate_hz = float(metadata["rate_hz"])
    time = source["time_sec"].to_numpy(float)
    dt = np.diff(time)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    if not np.isfinite(rate_hz) or rate_hz <= 0 or not len(positive_dt):
        raise ValueError("A positive frame rate and increasing timestamps are required")
    if not np.isclose(1 / np.median(positive_dt), rate_hz, rtol=1e-6):
        raise ValueError("Frame rate disagrees with the input timebase")
    frame, spectra, info = analyze_frame(source)
    band_ratios = band_amplitude_ratios(spectra["long_axis"])
    destination.mkdir(parents=True, exist_ok=True)
    tables = destination / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / f"{stem}_time_series.csv", index=False)
    for name, spectrum in spectra.items():
        spectrum.to_csv(tables / f"{stem}_{name}_fft.csv", index=False)
    band_ratios.to_csv(tables / f"{stem}_band_amplitude_ratio.csv", index=False)
    with (tables / f"{stem}_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                **metadata,
                **info,
                "window_widths_s": WINDOWS_S,
                "fft_input": "z-scored Gaussian-window SD(abs(long-axis angular velocity))/Mean(abs(long-axis angular velocity))",
                "fft_display_nyquist_hz": rate_hz / 2,
                "band_ratio_definition": "mean FFT amplitude for 0 < f < 1 Hz divided by mean FFT amplitude for 80 <= f < 100 Hz",
                "band_ratio_source": "z-scored Gaussian-window SD(abs(long-axis angular velocity))/Mean(abs(long-axis angular velocity))",
                "band_ratio_includes_stops": metadata["kind"] == "measured",
            },
            stream,
            indent=2,
            allow_nan=True,
        )
    figure(
        frame,
        spectra,
        title,
        destination / f"{stem}.png",
        rate_hz,
        band_ratios,
        includes_stops=metadata["kind"] == "measured",
    )
    row = {
        **metadata,
        "plot_file": f"{stem}.png",
        "frame_count": info["frame_count"],
        "detected_fraction": info["detected_count"] / info["frame_count"],
        "long_axis_omega_fraction": info["long_axis_omega_fraction"],
        "axis_ratio_p10": info["axis_ratio_p10"],
        "axis_step_abs_deg_p95": info["axis_step_abs_deg_p95"],
        "long_axis_peak_0.2s_hz": info["long_axis_fft"][1]["peak_frequency_hz"],
    }
    row["band_ratio_file"] = f"tables/{stem}_band_amplitude_ratio.csv"
    return row


def null_spectrum_summary(spectrum: pd.DataFrame, rate_hz: float) -> dict:
    """Compare random-only peaks over the full positive FFT band below Nyquist."""
    band = spectrum[
        np.isfinite(spectrum.frequency_hz)
        & np.isfinite(spectrum.amplitude)
        & (spectrum.frequency_hz > 0)
        & (spectrum.frequency_hz < rate_hz / 2)
    ]
    if band.empty:
        return {
            "nyquist_hz": rate_hz / 2,
            "peak_frequency_hz": np.nan,
            "peak_amplitude": np.nan,
            "peak_to_median_full_band": np.nan,
        }
    peak = band.loc[band.amplitude.idxmax()]
    median_amp = float(band.amplitude.median())
    return {
        "nyquist_hz": rate_hz / 2,
        "peak_frequency_hz": float(peak.frequency_hz),
        "peak_amplitude": float(peak.amplitude),
        "peak_to_median_full_band": float(peak.amplitude / median_amp) if median_amp > 0 else np.nan,
    }


def main(args: argparse.Namespace, files_by_day: dict, settings_by_day: dict, destination: Path) -> None:
    rows = []
    if args.mode in ("all", "measured"):
        duplicates = duplicate_source_map(files_by_day)
        for day in args.days:
            for sample_no, avi in enumerate(files_by_day[day], 1):
                duplicate_of = duplicates[(day, sample_no)]
                if duplicate_of:
                    rows.append(
                        {
                            "kind": "measured",
                            "condition": DATASETS[day],
                            "day": day,
                            "sample_no": sample_no,
                            "avi_filename": avi.name,
                            "duplicate_of": duplicate_of,
                            "analysis_status": "excluded_duplicate",
                        }
                    )
                    print(f"Excluding duplicate {day} No.{sample_no:02d} ({duplicate_of})", flush=True)
                    continue
                print(f"Analyzing {day} No.{sample_no:02d}", flush=True)
                source = read_avi(avi, settings_by_day[day])
                row = save_case(
                    source,
                    destination / "1_rotational_measurements" / day,
                    f"No.{sample_no:02d}",
                    f"{DATASETS[day]} | {day} | cell {sample_no:02d}",
                    {
                        "kind": "measured",
                        "condition": DATASETS[day],
                        "day": day,
                        "sample_no": sample_no,
                        "avi_filename": avi.name,
                        "duplicate_of": None,
                        "analysis_status": "analyzed",
                        "rate_hz": settings_by_day[day]["rate_hz"],
                        "duration_s": settings_by_day[day]["duration_s"],
                    },
                )
                row["plot_file"] = f"{day}/{row['plot_file']}"
                row["band_ratio_file"] = f"{day}/{row['band_ratio_file']}"
                rows.append(row)
    if args.mode in ("all", "synthetic"):
        for rate in args.synthetic_rates:
            for model in ("constant_angle_step_plus_noise", "random_angle_noise"):
                print(f"Simulating {model} at {rate:g} Hz", flush=True)
                source = synthetic_long_axis(model, rate, args.synthetic_duration, args.seed)
                name = f"{model}_{int(rate)}hz_seed{args.seed}"
                rows.append(
                    save_case(
                        source,
                        destination / "2_simulation",
                        f"simulation_{name}",
                        name,
                        {
                            "kind": "synthetic",
                            "model": model,
                            "rate_hz": rate,
                            "duration_s": args.synthetic_duration,
                            "seed": args.seed,
                            "base_hz": 8.0 if model == "constant_angle_step_plus_noise" else 0.0,
                            "angle_noise_sd_rad": 0.01,
                            "angle_model": (
                                "constant angular step plus independent Gaussian angle noise"
                                if model == "constant_angle_step_plus_noise"
                                else "independent Gaussian angle noise only"
                            ),
                        },
                    )
                )
        null_rows = []
        for rate in args.synthetic_rates:
            for seed in range(args.seed, args.seed + args.null_replicates):
                null_source = synthetic_long_axis("random_angle_noise", rate, args.synthetic_duration, seed)
                null_time = null_source.time_sec.to_numpy(float)
                _, null_omega = axis_angular_velocity(null_time, null_source.long_axis_deg.to_numpy(float))
                null_result = analyze(null_time, null_omega, windows=(0.2,))
                null_rows.append(
                    {
                        "rate_hz": rate,
                        "seed": seed,
                        **null_spectrum_summary(null_result.spectra, rate),
                    }
                )
        simulation_dir = destination / "2_simulation"
        simulation_dir.mkdir(parents=True, exist_ok=True)
        tables = simulation_dir / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(null_rows).to_csv(tables / "simulation_random_angle_noise_ensemble.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
        for rate, group in pd.DataFrame(null_rows).groupby("rate_hz"):
            axes[0].hist(group.peak_frequency_hz, bins=20, alpha=0.5, label=f"{rate:g} Hz")
            axes[1].hist(group.peak_to_median_full_band, bins=20, alpha=0.5, label=f"{rate:g} Hz")
        axes[0].set(
            xlabel="Peak frequency (Hz)", ylabel="Count", title="Random angle noise: strongest apparent FFT peak"
        )
        axes[0].set_xlim(0, max(args.synthetic_rates) / 2)
        axes[1].set(
            xlabel="Peak / median amplitude, full band", ylabel="Count", title="Random angle noise: peak contrast"
        )
        for ax in axes:
            ax.legend()
        style_axes(axes)
        fig.savefig(simulation_dir / "simulation_random_angle_noise_ensemble.png", dpi=180)
        plt.close(fig)
        for obsolete in (
            *simulation_dir.glob("simulation_periodic_plus_noise_*.png"),
            *tables.glob("simulation_periodic_plus_noise_*"),
            *simulation_dir.glob("simulation_steady_rotation_plus_noise_*.png"),
            *tables.glob("simulation_steady_rotation_plus_noise_*"),
            *simulation_dir.glob("simulation_random_only_*.png"),
            *tables.glob("simulation_random_only_*"),
        ):
            obsolete.unlink()
    destination.mkdir(parents=True, exist_ok=True)
    measured_rows = [r for r in rows if r["kind"] == "measured"]
    synthetic_rows = [r for r in rows if r["kind"] == "synthetic"]
    if args.mode in ("all", "measured"):
        measured_dir = destination / "1_rotational_measurements"
        measured_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(measured_rows).to_csv(measured_dir / "index.csv", index=False)
        by_cell = {(row["day"], row["sample_no"]): row for row in measured_rows if row.get("band_ratio_file")}
        if all(candidate in by_cell for candidate in POSTER_CANDIDATES):
            candidate_tables = []
            for day, sample_no in POSTER_CANDIDATES:
                table = pd.read_csv(measured_dir / by_cell[(day, sample_no)]["band_ratio_file"])
                table.insert(0, "sample_no", sample_no)
                table.insert(0, "day", day)
                candidate_tables.append(table)
            comparison = pd.concat(candidate_tables, ignore_index=True)
            comparison_tables = measured_dir / "tables"
            comparison_tables.mkdir(parents=True, exist_ok=True)
            comparison.to_csv(comparison_tables / "candidate_band_amplitude_ratio.csv", index=False)
            candidate_band_comparison(comparison, measured_dir / "candidate_band_amplitude_ratio.png")
    if args.mode in ("all", "synthetic"):
        simulation_dir = destination / "2_simulation"
        simulation_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(synthetic_rows).to_csv(simulation_dir / "index.csv", index=False)
    if measured_rows:
        selection = pd.DataFrame(
            [
                {
                    "day": r["day"],
                    "sample_no": r["sample_no"],
                    "condition": r["condition"],
                    "status": "exclude_duplicate" if r["duplicate_of"] else "needs_visual_review",
                    "reason": f"byte-identical to {r['duplicate_of']}" if r["duplicate_of"] else "",
                    "reviewer_note": "",
                }
                for r in measured_rows
            ]
        )
        selection.to_csv(measured_dir / "selection_template.csv", index=False)
    with (destination / "README.txt").open("w", encoding="utf-8") as stream:
        stream.write(
            "Private local poster analysis from binary AVI, not source TIFF. "
            "Measured plots are 1_rotational_measurements/<day>/No.XX.png; simulations are in 2_simulation/. "
            "Measured PNGs contain t-x, t-y, x-y; synthetic case PNGs omit coordinates. Both contain inferred relative phase wrapped to [-pi, pi), separate signed "
            "and absolute long-axis angular velocity, all six SD/Mean windows, standardized inputs, their FFTs, "
            "and side-by-side Low/High, Low-only and Low-High mean FFT amplitude metrics for 0<f<1 Hz and 80<=f<100 Hz. "
            "Measured ratios include stops; the ratio is unavailable when either band has no bins or the high-band mean is zero. "
            "A full measured run also writes the four candidate-cell comparison PNG and table under 1_rotational_measurements/. "
            "The FFT plot extends to half the frame rate and uses a logarithmic amplitude axis. "
            "Each case's numeric series, spectra and metadata are in its sibling tables/. "
            "Synthetic angles use a constant 8 Hz angle step plus independent Gaussian angle noise (SD 0.01 rad per frame), or independent Gaussian angle noise only without rotation. "
            "Both derive angular velocity from the generated 180-degree long-axis angle. Defaults are 200/250 fps, 6 s, seed 81. "
            "Each category has index.csv; "
            "1_rotational_measurements/selection_template.csv marks excluded duplicate AVIs. "
            "Long-axis angle has 180-degree ambiguity. Derived measured plots and series stay local.\n"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", nargs="+", choices=tuple(DATASETS), default=list(DATASETS))
    parser.add_argument("--mode", choices=("all", "measured", "synthetic"), default="all")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs" / "poster_fluctuation"))
    parser.add_argument("--synthetic-rates", nargs="+", type=float, default=[200.0, 250.0])
    parser.add_argument("--synthetic-duration", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=81)
    parser.add_argument("--null-replicates", type=int, default=40)
    args = parser.parse_args()
    if args.null_replicates < 1:
        parser.error("--null-replicates must be positive")
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    files_by_day, settings_by_day = {}, {}
    inputs = []
    if args.mode in ("all", "measured"):
        for day in args.days:
            files_by_day[day], settings_by_day[day] = dataset_files(data_root, day)
            inputs.extend([data_root / day / "config.ini", *files_by_day[day]])
    with analysis_run(parameters=vars(args), inputs=inputs, outputs=[output_root]) as run_root:
        main(args, files_by_day, settings_by_day, output_root)
    # Keep a private copy beside custom output roots when the source worktree is archived.
    archive = output_root / "_run_records" / run_root.parent.name / run_root.name
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run_root / "manifest.json", archive / "manifest.json")
    shutil.copy2(run_root / "run.log", archive / "run.log")
