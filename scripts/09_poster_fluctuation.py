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
    duplicate_source_map,
    major_axis_deg,
    orbit_angular_velocity,
    synthetic_trajectory,
)
from utils.run_context import analysis_run  # noqa: E402

DATASETS = {"20231005": "MotAB overexpression", "20230607": "Fixed rotational direction", "20231107": "Brake protein"}


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
    x = source["x_um"].to_numpy(float)
    y = source["y_um"].to_numpy(float)
    axis = source["long_axis_deg"].to_numpy(float)
    long_angle, long_omega = axis_angular_velocity(time, axis)
    orbit_angle, orbit_omega, center = orbit_angular_velocity(time, x, y)
    long_result = analyze(time, long_omega)
    orbit_result = analyze(time, orbit_omega)
    frame = source.copy()
    frame["long_axis_unwrapped_rad"] = long_angle
    frame["long_axis_omega_rad_s"] = long_omega
    frame["orbit_angle_rad"] = orbit_angle
    frame["orbit_omega_rad_s"] = orbit_omega
    for column in long_result.frame.columns:
        if column != "time_sec" and column != "angular_velocity_rad_s":
            frame[f"long_axis_{column}"] = long_result.frame[column]
            frame[f"orbit_{column}"] = orbit_result.frame[column]
    spectra = {"long_axis": long_result.spectra, "orbit": orbit_result.spectra}
    valid_axis = int(np.isfinite(long_omega).sum())
    valid_orbit = int(np.isfinite(orbit_omega).sum())
    ratios = source["axis_ratio"].to_numpy(float)
    finite_ratios = ratios[np.isfinite(ratios)]
    raw_delta = (np.diff(axis) + 90.0) % 180.0 - 90.0
    finite_delta = np.abs(raw_delta[np.isfinite(raw_delta)])
    truth = source["true_angular_velocity_rad_s"].to_numpy(float) if "true_angular_velocity_rad_s" in source else None

    def rmse(estimate: np.ndarray) -> float:
        if truth is None:
            return float("nan")
        mask = np.isfinite(truth) & np.isfinite(estimate)
        return float(np.sqrt(np.mean((estimate[mask] - truth[mask]) ** 2))) if mask.any() else float("nan")

    info = {
        "frame_count": len(frame),
        "detected_count": int(np.isfinite(x).sum()),
        "long_axis_omega_count": valid_axis,
        "orbit_omega_count": valid_orbit,
        "long_axis_omega_fraction": valid_axis / max(1, len(frame) - 1),
        "orbit_omega_fraction": valid_orbit / max(1, len(frame) - 1),
        "orbit_center_um": list(center),
        "axis_ratio_p10": float(np.percentile(finite_ratios, 10)) if len(finite_ratios) else np.nan,
        "axis_step_abs_deg_p95": float(np.percentile(finite_delta, 95)) if len(finite_delta) else np.nan,
        "long_axis_true_rmse_rad_s": rmse(long_omega),
        "orbit_true_rmse_rad_s": rmse(orbit_omega),
        "long_axis_fft": long_result.metadata,
        "orbit_fft": orbit_result.metadata,
        "angle_warning": "The unoriented long axis has a 180-degree ambiguity; angular changes of >=90 degrees per sample are not identifiable.",
    }
    return frame, spectra, info


def figure(frame: pd.DataFrame, spectra: dict[str, pd.DataFrame], title: str, target: Path) -> None:
    """One complete long-axis poster sheet per observed cell or synthetic case."""
    fig, axes = plt.subplots(4, 2, figsize=(15, 17), constrained_layout=True)
    t = frame["time_sec"].to_numpy(float)
    x = frame["x_um"].to_numpy(float)
    y = frame["y_um"].to_numpy(float)
    axes[0, 0].plot(t, x, linewidth=0.7)
    axes[0, 0].set(xlabel="Time (s)", ylabel="x (um)", title="Centroid x vs time")
    axes[0, 1].plot(t, y, linewidth=0.7)
    axes[0, 1].set(xlabel="Time (s)", ylabel="y (um, image down)", title="Centroid y vs time")
    axes[1, 0].plot(x, y, linewidth=0.7)
    axes[1, 0].set(xlabel="x (um)", ylabel="y (um, image down)", title="Centroid x-y trajectory")
    axes[1, 0].set_aspect("equal", adjustable="box")
    axes[1, 0].invert_yaxis()
    axes[1, 1].plot(t, frame["long_axis_unwrapped_rad"], linewidth=0.7)
    axes[1, 1].set(xlabel="Time (s)", ylabel="Relative angle (rad, CCW +)", title="Cell long-axis angle (unwrapped)")
    if "true_angular_velocity_rad_s" in frame:
        axes[2, 0].plot(
            t, frame["true_angular_velocity_rad_s"], color="black", linewidth=0.7, alpha=0.6, label="generated truth"
        )
    axes[2, 0].plot(t, frame["long_axis_omega_rad_s"], linewidth=0.7, label="long axis")
    axes[2, 0].set(xlabel="Time (s)", ylabel="Angular velocity (rad/s, CCW +)", title="Long-axis angular velocity")
    if "true_angular_velocity_rad_s" in frame:
        axes[2, 0].legend(fontsize=8)
    for width in WINDOWS_S:
        label = f"{width:g} s"
        prefix = f"long_axis_sd_per_mean_{width:g}s"
        axes[2, 1].plot(t, frame[prefix], linewidth=0.8, label=label)
        axes[3, 0].plot(t, frame[f"{prefix}_z"], linewidth=0.8, label=label)
        spectrum = spectra["long_axis"]
        part = spectrum[spectrum.window_s == width]
        axes[3, 1].plot(part.frequency_hz, part.amplitude, linewidth=0.8, label=label)
    axes[2, 1].set(xlabel="Time (s)", ylabel="SD / Mean", title="Gaussian-window SD / Mean: all widths")
    axes[3, 0].set(xlabel="Time (s)", ylabel="z-score", title="Standardized SD / Mean: FFT input")
    axes[3, 1].set(xlabel="Frequency (Hz)", ylabel="FFT amplitude", title="FFT for each standardized window series")
    axes[3, 1].set_xlim(0, 20)
    for ax in (axes[2, 1], axes[3, 0], axes[3, 1]):
        ax.legend(fontsize=8, ncol=2)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    fig.suptitle(title)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def save_case(source: pd.DataFrame, destination: Path, stem: str, title: str, metadata: dict) -> dict:
    frame, spectra, info = analyze_frame(source)
    destination.mkdir(parents=True, exist_ok=True)
    tables = destination / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables / f"{stem}_time_series.csv", index=False)
    for name, spectrum in spectra.items():
        spectrum.to_csv(tables / f"{stem}_{name}_fft.csv", index=False)
    with (tables / f"{stem}_metadata.json").open("w", encoding="utf-8") as stream:
        json.dump(
            {
                **metadata,
                **info,
                "window_widths_s": WINDOWS_S,
                "fft_input": "z-scored Gaussian-window SD(abs(angular velocity))/Mean(abs(angular velocity))",
            },
            stream,
            indent=2,
            allow_nan=True,
        )
    figure(frame, spectra, title, destination / f"{stem}.png")
    return {
        **metadata,
        "plot_file": f"{stem}.png",
        "frame_count": info["frame_count"],
        "detected_fraction": info["detected_count"] / info["frame_count"],
        "long_axis_omega_fraction": info["long_axis_omega_fraction"],
        "orbit_omega_fraction": info["orbit_omega_fraction"],
        "axis_ratio_p10": info["axis_ratio_p10"],
        "axis_step_abs_deg_p95": info["axis_step_abs_deg_p95"],
        "long_axis_true_rmse_rad_s": info["long_axis_true_rmse_rad_s"],
        "orbit_true_rmse_rad_s": info["orbit_true_rmse_rad_s"],
        "long_axis_peak_0.2s_hz": info["long_axis_fft"][1]["peak_frequency_hz"],
        "orbit_peak_0.2s_hz": info["orbit_fft"][1]["peak_frequency_hz"],
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
                rows.append(
                    save_case(
                        source,
                        destination,
                        f"{day}_No.{sample_no:02d}",
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
                )
    if args.mode in ("all", "synthetic"):
        for rate in args.synthetic_rates:
            for model in ("periodic_plus_noise", "random_only"):
                print(f"Simulating {model} at {rate:g} Hz", flush=True)
                source = synthetic_trajectory(
                    model,
                    rate,
                    args.synthetic_duration,
                    args.seed,
                    base_hz=8.0 if model == "periodic_plus_noise" else 0.0,
                )
                source["source_frame_1based"] = np.arange(1, len(source) + 1)
                source["long_axis_deg"] = (-np.rad2deg(source["true_angle_rad"])) % 180.0
                source["axis_ratio"] = np.nan
                source = source[
                    [
                        "source_frame_1based",
                        "time_sec",
                        "x_um",
                        "y_um",
                        "long_axis_deg",
                        "axis_ratio",
                        "true_angle_rad",
                        "true_angular_velocity_rad_s",
                        "true_x_um",
                        "true_y_um",
                    ]
                ]
                name = f"{model}_{int(rate)}hz_seed{args.seed}"
                rows.append(
                    save_case(
                        source,
                        destination,
                        f"simulation_{name}",
                        name,
                        {
                            "kind": "synthetic",
                            "model": model,
                            "rate_hz": rate,
                            "duration_s": args.synthetic_duration,
                            "seed": args.seed,
                            "base_hz": 8.0 if model == "periodic_plus_noise" else 0.0,
                            "periodic_modulation_hz": 2.0 if model == "periodic_plus_noise" else None,
                            "periodic_modulation_rad_s": 8.0 if model == "periodic_plus_noise" else 0.0,
                            "white_noise_sd_rad_s": 3.0,
                            "radius_um": 0.5,
                            "coordinate_noise_sd_um": 0.01,
                        },
                    )
                )
        null_rows = []
        for rate in args.synthetic_rates:
            for seed in range(args.seed, args.seed + args.null_replicates):
                null_source = synthetic_trajectory("random_only", rate, args.synthetic_duration, seed, base_hz=0.0)
                null_time = null_source.time_sec.to_numpy(float)
                null_axis = (-np.rad2deg(null_source.true_angle_rad.to_numpy(float))) % 180.0
                _, null_omega = axis_angular_velocity(null_time, null_axis)
                null_result = analyze(null_time, null_omega, windows=(0.2,))
                spectrum = null_result.spectra
                low = spectrum[spectrum.frequency_hz <= 20]
                amplitudes = low.amplitude.to_numpy(float)
                median_amp = float(np.median(amplitudes)) if len(amplitudes) else np.nan
                null_rows.append(
                    {
                        "rate_hz": rate,
                        "seed": seed,
                        "peak_frequency_hz": null_result.metadata[0]["peak_frequency_hz"],
                        "peak_amplitude": float(np.max(amplitudes)) if len(amplitudes) else np.nan,
                        "peak_to_median_0_20hz": (
                            float(np.max(amplitudes) / median_amp)
                            if np.isfinite(median_amp) and median_amp > 0
                            else np.nan
                        ),
                    }
                )
        destination.mkdir(parents=True, exist_ok=True)
        tables = destination / "tables"
        tables.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(null_rows).to_csv(tables / "simulation_random_only_ensemble.csv", index=False)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
        for rate, group in pd.DataFrame(null_rows).groupby("rate_hz"):
            axes[0].hist(group.peak_frequency_hz, bins=20, alpha=0.5, label=f"{rate:g} Hz")
            axes[1].hist(group.peak_to_median_0_20hz, bins=20, alpha=0.5, label=f"{rate:g} Hz")
        axes[0].set(xlabel="Peak frequency (Hz)", ylabel="Count", title="Random-only: strongest apparent FFT peak")
        axes[1].set(xlabel="Peak / median amplitude, 0-20 Hz", ylabel="Count", title="Random-only peak contrast")
        for ax in axes:
            ax.legend()
            ax.grid(alpha=0.2)
        fig.savefig(destination / "simulation_random_only_ensemble.png", dpi=150)
        plt.close(fig)
    destination.mkdir(parents=True, exist_ok=True)
    index_name = "index.csv" if args.mode == "all" else f"{args.mode}_index.csv"
    pd.DataFrame(rows).to_csv(destination / index_name, index=False)
    measured_rows = [r for r in rows if r["kind"] == "measured"]
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
        selection.to_csv(destination / "selection_template.csv", index=False)
    with (destination / "README.txt").open("w", encoding="utf-8") as stream:
        stream.write(
            "Private local poster analysis. One flat PNG per measured cell (<day>_No.XX.png) or synthetic case (simulation_*.png). Each PNG contains t-x, t-y, x-y, long-axis angle/velocity, all six SD/Mean windows, standardized inputs and their FFTs. Numeric series, spectra and metadata are in tables/ with matching stems. index.csv is the combined inventory; selection_template.csv marks excluded duplicate AVIs. Long-axis angle has 180-degree ambiguity. Derived measured plots and series stay local.\n"
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
