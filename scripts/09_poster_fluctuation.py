"""Local poster figures for long-axis fluctuation and synthetic null controls."""

import argparse
import configparser
import json
import re
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
    fig, axes = plt.subplots(3, 2, figsize=(14, 12), constrained_layout=True)
    x = frame["x_um"].to_numpy(float)
    y = frame["y_um"].to_numpy(float)
    t = frame["time_sec"].to_numpy(float)
    axes[0, 0].plot(x, y, linewidth=0.7)
    axes[0, 0].set(xlabel="x (um)", ylabel="y (um, image down)", title="Centroid trajectory")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].invert_yaxis()
    axes[0, 1].plot(t, frame["long_axis_deg"], linewidth=0.7)
    axes[0, 1].set(xlabel="Time (s)", ylabel="Long-axis orientation (deg, modulo 180)", title="Cell long axis")
    if "true_angular_velocity_rad_s" in frame:
        axes[1, 0].plot(
            t, frame["true_angular_velocity_rad_s"], linewidth=0.7, color="black", alpha=0.6, label="synthetic truth"
        )
    for name, color in (("long_axis", "tab:blue"), ("orbit", "tab:orange")):
        axes[1, 0].plot(t, frame[f"{name}_omega_rad_s"], linewidth=0.6, color=color, label=name)
        key = f"{name}_sd_per_mean_0.2s"
        axes[1, 1].plot(t, frame[key], linewidth=0.8, color=color, label=name)
        zkey = f"{name}_sd_per_mean_0.2s_z"
        axes[2, 0].plot(t, frame[zkey], linewidth=0.8, color=color, label=name)
        spec = spectra[name]
        part = spec[spec.window_s == 0.2]
        axes[2, 1].plot(part.frequency_hz, part.amplitude, linewidth=0.8, color=color, label=name)
    axes[1, 0].set(xlabel="Time (s)", ylabel="Angular velocity (rad/s, CCW +)", title="Angular velocity")
    axes[1, 1].set(xlabel="Time (s)", ylabel="SD / Mean", title="Gaussian window, 0.2 s")
    axes[2, 0].set(xlabel="Time (s)", ylabel="z-score", title="Standardized SD / Mean, 0.2 s")
    axes[2, 1].set(xlabel="Frequency (Hz)", ylabel="FFT amplitude", title="FFT of standardized SD / Mean (0-20 Hz)")
    axes[2, 1].set_xlim(0, 20)
    for ax in axes.flat:
        ax.grid(alpha=0.2)
    for ax in (axes[1, 0], axes[1, 1], axes[2, 0], axes[2, 1]):
        ax.legend(fontsize=8)
    fig.suptitle(title)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)


def save_case(source: pd.DataFrame, folder: Path, title: str, metadata: dict) -> dict:
    frame, spectra, info = analyze_frame(source)
    folder.mkdir(parents=True, exist_ok=True)
    frame.to_csv(folder / "time_series.csv", index=False)
    for name, spectrum in spectra.items():
        spectrum.to_csv(folder / f"{name}_fft.csv", index=False)
    with (folder / "metadata.json").open("w", encoding="utf-8") as stream:
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
    figure(frame, spectra, title, folder / "overview.png")
    return {
        **metadata,
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


def condition_summary(folder: Path, day: str) -> None:
    """Put all cells on matching axes for local screening."""
    cells = sorted(folder.glob("No.*"))
    if not cells:
        return
    loaded = []
    for cell in cells:
        frame = pd.read_csv(cell / "time_series.csv")
        spectrum = pd.read_csv(cell / "long_axis_fft.csv")
        part = spectrum[spectrum.window_s == 0.2]
        loaded.append((cell, frame, part))
    all_z = np.concatenate([frame["long_axis_sd_per_mean_0.2s_z"].to_numpy(float) for _, frame, _ in loaded])
    all_amp = np.concatenate([part.amplitude.to_numpy(float) for _, _, part in loaded])
    z_limit = max(3.0, float(np.nanmax(np.abs(all_z))) * 1.05) if np.isfinite(all_z).any() else 3.0
    amp_limit = max(0.1, float(np.nanmax(all_amp)) * 1.05) if np.isfinite(all_amp).any() else 0.1
    fig, axes = plt.subplots(len(cells), 2, figsize=(14, 2.2 * len(cells)), squeeze=False, constrained_layout=True)
    for row, (cell, frame, part) in enumerate(loaded):
        axes[row, 0].plot(frame.time_sec, frame["long_axis_sd_per_mean_0.2s_z"], linewidth=0.7)
        axes[row, 0].set_xlim(0, 6)
        axes[row, 0].set_ylim(-z_limit, z_limit)
        axes[row, 0].set_ylabel(cell.name)
        axes[row, 1].plot(part.frequency_hz, part.amplitude, linewidth=0.7)
        axes[row, 1].set_xlim(0, 20)
        axes[row, 1].set_ylim(0, amp_limit)
        for ax in axes[row]:
            ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Time (s): z-scored SD/Mean, 0.2 s window")
    axes[-1, 1].set_xlabel("Frequency (Hz): FFT amplitude")
    fig.suptitle(f"{DATASETS[day]} ({day}): long-axis fluctuation screening")
    fig.savefig(folder / "condition_summary.png", dpi=150)
    plt.close(fig)


def main(args: argparse.Namespace, files_by_day: dict, settings_by_day: dict, destination: Path) -> None:
    rows = []
    if args.mode in ("all", "measured"):
        for day in args.days:
            for sample_no, avi in enumerate(files_by_day[day], 1):
                print(f"Analyzing {day} No.{sample_no:02d}", flush=True)
                source = read_avi(avi, settings_by_day[day])
                rows.append(
                    save_case(
                        source,
                        destination / "measured" / day / f"No.{sample_no:02d}",
                        f"{DATASETS[day]} | {day} | cell {sample_no:02d}",
                        {
                            "kind": "measured",
                            "condition": DATASETS[day],
                            "day": day,
                            "sample_no": sample_no,
                            "avi_filename": avi.name,
                            "rate_hz": settings_by_day[day]["rate_hz"],
                            "duration_s": settings_by_day[day]["duration_s"],
                        },
                    )
                )
        for day in args.days:
            condition_summary(destination / "measured" / day, day)
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
                        destination / "synthetic" / name,
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
    destination.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(destination / "index.csv", index=False)
    with (destination / "README.txt").open("w", encoding="utf-8") as stream:
        stream.write(
            "Private local poster analysis. Each cell/model folder contains overview.png, time_series.csv, long_axis_fft.csv, orbit_fft.csv, and metadata.json. index.csv is a screening inventory, not an adoption decision. Long-axis angle is unoriented (180-degree symmetry). All derived measured plots and time series stay local.\n"
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
    args = parser.parse_args()
    data_root = Path(args.data_root).resolve()
    output_root = Path(args.output_root).resolve()
    files_by_day, settings_by_day = {}, {}
    inputs = []
    if args.mode in ("all", "measured"):
        for day in args.days:
            files_by_day[day], settings_by_day[day] = dataset_files(data_root, day)
            inputs.extend([data_root / day / "config.ini", *files_by_day[day]])
    with analysis_run(parameters=vars(args), inputs=inputs, outputs=[output_root]):
        main(args, files_by_day, settings_by_day, output_root)
