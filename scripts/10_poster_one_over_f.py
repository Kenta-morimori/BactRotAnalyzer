"""Local six-case FFT-amplitude envelope fits and random-phase inverse FFT figures."""

import argparse
import json
import shutil
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.functions.frequency_analysis import fft  # noqa: E402
from utils.functions.poster_one_over_f import (  # noqa: E402
    amplitude_envelope,
    fit_amplitude,
    log_frequency_bins,
    random_phase_inverse_fft,
)
from utils.run_context import analysis_run  # noqa: E402

WINDOW_S = 0.5
SPECTRUM_SUFFIX = "long_axis_fft.csv"
SERIES_SUFFIX = "time_series.csv"
METADATA_SUFFIX = "metadata.json"


@dataclass(frozen=True)
class CaseSpec:
    key: str
    label: str
    table_folder: str
    stem: str
    kind: str
    expected_rate_hz: float | None = None


CASES = (
    CaseSpec("20230607_No06", "20230607 No.06", "1_rotational_measurements/20230607/tables", "No.06", "measured"),
    CaseSpec("20230607_No08", "20230607 No.08", "1_rotational_measurements/20230607/tables", "No.08", "measured"),
    CaseSpec(
        "20231005_No01",
        "20231005 No.01\n(stops included)",
        "1_rotational_measurements/20231005/tables",
        "No.01",
        "measured",
    ),
    CaseSpec("20231107_No06", "20231107 No.06", "1_rotational_measurements/20231107/tables", "No.06", "measured"),
    CaseSpec(
        "constant_step",
        "8 Hz step + angle noise\n200 fps",
        "2_simulation/tables",
        "simulation_constant_angle_step_plus_noise_200hz_seed81",
        "synthetic",
        200.0,
    ),
    CaseSpec(
        "random_angle",
        "Random angle noise\n200 fps",
        "2_simulation/tables",
        "simulation_random_angle_noise_200hz_seed81",
        "synthetic",
        200.0,
    ),
)


def input_paths(source_root: Path, cases: tuple[CaseSpec, ...] = CASES) -> list[Path]:
    return [
        source_root / case.table_folder / f"{case.stem}_{suffix}"
        for case in cases
        for suffix in (SERIES_SUFFIX, SPECTRUM_SUFFIX, METADATA_SUFFIX)
    ]


def load_case(source_root: Path, case: CaseSpec) -> dict:
    folder = source_root / case.table_folder
    series = pd.read_csv(folder / f"{case.stem}_{SERIES_SUFFIX}")
    spectrum = pd.read_csv(folder / f"{case.stem}_{SPECTRUM_SUFFIX}")
    metadata = json.loads((folder / f"{case.stem}_{METADATA_SUFFIX}").read_text(encoding="utf-8"))
    if metadata["kind"] != case.kind:
        raise ValueError(f"{case.key}: source kind disagrees with the case specification")
    rate_hz = float(metadata["rate_hz"])
    if not np.isfinite(rate_hz) or rate_hz <= 0:
        raise ValueError(f"{case.key}: invalid frame rate")
    if case.expected_rate_hz is not None and not np.isclose(rate_hz, case.expected_rate_hz):
        raise ValueError(f"{case.key}: frame rate disagrees with the case specification")
    rows = [item for item in metadata["long_axis_fft"] if np.isclose(item["window_s"], WINDOW_S)]
    if len(rows) != 1:
        raise ValueError(f"{case.key}: 0.5 s FFT segment metadata is unavailable")
    segment = rows[0]
    start, end, count = (int(segment[name]) for name in ("start_index", "end_index", "sample_count"))
    if start < 0 or end >= len(series) or end - start + 1 != count:
        raise ValueError(f"{case.key}: FFT segment does not match the saved series")
    time = series["time_sec"].to_numpy(float)
    frames = series["source_frame_1based"].to_numpy(float)
    if not np.array_equal(frames, np.arange(1, len(series) + 1)):
        raise ValueError(f"{case.key}: source-frame mapping is not the expected complete series")
    raw = series["long_axis_sd_per_mean_0.5s"].to_numpy(float)
    z = series["long_axis_sd_per_mean_0.5s_z"].to_numpy(float)[start : end + 1]
    segment_time = time[start : end + 1]
    dt = np.diff(segment_time)
    if (
        not np.isfinite(z).all()
        or not np.isfinite(segment_time).all()
        or not np.allclose(dt, 1 / rate_hz, rtol=1e-6, atol=1e-10)
    ):
        raise ValueError(f"{case.key}: FFT source has missing or irregular samples")
    saved = spectrum[np.isclose(spectrum["window_s"], WINDOW_S)].sort_values("frequency_hz")
    calculated_frequency, calculated_amplitude = fft(z, float(np.median(dt)))
    frequency_hz = saved["frequency_hz"].to_numpy(float)
    amplitude = saved["amplitude"].to_numpy(float)
    if not (
        frequency_hz.shape == calculated_frequency.shape
        and np.allclose(frequency_hz, calculated_frequency, rtol=1e-8, atol=1e-10)
        and np.allclose(amplitude, calculated_amplitude, rtol=1e-8, atol=1e-10)
    ):
        raise ValueError(f"{case.key}: saved FFT does not match the 0.5 s z-score segment")
    return {
        "case": case,
        "time_sec": time,
        "sd_per_mean": raw,
        "segment_time_sec": segment_time,
        "frequency_hz": frequency_hz,
        "amplitude": amplitude,
        "rate_hz": rate_hz,
        "sample_count": count,
        "segment_start_frame_1based": int(frames[start]),
        "segment_end_frame_1based": int(frames[end]),
    }


def draw_comparison(cases: list[dict], target: Path) -> None:
    fig, axes = plt.subplots(3, len(cases), figsize=(34, 15), constrained_layout=True)
    for column, item in enumerate(cases):
        first, second, third = axes[:, column]
        first.plot(item["time_sec"], item["sd_per_mean"], linewidth=0.8)
        first.set(title=item["case"].label, xlabel="Time (s)", ylabel="SD / Mean")
        positive = np.isfinite(item["amplitude"]) & (item["amplitude"] > 0)
        second.plot(item["frequency_hz"][positive], item["amplitude"][positive], linewidth=0.8, label="0.5 s FFT")
        second.plot(
            item["frequency_hz"],
            item["fitted_amplitude"],
            color="red",
            linewidth=1.5,
            label="Fit: A[(f + Δf)/1 Hz]^(-α)",
        )
        second.scatter(
            item["fit_bin_frequency_hz"],
            item["fit_bin_amplitude"],
            s=24,
            color="darkorange",
            zorder=3,
            label="Log-band medians used for fit",
        )
        second.set(xlabel="Frequency (Hz)", ylabel="FFT amplitude (log)", xlim=(0, item["rate_hz"] / 2))
        second.set_yscale("log")
        second.legend(fontsize=9)
        second.text(
            0.98,
            0.05,
            f"A = {item['fit'].scale:.3g}, α = {item['fit'].exponent:.2f}\n"
            f"Δf = {item['fit'].offset_hz:.3g} Hz, log R² = {item['fit'].log_bin_r_squared:.2f}",
            transform=second.transAxes,
            ha="right",
            fontsize=10,
        )
        third.plot(item["segment_time_sec"], item["surrogate"], linewidth=0.8)
        third.axhline(0, color="gray", linewidth=0.5)
        third.set(xlabel="Time (s)", ylabel="Inverse FFT (z units)")
        for ax in (first, second, third):
            ax.tick_params(labelsize=10)
            ax.xaxis.label.set_fontsize(11)
            ax.yaxis.label.set_fontsize(11)
            ax.title.set_fontsize(13)
            ax.grid(alpha=0.2)
    fig.suptitle("0.5 s Gaussian window: SD/Mean, FFT-amplitude fit, random-phase inverse FFT", fontsize=19)
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=170)
    plt.close(fig)


def generate(source_root: Path, output_root: Path, phase_seed: int, cases: tuple[CaseSpec, ...] = CASES) -> None:
    loaded = []
    for index, case in enumerate(cases):
        item = load_case(source_root, case)
        fit = fit_amplitude(item["frequency_hz"], item["amplitude"])
        bin_frequency, bin_amplitude, bin_count = log_frequency_bins(item["frequency_hz"], item["amplitude"])
        seed = phase_seed + index
        frequency, target_amplitude, phase, surrogate = random_phase_inverse_fft(
            item["sample_count"], 1 / item["rate_hz"], fit.scale, fit.exponent, fit.offset_hz, seed
        )
        if not np.allclose(frequency, item["frequency_hz"], rtol=1e-8, atol=1e-10):
            raise ValueError(f"{case.key}: inverse FFT frequency grid differs from the source FFT")
        recovered_amplitude = 2 * np.abs(np.fft.rfft(surrogate)[1 : item["sample_count"] // 2]) / item["sample_count"]
        if not np.allclose(recovered_amplitude, target_amplitude, rtol=1e-10, atol=1e-12):
            raise ValueError(f"{case.key}: inverse FFT did not reproduce the fitted amplitude")
        item.update(
            fit=fit,
            phase_seed=seed,
            fitted_amplitude=amplitude_envelope(
                item["frequency_hz"], fit.scale, exponent=fit.exponent, offset_hz=fit.offset_hz
            ),
            fit_bin_frequency_hz=bin_frequency,
            fit_bin_amplitude=bin_amplitude,
            fit_bin_count=bin_count,
            phase_rad=phase,
            surrogate=surrogate,
        )
        loaded.append(item)
    tables = output_root / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    summary = []
    for item in loaded:
        case = item["case"]
        pd.DataFrame(
            {
                "frequency_hz": item["frequency_hz"],
                "observed_fft_amplitude": item["amplitude"],
                "fitted_fft_amplitude": item["fitted_amplitude"],
                "inverse_fft_phase_rad": item["phase_rad"],
            }
        ).to_csv(tables / f"{case.key}_fit.csv", index=False)
        pd.DataFrame(
            {
                "frequency_hz": item["fit_bin_frequency_hz"],
                "median_fft_amplitude": item["fit_bin_amplitude"],
                "fft_point_count": item["fit_bin_count"],
                "fitted_fft_amplitude": amplitude_envelope(
                    item["fit_bin_frequency_hz"],
                    item["fit"].scale,
                    exponent=item["fit"].exponent,
                    offset_hz=item["fit"].offset_hz,
                ),
            }
        ).to_csv(tables / f"{case.key}_fit_bins.csv", index=False)
        pd.DataFrame({"time_sec": item["segment_time_sec"], "inverse_fft_z_units": item["surrogate"]}).to_csv(
            tables / f"{case.key}_inverse_fft.csv", index=False
        )
        summary.append(
            {
                "case": case.key,
                "label": case.label.replace("\n", " "),
                "kind": case.kind,
                "rate_hz": item["rate_hz"],
                "sample_count": item["sample_count"],
                "segment_start_frame_1based": item["segment_start_frame_1based"],
                "segment_end_frame_1based": item["segment_end_frame_1based"],
                "phase_seed": item["phase_seed"],
                **asdict(item["fit"]),
                "fit_file": f"tables/{case.key}_fit.csv",
                "fit_bins_file": f"tables/{case.key}_fit_bins.csv",
                "inverse_fft_file": f"tables/{case.key}_inverse_fft.csv",
            }
        )
    pd.DataFrame(summary).to_csv(output_root / "index.csv", index=False)
    draw_comparison(loaded, output_root / "six_case_one_over_f_0.5s.png")
    (output_root / "README.txt").write_text(
        "Private local comparison for four measured cells and two synthetic angle controls. "
        "Row 1 is the raw 0.5 s Gaussian-window SD/Mean on original observation times. "
        "Row 2 fits the z-scored SD/Mean FFT amplitude with A*((f+delta_f)/1 Hz)^(-alpha), "
        "where delta_f is the first positive FFT frequency. Nine equal log-frequency bands summarize the "
        "positive amplitudes below Nyquist by their medians; log-amplitude residuals of those medians "
        "determine A and alpha. The raw spectrum and fit are displayed on a log-amplitude axis. "
        "This fitted colored-noise spectrum is not a claim of 1/f power spectral density. "
        "Row 3 is a new zero-mean real-valued series in z-score units from independent uniform random phases. "
        "DC and the final positive FFT bin are zero; the time length equals the original FFT segment, not the full 6 s. "
        "Seeds and fit diagnostics are in index.csv. Measured results must remain local.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", default=str(ROOT / "outputs" / "poster_fluctuation"))
    parser.add_argument("--output-root")
    parser.add_argument("--phase-seed", type=int, default=84)
    args = parser.parse_args()
    source_root = Path(args.source_root).resolve()
    output_root = Path(args.output_root).resolve() if args.output_root else source_root / "2_simulation" / "one_over_f"
    with analysis_run(parameters=vars(args), inputs=input_paths(source_root), outputs=[output_root]) as run_root:
        generate(source_root, output_root, args.phase_seed)
    archive = output_root / "_run_records" / run_root.parent.name / run_root.name
    archive.mkdir(parents=True, exist_ok=False)
    shutil.copy2(run_root / "manifest.json", archive / "manifest.json")
    shutil.copy2(run_root / "run.log", archive / "run.log")
