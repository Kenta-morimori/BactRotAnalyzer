"""Descriptive rotation-only fluctuation comparison for poster measurements."""

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

from utils.functions.poster_rotation_only import (  # noqa: E402
    HIGH_BANDS_HZ,
    WINDOWS_S,
    detect_stops,
    evaluate_interval,
    longest_rotation_interval,
    summarize_groups,
)
from utils.run_context import analysis_run  # noqa: E402

CONDITIONS = {
    "20230607": "YSC2123/MotAB",
    "20231005": "SJW46/MotAB",
    "20231107": "YcgR-tet",
}
COLORS = {"20230607": "#2b6cb0", "20231005": "#dd6b20", "20231107": "#b83280"}


def _style(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=21)
    ax.set_xlabel("Window width [s]", fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xticks(WINDOWS_S)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.2)


def group_figures(metrics: pd.DataFrame, summary: pd.DataFrame, output: Path) -> None:
    accepted = metrics[metrics.selection_reason.eq("included")].copy()
    if accepted.empty:
        return
    accepted["normalized_ratio"] = accepted.low_to_high_amplitude_ratio / accepted.groupby(
        ["day", "sample_no"]
    ).low_to_high_amplitude_ratio.transform("max")
    plt.rcParams["font.family"] = "Arial"
    definitions = (
        ("low_to_high_amplitude_ratio", "Low / High (80–100 Hz)", "Amp ratio", "raw_ratio.png", True),
        ("normalized_ratio", "Each cell maximum = 1", "Normalized amp ratio", "normalized_ratio.png", False),
        ("low_mean_amplitude", "Low band (0–1 Hz)", "Mean FFT amp", "low_amplitude.png", False),
        ("high_mean_amplitude", "High band (80–100 Hz)", "Mean FFT amp", "high_amplitude.png", False),
        (
            "low_to_high_amplitude_ratio_60_80hz",
            "Low / High (60–80 Hz)",
            "Amp ratio",
            "high_band_sensitivity.png",
            True,
        ),
    )
    for column, title, ylabel, filename, log in definitions:
        fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
        for day, label in CONDITIONS.items():
            part = accepted[accepted.day.eq(day)]
            if part.empty:
                continue
            for cell in part.sample_no.unique():
                item = part[part.sample_no.eq(cell)].sort_values("window_s")
                ax.plot(item.window_s, item[column], color=COLORS[day], alpha=0.26, linewidth=1.2)
                ax.scatter(item.window_s, item[column], color=COLORS[day], alpha=0.5, s=32)
            center = part.groupby("window_s")[column].apply(
                lambda values: float(np.exp(np.mean(np.log(values)))) if log else float(values.mean())
            )
            spread = part.groupby("window_s")[column].apply(
                lambda values: (
                    float(np.exp(np.std(np.log(values), ddof=1)))
                    if log and len(values) > 1
                    else (float(values.std(ddof=1)) if len(values) > 1 else (1.0 if log else 0.0))
                )
            )
            if log:
                yerr = np.vstack(
                    [center.to_numpy() * (1 - 1 / spread.to_numpy()), center.to_numpy() * (spread.to_numpy() - 1)]
                )
            else:
                yerr = spread.to_numpy()
            ax.errorbar(
                center.index,
                center.to_numpy(),
                yerr=yerr,
                fmt="-o",
                color=COLORS[day],
                linewidth=3.1,
                markersize=9,
                capsize=5,
                label=f"{label} (n={part.sample_no.nunique()})",
            )
        _style(ax, title, ylabel)
        if log:
            ax.set_yscale("log")
        fig.legend(loc="outside lower center", ncol=1, fontsize=17)
        fig.savefig(output / filename, dpi=200)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 7), layout="constrained")
    for day, label in CONDITIONS.items():
        part = accepted[accepted.day.eq(day)]
        if part.empty:
            continue
        ax.scatter(
            part.low_to_high_amplitude_ratio,
            part.low_to_high_amplitude_ratio_60_80hz,
            label=label,
            color=COLORS[day],
            s=75,
            alpha=0.75,
        )
    values = accepted[["low_to_high_amplitude_ratio", "low_to_high_amplitude_ratio_60_80hz"]].to_numpy(float)
    lower, upper = float(np.nanmin(values)), float(np.nanmax(values))
    ax.plot([lower, upper], [lower, upper], "k--", alpha=0.5)
    ax.set(xscale="log", yscale="log")
    ax.set_xlabel("Low / High 80–100 Hz", fontsize=18)
    ax.set_ylabel("Low / High 60–80 Hz", fontsize=18)
    ax.set_title("High-band sensitivity: all cells and windows", fontsize=21)
    ax.tick_params(labelsize=16)
    ax.grid(alpha=0.2)
    fig.legend(loc="outside lower center", ncol=1, fontsize=17)
    fig.savefig(output / "high_band_sensitivity_scatter.png", dpi=200)
    plt.close(fig)


def run(measurements: Path, output: Path) -> None:
    index = pd.read_csv(measurements / "index.csv")
    selection_rows: list[dict] = []
    metric_rows: list[dict] = []
    stop_rows: list[dict] = []
    spectra_rows: list[pd.DataFrame] = []
    for row in index.itertuples():
        if row.analysis_status != "analyzed" or str(row.day) not in CONDITIONS:
            continue
        day, number = str(row.day), int(row.sample_no)
        frame = pd.read_csv(measurements / day / "tables" / f"No.{number:02d}_time_series.csv")
        v2, stops = detect_stops(frame)
        selected = longest_rotation_interval(frame, stops)
        if day == "20230607" and number == 8:
            reason = "user_excluded_no08"
            measured, spectra, bands = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
        else:
            measured, spectra, bands, reason = evaluate_interval(selected)
        start_frame = int(selected.source_frame_1based.iloc[0]) if len(selected) else np.nan
        end_frame = int(selected.source_frame_1based.iloc[-1]) if len(selected) else np.nan
        start_s = float(selected.time_sec.iloc[0]) if len(selected) else np.nan
        end_s = float(selected.time_sec.iloc[-1]) if len(selected) else np.nan
        duration_s = end_s - start_s + 1 / float(row.rate_hz) if len(selected) else 0.0
        selection_rows.append(
            {
                "day": day,
                "condition": CONDITIONS[day],
                "sample_no": number,
                "selection_reason": reason,
                "start_frame_1based": start_frame,
                "end_frame_1based": end_frame,
                "start_s": start_s,
                "end_s": end_s,
                "duration_s": duration_s,
                "frame_count": len(selected),
                "frame_rate_hz": row.rate_hz,
                "v2_rows": len(v2),
                "stop_count": len(stops),
                "angle_definition": "unoriented_long_axis_180deg",
                "visual_qc": "pending_source_avi_review",
            }
        )
        for stop in stops.itertuples():
            stop_rows.append(
                {"day": day, "sample_no": number, "start_s": stop.start_s, "end_s": stop.end_s, "method": stop.method}
            )
        if bands.empty:
            continue
        for band in bands.to_dict("records"):
            metric_rows.append(
                {
                    "day": day,
                    "condition": CONDITIONS[day],
                    "sample_no": number,
                    "selection_reason": reason,
                    "frame_rate_hz": row.rate_hz,
                    "selected_duration_s": duration_s,
                    **band,
                }
            )
        spectra = spectra.copy()
        spectra.insert(0, "sample_no", number)
        spectra.insert(0, "day", day)
        spectra_rows.append(spectra)
        # Keep derived measured series in private output only, with source identity.
        if not measured.empty:
            measured.insert(0, "source_frame_1based", selected.source_frame_1based.to_numpy(int))
            measured.to_csv(output / "tables" / f"{day}_No.{number:02d}_rotation_only_series.csv", index=False)
    selections = pd.DataFrame(selection_rows)
    metrics = pd.DataFrame(metric_rows)
    summary = summarize_groups(metrics)
    selections.to_csv(output / "selection.csv", index=False)
    metrics.to_csv(output / "cell_band_metrics.csv", index=False)
    summary.to_csv(output / "group_summary.csv", index=False)
    pd.DataFrame(stop_rows, columns=["day", "sample_no", "start_s", "end_s", "method"]).to_csv(
        output / "stop_intervals.csv", index=False
    )
    if spectra_rows:
        pd.concat(spectra_rows, ignore_index=True).to_csv(output / "tables" / "rotation_only_fft.csv", index=False)
    group_figures(metrics, summary, output)
    (output / "method.json").write_text(
        json.dumps(
            {
                "windows_s": WINDOWS_S,
                "low_band_hz": [0, 1],
                "high_bands_hz": HIGH_BANDS_HZ,
                "minimum_low_bins_per_window": 3,
                "stop_margin_s": 0.1,
                "interval": "longest observed continuous rotation run",
                "group_unit": "cell; one experimental day per condition; descriptive only",
                "ratio_center": "geometric mean of cell ratios",
                "normalization": "each cell maximum over 3 windows",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--measurements-root", type=Path, default=ROOT / "outputs/poster_fluctuation/1_rotational_measurements"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT / "outputs/poster_fluctuation/1_rotational_measurements/rotation_only_stats",
    )
    args = parser.parse_args()
    index = pd.read_csv(args.measurements_root / "index.csv")
    inputs = [args.measurements_root / "index.csv"]
    for row in index.itertuples():
        if row.analysis_status == "analyzed" and str(row.day) in CONDITIONS:
            inputs.append(
                args.measurements_root / str(row.day) / "tables" / f"No.{int(row.sample_no):02d}_time_series.csv"
            )
    with analysis_run(
        parameters={"measurements_root": str(args.measurements_root), "output_root": str(args.output_root)},
        inputs=inputs,
        outputs=[args.output_root],
    ) as run_record:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "tables").mkdir(parents=True, exist_ok=True)
        run(args.measurements_root, args.output_root)
    records = args.output_root / "_run_records"
    records.mkdir(exist_ok=True)
    shutil.copy2(run_record / "manifest.json", records / f"{run_record.name}.json")


if __name__ == "__main__":
    main()
