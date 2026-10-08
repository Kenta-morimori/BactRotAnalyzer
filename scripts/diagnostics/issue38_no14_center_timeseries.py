"""Plot the No.14 centroid and current moving-window center on one time axis."""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from utils import param  # noqa: E402
from utils.functions import (  # noqa: E402
    frequency_analysis,
    get_centroid_coordinate,
    raw_centroid,
    read_csv,
    repellent_response,
)

DAY = "repellent-response/23"
SAMPLE_NO = 14
START_FRAME, END_FRAME = 2049, 16384
RISE_TIME_SEC = 40.0
FIGURE_DIR = Path(__file__).resolve().parents[2] / "docs/analysis/issue-38"


def window_width(time: np.ndarray, x: np.ndarray, y: np.ndarray) -> float:
    """Match the current pre-rise FFT rule for a thirty-rotation window."""
    before = time < RISE_TIME_SEC
    t = time[before]
    frame_rate = len(t) / (t[-1] - t[0])
    peaks = []
    for values in (x[before], y[before]):
        frequency, amplitude = frequency_analysis.fft(values, 1.0 / frame_rate)
        mask = frequency > 5.0
        if mask.any() and np.isfinite(amplitude[mask]).any():
            peaks.append(float(frequency[mask][np.nanargmax(amplitude[mask])]))
    if not peaks:
        raise ValueError("The pre-rise centroid does not provide a valid FFT peak above 5 Hz")
    return float(param.n_rotations / max(peaks))


def uncorrected_centers(time: np.ndarray, x: np.ndarray, y: np.ndarray, width: float) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce the current forward fits before outlier correction or tail fill."""
    n = len(time)
    dt = (time[-1] - time[0]) / n
    start = float(time[0])
    raw_x, raw_y = [], []
    while True:
        selected = (time >= start) & (time < start + width)
        if np.count_nonzero(selected) < param.min_ref_centroid_num:
            cx, cy = np.nan, np.nan
        else:
            cx, cy, _, _, _ = get_centroid_coordinate.calculate_ellipse_properties(
                x[selected].reshape(-1, 1), y[selected].reshape(-1, 1)
            )
        raw_x.append(cx)
        raw_y.append(cy)
        start += dt
        if start + width >= time[-1]:
            break
    missing = n - len(raw_x)
    if missing < 0:
        raise ValueError("Uncorrected center count exceeds source frames")
    return np.asarray(raw_x + [np.nan] * missing), np.asarray(raw_y + [np.nan] * missing)


def plot_timeseries(data: pd.DataFrame, width: float, path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(22, 7), sharex=True)
    time = data.time_sec.to_numpy(float)
    cutoff = time[-1] - width
    colors = {"centroid": "#455468", "center": "#d95f02", "raw": "#0072b2"}
    for row, axis_name in enumerate("xy"):
        axis = axes[row]
        values = np.r_[data[f"{axis_name}_um"].to_numpy(float),
                       data[f"center_{axis_name}_um"].to_numpy(float)]
        lo, hi = np.nanmin(values), np.nanmax(values)
        padding = 0.045 * (hi - lo)
        lo, hi = lo - padding, hi + padding
        axis.plot(time, data[f"{axis_name}_um"], color=colors["centroid"], alpha=0.38,
                  lw=0.45, rasterized=True, label="centroid")
        raw = data[f"raw_center_{axis_name}_um"].to_numpy(float)
        raw_plot = np.where(np.isfinite(raw) & (raw >= lo) & (raw <= hi), raw, np.nan)
        axis.plot(time, raw_plot, color=colors["raw"], lw=0.6, alpha=0.7,
                  label="uncorrected fit")
        fitted = time <= cutoff
        axis.plot(time[fitted], data.loc[fitted, f"center_{axis_name}_um"],
                  color=colors["center"], lw=1.7, label="corrected center")
        held = time > cutoff
        if held.any():
            first = np.flatnonzero(held)[0]
            bridge = np.r_[first - 1, np.flatnonzero(held)]
            axis.plot(time[bridge], data.iloc[bridge][f"center_{axis_name}_um"],
                      color=colors["center"], lw=1.7, ls="--", label="terminal hold")
        axis.axvspan(cutoff, time[-1], color="#d95f02", alpha=0.06, lw=0)
        axis.axvline(RISE_TIME_SEC, color="#7b3294", lw=1.1, ls=":")
        axis.set_ylim(lo, hi)
        axis.set_ylabel(f"{axis_name} in AVI frame (µm)")
        axis.grid(alpha=0.16)
    axes[0].legend(loc="upper right", ncol=4, frameon=False)
    axes[-1].set_xlabel("Time (s)")
    axes[-1].set_xlim(time[0], time[-1])
    fig.suptitle(f"No.14 centroid and center | {width:.3f} s forward window", y=1.015, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    full_time = read_csv.get_timelist(DAY)[SAMPLE_NO - 1]
    frame = raw_centroid.load(DAY, SAMPLE_NO, full_time).iloc[START_FRAME - 1:END_FRAME].copy()
    time = frame.time_sec.to_numpy(float)
    x = frame.x_um.to_numpy(float)
    y = frame.y_um.to_numpy(float)
    width = window_width(time, x, y)
    rise_index = int(np.searchsorted(time, RISE_TIME_SEC))
    centers_x, centers_y = repellent_response.estimate_rotation_center_like_standard(
        [time], [x], [y], rise_indices=[rise_index]
    )
    raw_x, raw_y = uncorrected_centers(time, x, y, width)
    if len(centers_x[0]) != len(frame) or len(centers_y[0]) != len(frame):
        raise ValueError("Current center estimate is not aligned with the canonical centroid")
    data = frame[["source_frame_1based", "time_sec", "x_px", "y_px", "x_um", "y_um", "detected"]].reset_index(drop=True)
    data["center_x_um"] = centers_x[0]
    data["center_y_um"] = centers_y[0]
    data["raw_center_x_um"] = raw_x
    data["raw_center_y_um"] = raw_y
    data["center_source"] = np.where(time <= time[-1] - width, "window", "terminal_hold")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(FIGURE_DIR / "no14_centroid_center_timeseries.csv", index=False)
    plot_timeseries(data, width, FIGURE_DIR / "no14_centroid_center_timeseries.png")
    print(f"No.14: {len(data)} frames, window={width:.3f}s")


if __name__ == "__main__":
    main()
