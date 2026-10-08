"""Plot No.14 centroid and current moving-window center in five-second columns."""

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


def plot_grid(data: pd.DataFrame, width: float, starts: list[int], path: Path) -> None:
    ncols = len(starts)
    fig, axes = plt.subplots(2, ncols, figsize=(3.6 * ncols, 6.4), sharey="row", squeeze=False)
    time = data.time_sec.to_numpy(float)
    cutoff = time[-1] - width
    colors = {"centroid": "#455468", "center": "#d95f02", "raw": "#0072b2"}
    limits = {}
    for axis_name in "xy":
        values = np.r_[data[f"{axis_name}_um"].to_numpy(float),
                       data[f"center_{axis_name}_um"].to_numpy(float)]
        lo, hi = np.nanmin(values), np.nanmax(values)
        padding = 0.045 * (hi - lo)
        limits[axis_name] = (lo - padding, hi + padding)
    for col, begin in enumerate(starts):
        end = begin + 5
        mask = (time >= begin) & (time < end)
        for row, axis_name in enumerate("xy"):
            axis = axes[row, col]
            axis.plot(time[mask], data.loc[mask, f"{axis_name}_um"], color=colors["centroid"],
                      alpha=0.52, lw=0.55, rasterized=True, label="centroid" if col == 0 else None)
            fitted = mask & (time <= cutoff)
            held = mask & (time > cutoff)
            raw = data[f"raw_center_{axis_name}_um"].to_numpy(float)
            lo, hi = limits[axis_name]
            raw_visible = mask & np.isfinite(raw) & (raw >= lo) & (raw <= hi)
            raw_plot = np.where(raw_visible, raw, np.nan)
            axis.plot(time[mask], raw_plot[mask], color=colors["raw"], lw=0.6,
                      alpha=0.7, label="uncorrected fit" if col == 0 else None)
            axis.plot(time[fitted], data.loc[fitted, f"center_{axis_name}_um"],
                      color=colors["center"], lw=1.65, label="window center" if col == 0 else None)
            if held.any():
                first = np.flatnonzero(held)[0]
                bridge = np.r_[first - 1, np.flatnonzero(held)] if first > 0 else np.flatnonzero(held)
                axis.plot(time[bridge], data.iloc[bridge][f"center_{axis_name}_um"],
                          color=colors["center"], lw=1.65, ls="--", label="held center" if col == 0 else None)
                axis.axvspan(max(begin, cutoff), end, color="#d95f02", alpha=0.06, lw=0)
            if begin <= RISE_TIME_SEC < end:
                axis.axvline(RISE_TIME_SEC, color="#7b3294", lw=1.1, ls=":")
            axis.set_xlim(begin, end)
            axis.set_xticks([begin, begin + 2.5, end])
            axis.grid(alpha=0.16)
            if row == 0:
                axis.set_title(f"{begin}–{end} s", fontsize=10)
            else:
                axis.set_xlabel("Time (s)")
            if col == 0:
                axis.set_ylabel(f"{axis_name} in AVI frame (µm)")
    for row, axis_name in enumerate("xy"):
        lo, hi = limits[axis_name]
        for axis in axes[row]:
            axis.set_ylim(lo, hi)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9)
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
    data = frame[["source_frame_1based", "time_sec", "x_um", "y_um"]].reset_index(drop=True)
    data["center_x_um"] = centers_x[0]
    data["center_y_um"] = centers_y[0]
    data["raw_center_x_um"] = raw_x
    data["raw_center_y_um"] = raw_y
    data["center_source"] = np.where(time <= time[-1] - width, "window", "terminal_hold")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    data.to_csv(FIGURE_DIR / "no14_centroid_center_5s.csv", index=False)
    starts = list(range(15, 90, 5))
    plot_grid(data, width, starts, FIGURE_DIR / "no14_centroid_center_5s_full.png")
    for begin in (15, 40, 65):
        plot_grid(data, width, list(range(begin, begin + 25, 5)),
                  FIGURE_DIR / f"no14_centroid_center_5s_{begin:02d}-{begin + 25:02d}.png")
    print(f"No.14: {len(data)} frames, window={width:.3f}s, columns={len(starts)}")


if __name__ == "__main__":
    main()
