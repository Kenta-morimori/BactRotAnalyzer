import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

if hasattr(cv2, "VideoWriter_fourcc"):
    _cv2_fourcc: Callable[..., int] = getattr(cv2, "VideoWriter_fourcc")

    def cv2_video_writer_fourcc(*args: str) -> int:
        return _cv2_fourcc(*args)

else:  # pragma: no cover

    def cv2_video_writer_fourcc(*_args: str) -> int:
        raise RuntimeError("OpenCV VideoWriter_fourcc is unavailable")


sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from utils import param

DATA_KEYS = [
    "SJW46_10",
    "SJW46_23",
    "SJW46_30",
]
DEFAULT_HIST_BINS = 40
AV_DIR_DICT = {  # Angular velocity
    "SJW46_10": "outputs/SJW46_temp=10/angular_velocity/angular-velocity_time-series.csv",
    "SJW46_23": "outputs/SJW46_temp=23/angular_velocity/angular-velocity_time-series.csv",  # SJW46 23°C
    "SJW46_30": "outputs/SJW46_temp=30/angular_velocity/angular-velocity_time-series.csv",  # SJW46 30°C
}
TIME_DIR_DICT = {
    "SJW46_10": "outputs/SJW46_temp=10/time_list.csv",
    "SJW46_23": "outputs/SJW46_temp=23/time_list.csv",
    "SJW46_30": "outputs/SJW46_temp=30/time_list.csv",
}


def plot_distribution(df: pd.DataFrame, out_dir: Path) -> None:
    """Plot angular-velocity histograms per strain/No with shared binning and save the grids.

    Args:
        df (pd.DataFrame): Time-series angular velocity dataframe with columns ['label', 'No', 'av'].
        out_dir (Path): Directory where distribution figures are written.

    Returns:
        None
    """
    out_dir = out_dir / "av_distribution"
    out_dir.mkdir(parents=True, exist_ok=True)

    av_values = df["av"].to_numpy(dtype=float)
    av_values = av_values[np.isfinite(av_values)]
    if av_values.size > 0:
        global_min, global_max = av_values.min(), av_values.max()
        if global_min == global_max:
            global_min -= 0.5
            global_max += 0.5
        bin_edges = np.linspace(global_min, global_max, DEFAULT_HIST_BINS + 1)
    else:
        bin_edges = np.linspace(-1, 1, DEFAULT_HIST_BINS + 1)

    for save_i, data_key in enumerate(DATA_KEYS):
        df_selected = df[df["label"] == data_key]
        No_list = df_selected["No"].unique().tolist()
        No_num = len(No_list)

        if No_num == 0:
            continue

        # plot
        cols = 2
        rows = max(1, math.ceil(No_num / cols))
        fig, axs = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axs = np.array(axs).ravel()

        for i, No_i in enumerate(No_list):
            axes = axs[i]

            data = df_selected[df_selected["No"] == No_i]["av"]
            axes.hist(data, bins=bin_edges)
            axes.axvline(0, color="red", linewidth=2)
            axes.grid(True)
            axes.set_title(f"No.{i+1}")
            axes.set_xlabel("Angular Velocity")
            axes.set_ylabel("Counts")
        for j in range(No_num, rows * cols):
            fig.delaxes(axs[j])
        fig.suptitle(f"{data_key} Angular Velocity Distribution")
        plt.tight_layout()
        plt.savefig(f"{out_dir}/{save_i + 1}_{data_key}_av_distribution.png")
        plt.close(fig)


def plot_stats(df_stats: pd.DataFrame, out_dir: Path) -> None:
    """Scatter the mean and median of absolute angular velocity across strains.

    Args:
        df_stats (pd.DataFrame): Summary dataframe containing 'label', 'No', 'av_abs_mean', and 'av_abs_median'.
        out_dir (Path): Directory where summary plots are written.

    Returns:
        None
    """

    out_dir = out_dir / "av_stats"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs = np.array(axs).ravel()

    for save_i, data_key in enumerate(DATA_KEYS):
        df_selected = df_stats[df_stats["label"] == data_key]
        data_len = len(df_selected)

        # Mean
        axs[0].scatter([save_i] * data_len, df_selected["av_abs_mean"])

        # Median
        axs[1].scatter([save_i] * data_len, df_selected["av_abs_median"])

    x_positions = np.arange(len(DATA_KEYS))
    for ax in axs:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(DATA_KEYS, rotation=45, ha="right")
    axs[0].set_ylabel("Mean of angular velocity")
    axs[0].set_title("Mean of angular velocity")
    axs[1].set_ylabel("Median of angular velocity")
    axs[1].set_title("Median of angular velocity")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/av_stats.png")
    plt.close(fig)


def plot_gaussian_stats(df_gauss: pd.DataFrame, out_dir: Path) -> None:
    """Visualize Gaussian fit mean/std distributions and their mean-vs-std relationship.

    Args:
        df_gauss (pd.DataFrame): Gaussian fit results with columns such as 'label', 'mean', and 'std'.
        out_dir (Path): Directory where Gaussian summary plots are written.

    Returns:
        None
    """

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs = np.array(axs).ravel()

    for save_i, data_key in enumerate(DATA_KEYS):
        df_selected = df_gauss[(df_gauss["label"] == data_key) & df_gauss["mean"].notna() & df_gauss["std"].notna()]
        data_len = len(df_selected)
        if data_len == 0:
            continue

        axs[0].scatter([save_i] * data_len, df_selected["mean"])
        axs[1].scatter([save_i] * data_len, df_selected["std"])
        axs[2].scatter(df_selected["mean"], df_selected["std"], label=data_key)

    x_positions = np.arange(len(DATA_KEYS))
    for ax in axs[:2]:
        ax.set_xticks(x_positions)
        ax.set_xticklabels(DATA_KEYS, rotation=45, ha="right")
    axs[2].set_xticks([])
    axs[0].set_ylabel("Gaussian mean")
    axs[0].set_title("Gaussian mean distribution")
    axs[1].set_ylabel("Gaussian std")
    axs[1].set_title("Gaussian std distribution")
    axs[2].set_xlabel("Gaussian mean")
    axs[2].set_ylabel("Gaussian std")
    axs[2].set_title("Mean vs Std")
    axs[2].legend()
    plt.tight_layout()
    plt.savefig(out_dir / "gaussian_fit_stats.png")
    plt.close(fig)


def gaussian(x: np.ndarray, amplitude: float, mean: float, std: float) -> np.ndarray:
    """Evaluate a 1D Gaussian at x for given amplitude, mean, and std.

    Args:
        x (np.ndarray): Sample positions.
        amplitude (float): Peak height of the Gaussian.
        mean (float): Center of the Gaussian.
        std (float): Standard deviation of the Gaussian.

    Returns:
        np.ndarray: Gaussian values at x.
    """
    return amplitude * np.exp(-((x - mean) ** 2) / (2 * std**2))


def _format_param(value: float) -> str:
    """Format numeric parameter for display (2 decimals, 'NaN' if non-finite).

    Args:
        value (float): Numeric value to format.

    Returns:
        str: Formatted string.
    """
    return f"{value:.2f}" if np.isfinite(value) else "NaN"


def generate_sliding_window_animation(
    df: pd.DataFrame,
    out_dir: Path,
    window_seconds: float = 5.0,
    step_seconds: float = 0.5,
    hist_bins: int = 40,
    video_fps: int = 6,
    fix_hist_ylim: bool = True,
) -> None:
    """Slide a fixed window over time series to show distribution dynamics and export as a video.

    Args:
        df (pd.DataFrame): Time-series angular velocity dataframe with ['label', 'No', 'time', 'av'].
        out_dir (Path): Directory where frames and videos are written.
        window_seconds (float): Width of the sliding window in seconds.
        step_seconds (float): Step size between windows in seconds.
        hist_bins (int): Number of histogram bins.
        video_fps (int): Frames per second for the output video.
        fix_hist_ylim (bool): Whether to fix histogram y-limits to the global max across frames.

    Returns:
        None
    """

    slide_dir = out_dir / "av_sliding_window"
    slide_dir.mkdir(parents=True, exist_ok=True)

    for data_key in DATA_KEYS:
        df_label = df[df["label"] == data_key]
        No_list = df_label["No"].unique().tolist()

        for No_i in No_list:
            df_no = df_label[df_label["No"] == No_i].dropna(subset=["time", "av"]).sort_values("time")

            if len(df_no) < 2:
                print(f"[SlidingWindow] Skip {data_key}-{No_i}: insufficient data points")
                continue

            times = df_no["time"].values
            avs = df_no["av"].values

            time_min, time_max = times.min(), times.max()
            if time_max - time_min < window_seconds:
                print(f"[SlidingWindow] Skip {data_key}-{No_i}: duration shorter than window")
                continue

            av_min, av_max = avs.min(), avs.max()
            if av_min == av_max:
                av_min -= 1
                av_max += 1
            av_margin = (av_max - av_min) * 0.05
            y_min = av_min - av_margin
            y_max = av_max + av_margin

            bin_edges = np.linspace(av_min, av_max, hist_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            frames_dir = slide_dir / f"{data_key}_{No_i}"
            frames_dir.mkdir(parents=True, exist_ok=True)
            for frame_file in frames_dir.glob("frame_*.png"):
                frame_file.unlink()

            window_starts = []
            current = time_min
            epsilon = 1e-8
            while current <= (time_max - window_seconds + epsilon):
                window_starts.append(current)
                current += step_seconds
            if not window_starts:
                window_starts = [time_min]
            total_frames = len(window_starts)

            hist_counts_list = []
            hist_max = 0
            for start_time in window_starts:
                end_time = start_time + window_seconds
                mask = (times >= start_time) & (times < end_time)
                window_av = avs[mask]
                hist_counts, _ = np.histogram(window_av, bins=bin_edges)
                hist_counts_list.append(hist_counts)
                if hist_counts.size > 0:
                    hist_max = max(hist_max, hist_counts.max())

            if hist_max == 0:
                hist_max = 1

            frame_paths = []
            print(f"[SlidingWindow] Start {data_key}-{No_i}: {total_frames} frames")

            for frame_idx, (start_time, hist_counts) in enumerate(zip(window_starts, hist_counts_list)):
                end_time = start_time + window_seconds
                fig, (ax_ts, ax_hist) = plt.subplots(
                    2,
                    1,
                    figsize=(10, 6),
                    gridspec_kw={"height_ratios": [2, 1]},
                )

                ax_ts.plot(times, avs, color="tab:blue", linewidth=1)
                ax_ts.axvspan(start_time, end_time, color="orange", alpha=0.3)
                ax_ts.set_xlim(time_min, time_max)
                ax_ts.set_ylim(y_min, y_max)
                ax_ts.set_title(f"{data_key} {No_i} ({window_seconds:.1f}s window) " f"{frame_idx + 1}/{total_frames}")
                ax_ts.set_xlabel("Time [s]")
                ax_ts.set_ylabel("Angular velocity")
                ax_ts.grid(True)

                ax_hist.bar(
                    bin_centers,
                    hist_counts,
                    width=np.diff(bin_edges),
                    color="tab:green",
                    align="center",
                )
                ax_hist.set_xlim(av_min, av_max)
                if fix_hist_ylim:
                    ax_hist.set_ylim(0, hist_max * 1.1)
                ax_hist.set_title("Windowed distribution")
                ax_hist.set_xlabel("Angular velocity")
                ax_hist.set_ylabel("Counts")

                plt.tight_layout()
                frame_path = frames_dir / f"frame_{frame_idx:04d}.png"
                fig.savefig(frame_path)
                plt.close(fig)
                frame_paths.append(frame_path)

                print(f"[SlidingWindow] {data_key}-{No_i}: frame {frame_idx + 1}/{total_frames}")

            if not frame_paths:
                continue

            first_image = cv2.imread(str(frame_paths[0]))
            if first_image is None:
                print(f"[SlidingWindow] Failed to read first frame for {data_key}-{No_i}")
                continue

            height, width, _ = first_image.shape
            fourcc = cv2_video_writer_fourcc(*"mp4v")
            video_path = slide_dir / f"{data_key}_{No_i}.mov"
            writer = cv2.VideoWriter(str(video_path), fourcc, video_fps, (width, height))

            for frame_path in frame_paths:
                img = cv2.imread(str(frame_path))
                if img is None:
                    continue
                writer.write(img)
            writer.release()
            print(f"[SlidingWindow] Saved video: {video_path}")


def plot_gaussian_fit(df: pd.DataFrame, out_dir: Path, hist_bins: int = DEFAULT_HIST_BINS) -> None:
    """Fit Gaussians to positive angular velocities, plot histogram+fit+residual panels, and save parameters.

    Args:
        df (pd.DataFrame): Time-series angular velocity dataframe with ['label', 'No', 'av'].
        out_dir (Path): Directory where Gaussian fit figures and CSV outputs are written.
        hist_bins (int): Number of histogram bins to use for fitting and plotting.

    Returns:
        None
    """

    out_dir = out_dir / "av_gaussian_fit"
    out_dir.mkdir(parents=True, exist_ok=True)

    gauss_param_list = []

    for save_i, data_key in enumerate(DATA_KEYS):
        df_selected = df[df["label"] == data_key]
        No_list = df_selected["No"].unique().tolist()
        No_num = len(No_list)

        if No_num == 0:
            continue

        positive_values = df_selected[df_selected["av"] > 0]["av"].dropna().to_numpy(dtype=float)
        if positive_values.size > 0:
            global_min, global_max = positive_values.min(), positive_values.max()
            if global_min == global_max:
                global_min -= 0.5
                global_max += 0.5
            bin_edges = np.linspace(global_min, global_max, hist_bins + 1)
        else:
            bin_edges = np.linspace(0, 1, hist_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        cols = 2
        rows = max(1, math.ceil(No_num / cols))
        height_ratios = []
        for _ in range(rows):
            height_ratios.extend([3, 1])
        fig = plt.figure(figsize=(15, 4 * rows))
        gs = fig.add_gridspec(nrows=rows * 2, ncols=cols, height_ratios=height_ratios)
        axs_main = []
        axs_resid = []
        for idx in range(rows * cols):
            row = idx // cols
            col = idx % cols
            axs_main.append(fig.add_subplot(gs[row * 2, col]))
            axs_resid.append(fig.add_subplot(gs[row * 2 + 1, col]))

        for i, No_i in enumerate(No_list):
            axes = axs_main[i]
            resid_ax = axs_resid[i]
            data = df_selected[(df_selected["No"] == No_i) & (df_selected["av"] > 0)]["av"].dropna()
            axes.axvline(0, color="red", linewidth=2)
            axes.grid(True)
            resid_ax.grid(True)
            axes.set_ylabel("Counts")
            resid_ax.set_ylabel("Residual")
            resid_ax.set_xlabel("Angular Velocity")
            axes.set_xlim(bin_edges[0], bin_edges[-1])
            resid_ax.set_xlim(bin_edges[0], bin_edges[-1])
            resid_ax.axhline(0, color="black", linewidth=1)

            fit_params = {"amplitude": np.nan, "mean": np.nan, "std": np.nan}
            hist_counts = np.array([])

            if len(data) == 0:
                axes.text(0.5, 0.5, "No data", transform=axes.transAxes, ha="center", va="center")
                resid_ax.text(0.5, 0.5, "No data", transform=resid_ax.transAxes, ha="center", va="center")
            else:
                hist_counts, _ = np.histogram(data, bins=bin_edges)
                axes.bar(
                    bin_centers,
                    hist_counts,
                    width=np.diff(bin_edges),
                    alpha=0.6,
                    color="tab:blue",
                    align="center",
                )

                if len(data) < 5:
                    axes.text(0.5, 0.5, "Insufficient data", transform=axes.transAxes, ha="center", va="center")
                    resid_ax.text(0.5, 0.5, "Insufficient data", transform=resid_ax.transAxes, ha="center", va="center")
                elif np.all(hist_counts == 0):
                    axes.text(0.5, 0.5, "No counts", transform=axes.transAxes, ha="center", va="center")
                    resid_ax.text(0.5, 0.5, "No counts", transform=resid_ax.transAxes, ha="center", va="center")
                else:
                    std_guess = data.std()
                    if std_guess == 0:
                        std_guess = 1e-3
                    initial_guess = (hist_counts.max(), data.mean(), std_guess)
                    try:
                        popt, _ = curve_fit(gaussian, bin_centers, hist_counts, p0=initial_guess, maxfev=5000)
                        x_fit = np.linspace(bin_edges[0], bin_edges[-1], 200)
                        axes.plot(x_fit, gaussian(x_fit, *popt), color="orange", linewidth=2)
                        fit_params = {"amplitude": popt[0], "mean": popt[1], "std": abs(popt[2])}
                        expected_counts = gaussian(bin_centers, *popt)
                        residuals = hist_counts - expected_counts
                        resid_ax.bar(
                            bin_centers,
                            residuals,
                            width=np.diff(bin_edges),
                            align="center",
                            color="tab:gray",
                        )
                        resid_max = np.max(np.abs(residuals)) if residuals.size else 0
                        if resid_max == 0:
                            resid_max = 1
                        resid_ax.set_ylim(-resid_max * 1.1, resid_max * 1.1)
                        resid_ax.set_title("Data - Gaussian", fontsize=9)
                        if np.isfinite(fit_params["mean"]):
                            resid_ax.axvline(fit_params["mean"], color="red", linewidth=1)
                    except (RuntimeError, ValueError):
                        axes.text(
                            0.5,
                            0.5,
                            "Fit failed",
                            transform=axes.transAxes,
                            ha="center",
                            va="center",
                            color="red",
                        )
                        resid_ax.text(
                            0.5,
                            0.5,
                            "Fit failed",
                            transform=resid_ax.transAxes,
                            ha="center",
                            va="center",
                            color="red",
                        )

            gauss_param_list.append(
                {
                    "label": data_key,
                    "No": No_i,
                    "positive_count": len(data),
                    "amplitude": fit_params["amplitude"],
                    "mean": fit_params["mean"],
                    "std": fit_params["std"],
                }
            )

            axes.set_title(
                f"No.{i+1} | amp={_format_param(fit_params['amplitude'])}, "
                f"mean={_format_param(fit_params['mean'])}, std={_format_param(fit_params['std'])}"
            )

        for j in range(No_num, rows * cols):
            axs_main[j].remove()
            axs_resid[j].remove()
        fig.suptitle(f"{data_key} Gaussian Fit (Positive AV)")
        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(f"{out_dir}/{save_i + 1}_{data_key}_gaussian_fit.png")
        plt.close(fig)

    if gauss_param_list:
        df_gauss = pd.DataFrame(gauss_param_list)
        df_gauss.to_csv(out_dir / "gaussian_fit_parameters.csv", index=False)
        plot_gaussian_stats(df_gauss, out_dir)


def main():
    ############
    # Load Data
    ############
    df_dict = defaultdict(list)
    for data_key in DATA_KEYS:
        df_av = pd.read_csv(AV_DIR_DICT[data_key])
        df_time = pd.read_csv(TIME_DIR_DICT[data_key])

        # obtain data number
        cols = df_av.columns.tolist()

        # make dataset
        for col in cols:  # No
            data_len = len(df_av[col])
            # data param
            df_dict["label"].extend([data_key] * data_len)
            df_dict["No"].extend([col] * data_len)

            # angular velocity and time
            df_dict["av"].extend(df_av[col])
            df_dict["time"].extend(df_time[col][:data_len])
    df = pd.DataFrame(df_dict)

    ############
    # Data analysis
    ############
    out_dir = Path(param.save_dir_bef) / "03_av_analysis"

    # 1. Angular Velocity Distribution
    plot_distribution(df, out_dir)

    # 2. Mean and Median analysis
    stats_dict = defaultdict(list)
    df["av_abs"] = np.abs(df["av"])
    for data_key in DATA_KEYS:
        cols = pd.read_csv(AV_DIR_DICT[data_key]).columns.tolist()
        for col in cols:  # No
            df_selected = df[(df["label"] == data_key) & (df["No"] == col)]
            stats_dict["label"].append(data_key)
            stats_dict["No"].append(col)
            # record stats(mean, median)
            stats_dict["av_abs_mean"].append(df_selected["av_abs"].mean())
            stats_dict["av_abs_median"].append(df_selected["av_abs"].median())
    df_stats = pd.DataFrame(stats_dict)
    # plot
    plot_stats(df_stats, out_dir)

    # 3. Gaussian Fit
    plot_gaussian_fit(df, out_dir)
    # 4. Sliding window animation
    generate_sliding_window_animation(df, out_dir)


if __name__ == "__main__":
    # TODO: select directory (and data number) to analyze
    main()
