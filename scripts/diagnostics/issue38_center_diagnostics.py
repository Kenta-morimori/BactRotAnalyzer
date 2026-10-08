"""Reproduce the centre-first diagnostics for BactRotAnalyzer issue #38.

Run from the repository root. The script reads existing analysis outputs and AVIs;
it does not rerun or change the production analysis.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data/repellent-response/23"
OUTPUT = REPO / "outputs/repellent-response/23"
STAGE = OUTPUT / "repellent_response/00_all_rotational_analysis"
DEFAULT_DEST = REPO / "docs/analysis/issue-38/archive/legacy"
VIDEOS = {10: "2026_0606_221330.avi", 14: "2026_0802_182430.avi"}
PLOT_TIMES = {10: (10, 78, 82, 90), 14: (22, 35, 42, 65)}


@dataclass
class Track:
    sample: int
    video: str
    source_start: int
    time: np.ndarray
    x: np.ndarray
    y: np.ndarray
    cx: np.ndarray
    cy: np.ndarray
    initial_uncorrected_cx: np.ndarray
    initial_uncorrected_cy: np.ndarray
    initial_corrected_cx: np.ndarray
    initial_corrected_cy: np.ndarray
    initial_raw_x: np.ndarray
    initial_raw_y: np.ndarray


@dataclass
class Ellipse:
    center: np.ndarray
    axes: np.ndarray
    basis: np.ndarray
    condition: float


def read_track(sample: int) -> Track:
    mapping = pd.read_csv(OUTPUT / "centroid_coordinate_sample_map.csv")
    entry = mapping.loc[mapping.sample_no == sample].iloc[0]
    video = str(entry.avi_filename)
    assert video == VIDEOS[sample], (sample, video)
    ranges = pd.read_csv(OUTPUT / "repellent_response/00_time_list/analysis_frame_ranges.csv")
    source_start = int(ranges.loc[ranges.sample_no == sample, "source_start_index_0based"].iloc[0])
    relative = pd.read_csv(STAGE / "centroid_coordinate/centroid_time_series.csv")
    center = pd.read_csv(STAGE / "center_coordinate/center_coordinate.csv")
    fields = [relative[f"No.{sample}_{name}"].to_numpy(float) for name in ("time", "x", "y")]
    fields += [center[f"No.{sample}_{name}"].to_numpy(float) for name in ("x", "y")]
    n = int(ranges.loc[ranges.sample_no == sample, "selected_frame_count"].iloc[0])
    t, rx, ry, cx, cy = [field[:n] for field in fields]
    initial_dir = OUTPUT / "center_coordinate/bef_correction"
    before = pd.read_csv(initial_dir / "center_coordinate_bef_correct.csv")
    after = pd.read_csv(initial_dir / "center_coordinate.csv")
    beginning = source_start
    ending = source_start + n
    bx, by = [before[f"No.{sample}_{axis}"].to_numpy(float)[beginning:ending] for axis in "xy"]
    ax, ay = [after[f"No.{sample}_{axis}"].to_numpy(float)[beginning:ending] for axis in "xy"]
    initial_relative = pd.read_csv(OUTPUT / "centroid_coordinate.csv")
    initial_x = initial_relative[f"x_{sample}"].to_numpy(float)[beginning:ending] + ax
    initial_y = initial_relative[f"y_{sample}"].to_numpy(float)[beginning:ending] + ay
    assert all(np.isfinite(field).all() for field in (t, rx, ry, cx, cy))
    assert np.all(np.diff(t) > 0)
    return Track(sample, video, source_start, t, rx + cx, ry + cy, cx, cy, bx, by, ax, ay, initial_x, initial_y)


def fit_ellipse(x: np.ndarray, y: np.ndarray) -> Ellipse | None:
    """Fit the same quadratic form as the current algorithm, then resolve its geometry."""
    if len(x) < 30 or not np.isfinite(x).all() or not np.isfinite(y).all():
        return None
    design = np.column_stack((x * x, x * y, y * y, x, y))
    coef, _, rank, _ = np.linalg.lstsq(design, np.ones(len(x)), rcond=None)
    if rank < 5:
        return None
    a, b, c, d, e = coef
    quadratic = np.array([[a, b / 2], [b / 2, c]])
    eigval, basis = np.linalg.eigh(quadratic)
    if min(abs(eigval)) < 1e-12:
        return None
    center = -0.5 * np.linalg.solve(quadratic, np.array([d, e]))
    level = 1 + center @ quadratic @ center
    axis_squared = level / eigval
    if np.any(axis_squared <= 0) or not np.isfinite(axis_squared).all():
        return None
    # Normalization prevents microscope-origin offsets from dominating the condition number.
    scale = max(float(np.median(np.hypot(x - np.median(x), y - np.median(y)))), 1e-6)
    u, v = (x - np.median(x)) / scale, (y - np.median(y)) / scale
    normalized_design = np.column_stack((u * u, u * v, v * v, u, v))
    condition = float(np.linalg.cond(normalized_design))
    return Ellipse(center, np.sqrt(axis_squared), basis, condition)


def ellipse_points(ellipse: Ellipse, count: int = 1440) -> np.ndarray:
    phase = np.linspace(0, 2 * np.pi, count, endpoint=False)
    return ellipse.center + np.column_stack((np.cos(phase), np.sin(phase))) @ np.diag(ellipse.axes) @ ellipse.basis.T


def ellipse_distance(x: np.ndarray, y: np.ndarray, ellipse: Ellipse) -> np.ndarray:
    # A dense polyline approximates the shortest Euclidean distance to the ellipse.
    return cKDTree(ellipse_points(ellipse)).query(np.column_stack((x, y)))[0]


def translated_ellipse(ellipse: Ellipse, center: np.ndarray) -> Ellipse:
    return Ellipse(center, ellipse.axes, ellipse.basis, ellipse.condition)


def angular_coverage(x: np.ndarray, y: np.ndarray, center: np.ndarray) -> tuple[float, float]:
    phase = np.sort(np.mod(np.arctan2(y - center[1], x - center[0]), 2 * np.pi))
    gaps = np.diff(np.r_[phase, phase[0] + 2 * np.pi])
    biggest = float(np.degrees(gaps.max()))
    return 360 - biggest, biggest


def legacy_window_width(sample: int) -> float:
    summary = pd.read_csv(OUTPUT / "center_coordinate/rotation_center_fft_window_summary.csv")
    row = summary.loc[(summary.analysis_stage == "repellent_response") & (summary.sample_no == sample)].iloc[0]
    return float(row.applied_width_time_sec)


def classify(row: dict) -> str:
    # Exploratory flags, not acceptance thresholds for a new method in issue #39.
    if not np.isfinite(row["residual_p95_over_radius"]):
        return "fit_failed"
    if row["jackknife_valid_fits"] < 4:
        return "center_unidentifiable"
    if row["coverage_deg"] < 180 or row["jackknife_max_over_radius"] > 0.25:
        return "center_unidentifiable"
    if row["residual_p95_over_radius"] > 0.2:
        return "nonelliptic_or_noisy"
    if row["saved_vs_refit_over_radius"] > 0.25:
        return "saved_center_displaced"
    if row["coverage_deg"] < 270 or row["normalized_condition"] > 1e4:
        return "review"
    return "supported"


def evaluate_windows(track: Track, stride: int, source: str = "stage") -> pd.DataFrame:
    width = legacy_window_width(track.sample)
    times = track.time
    source_x, source_y = (track.x, track.y) if source == "stage" else (track.initial_raw_x, track.initial_raw_y)
    records = []
    previous_center = None
    for start in range(0, len(times), stride):
        end = int(np.searchsorted(times, times[start] + width, side="left"))
        if end - start < 30:
            continue
        xx, yy = source_x[start:end], source_y[start:end]
        ellipse = fit_ellipse(xx, yy)
        saved = np.array([track.cx[start], track.cy[start]])
        radius = float(np.median(np.hypot(xx - saved[0], yy - saved[1])))
        if ellipse is None or radius <= 0:
            records.append(
                {
                    "sample_no": track.sample,
                    "start_index_0based": start,
                    "end_index_exclusive_0based": end,
                    "source_start_frame_1based": start + track.source_start + 1,
                    "source_end_frame_1based": end + track.source_start,
                    "start_time_sec": times[start],
                    "end_time_sec": times[end - 1],
                    "n_points": end - start,
                    "source": source,
                    "assessment": "fit_failed",
                }
            )
            previous_center = saved
            continue
        coverage, gap = angular_coverage(xx, yy, ellipse.center)
        # Leave one contiguous quarter out. This keeps most of each rotation in every fit.
        quarters = np.array_split(np.arange(len(xx)), 4)
        jackknife = []
        for quarter in quarters:
            keep = np.ones(len(xx), dtype=bool)
            keep[quarter] = False
            part = fit_ellipse(xx[keep], yy[keep])
            if part is not None:
                jackknife.append(part.center)
        jackknife_array = np.array(jackknife) if jackknife else np.empty((0, 2))
        jackknife_max = float(np.max(np.linalg.norm(jackknife - ellipse.center, axis=1))) if len(jackknife) else np.nan
        distances = ellipse_distance(xx, yy, ellipse)
        saved_distances = ellipse_distance(xx, yy, translated_ellipse(ellipse, saved))
        initial_correction = np.linalg.norm(
            np.array([track.initial_corrected_cx[start], track.initial_corrected_cy[start]])
            - np.array([track.initial_uncorrected_cx[start], track.initial_uncorrected_cy[start]])
        )
        record = {
            "sample_no": track.sample,
            "source": source,
            "start_index_0based": start,
            "end_index_exclusive_0based": end,
            "source_start_frame_1based": start + track.source_start + 1,
            "source_end_frame_1based": end + track.source_start,
            "start_time_sec": times[start],
            "end_time_sec": times[end - 1],
            "n_points": end - start,
            "saved_center_x": saved[0],
            "saved_center_y": saved[1],
            "refit_center_x": ellipse.center[0],
            "refit_center_y": ellipse.center[1],
            "median_orbit_radius": radius,
            "residual_median_over_radius": np.median(distances) / radius,
            "residual_p95_over_radius": np.quantile(distances, 0.95) / radius,
            "saved_center_residual_p95_over_radius": np.quantile(saved_distances, 0.95) / radius,
            "coverage_deg": coverage,
            "max_gap_deg": gap,
            "normalized_condition": ellipse.condition,
            "jackknife_valid_fits": len(jackknife),
            "jackknife_sd_x_over_radius": np.std(jackknife_array[:, 0]) / radius if len(jackknife) else np.nan,
            "jackknife_sd_y_over_radius": np.std(jackknife_array[:, 1]) / radius if len(jackknife) else np.nan,
            "jackknife_max_over_radius": jackknife_max / radius,
            "saved_vs_refit_over_radius": np.linalg.norm(saved - ellipse.center) / radius,
            "initial_center_correction_over_radius": initial_correction / radius,
            "previous_saved_center_step_over_radius": (
                np.linalg.norm(saved - previous_center) / radius if previous_center is not None else np.nan
            ),
        }
        record["assessment"] = classify(record)
        records.append(record)
        previous_center = saved
    return pd.DataFrame.from_records(records)


def video_check(track: Track, dest: Path) -> pd.DataFrame:
    cap = cv2.VideoCapture(str(DATA / track.video))
    assert cap.isOpened(), track.video
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) >= track.source_start + len(track.time)
    indices = sorted(set(np.linspace(0, len(track.time) - 1, 80, dtype=int).tolist()))
    rows = []
    for i in indices:
        frame_index = track.source_start + i
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        contour = max(contours, key=cv2.contourArea)
        rows.append(
            {
                "sample_no": track.sample,
                "analysis_index": i,
                "source_frame_1based": frame_index + 1,
                "time_sec": track.time[i],
                "avi_local_x_px": float(contour[:, 0, 0].mean()),
                "avi_local_y_px": float(contour[:, 0, 1].mean()),
                "reconstructed_x": track.x[i],
                "reconstructed_y": track.y[i],
                "initial_x": track.initial_raw_x[i],
                "initial_y": track.initial_raw_y[i],
            }
        )
    cap.release()
    df = pd.DataFrame(rows)
    # AVI uses a cropped local image. Fit an affine registration on alternating frames.
    # Validate on the other alternating frames. This is a registration check,
    # not an independent ground-truth measurement of absolute microscope coordinates.
    train = df.iloc[::2]
    test = df.iloc[1::2].copy()
    for source, prefix in (("stage", "reconstructed"), ("initial", "initial")):
        for axis in "xy":
            slope, offset = np.polyfit(train[f"avi_local_{axis}_px"], train[f"{prefix}_{axis}"], 1)
            df[f"{source}_avi_fit_{axis}"] = slope * df[f"avi_local_{axis}_px"] + offset
            df[f"{source}_avi_error_{axis}"] = df[f"{prefix}_{axis}"] - df[f"{source}_avi_fit_{axis}"]
            test[f"{source}_error_{axis}"] = test[f"{prefix}_{axis}"] - (slope * test[f"avi_local_{axis}_px"] + offset)
    df.to_csv(dest / f"no{track.sample:02d}_avi_registration.csv", index=False)
    fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True, layout="constrained")
    for plot_axis, coordinate in zip(axes, "xy"):
        plot_axis.scatter(df.time_sec, df[f"stage_avi_error_{coordinate}"], s=10, label="stage")
        plot_axis.scatter(df.time_sec, df[f"initial_avi_error_{coordinate}"], s=10, label="initial")
        plot_axis.axhline(0, color="black", lw=0.7)
        plot_axis.set_ylabel(f"{coordinate} residual")
        plot_axis.legend(fontsize=8)
    axes[-1].set_xlabel("TIFF time (s)")
    fig.suptitle(f"No.{track.sample} — AVI contour against reconstructed centroid (registered crop)")
    fig.savefig(dest / f"no{track.sample:02d}_avi_registration.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame(
        [
            {
                "sample_no": track.sample,
                "validation_frames": len(test),
                "stage_heldout_median_error": float(np.median(np.hypot(test.stage_error_x, test.stage_error_y))),
                "stage_heldout_p95_error": float(np.quantile(np.hypot(test.stage_error_x, test.stage_error_y), 0.95)),
                "initial_heldout_median_error": float(np.median(np.hypot(test.initial_error_x, test.initial_error_y))),
                "initial_heldout_p95_error": float(
                    np.quantile(np.hypot(test.initial_error_x, test.initial_error_y), 0.95)
                ),
                "fitted_x_units_per_px": float(np.polyfit(train.avi_local_x_px, train.reconstructed_x, 1)[0]),
                "fitted_y_units_per_px": float(np.polyfit(train.avi_local_y_px, train.reconstructed_y, 1)[0]),
            }
        ]
    )


def plot_overview(track: Track, windows: pd.DataFrame, dest: Path) -> None:
    fig, ax = plt.subplots(4, 1, figsize=(12, 10), sharex=True, layout="constrained")
    ax[0].plot(track.time, track.x, lw=0.45, label="centroid x")
    ax[0].plot(track.time, track.cx, lw=0.8, label="saved center x")
    ax[0].plot(track.time, track.initial_uncorrected_cx, lw=0.35, alpha=0.5, label="initial uncorrected x")
    ax[0].plot(track.time, track.initial_corrected_cx, lw=0.35, alpha=0.5, label="initial corrected x")
    ax[1].plot(track.time, track.y, lw=0.45, label="centroid y")
    ax[1].plot(track.time, track.cy, lw=0.8, label="saved center y")
    ax[1].plot(track.time, track.initial_uncorrected_cy, lw=0.35, alpha=0.5, label="initial uncorrected y")
    ax[1].plot(track.time, track.initial_corrected_cy, lw=0.35, alpha=0.5, label="initial corrected y")
    ax[2].plot(windows.start_time_sec, windows.coverage_deg, label="angular coverage (deg)")
    ax[2].plot(windows.start_time_sec, windows.max_gap_deg, label="largest gap (deg)")
    ax[3].plot(windows.start_time_sec, windows.residual_p95_over_radius, label="ellipse residual p95 / radius")
    ax[3].plot(windows.start_time_sec, windows.jackknife_max_over_radius, label="jackknife center shift / radius")
    ax[3].plot(windows.start_time_sec, windows.saved_vs_refit_over_radius, label="saved vs refit / radius")
    ax[3].plot(
        windows.start_time_sec, windows.initial_center_correction_over_radius, label="initial correction / radius"
    )
    for axis in ax:
        axis.legend(loc="upper right", fontsize=8)
        axis.grid(alpha=0.2)
    ax[0].set_ylabel("x (source units)")
    ax[1].set_ylabel("y (source units)")
    ax[2].set_ylabel("degrees")
    ax[3].set_ylabel("ratio")
    ax[3].set_xlabel("TIFF time (s)")
    fig.suptitle(f"No.{track.sample} — {track.video}")
    fig.savefig(dest / f"no{track.sample:02d}_overview.png", dpi=150)
    plt.close(fig)


def plot_windows(track: Track, windows: pd.DataFrame, dest: Path, source: str = "stage") -> None:
    source_x, source_y = (track.x, track.y) if source == "stage" else (track.initial_raw_x, track.initial_raw_y)
    fig, axes = plt.subplots(2, 2, figsize=(10, 10), layout="constrained")
    for axis, target in zip(axes.flat, PLOT_TIMES[track.sample]):
        row = windows.iloc[(windows.start_time_sec - target).abs().argmin()]
        start, end = int(row.start_index_0based), int(row.end_index_exclusive_0based)
        xx, yy = source_x[start:end], source_y[start:end]
        ellipse = fit_ellipse(xx, yy)
        axis.plot(xx, yy, ".", ms=2, alpha=0.5, label="centroid")
        if ellipse is not None:
            points = ellipse_points(ellipse)
            axis.plot(points[:, 0], points[:, 1], lw=1, label="refit ellipse")
            axis.plot(*ellipse.center, "kx", ms=8, label="refit center")
        axis.plot(track.cx[start], track.cy[start], "r+", ms=10, label="saved center")
        axis.set_aspect("equal", adjustable="datalim")
        if row.assessment == "fit_failed":
            detail = "ellipse fit failed"
        else:
            detail = f"coverage {row.coverage_deg:.0f}°; p95 residual {row.residual_p95_over_radius:.2f}R"
        axis.set_title(f"{row.start_time_sec:.2f}–{row.end_time_sec:.2f}s  {row.assessment}\n{detail}")
        axis.set_xlabel("x (source units)")
        axis.set_ylabel("y (source units)")
        axis.legend(fontsize=7)
    fig.suptitle(f"No.{track.sample} {source} centroid series")
    fig.savefig(dest / f"no{track.sample:02d}_{'initial_' if source == 'initial' else ''}windows.png", dpi=150)
    plt.close(fig)


def plot_video_frames(track: Track, dest: Path) -> None:
    target = 90 if track.sample == 10 else 35
    first = int(np.searchsorted(track.time, target))
    indices = first + np.arange(0, 60, 5)
    cap = cv2.VideoCapture(str(DATA / track.video))
    fig, axes = plt.subplots(3, 4, figsize=(10, 8), layout="constrained")
    for axis, index in zip(axes.flat, indices):
        source_index = track.source_start + int(index)
        cap.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = cap.read()
        if ok:
            axis.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axis.set_title(f"{track.time[index]:.3f}s, frame {source_index + 1}", fontsize=8)
        axis.axis("off")
    cap.release()
    fig.suptitle(f"No.{track.sample} AVI frames (source image coordinates)")
    fig.savefig(dest / f"no{track.sample:02d}_video_frames.png", dpi=120)
    plt.close(fig)


def source_integrity(track: Track) -> dict:
    raw_discrepancy = np.hypot(track.x - track.initial_raw_x, track.y - track.initial_raw_y)
    difference = np.hypot(
        track.initial_corrected_cx - track.initial_uncorrected_cx,
        track.initial_corrected_cy - track.initial_uncorrected_cy,
    )
    same = (np.diff(track.cx) == 0) & (np.diff(track.cy) == 0)
    run_starts = np.r_[0, np.where(~same)[0] + 1]
    run_ends = np.r_[run_starts[1:], len(track.cx)]
    longest = int(np.argmax(run_ends - run_starts))
    start, end = int(run_starts[longest]), int(run_ends[longest])
    return {
        "sample_no": track.sample,
        "video": track.video,
        "source_first_frame_1based": track.source_start + 1,
        "source_last_frame_1based": track.source_start + len(track.time),
        "time_first_sec": track.time[0],
        "time_last_sec": track.time[-1],
        "initial_center_corrected_frames": int(np.count_nonzero(difference > 1e-12)),
        "initial_correction_p99": float(np.quantile(difference, 0.99)),
        "initial_correction_max": float(np.max(difference)),
        "stage_vs_initial_raw_median": float(np.median(raw_discrepancy)),
        "stage_vs_initial_raw_p95": float(np.quantile(raw_discrepancy, 0.95)),
        "longest_constant_center_frames": end - start,
        "constant_center_start_sec": track.time[start],
        "constant_center_end_sec": track.time[end - 1],
    }


def angle_sensitivity(track: Track, windows: pd.DataFrame) -> pd.DataFrame:
    """Compare angle diagnostics only after assessing the centre in each window."""
    rows = []
    for target in PLOT_TIMES[track.sample]:
        row = windows.iloc[(windows.start_time_sec - target).abs().argmin()]
        if row.assessment == "fit_failed":
            continue
        start, end = int(row.start_index_0based), int(row.end_index_exclusive_0based)
        for label, center in (
            ("saved", (row.saved_center_x, row.saved_center_y)),
            ("refit", (row.refit_center_x, row.refit_center_y)),
        ):
            phase = np.unwrap(np.arctan2(track.y[start:end] - center[1], track.x[start:end] - center[0]))
            speed = -np.diff(phase) / np.diff(track.time[start:end])
            rows.append(
                {
                    "sample_no": track.sample,
                    "target_time_sec": target,
                    "start_time_sec": row.start_time_sec,
                    "source_start_frame_1based": row.source_start_frame_1based,
                    "assessment": row.assessment,
                    "center": label,
                    "signed_turns": float((phase[-1] - phase[0]) / (2 * np.pi)),
                    "speed_sign_changes": int(np.count_nonzero(np.sign(speed[1:]) * np.sign(speed[:-1]) < 0)),
                    "median_abs_speed_rad_s": float(np.median(np.abs(speed))),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--stride", type=int, default=20, help="Number of source frames between diagnostic windows")
    args = parser.parse_args()
    assert args.stride > 0
    args.dest.mkdir(parents=True, exist_ok=True)
    all_windows, registrations, integrity = [], [], []
    for sample in VIDEOS:
        track = read_track(sample)
        windows = evaluate_windows(track, args.stride)
        windows.to_csv(args.dest / f"no{sample:02d}_windows.csv", index=False)
        plot_overview(track, windows, args.dest)
        plot_windows(track, windows, args.dest)
        if sample == 14:
            initial_windows = evaluate_windows(track, args.stride, source="initial")
            initial_windows.to_csv(args.dest / "no14_initial_windows.csv", index=False)
            plot_windows(track, initial_windows, args.dest, source="initial")
        plot_video_frames(track, args.dest)
        all_windows.append(windows)
        registrations.append(video_check(track, args.dest))
        integrity.append(source_integrity(track))
        angle_sensitivity(track, windows).to_csv(args.dest / f"no{sample:02d}_angle_sensitivity.csv", index=False)
        print(sample, track.video, len(track.time), len(windows), windows.assessment.value_counts().to_dict())
    pd.concat(registrations).to_csv(args.dest / "avi_registration_summary.csv", index=False)
    pd.DataFrame(integrity).to_csv(args.dest / "source_integrity_summary.csv", index=False)
    pd.concat(all_windows).groupby(["sample_no", "assessment"]).size().rename("window_count").reset_index().to_csv(
        args.dest / "assessment_counts.csv", index=False
    )


if __name__ == "__main__":
    main()
