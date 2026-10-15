"""Align the canonical No.14 centroid, legacy center, angle, and angular velocity."""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from issue38_paths import FIGURES, TABLES, ensure_output_dirs  # noqa: E402

SOURCE = TABLES / "no14_centroid_center_timeseries.csv"
OUTPUT = TABLES / "no14_legacy_angle_velocity_timeseries.csv"
RISE_TIME_SEC = 40.0
TERMINAL_HOLD_SEC = 82.5


def calculate_angle_velocity(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Use the existing analysis sign and one-frame-difference convention."""
    x = frame["x_um"].to_numpy(float) - frame["center_x_um"].to_numpy(float)
    y = frame["y_um"].to_numpy(float) - frame["center_y_um"].to_numpy(float)
    time = frame["time_sec"].to_numpy(float)
    angle = np.arctan2(y, x)
    velocity = np.full(len(frame), np.nan)
    dt = np.diff(time)
    valid_dt = dt[np.isfinite(dt) & (dt > 0)]
    nominal_dt = float(np.median(valid_dt)) if valid_dt.size else np.nan
    valid = np.isfinite(angle[:-1]) & np.isfinite(angle[1:]) & np.isfinite(dt) & (dt > 0)
    if np.isfinite(nominal_dt):
        valid &= dt <= 1.5 * nominal_dt
    angle_diff = np.diff(angle)
    angle_diff = np.where(
        angle_diff > np.pi,
        angle_diff - 2 * np.pi,
        np.where(angle_diff < -np.pi, angle_diff + 2 * np.pi, angle_diff),
    )
    velocity[np.flatnonzero(valid)] = -angle_diff[valid] / dt[valid]
    return angle, velocity


def plot(frame: pd.DataFrame, angle: np.ndarray, velocity: np.ndarray, path, bounds=None) -> None:
    time = frame["time_sec"].to_numpy(float)
    fig, axes = plt.subplots(4, 1, figsize=(17, 10), sharex=True, constrained_layout=True)
    for axis, coordinate in zip(axes[:2], "xy"):
        axis.plot(time, frame[f"{coordinate}_um"], color="#4c6475", lw=0.5, alpha=0.65, rasterized=True)
        axis.plot(time, frame[f"center_{coordinate}_um"], color="#db6a28", lw=1.0, rasterized=True)
        axis.set_ylabel(f"{coordinate} (µm)")
        axis.grid(alpha=0.2)
    axes[0].legend(["centroid", "legacy center"], loc="upper right", ncol=2, frameon=False)
    axes[2].plot(time, angle, color="#3d876f", lw=0.55, rasterized=True)
    axes[2].set_ylabel("angle (rad)")
    axes[2].set_ylim(-np.pi, np.pi)
    axes[2].grid(alpha=0.2)
    axes[3].plot(time, velocity, color="#6654a4", lw=0.5, rasterized=True)
    axes[3].axhline(0, color="#333333", lw=0.7)
    axes[3].set_ylabel("signed ω (rad/s)")
    axes[3].set_xlabel("time in source AVI (s)")
    axes[3].grid(alpha=0.2)
    for axis in axes:
        axis.axvline(RISE_TIME_SEC, color="#9a4ba6", ls=":", lw=0.9)
        axis.axvspan(TERMINAL_HOLD_SEC, time[-1], color="#db6a28", alpha=0.08, lw=0)
        if bounds is not None:
            axis.set_xlim(*bounds)
    title = "No.14 | canonical centroid and legacy center; raw orbital angle and signed velocity"
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    frame = pd.read_csv(SOURCE)
    if len(frame) != 14336 or not np.array_equal(frame["source_frame_1based"], np.arange(2049, 16385)):
        raise ValueError("No.14 frame correspondence differs from the canonical 2049–16384 range")
    angle, velocity = calculate_angle_velocity(frame)
    pd.DataFrame(
        {
            "source_frame_1based": frame["source_frame_1based"],
            "time_sec": frame["time_sec"],
            "angle_rad": angle,
            "angular_velocity_rad_s": velocity,
        }
    ).to_csv(OUTPUT, index=False)
    plot(frame, angle, velocity, FIGURES / "no14_legacy_centroid_center_angle_velocity.png")
    plot(frame, angle, velocity, FIGURES / "no14_legacy_centroid_center_angle_velocity_38_42s.png", (38, 42))
    print(f"No.14: {len(frame)} aligned frames; {np.count_nonzero(np.isfinite(velocity))} valid intervals")


if __name__ == "__main__":
    main()
