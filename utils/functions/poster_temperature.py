"""Read existing SJW46 temperature tables without changing source-frame identity."""

from __future__ import annotations

import configparser
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from utils.functions.frequency_analysis import CONTINUITY_GAP_FACTOR
from utils.functions.poster_fluctuation import WINDOWS_S, analyze, band_amplitude_ratios


@dataclass(frozen=True)
class TemperatureSources:
    temperature_c: int
    frame_rate_hz: float
    sample_count: int
    time: pd.DataFrame
    centered_centroid: pd.DataFrame
    corrected_center: pd.DataFrame
    original_center: pd.DataFrame
    pixel_size_um: tuple[float, float]


def source_paths(source_root: Path, config_root: Path, temperature_c: int) -> tuple[Path, ...]:
    label = f"SJW46_temp={temperature_c}"
    folder = source_root / label
    return (
        config_root / label / "config.ini",
        folder / "time_list.csv",
        folder / "centroid_coordinate.csv",
        folder / "center_coordinate/bef_correction/center_coordinate.csv",
        folder / "center_coordinate/bef_correction/center_coordinate_bef_correct.csv",
    )


def read_sources(source_root: Path, config_root: Path, temperature_c: int) -> TemperatureSources:
    paths = source_paths(source_root, config_root, temperature_c)
    config = configparser.ConfigParser()
    if not config.read(paths[0]):
        raise FileNotFoundError(paths[0])
    count = config.getint("Settings", "sample_num")
    frame_rate = config.getfloat("Settings", "FrameRate")
    pixel_size = (config.getfloat("Settings", "px2um_x"), config.getfloat("Settings", "px2um_y"))
    if count < 1 or not np.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"{temperature_c}°C: invalid sample count or frame rate")
    if not np.isfinite(pixel_size).all() or min(pixel_size) <= 0:
        raise ValueError(f"{temperature_c}°C: invalid pixel calibration")
    time, centroid, center, before = (pd.read_csv(path) for path in paths[1:])
    for no in range(1, count + 1):
        required = (
            (time, f"No.{no}"),
            (centroid, f"x_{no}"),
            (centroid, f"y_{no}"),
            (center, f"No.{no}_x"),
            (center, f"No.{no}_y"),
            (before, f"No.{no}_x"),
            (before, f"No.{no}_y"),
        )
        for table, column in required:
            if column not in table:
                raise ValueError(f"{temperature_c}°C No.{no}: missing source column {column}")
    if len(time.columns) != count:
        raise ValueError(f"{temperature_c}°C: time-list cell count disagrees with config")
    return TemperatureSources(temperature_c, frame_rate, count, time, centroid, center, before, pixel_size)


def _aligned(table: pd.DataFrame, column: str, count: int) -> np.ndarray:
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(float)
    if len(values) < count:
        values = np.pad(values, (0, count - len(values)), constant_values=np.nan)
    return values[:count]


def cell_coordinates(sources: TemperatureSources, sample_no: int) -> pd.DataFrame:
    """Return observed centered and absolute coordinates on original time rows."""
    if sample_no < 1 or sample_no > sources.sample_count:
        raise ValueError("sample number is outside the temperature condition")
    full_time = pd.to_numeric(sources.time[f"No.{sample_no}"], errors="coerce").to_numpy(float)
    finite_rows = np.flatnonzero(np.isfinite(full_time))
    if len(finite_rows) == 0:
        raise ValueError(f"{sources.temperature_c}°C No.{sample_no}: no finite timestamps")
    count = int(finite_rows[-1] + 1)
    time = full_time[:count]
    x = _aligned(sources.centered_centroid, f"x_{sample_no}", count)
    y = _aligned(sources.centered_centroid, f"y_{sample_no}", count)
    cx = _aligned(sources.corrected_center, f"No.{sample_no}_x", count)
    cy = _aligned(sources.corrected_center, f"No.{sample_no}_y", count)
    before_x = _aligned(sources.original_center, f"No.{sample_no}_x", count)
    before_y = _aligned(sources.original_center, f"No.{sample_no}_y", count)
    valid = np.isfinite(time) & np.isfinite(x) & np.isfinite(y) & np.isfinite(cx) & np.isfinite(cy)
    changed = (
        np.isfinite(cx)
        & np.isfinite(cy)
        & (
            ~np.isfinite(before_x)
            | ~np.isfinite(before_y)
            | ~np.isclose(cx, before_x, atol=1e-9, rtol=0)
            | ~np.isclose(cy, before_y, atol=1e-9, rtol=0)
        )
    )
    centered_x = np.where(valid, x, np.nan)
    centered_y = np.where(valid, y, np.nan)
    return pd.DataFrame(
        {
            "source_frame_1based": np.arange(1, count + 1),
            "time_sec": time,
            "centroid_x_um": np.where(valid, x + cx, np.nan),
            "centroid_y_um": np.where(valid, y + cy, np.nan),
            "center_x_um": np.where(valid, cx, np.nan),
            "center_y_um": np.where(valid, cy, np.nan),
            "centered_x_um": centered_x,
            "centered_y_um": centered_y,
            "radius_um": np.hypot(centered_x, centered_y),
            "source_center_adjusted": changed,
        }
    )


def local_conic_coordinates(sources: TemperatureSources, sample_no: int) -> tuple[pd.DataFrame, dict]:
    """Refit the repellent default center using the same saved raw centroids."""
    from utils.functions.repellent_workflow import analyze_sample

    stored = cell_coordinates(sources, sample_no)
    time = stored["time_sec"].to_numpy(float)
    x = stored["centroid_x_um"].to_numpy(float)
    y = stored["centroid_y_um"].to_numpy(float)
    dt = np.diff(time)
    positive_dt = dt[np.isfinite(dt) & (dt > 0)]
    if not len(positive_dt):
        raise ValueError(f"{sources.temperature_c}°C No.{sample_no}: no positive frame interval")
    # The original standard output stores (raw centroid - corrected center).
    # Re-add that center, then use the configured pixel scale for the same
    # pixel-space fit used by the repellent local_conic workflow.
    raw = pd.DataFrame(
        {
            "source_frame_1based": stored["source_frame_1based"],
            "time_sec": time,
            "x_um": x,
            "y_um": y,
            "x_px": x / sources.pixel_size_um[0],
            "y_px": y / sources.pixel_size_um[1],
            "detected": np.isfinite(x) & np.isfinite(y),
        }
    )
    observation, _, method = analyze_sample(
        raw,
        sample_no=sample_no,
        source_id=f"SJW46_temp={sources.temperature_c}",
        centroid_method="stored_standard_centroid_reconstructed_from_centered_output",
        frame_period_s=float(np.median(positive_dt)),
        timestamp_quantum_s=1e-6,
        identity="one original AVI per cell; no track switching assumed",
    )
    if not np.array_equal(
        observation["source_frame_1based"].to_numpy(int), stored["source_frame_1based"].to_numpy(int)
    ) or not np.allclose(observation["time_sec"], time, equal_nan=True):
        raise ValueError("local_conic output changed source frame or timestamp identity")
    frame = stored.copy()
    frame["stored_center_x_um"] = stored["center_x_um"]
    frame["stored_center_y_um"] = stored["center_y_um"]
    frame["center_x_um"] = observation["center_x_um"].to_numpy(float)
    frame["center_y_um"] = observation["center_y_um"].to_numpy(float)
    frame["centered_x_um"] = x - frame["center_x_um"]
    frame["centered_y_um"] = y - frame["center_y_um"]
    frame["radius_um"] = np.hypot(frame["centered_x_um"], frame["centered_y_um"])
    frame["local_conic_center_status"] = observation["center_status"].to_numpy()
    frame["local_conic_center_estimate_status"] = observation["center_estimate_status"].to_numpy()
    frame["local_conic_center_estimate_reason"] = observation["center_estimate_reason"].to_numpy()
    frame["local_conic_center_estimate_window_s"] = observation["center_estimate_window_s"].to_numpy()
    frame["local_conic_center_estimate_degree"] = observation["center_estimate_degree"].to_numpy()
    frame["local_conic_center_review_required"] = observation["center_review_required"].to_numpy()
    return frame, method


def orbit_kinematics(frame: pd.DataFrame) -> pd.DataFrame:
    """Calculate CCW-positive orbit velocity from centered image coordinates.

    Coordinates retain the image's downward-positive y axis. Each signed
    increment is wrapped to [-pi, pi); a time gap or absent source row yields
    a missing velocity. The plotted angle is the instantaneous orbit phase.
    """
    result = frame.copy()
    time = result["time_sec"].to_numpy(float)
    x = result["centered_x_um"].to_numpy(float)
    y = result["centered_y_um"].to_numpy(float)
    valid = np.isfinite(time) & np.isfinite(x) & np.isfinite(y) & (np.hypot(x, y) > 0)
    phase = np.full(len(time), np.nan)
    raw_phase = np.arctan2(-y[valid], x[valid])
    phase[valid] = (raw_phase + np.pi) % (2 * np.pi) - np.pi
    omega = np.full(len(time), np.nan)
    dt = np.diff(time)
    positive = dt[np.isfinite(dt) & (dt > 0)]
    nominal_dt = float(np.median(positive)) if len(positive) else np.nan
    if len(time) > 1 and np.isfinite(nominal_dt):
        good = (
            np.isfinite(phase[1:])
            & np.isfinite(phase[:-1])
            & np.isfinite(dt)
            & (dt > 0)
            & (dt <= CONTINUITY_GAP_FACTOR * nominal_dt)
        )
        delta = (np.diff(phase) + np.pi) % (2 * np.pi) - np.pi
        omega[np.flatnonzero(good) + 1] = delta[good] / dt[good]
    result["orbit_angle_rad"] = phase
    result["orbit_omega_rad_s"] = omega
    result["orbit_abs_omega_rad_s"] = np.abs(omega)
    result["valid_velocity"] = np.isfinite(omega)
    return result


def analyze_cell(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    """Apply the established poster Gaussian-window analysis to orbit speed."""
    kinematics = orbit_kinematics(frame)
    measured = analyze(
        kinematics["time_sec"].to_numpy(float),
        kinematics["orbit_omega_rad_s"].to_numpy(float),
        WINDOWS_S,
    )
    for column in measured.frame:
        if column not in ("time_sec", "angular_velocity_rad_s"):
            kinematics[column] = measured.frame[column]
    ratios = band_amplitude_ratios(measured.spectra)
    return kinematics, measured.spectra, ratios, measured.metadata


def cell_summary(frame: pd.DataFrame, ratios: pd.DataFrame, temperature_c: int, sample_no: int) -> dict:
    """Summarize one cell without treating within-cell frames as replicates."""
    speed = frame["orbit_abs_omega_rad_s"].to_numpy(float)
    radius = frame["radius_um"].to_numpy(float)
    nominal_dt = np.diff(frame["time_sec"].to_numpy(float))
    positive_dt = nominal_dt[np.isfinite(nominal_dt) & (nominal_dt > 0)]
    ratio_half_second = ratios.loc[np.isclose(ratios["window_s"], 0.5), "low_to_high_amplitude_ratio"]
    valid_speed = speed[np.isfinite(speed)]
    valid_radius = radius[np.isfinite(radius)]
    return {
        "temperature_c": temperature_c,
        "sample_no": sample_no,
        "source_frame_count": len(frame),
        "valid_velocity_count": int(np.isfinite(speed).sum()),
        "timestamp_gap_count": (
            int(np.sum(nominal_dt > CONTINUITY_GAP_FACTOR * np.median(positive_dt))) if len(positive_dt) else 0
        ),
        "corrected_center_frame_count": int(frame["source_center_adjusted"].sum()),
        "mean_absolute_angular_velocity_rad_s": float(np.mean(valid_speed)) if len(valid_speed) else float("nan"),
        "mean_rotation_radius_um": float(np.mean(valid_radius)) if len(valid_radius) else float("nan"),
        "low_to_high_amplitude_ratio_0.5s": float(ratio_half_second.iloc[0]),
    }
