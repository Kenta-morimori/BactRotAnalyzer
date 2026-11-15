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
    if count < 1 or not np.isfinite(frame_rate) or frame_rate <= 0:
        raise ValueError(f"{temperature_c}°C: invalid sample count or frame rate")
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
    return TemperatureSources(temperature_c, frame_rate, count, time, centroid, center, before)


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
    return {
        "temperature_c": temperature_c,
        "sample_no": sample_no,
        "source_frame_count": len(frame),
        "valid_velocity_count": int(np.isfinite(speed).sum()),
        "timestamp_gap_count": (
            int(np.sum(nominal_dt > CONTINUITY_GAP_FACTOR * np.median(positive_dt))) if len(positive_dt) else 0
        ),
        "corrected_center_frame_count": int(frame["source_center_adjusted"].sum()),
        "mean_absolute_angular_velocity_rad_s": float(np.nanmean(speed)),
        "mean_rotation_radius_um": float(np.nanmean(radius)),
        "low_to_high_amplitude_ratio_0.5s": float(ratio_half_second.iloc[0]),
    }
