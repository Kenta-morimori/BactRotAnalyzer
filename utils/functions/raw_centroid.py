"""Canonical, center-independent centroid measurements from source AVIs."""

import hashlib
import json
import os
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from utils import param
from utils.functions import input_data
from utils.functions.get_centroid_coordinate import contours

METHOD = "standard-largest-contour-mean-threshold-120-v2"
COLUMNS = ["source_frame_1based", "time_sec", "x_px", "y_px", "x_um", "y_um", "ellipse_angle_deg", "detected"]


def source_path(day: str, sample_no: int) -> Path:
    paths = input_data.get_ordered_avi_paths(day)
    if not 1 <= sample_no <= len(paths):
        raise ValueError(f"No.{sample_no} is outside the AVI sample map")
    return Path(paths[sample_no - 1])


def coordinate_path(day: str, sample_no: int) -> Path:
    return Path(param.save_dir_bef) / day / "raw_centroid" / f"{source_path(day, sample_no).stem}.csv"


def _provenance(day: str, sample_no: int, frame_count: int) -> dict:
    avi = source_path(day, sample_no)
    stat = avi.stat()
    px2um_x, px2um_y = param.get_px2um_config(day)
    return {
        "method": METHOD,
        "avi_filename": avi.name,
        "avi_size": stat.st_size,
        "avi_mtime_ns": stat.st_mtime_ns,
        "sample_no": sample_no,
        "frame_count": frame_count,
        "px2um_x": px2um_x,
        "px2um_y": px2um_y,
    }


def extract_or_load(day: str, sample_no: int, time_values: Sequence[float], force: bool = False) -> pd.DataFrame:
    """Extract once, then reuse only when AVI, calibration and frame axis match."""
    path = coordinate_path(day, sample_no)
    meta_path = path.with_suffix(".json")
    expected = _provenance(day, sample_no, len(time_values))
    if path.exists() and meta_path.exists() and not force:
        with meta_path.open(encoding="utf-8") as stream:
            saved = json.load(stream)
        if saved == expected:
            return load(day, sample_no, time_values)

    capture = cv2.VideoCapture(str(source_path(day, sample_no)))
    if not capture.isOpened():
        raise OSError(f"Cannot open AVI: {source_path(day, sample_no)}")
    px2um_x, px2um_y = param.get_px2um_config(day)
    rows: list[tuple[int, float, float, float, float, float, float, bool]] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            x_px, y_px, ellipse = contours(frame)
            detected = bool(np.isfinite(x_px) and np.isfinite(y_px))
            index = len(rows)
            rows.append(
                (
                    index + 1,
                    float(time_values[index]) if index < len(time_values) else np.nan,
                    x_px,
                    y_px,
                    x_px * px2um_x,
                    y_px * px2um_y,
                    ellipse[2] if ellipse is not None else np.nan,
                    detected,
                )
            )
    finally:
        capture.release()
    if len(rows) != len(time_values):
        raise ValueError(f"No.{sample_no} AVI has {len(rows)} frames, TIFF time list has {len(time_values)}")
    frame = pd.DataFrame(rows, columns=COLUMNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    csv_tmp = path.with_suffix(".csv.tmp")
    meta_tmp = meta_path.with_suffix(".json.tmp")
    frame.to_csv(csv_tmp, index=False, float_format="%.17g")
    with meta_tmp.open("w", encoding="utf-8") as stream:
        json.dump(expected, stream, indent=2)
    os.replace(csv_tmp, path)
    os.replace(meta_tmp, meta_path)
    return load(day, sample_no, time_values)


def load(day: str, sample_no: int, time_values: Sequence[float]) -> pd.DataFrame:
    path = coordinate_path(day, sample_no)
    frame = pd.read_csv(path)
    expected = _provenance(day, sample_no, len(time_values))
    with path.with_suffix(".json").open(encoding="utf-8") as stream:
        saved = json.load(stream)
    if saved != expected or list(frame.columns) != COLUMNS:
        raise ValueError(f"Stale or malformed raw centroid: {path}")
    frame_no = np.arange(1, len(time_values) + 1)
    if not np.array_equal(frame["source_frame_1based"].to_numpy(), frame_no):
        raise ValueError(f"Frame numbers do not match TIFF time list: {path}")
    if not np.allclose(frame["time_sec"].to_numpy(dtype=float), time_values, atol=1e-9, rtol=0):
        raise ValueError(f"Timestamps do not match TIFF time list: {path}")
    if not np.array_equal(
        frame["detected"].to_numpy(dtype=bool),
        np.isfinite(frame["x_px"].to_numpy(dtype=float)) & np.isfinite(frame["y_px"].to_numpy(dtype=float)),
    ):
        raise ValueError(f"Detection flags do not match centroid values: {path}")
    return frame


def load_all(day: str, time_list: Sequence[Sequence[float]], extract_missing: bool = True):
    if len(time_list) != len(input_data.get_ordered_avi_paths(day)):
        raise ValueError("TIFF time series and AVI sample counts differ")
    frames = []
    for sample_no, times in enumerate(time_list, start=1):
        if extract_missing:
            frames.append(extract_or_load(day, sample_no, times))
        else:
            frames.append(load(day, sample_no, times))
    return frames


def _standard_manifest(day: str, time_list: Sequence[Sequence[float]]) -> list[dict]:
    entries = []
    for sample_no, times in enumerate(time_list, start=1):
        entry = _provenance(day, sample_no, len(times))
        entry["time_sha256"] = hashlib.sha256(np.asarray(times, dtype="<f8").tobytes()).hexdigest()
        entries.append(entry)
    return entries


def save_standard_manifest(day: str, time_list: Sequence[Sequence[float]]) -> None:
    path = Path(param.save_dir_bef) / day / "centroid_coordinate_source.json"
    with path.open("w", encoding="utf-8") as stream:
        json.dump(_standard_manifest(day, time_list), stream, indent=2)


def standard_cache_matches(day: str, time_list: Sequence[Sequence[float]]) -> bool:
    root = Path(param.save_dir_bef) / day
    path = root / "centroid_coordinate_source.json"
    if not (root / "centroid_coordinate.csv").exists() or not path.exists():
        return False
    try:
        with path.open(encoding="utf-8") as stream:
            saved = json.load(stream)
        return saved == _standard_manifest(day, time_list)
    except (OSError, ValueError, KeyError):
        return False
