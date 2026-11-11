"""Explicit image inputs for the source-aligned repellent measurement workflow.

Local settings describe user choices; immutable CSV/JSON cache pairs describe
measurements. Background extraction and rise detection belong to the user.
"""

import configparser
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils.functions import tiff_roi_io as io
from utils.functions.get_centroid_coordinate import contours
from utils.functions.repellent_workflow import acquisition_policy
from utils.functions.tiff_roi import ROI, Tracker, nominal_max_dt

TIFF_METHOD = "tiff-seeded-fixed-roi-area-centroid-v1"
AVI_METHOD = "avi-largest-contour-area-centroid-threshold-120-v3"


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _resolve(settings_path, value):
    return (Path(settings_path).resolve().parent / value).resolve()


def draft_settings(day, source, data_root, *, px2um=None):
    """List recordings without selecting a target or reading whole recordings."""
    if source not in {"tiff", "avi"}:
        raise ValueError("source must be tiff or avi")
    day_path = Path(day)
    if day_path.is_absolute() or ".." in day_path.parts or day_path.parts[:1] == ("outputs",):
        raise ValueError("day must be a data identifier, without outputs/ or parent traversal")
    root = Path(data_root).resolve() / day
    directories = {p.name: p for p in (root / "tiff_data").iterdir() if p.is_dir()}
    config = configparser.ConfigParser()
    config.read(root / "config.ini")
    ordered = config.get("Tiff_info", "tiff_data", fallback="")
    names = [name.strip() for name in ordered.split(",")] if ordered else sorted(directories)
    if not names or len(set(names)) != len(names) or set(names) != set(directories):
        raise ValueError("explicit TIFF order must identify every recording exactly once")
    if px2um is None and source == "avi":
        px2um = [config.getfloat("Settings", "px2um_x"), config.getfloat("Settings", "px2um_y")]
    if px2um is None or len(px2um) != 2 or not np.isfinite(px2um).all() or min(px2um) <= 0:
        raise ValueError("explicit positive source-pixel calibration required; AVI calibration is not TIFF calibration")
    ranges = config.get("RepellentResponse", "analysis_frame_ranges", fallback="")
    scopes = [item.strip() for item in ranges.split(",")] if ranges else ["auto"] * len(names)
    if len(scopes) != len(names):
        raise ValueError("configured frame ranges and recording count differ")
    samples = []
    for number, (name, scope) in enumerate(zip(names, scopes), 1):
        frame_range = None if scope.lower() in {"auto", "none", "nan", ""} else [int(v) for v in scope.split("-")]
        sample = dict(
            sample_no=number,
            input_id=name,
            tiff_directory=str(directories[name]),
            px2um_x=float(px2um[0]),
            px2um_y=float(px2um[1]),
            analysis_frame_range=frame_range,
            background=None,
        )
        if source == "tiff":
            sample.update(threshold=None, roi=None, seeds={}, target_confirmed=False)
        else:
            avi = root / (name + ".avi")
            if not avi.is_file():
                raise ValueError(f"missing AVI corresponding to recording: {name}")
            sample["avi"] = str(avi)
        samples.append(sample)
    return dict(
        version=1,
        day=day,
        source=source,
        order_confirmed=bool(ordered),
        order_origin="config.ini" if ordered else "directory names; awaiting user confirmation",
        background_policy="external_only",
        samples=samples,
    )


def selected_samples(settings, numbers=None):
    samples = settings["samples"]
    ids = [s["sample_no"] for s in samples]
    if not ids or any(type(n) is not int or n < 1 for n in ids) or len(set(ids)) != len(ids):
        raise ValueError("unique positive original sample numbers required")
    inputs = [s["input_id"] for s in samples]
    if len(set(inputs)) != len(inputs):
        raise ValueError("unique recording identifiers required")
    if numbers is not None and (not numbers or len(set(numbers)) != len(numbers) or not set(numbers) <= set(ids)):
        raise ValueError("sample selection must identify existing original numbers without duplicates")
    return [s for s in samples if numbers is None or s["sample_no"] in numbers]


def _load(path, source=None, day=None):
    settings = json.loads(Path(path).read_text())
    if settings.get("version") != 1 or settings.get("source") not in {"tiff", "avi"}:
        raise ValueError("unsupported image input settings")
    if source is not None and source != settings["source"]:
        raise ValueError("CLI source differs from settings source")
    if day is not None and day != settings["day"]:
        raise ValueError("CLI day differs from settings day")
    if settings.get("background_policy") != "external_only":
        raise ValueError("background must be externally supplied")
    return settings


def choose_threshold(image, seed, initial=2000):
    """Preview the actual 10x mask and explicitly accept its fixed threshold."""
    from utils.functions.centroid_review import display_limits, gray_display
    from utils.functions.tiff_roi import binary_regions

    window = "Threshold: slider; ENTER accept; ESC cancel"
    maximum = int(np.iinfo(image.dtype).max) if np.issubdtype(image.dtype, np.integer) else int(image.max())
    cv2.namedWindow(window)
    getattr(cv2, "createTrackbar")("Threshold", window, max(1, int(initial)), max(1, maximum), lambda value: None)
    original = gray_display(image, display_limits(image))
    try:
        while True:
            threshold = max(1, cv2.getTrackbarPos("Threshold", window))
            mask, _, _ = binary_regions(image, ROI(0, 0, image.shape[1], image.shape[0]), threshold)
            binary = cv2.resize(mask.astype(np.uint8) * 255, (image.shape[1], image.shape[0]))
            canvas = cv2.cvtColor(np.hstack([original, binary]), cv2.COLOR_GRAY2BGR)
            for offset in (0, image.shape[1]):
                cv2.drawMarker(canvas, (int(round(seed[0])) + offset, int(round(seed[1]))), (0, 0, 255))
            cv2.imshow(window, cv2.resize(canvas, None, fx=6, fy=6, interpolation=cv2.INTER_NEAREST))
            key = cv2.waitKey(30) & 255
            if key in (10, 13):
                return float(threshold)
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Threshold confirmation cancelled")
    finally:
        cv2.destroyWindow(window)


def configure_settings(path, *, numbers=None, confirm_order=False):
    """Confirm native-pixel choices; gaps require explicit target reselection."""
    settings = _load(path)
    if settings["order_confirmed"] is not True and not confirm_order:
        raise ValueError("review sample order and pass --confirm-order, or edit the settings explicitly")
    settings["order_confirmed"] = True
    for sample in selected_samples(settings, numbers):
        if settings["source"] == "avi":
            continue
        rows = io.directory_rows(_resolve(path, sample["tiff_directory"]))
        image = io.load_image(rows[0]["path"])
        seed = io.pick_target(image)
        threshold = choose_threshold(image, seed, sample.get("threshold") or 2000)
        configured = dict(sample, frames=rows, seeds={str(rows[0]["source_frame_1based"]): seed}, threshold=threshold)
        roi, seeds, _, _, _ = io.prepare(configured, interactive=True)
        maximum = nominal_max_dt(rows)
        for previous, row in zip(rows, rows[1:]):
            if row["time_sec"] is not None and io.recording_boundary(previous, row, maximum):
                seeds[str(row["source_frame_1based"])] = io.pick_target(io.load_image(row["path"]))
        for row in rows:
            seed = seeds.get(str(row["source_frame_1based"]))
            if seed is not None:
                result = Tracker(roi, threshold, max_dt=maximum).step(
                    io.load_image(row["path"]), row["source_frame_1based"], row["time_sec"], row["segment_id"], seed
                )
                if result.point is None:
                    raise ValueError(f"target seed is invalid inside the fixed ROI: {result.reason}")
        sample.update(
            threshold=threshold, roi=[roi.x, roi.y, roi.width, roi.height], seeds=seeds, target_confirmed=True
        )
        _write(path, settings)
    _write(path, settings)
    return settings


def _scope(raw, sample, all_frames):
    bounds = None if all_frames else sample.get("analysis_frame_range")
    if bounds is None:
        return raw.copy()
    if (
        len(bounds) != 2
        or any(type(n) is not int or n < 1 for n in bounds)
        or bounds[0] > bounds[1]
        or not set(bounds) <= set(raw.source_frame_1based)
    ):
        raise ValueError("analysis range endpoints must be existing original frames")
    return raw[raw.source_frame_1based.between(*bounds)].copy()


def _extract(sample, rows, source, settings_path):
    records = []
    capture = None
    selection = None
    if source == "tiff":
        io.validate_sample(dict(sample, frames=rows))
        tracker = Tracker(ROI(*sample["roi"]), sample["threshold"], max_dt=nominal_max_dt(rows))
    else:
        capture = cv2.VideoCapture(str(_resolve(settings_path, sample["avi"])))
        if not capture.isOpened():
            capture.release()
            raise ValueError("could not open source AVI")
    try:
        for index, row in enumerate(rows):
            if source == "tiff":
                stamp = np.nan if row["time_sec"] is None else row["time_sec"]
                seed = sample["seeds"].get(str(row["source_frame_1based"]))
                if seed is not None:
                    selection = row["source_frame_1based"]
                result = tracker.step(
                    io.load_image(row["path"]),
                    row["source_frame_1based"],
                    stamp,
                    row["segment_id"],
                    seed,
                )
                x, y = result.point if result.point is not None else (np.nan, np.nan)
                detected, reason, reacquired = result.point is not None, result.reason, result.reacquired
            else:
                assert capture is not None
                ok, image = capture.read()
                if not ok:
                    raise ValueError("AVI/TIFF observation counts differ; original mapping cannot be assumed")
                x, y, _ = contours(image)
                detected = bool(np.isfinite([x, y]).all() and row["time_sec"] is not None)
                if not detected:
                    x = y = np.nan
                reason, reacquired = ("" if detected else "missing_centroid_or_time"), False
            records.append(
                dict(
                    source_frame_1based=row["source_frame_1based"],
                    time_sec=row["time_sec"],
                    source_timestamp=row["source_timestamp"],
                    segment_id=row["segment_id"],
                    x_px=x,
                    y_px=y,
                    x_um=x * sample["px2um_x"],
                    y_um=y * sample["px2um_y"],
                    detected=detected,
                    reason=reason,
                    reacquired=reacquired,
                    target_id=f"{sample['input_id']}@selection-{selection}" if source == "tiff" else sample["input_id"],
                    source_input=row["path"],
                    avi_frame_1based=index + 1 if source == "avi" else None,
                )
            )
        if capture is not None and capture.read()[0]:
            raise ValueError("AVI/TIFF observation counts differ; original mapping cannot be assumed")
    finally:
        if capture is not None:
            capture.release()
    return pd.DataFrame(records)


def prepare_measurements(path, output_root, *, source, day, numbers=None, all_frames=False, force=False):
    """Generate canonical measurements and a manifest, without touching legacy caches."""
    path, output_root = Path(path).resolve(), Path(output_root).resolve()
    settings = _load(path, source, day)
    if settings.get("order_confirmed") is not True:
        raise ValueError("recording order has not been confirmed")
    samples = selected_samples(settings, numbers)
    for sample in samples:
        if source == "tiff" and (
            sample.get("target_confirmed") is not True
            or sample.get("threshold") is None
            or sample.get("roi") is None
            or not sample.get("seeds")
        ):
            raise ValueError(f"No.{sample['sample_no']}: confirm target, threshold and fixed ROI first")
        if (
            not np.isfinite([sample["px2um_x"], sample["px2um_y"]]).all()
            or min(sample["px2um_x"], sample["px2um_y"]) <= 0
        ):
            raise ValueError("source-pixel calibration must be positive")
    method = TIFF_METHOD if source == "tiff" else AVI_METHOD
    code_paths = [
        Path(__file__),
        Path(io.__file__),
        Path(io.__file__).with_name("tiff_roi.py"),
        Path(io.__file__).with_name("centroid_diagnostics.py"),
        Path(io.__file__).with_name("get_centroid_coordinate.py"),
    ]
    code = {p.name: _hash(p) for p in code_paths}
    input_hashes = {str(path): _hash(path)}
    entries = []
    for sample in samples:
        rows = io.directory_rows(_resolve(path, sample["tiff_directory"]))
        policy = acquisition_policy(rows[0]["path"])
        paths = [Path(row["path"]) for row in rows]
        if source == "avi":
            paths.append(_resolve(path, sample["avi"]))
        hashes = {str(p): _hash(p) for p in paths}
        input_hashes.update(hashes)
        provenance = dict(
            source=source,
            method=method,
            sample={name: value for name, value in sample.items() if name != "background"},
            input_hashes=hashes,
            code_sha256=code,
            all_frames=all_frames,
            acquisition_policy=policy,
            runtime_versions={
                name: version(name) for name in ("opencv-python", "numpy", "pandas", "Pillow", "tifffile")
            },
        )
        key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
        cache = output_root / "raw_centroid" / method / key
        csv, metadata = cache.with_suffix(".csv"), cache.with_suffix(".json")
        reusable = False
        if not force and csv.is_file() and metadata.is_file():
            try:
                saved = json.loads(metadata.read_text())
                reusable = saved.get("provenance") == provenance and saved.get("csv_sha256") == _hash(csv)
            except (ValueError, OSError):
                reusable = False
        if not reusable:
            raw = _scope(_extract(sample, rows, source, path), sample, all_frames).reset_index(drop=True)
            cache.parent.mkdir(parents=True, exist_ok=True)
            temporary = csv.with_suffix(".csv.tmp")
            raw.to_csv(temporary, index=False, float_format="%.17g")
            temporary.replace(csv)
            _write(
                metadata,
                dict(
                    sample_no=sample["sample_no"],
                    frame_count=len(raw),
                    method=method,
                    px2um_x=sample["px2um_x"],
                    px2um_y=sample["px2um_y"],
                    source_id=sample["input_id"],
                    provenance=provenance,
                    csv_sha256=_hash(csv),
                ),
            )
        raw = pd.read_csv(csv, float_precision="round_trip")
        entry = dict(
            sample_no=sample["sample_no"],
            centroids=str(csv),
            metadata=str(metadata),
            acquisition_tiff=rows[0]["path"],
            identity=(
                "user selected observed target within continuous runs; unverified physical identity"
                if source == "tiff"
                else "AVI ordinal matched to TIFF suffix order by recording name and count; unverified target identity"
            ),
        )
        background = sample.get("background")
        if background is not None:
            bg_path = _resolve(path, background["csv"])
            input_hashes[str(bg_path)] = _hash(bg_path)
            bg = pd.read_csv(bg_path, float_precision="round_trip")
            bg = _scope(bg, sample, all_frames)
            if not np.array_equal(bg.source_frame_1based, raw.source_frame_1based) or not np.array_equal(
                bg.time_sec.to_numpy(), raw.time_sec.to_numpy(), equal_nan=True
            ):
                raise ValueError("external background must match original frames and timestamps exactly")
            bg_cache = cache.with_suffix(".background.csv")
            bg_key = hashlib.sha256((key + _hash(bg_path)).encode()).hexdigest()
            bg_cache = bg_cache.with_name(bg_key + ".background.csv")
            bg.to_csv(bg_cache, index=False, float_format="%.17g")
            entry.update(
                background=str(bg_cache),
                background_time_column="time_sec",
                background_column="background_intensity",
                rise_source_frame=background.get("rise_source_frame"),
            )
        entries.append(entry)
    if any(_hash(name) != value for name, value in input_hashes.items()):
        raise RuntimeError("input changed during image extraction")
    manifest = dict(
        samples=entries,
        image_input=dict(
            source=source,
            method=method,
            input_hashes=input_hashes,
            source_settings=str(path),
            background_policy="external_only",
        ),
    )
    for option in ("details", "samples_per_page", "zoom_ranges"):
        if option in settings:
            manifest[option] = settings[option]
    manifest_key = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    destination = output_root / "measurement_input" / (source + "-" + manifest_key + ".json")
    _write(destination, manifest)
    return destination
