"""Local CSV/movie export and optional one-click setup for fixed-ROI tracking."""

import csv
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from utils.functions import centroid_diagnostics as diagnostic
from utils.functions import centroid_review as review
from utils.functions.tiff_roi import ROI, Tracker, infer_roi, nominal_max_dt


def load_image(path):
    with Image.open(path) as plane:
        if getattr(plane, "n_frames", 1) != 1:
            raise ValueError("Each row must identify a single TIFF plane")
        image = np.array(plane)
    if image.ndim != 2:
        raise ValueError("Expected a grayscale TIFF")
    return image


def directory_rows(folder):
    paths = list(Path(folder).glob("*.tif"))
    if not paths:
        raise ValueError("No single-plane .tif files found")
    original = diagnostic.timeline(paths, 1.0)
    rows = [
        dict(
            path=r["tiff_path"],
            source_frame_1based=r["tiff_suffix_0based"] + 1,
            time_sec=r["time_sec"],
            source_timestamp=r["source_timestamp"],
            segment_id=0,
        )
        for r in original
    ]
    maximum = nominal_max_dt(rows)
    segment = 0
    for index, row in enumerate(rows):
        if index:
            previous = rows[index - 1]
            a, b = previous["time_sec"], row["time_sec"]
            if (
                row["source_frame_1based"] != previous["source_frame_1based"] + 1
                or a is None
                or b is None
                or not 0 < b - a <= maximum
            ):
                segment += 1
        row["segment_id"] = segment
    return rows


def validate_sample(sample):
    rows = sample["frames"]
    ids = [r["source_frame_1based"] for r in rows]
    if not ids or any(type(value) is not int or value < 1 for value in ids) or np.any(np.diff(ids) <= 0):
        raise ValueError("Original frame IDs must be increasing positive integers")
    if len({str(Path(row["path"]).resolve()) for row in rows}) != len(rows):
        raise ValueError("One source file cannot represent multiple observations")
    for row in rows:
        if type(row["segment_id"]) is not int:
            raise ValueError("Recording segment IDs must be integers")
        if row["time_sec"] is not None and not np.isfinite(row["time_sec"]):
            raise ValueError("Unavailable timestamps must be null")
    seeds = sample.get("seeds", {})
    if any(str(int(key)) != key or int(key) not in ids for key in seeds):
        raise ValueError("Seeds must name existing original frames with canonical string IDs")
    if not np.isfinite(sample.get("threshold", 2000)) or sample.get("threshold", 2000) <= 0:
        raise ValueError("Threshold must be finite and positive")


def pick_target(image):
    """One click in original-image coordinates; Escape cancels setup."""
    zoom = 8
    gray = review.gray_display(image, review.display_limits(image))
    displayed = cv2.resize(gray, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    window = "Click target foreground; ESC cancel"
    point: list[float] = []

    def callback(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            point[:] = [(x + 0.5) / zoom - 0.5, (y + 0.5) / zoom - 0.5]

    cv2.namedWindow(window)
    getattr(cv2, "setMouseCallback")(window, callback)
    try:
        while not point:
            cv2.imshow(window, displayed)
            if cv2.waitKey(30) & 255 == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("Target selection cancelled")
        return point
    finally:
        cv2.destroyWindow(window)


def confirm_roi(image, roi):
    """Enter accepts; R edits the fixed rectangle, Escape cancels."""
    zoom = 8
    raw = cv2.cvtColor(review.gray_display(image, review.display_limits(image)), cv2.COLOR_GRAY2BGR)
    raw = cv2.resize(raw, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST)
    window = "ROI: ENTER accept, R edit, ESC cancel"
    try:
        while True:
            canvas = raw.copy()
            if roi is not None:
                cv2.rectangle(
                    canvas,
                    (roi.x * zoom, roi.y * zoom),
                    ((roi.x + roi.width) * zoom - 1, (roi.y + roi.height) * zoom - 1),
                    (0, 220, 255),
                    1,
                )
            cv2.imshow(window, canvas)
            key = cv2.waitKey(30) & 255
            if key == 27 or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise ValueError("ROI confirmation cancelled")
            if key in (10, 13) and roi is not None:
                return roi
            if key in (ord("r"), ord("R")) or roi is None:
                x, y, w, h = cv2.selectROI(window, raw, showCrosshair=True, fromCenter=False)
                if not w or not h:
                    raise ValueError("ROI selection cancelled")
                if w and h:
                    x0, y0 = int(x // zoom), int(y // zoom)
                    right, bottom = min(image.shape[1], int(np.ceil((x + w) / zoom))), min(
                        image.shape[0], int(np.ceil((y + h) / zoom))
                    )
                    roi = ROI(x0, y0, right - x0, bottom - y0)
    finally:
        cv2.destroyAllWindows()


def prepare(sample, interactive=False):
    rows = sample["frames"]
    images = [load_image(row["path"]) for row in rows[:200]]
    first = str(rows[0]["source_frame_1based"])
    seeds = dict(sample.get("seeds", {}))
    if first not in seeds:
        if not interactive:
            raise ValueError("Specify an initial target point or use --interactive")
        seeds[first] = pick_target(images[0])
    threshold = sample.get("threshold", 2000)
    if sample.get("roi") is None:
        roi, calibration = infer_roi(images, rows[: len(images)], seeds[first], threshold)
    else:
        roi = ROI(*sample["roi"])
        calibration = dict(reason="manual")
    if interactive:
        chosen = confirm_roi(images[0], roi)
        if roi != chosen:
            calibration = dict(calibration, manual_override=True)
        roi = chosen
    if roi is None:
        raise ValueError("Automatic ROI unavailable: specify roi=[x,y,width,height] or use --interactive")
    roi.validate(images[0].shape)
    return roi, seeds, calibration, review.display_limits(np.stack(images)), images[0].shape


def video_frame(image, result, roi, row, input_id, limits):
    zoom = 4
    h, w = image.shape
    all_mask = np.zeros(image.shape, np.float32)
    selected = all_mask.copy()
    for array, mask in ((all_mask, result.foreground), (selected, result.mask)):
        array[roi.y : roi.y + roi.height, roi.x : roi.x + roi.width] = cv2.resize(
            mask.astype(np.float32), (roi.width, roi.height), interpolation=cv2.INTER_AREA
        )
    raw = cv2.cvtColor(
        cv2.resize(review.gray_display(image, limits), None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )
    mask = cv2.cvtColor(
        cv2.resize((all_mask > 0).astype(np.uint8) * 255, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST),
        cv2.COLOR_GRAY2BGR,
    )
    point = result.point if result.point is not None else np.array([np.nan, np.nan])
    overlay = review.overlay(image, all_mask, selected, point, limits, zoom)
    for panel in (raw, mask, overlay):
        cv2.rectangle(
            panel,
            (roi.x * zoom, roi.y * zoom),
            ((roi.x + roi.width) * zoom - 1, (roi.y + roi.height) * zoom - 1),
            (40, 220, 40),
            1,
        )
    canvas = np.full((h * zoom + 100, w * zoom * 3, 3), 18, np.uint8)
    stamp = "unavailable" if row["time_sec"] is None else f"{row['time_sec']:.6f}s"
    review.text(
        canvas,
        f"{input_id} | original frame {row['source_frame_1based']} | {stamp} | segment {row['segment_id']}",
        (8, 20),
        scale=0.48,
    )
    review.text(
        canvas,
        f"{result.reason} | green: fixed ROI; cyan: selected region; red +: measured centroid",
        (8, 42),
        scale=0.46,
    )
    for index, (title, panel) in enumerate(
        [("Original TIFF", raw), ("10x linear, ROI threshold", mask), ("Selected region / measured centroid", overlay)]
    ):
        review.text(canvas, title, (index * w * zoom + 6, 70), scale=0.4)
        canvas[90 : 90 + h * zoom, index * w * zoom : (index + 1) * w * zoom] = panel
    return canvas


def recording_boundary(previous, current, maximum):
    a, b = previous["time_sec"], current["time_sec"]
    return (
        current["segment_id"] != previous["segment_id"]
        or current["source_frame_1based"] != previous["source_frame_1based"] + 1
        or a is None
        or b is None
        or not 0 < b - a <= maximum
    )


def process(manifest, output, interactive=False, video=True, video_fps=50):
    """New run only; never call ordinary extraction or modify input caches."""
    if not manifest.get("samples"):
        raise ValueError("No samples supplied")
    if not np.isfinite(video_fps) or video_fps <= 0 or int(video_fps) != video_fps:
        raise ValueError("Video FPS must be a positive integer")
    for sample in manifest["samples"]:
        validate_sample(sample)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    code_paths = [Path(__file__), Path(__file__).with_name("tiff_roi.py")]
    code = [diagnostic.fingerprint(path) for path in code_paths]
    diagnostic.write_json(output / "run.json", dict(status="running", code=code))
    started = time.perf_counter()
    summaries = []
    try:
        for number, sample in enumerate(manifest["samples"], 1):
            inputs = [diagnostic.fingerprint(row["path"]) for row in sample["frames"]]
            roi, seeds, calibration, limits, shape = prepare(sample, interactive)
            folder = output / f"sample{number}"
            folder.mkdir()
            diagnostic.write_json(
                folder / "setup.json",
                dict(
                    input_id=sample["input_id"],
                    roi=asdict(roi),
                    seeds=seeds,
                    threshold=sample.get("threshold", 2000),
                    scale=10,
                    interpolation="linear",
                    calibration=calibration,
                    display_limits=limits,
                    coordinate_frame="original TIFF pixels: x right, y down; native=(enlarged+0.5)/10-0.5",
                    inputs=inputs,
                ),
            )
            tracker = Tracker(roi, sample.get("threshold", 2000), max_dt=nominal_max_dt(sample["frames"]))
            counts: Counter[str] = Counter()
            reacquired = []
            video_map: list[Optional[int]] = []
            processing_sec = 0.0
            columns = [
                "source_frame_1based",
                "time_sec",
                "source_timestamp",
                "segment_id",
                "x_px",
                "y_px",
                "detected",
                "reason",
                "reacquired",
                "source_input",
            ]
            with (folder / "centroids.csv").open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()

                def frames():
                    nonlocal processing_sec
                    for index, row in enumerate(sample["frames"]):
                        image = load_image(row["path"])
                        if image.shape != shape:
                            raise ValueError("Source image shape changed")
                        seed = seeds.get(str(row["source_frame_1based"]))
                        stamp = np.nan if row["time_sec"] is None else row["time_sec"]
                        if interactive and index and seed is None:
                            gap = recording_boundary(sample["frames"][index - 1], row, tracker.max_dt)
                            if gap and row["time_sec"] is not None:
                                seed = pick_target(image)
                                seeds[str(row["source_frame_1based"])] = seed
                        before = time.perf_counter()
                        result = tracker.step(image, row["source_frame_1based"], stamp, row["segment_id"], seed)
                        processing_sec += time.perf_counter() - before
                        counts[result.reason] += 1
                        if result.reacquired:
                            reacquired.append(row["source_frame_1based"])
                        point = result.point if result.point is not None else [None, None]
                        writer.writerow(
                            dict(
                                source_frame_1based=row["source_frame_1based"],
                                time_sec=row["time_sec"],
                                source_timestamp=row.get("source_timestamp"),
                                segment_id=row["segment_id"],
                                x_px=point[0],
                                y_px=point[1],
                                detected=result.point is not None,
                                reason=result.reason,
                                reacquired=result.reacquired,
                                source_input=str(Path(row["path"]).resolve()),
                            )
                        )
                        if video:
                            if index and recording_boundary(sample["frames"][index - 1], row, tracker.max_dt):
                                card = review.card(
                                    (shape[0] * 4 + 100, shape[1] * 12, 3),
                                    [
                                        "Recording gap: no measurements across this interval",
                                        "Next frame requires explicit target reselection",
                                    ],
                                )
                                for _ in range(int(video_fps)):
                                    video_map.append(None)
                                    yield card
                            video_map.append(row["source_frame_1based"])
                            yield video_frame(image, result, roi, row, sample["input_id"], limits)

                if video:
                    review.encode_video(folder / "review.mp4", frames(), fps=video_fps)
                else:
                    for _ in frames():
                        pass
            if any(diagnostic.fingerprint(item["path"])["sha256"] != item["sha256"] for item in inputs):
                raise ValueError("Source changed during extraction")
            diagnostic.write_json(folder / "presentation.json", dict(fps=video_fps, source_frames=video_map))
            diagnostic.write_json(folder / "resolved_seeds.json", seeds)
            summaries.append(
                dict(
                    input_id=sample["input_id"],
                    folder=folder.name,
                    rows=len(sample["frames"]),
                    reasons=dict(counts),
                    reacquired_source_frames=reacquired,
                    roi=asdict(roi),
                    calibration=calibration,
                    processing_sec=processing_sec,
                )
            )
            print(f"{sample['input_id']}: {dict(counts)}; reacquired={len(reacquired)}", flush=True)
        if code != [diagnostic.fingerprint(path) for path in code_paths]:
            raise RuntimeError("Implementation changed during extraction")
        diagnostic.write_json(
            output / "run.json",
            dict(
                status="complete",
                method="fixed-roi-tiff-centroid-v1",
                runtime=dict(python=sys.version, numpy=np.__version__, opencv=getattr(cv2, "__version__")),
                samples=summaries,
                code=code,
                elapsed_sec=time.perf_counter() - started,
                manifest=manifest,
                policy="No normal-analysis integration; missing remains unavailable; reacquisition is a spatial hypothesis requiring visual verification",
            ),
        )
    except BaseException as error:
        diagnostic.write_json(output / "run.json", dict(status="failed", error=str(error), code=code))
        raise
    return output
