"""Register saved binary AVI/stack to TIFF and export local comparison movies."""

import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
import tifffile
from PIL import Image

from utils.functions import centroid_diagnostics as diagnostic
from utils.functions import centroid_review as review
from utils.functions import image_registration as registration

ROOT = Path(__file__).resolve().parents[1]


def load_plane(row):
    with Image.open(row["tiff_path"]) as image:
        return np.array(image)


def explain_mask(image, mask):
    # An empty/full reference has no spatial texture with which to register.
    if not mask.any() or mask.all():
        return None, None
    return registration.optimal_threshold(image, mask)


def describe(mask):
    """Independent mask mean; do not silently impose ellipse-fit eligibility."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    selected = np.zeros(mask.shape, np.uint8)
    if not contours:
        return selected.astype(bool), None, False
    contour = max(contours, key=cv2.contourArea)
    _, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    label = labels[contour[0, 0, 1], contour[0, 0, 0]]
    selected = labels == label
    y, x = np.nonzero(selected)
    return selected, [float(x.mean()), float(y.mean())], len(contour) >= 5


def fit_and_verify(dataset, sample, timeline, output):
    if [row["tiff_suffix_0based"] for row in timeline] != list(range(len(timeline))):
        raise ValueError("This labelled-stack review requires consecutive TIFF suffixes starting at zero")
    stack_path, avi_path = dataset / f"{sample}.tif", dataset / f"{sample}.avi"
    with tifffile.TiffFile(stack_path) as stack_file:
        metadata = stack_file.imagej_metadata
        if metadata is None or "Labels" not in metadata:
            raise ValueError("Saved stack lacks independent ImageJ frame labels")
        labels = metadata["Labels"]
    registration.verify_labels(labels, timeline)
    stack = tifffile.memmap(stack_path)
    cap = cv2.VideoCapture(str(avi_path))
    decoded = 0
    mismatch = 0
    for index in range(len(stack)):
        ok, frame = cap.read()
        if not ok:
            raise ValueError("AVI is shorter than the labelled binary stack")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if gray.shape != stack[index].shape:
            raise ValueError("AVI/stack sizes differ")
        mismatch += int(np.count_nonzero(gray != stack[index]))
        decoded += 1
    extra, _ = cap.read()
    cap.release()
    if extra or mismatch:
        raise ValueError("AVI does not exactly match the labelled binary stack")
    # Spatial fitting uses only the first image, never centroids or saved tracks.
    candidates = registration.register_crop(load_plane(timeline[0]), stack[0] > 127)
    best = candidates[0]
    h, w = stack.shape[1:]
    transform = dict(best, scale=10, width=w, height=h)
    indices = sorted(set(np.linspace(1, len(timeline) - 1, 64, dtype=int).tolist()))
    checks = []
    for index in indices:
        reference = stack[index] > 127
        image = load_plane(timeline[index])
        patch = registration.crop_image(image, transform)
        threshold, iou = explain_mask(patch, reference)
        shift_scores = {}
        for shift in range(-3, 4):
            other = index + shift
            if 0 <= other < len(stack):
                shift_scores[str(shift)] = explain_mask(patch, stack[other] > 127)[1]
        checks.append(
            dict(tiff_suffix_0based=index, best_threshold=threshold, iou=iou, temporal_shift_iou=shift_scores)
        )
    usable = [check["iou"] for check in checks if check["iou"] is not None]
    if len(usable) < 32 or min(usable) < 0.97:
        raise ValueError("A single crop/resize transform does not validate throughout this recording")
    record = dict(
        status="verified_labelled_stack_and_pixel_correspondence",
        frame_identity=dict(
            all_source_filenames_and_timestamps_match=True,
            avi_frames_exactly_equal_labelled_stack=decoded,
            avi_stack_mismatched_pixels=mismatch,
            avi_frame_1based="TIFF suffix + 1",
        ),
        transform=transform,
        affine_native_xy="(AVI_xy + crop_xy + 0.5)/10 - 0.5",
        fit=dict(source_suffix=0, top_candidates=candidates[:8]),
        spatial_validation=checks,
        note="Thresholds fitted here explain correspondence only; no target accuracy is inferred. Fiji settings remain inferred, not recovered provenance.",
        sources=[diagnostic.fingerprint(stack_path), diagnostic.fingerprint(avi_path)],
    )
    diagnostic.write_json(output / f"{sample}_registration.json", record)
    return stack, transform, record


def masks_for(image, transform):
    masks = {}
    for mode in ("nearest", "linear", "cubic"):
        patch = registration.crop_image(image, dict(transform, interpolation=mode))
        masks[f"{mode}: T=2000"] = patch > 2000
    patch = registration.crop_image(image, dict(transform, interpolation="linear"))
    masks["linear: T=median+4000"] = patch > (float(np.median(image)) + 4000)
    return masks


def scan_windows(stack, transform, timeline):
    """Screen throughout the recording; rankings are sensitivity, not true error."""
    scores = []
    for index in range(0, len(timeline), 64):
        image = load_plane(timeline[index])
        old = stack[index] > 127
        old_selected, old_point, _ = describe(old)
        candidates = masks_for(image, transform)
        selected, point, _ = describe(candidates["linear: T=median+4000"])
        distance = (
            None if old_point is None or point is None else float(np.linalg.norm(np.subtract(point, old_point)) / 10)
        )
        scores.append(
            dict(
                index=index,
                centroid_difference_native_px=distance,
                old_area_native_px=float(old_selected.sum() / 100),
                candidate_area_native_px=float(selected.sum() / 100),
                old_selected_touches_crop=bool(old_selected[[0, -1], :].any() or old_selected[:, [0, -1]].any()),
                old_candidate_iou=registration.mask_iou(old, candidates["linear: T=median+4000"]),
            )
        )
    ranked = sorted(scores, key=lambda r: r["centroid_difference_native_px"] or 0, reverse=True)
    chosen: list[int] = []
    for row in ranked:
        if all(abs(row["index"] - other) > 200 for other in chosen):
            chosen.append(row["index"])
        if len(chosen) == 2:
            break
    return scores, chosen


def draw_panel(raw, mask, limits, title, binary=False):
    gray = review.gray_display(raw, limits)
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    selected, point, eligible = describe(mask) if mask is not None else (None, None, False)
    if binary:
        if mask is None:
            raise ValueError("Binary display requires a mask")
        canvas = cv2.cvtColor(mask.astype(np.uint8) * 255, cv2.COLOR_GRAY2BGR)
    canvas = cv2.resize(canvas, None, fx=2, fy=2, interpolation=cv2.INTER_NEAREST)
    if mask is not None and not binary:
        assert selected is not None
        for region, color in ((mask, (0, 220, 255)), (selected, (255, 220, 0))):
            outlines, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(canvas, [c * 2 for c in outlines], -1, color, 1)
    if point is not None:
        cv2.drawMarker(canvas, review.display_point(point, 2), (0, 0, 255), cv2.MARKER_CROSS, 13, 1)
    cell = np.full((420, 360, 3), 22, np.uint8)
    review.text(cell, title, (6, 18), scale=0.43)
    if point is not None:
        review.text(cell, f"mask mean ({point[0]:.2f}, {point[1]:.2f})", (6, 36), scale=0.39)
        review.text(cell, "ellipse contour OK" if eligible else "ellipse contour insufficient", (6, 53), scale=0.36)
    cell[60 : 60 + canvas.shape[0], : canvas.shape[1]] = canvas
    return cell, dict(
        mask_mean_crop_px=point,
        ellipse_contour_eligible=eligible,
        area_native_px=None if selected is None else float(selected.sum() / 100),
    )


def export_clip(sample_no, center, stack, transform, timeline, output, label, cached):
    start, stop = max(0, center - 50), min(len(timeline), center + 51)
    images = [load_plane(row) for row in timeline[start:stop]]
    patches = [registration.crop_image(image, transform) for image in images]
    limits = review.display_limits(np.stack(patches))
    name = f"sample{sample_no}_tiff_{start:05}_{stop-1:05}"
    observations = []
    presentation = []
    rendered = []
    for offset, (image, raw) in enumerate(zip(images, patches)):
        index = start + offset
        row = timeline[index]
        old = stack[index] > 127
        masks = masks_for(image, transform)
        canvas = np.full((960, 1440, 3), 18, np.uint8)
        review.text(
            canvas,
            f"Sample {sample_no} | TIFF {index:05} = AVI frame {index+1} | {row['time_sec']:.6f} s | segment {row['segment_id']}",
            (10, 22),
            scale=0.65,
        )
        review.text(
            canvas,
            "Matched 10x crop | yellow: foreground  cyan: largest region  red +: region pixel mean (not target truth)",
            (10, 45),
            scale=0.55,
        )
        review.text(
            canvas,
            "Original timestamps retained | 10 fps, nominal 20x slow | no production results changed",
            (10, 67),
            scale=0.5,
        )
        panels = [
            ("Original TIFF: linear 10x", None, False),
            ("Saved AVI: binary", old, True),
            ("Saved AVI on original TIFF", old, False),
            ("Old (yellow) / linear T2000 (cyan)", None, False),
        ]
        panels.extend((key, mask, False) for key, mask in masks.items())
        frame_details = []
        for position, (title, mask, binary) in enumerate(panels):
            cell, details = draw_panel(raw, mask, limits, title, binary)
            if position == 3:
                for region, color in ((old, (0, 220, 255)), (masks["linear: T=2000"], (255, 220, 0))):
                    outlines, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(cell, [c * 2 + np.array([0, 60]) for c in outlines], -1, color, 1)
            x = (position % 4) * 360
            y = 90 + (position // 4) * 420
            canvas[y : y + 420, x : x + 360] = cell
            if details["mask_mean_crop_px"] is not None:
                details["mask_mean_native_px"] = registration.reference_to_native(
                    details["mask_mean_crop_px"], transform
                ).tolist()
            frame_details.append(dict(panel=title, **details))
        review.text(
            canvas,
            "ROI in expanded TIFF: "
            + str([transform[k] for k in ("x", "y", "width", "height")])
            + " | all panels share crop coordinates",
            (10, 950),
            scale=0.5,
        )
        old_point = [float(cached[index][key]) for key in ("x_px", "y_px")]
        detected = bool(np.isfinite(old_point).all())
        review.text(
            canvas,
            "Old normal extraction: " + ("detected" if detected else "MISSING"),
            (1000, 67),
            (100, 255, 100) if detected else (100, 150, 255),
            scale=0.45,
        )
        if detected:
            cv2.drawMarker(
                canvas,
                tuple(np.asarray(review.display_point(old_point, 2)) + [720, 150]),
                (50, 255, 50),
                cv2.MARKER_DIAMOND,
                11,
                1,
            )
        threshold, iou = explain_mask(raw, old)
        observations.append(
            dict(
                tiff_suffix_0based=index,
                avi_frame_1based=index + 1,
                time_sec=row["time_sec"],
                source_timestamp=row["source_timestamp"],
                segment_id=row["segment_id"],
                gap_before=row["gap_before"],
                old_normal_centroid_crop_px=old_point if detected else None,
                old_reconstruction_iou=iou,
                old_reconstruction_mismatched_pixels=(
                    None if threshold is None else int(np.count_nonzero((raw > threshold) != old))
                ),
                old_foreground_pixels=int(old.sum()),
                old_explanatory_threshold=threshold,
                panels=frame_details,
                source=diagnostic.fingerprint(row["tiff_path"]),
            )
        )
        rendered.append(canvas)

    def frames():
        for repeat in range(2):
            if repeat:
                card = review.card(rendered[0].shape, ["Replay: same source observations"])
                for _ in range(10):
                    presentation.append(dict(kind="replay_card", source=None))
                    yield card
            for offset, canvas in enumerate(rendered):
                index = start + offset
                if offset and timeline[index]["gap_before"]:
                    card = review.card(
                        canvas.shape,
                        [
                            "Recording gap: no observations between these frames",
                            f"TIFF {index-1} -> {index}",
                            "No interpolation / no direction calculation",
                        ],
                    )
                    for _ in range(10):
                        presentation.append(dict(kind="gap_card", source=None))
                        yield card
                presentation.append(dict(kind="observation", tiff_suffix_0based=index, avi_frame_1based=index + 1))
                yield canvas

    review.encode_video(output / f"{name}.mp4", frames())
    if not cv2.imwrite(str(output / f"{name}.png"), rendered[center - start]):
        raise OSError("Cannot save registered review poster")
    diagnostic.write_json(
        output / f"{name}.json",
        dict(
            label=label,
            transform=transform,
            display_limits=limits,
            observations=observations,
            presentation=presentation,
            methods=dict(
                scale=10,
                threshold_fixed=2000,
                threshold_relative="median(original TIFF)+4000",
                interpolations=["nearest", "linear", "cubic"],
            ),
            limitations="Largest-region selection is not proof of target identity. Mask means are diagnostic, not production detections. Relative-threshold 10x is a visualization condition, not a validated adopted method.",
        ),
    )
    print("exported", name, flush=True)
    return dict(
        name=name,
        label=label,
        sample_no=sample_no,
        center=center,
        source_frames=stop - start,
        min_old_reconstruction_iou=min(
            (r["old_reconstruction_iou"] for r in observations if r["old_reconstruction_iou"] is not None), default=None
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--prior-review", type=Path, required=True)
    parser.add_argument("--raw-centroids", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.output is None:
        args.output = (
            ROOT
            / "outputs"
            / "centroid_evaluation"
            / ("registered_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
        )
    try:
        args.output = diagnostic.private_output_root(ROOT, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    diagnostic.write_json(args.output / "run.json", dict(status="running", purpose="image correspondence review"))
    try:
        run(args)
    except (Exception, KeyboardInterrupt) as exc:
        diagnostic.write_json(args.output / "run.json", dict(status="failed", error=str(exc)))
        raise
    diagnostic.write_json(args.output / "run.json", dict(status="complete", purpose="image correspondence review"))


def run(args):
    code_paths = [__file__, registration.__file__, review.__file__, diagnostic.__file__]
    code_fingerprints = [diagnostic.fingerprint(path) for path in code_paths]
    existing = json.loads((args.prior_review / "index.json").read_text())
    entries = []
    samples = sorted(p.stem for p in args.dataset.glob("*.avi"))
    if not samples:
        raise ValueError("No AVI inputs in the requested dataset")
    for sample_no, sample in enumerate(samples, 1):
        timeline = json.loads((args.diagnostics / sample / "timeline.json").read_text())
        stack, transform, record = fit_and_verify(args.dataset, sample, timeline, args.output)
        print(
            sample,
            "registered",
            transform,
            "validation minimum",
            min(r["iou"] for r in record["spatial_validation"] if r["iou"] is not None),
            flush=True,
        )
        scores, extra = scan_windows(stack, transform, timeline)
        diagnostic.write_json(
            args.output / f"{sample}_screening.json",
            dict(
                step=64, rank="old versus candidate centroid difference; not error", observations=scores, chosen=extra
            ),
        )
        centers = {
            row["start_index"] + row["source_frames"] // 2: "prior normal/problem/gap review window"
            for row in existing
            if row["sample_no"] == sample_no
        }
        # Inspect old extraction outliers and missing frames without treating
        # displacement from a median as ground-truth error.
        cached_path = args.raw_centroids / f"{sample}.csv"
        # NumPy's scalar/float-array overload does not describe named CSV fields.
        cached = cast(Any, np.genfromtxt(cached_path, delimiter=",", names=True, dtype=None, encoding="utf-8"))
        if not np.array_equal(cached["source_frame_1based"], np.arange(1, len(timeline) + 1)):
            raise ValueError("Cached original-frame identity differs")
        if not np.allclose(cached["time_sec"], [row["time_sec"] for row in timeline], atol=1e-9, rtol=0):
            raise ValueError("Cached timestamps differ")
        points = np.column_stack([cached["x_px"], cached["y_px"]])
        distance = np.linalg.norm(points - np.nanmedian(points, axis=0), axis=1)
        old_extreme = int(np.nanargmax(distance)) if np.isfinite(distance).any() else None
        missing = np.flatnonzero(~np.isfinite(points).all(axis=1))
        for index in ([old_extreme] if old_extreme is not None else []) + ([int(missing[0])] if len(missing) else []):
            if all(abs(index - center) > 25 for center in centers):
                centers[index] = "old extraction: displacement outlier / missing observation for inspection"
        diagnostic.write_json(
            args.output / f"{sample}_old_screening.json",
            dict(
                source=diagnostic.fingerprint(cached_path),
                finite_observations=int(np.isfinite(points).all(axis=1).sum()),
                missing_observations=len(missing),
                outlier=old_extreme,
                first_missing=int(missing[0]) if len(missing) else None,
            ),
        )
        for index in extra:
            if all(abs(index - center) > 50 for center in centers):
                centers[index] = "screening: large old/candidate mask-mean difference (not proven failure)"
        for center, label in sorted(centers.items()):
            entries.append(export_clip(sample_no, center, stack, transform, timeline, args.output, label, cached))
    for entry in entries:
        cap = cv2.VideoCapture(str(args.output / (entry["name"] + ".mp4")))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame.shape != (960, 1440, 3):
                raise ValueError("Wrong encoded video shape")
            n += 1
        cap.release()
        metadata = json.loads((args.output / (entry["name"] + ".json")).read_text())
        if n != len(metadata["presentation"]):
            raise ValueError("Incomplete encoded video")
        entry["decoded_presentation_frames"] = n
    diagnostic.write_json(args.output / "index.json", entries)
    sections = []
    for entry in entries:
        name = entry["name"]
        sections.append(
            f'<h2>{name}</h2><p>{html.escape(entry["label"])}</p><video controls preload="none" poster="{name}.png" src="{name}.mp4"></video>'
        )
    (args.output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Registered AVI / TIFF review</title><style>body{background:#181818;color:#eee;font:16px system-ui;margin:24px}video{width:100%;max-width:1440px}</style><h1>旧AVIと元TIFF・10倍拡大後の二値化</h1><p>対象とそれ以外を区別できるかを確認するためのローカル動画。重心差は真の誤差ではありません。赤十字は各マスクの画素平均で、通常解析の採用判定とは別です。緑の菱形は旧通常解析の保存済み重心です。下段は全て10倍拡大後に二値化。右上は黄色が旧AVI、シアンが線形補間・閾値2000。</p>'
        + "".join(sections),
        encoding="utf-8",
    )
    if code_fingerprints != [diagnostic.fingerprint(path) for path in code_paths]:
        raise RuntimeError("Exporter code changed during this run; rerun into a new directory")
    diagnostic.write_json(
        args.output / "provenance.json",
        dict(
            created_utc=datetime.now(timezone.utc).isoformat(),
            arguments={k: str(v.resolve()) for k, v in vars(args).items()},
            code=code_fingerprints,
            boundary="Local measured artifacts only. No ordinary cache generated or updated. No numerical comparison benchmark or adoption decision rerun.",
        ),
    )
    print(args.output.resolve(), flush=True)


if __name__ == "__main__":
    main()
