"""Export local TIFF/mask/centroid review movies from frozen comparison results."""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from utils.functions import centroid_comparison as comparison  # noqa: E402
from utils.functions import centroid_diagnostics as diagnostic  # noqa: E402
from utils.functions import centroid_review as review  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def export_window(sample, start, entries, timeline, comparison_root, output, repeats):
    n = entries[0]["frames"]
    if any(entry["frames"] != n for entry in entries):
        raise ValueError("Candidate lengths differ")
    rows = [dict(row, sample_no=sample) for row in timeline[start : start + n]]
    if len(rows) != n:
        raise ValueError("Source window is incomplete")
    planes = []
    for row in rows:
        with Image.open(row["tiff_path"]) as image:
            planes.append(np.array(image))
    images = np.stack(planes)
    if images.ndim != 3:
        raise ValueError("Expected grayscale TIFF planes")
    methods: list[dict] = []
    for entry in entries:
        if entry["method"]["name"] not in [method["name"] for method in methods]:
            methods.append(entry["method"])
    if len(methods) != 5 or len(entries) != 10:
        raise ValueError("This review layout requires all five methods and both selectors")
    saved_results = {}
    source_results = []
    foreground: dict[str, list[np.ndarray]] = {}
    for entry in entries:
        method, selector = entry["method"], entry["selector"]
        path = comparison_root / f"sample{sample}_{start}_{method['name']}_{selector}.npz"
        with np.load(path, allow_pickle=False) as saved:
            result = {key: saved[key] for key in saved.files}
        review.validate_saved(result, rows, images.shape[1:])
        saved_results[(method["name"], selector)] = result
        source_results.append(diagnostic.fingerprint(path))
        if method["name"] not in foreground:
            foreground[method["name"]] = []
            for image in images:
                binary = comparison.segment(image, method)
                coverage = cv2.resize(
                    binary.astype(np.float32), (images.shape[2], images.shape[1]), interpolation=cv2.INTER_AREA
                )
                foreground[method["name"]].append(coverage)
    limits, context_limits = review.display_limits(images), review.display_limits(images, 99.5)
    per_frame = []
    for index, row in enumerate(rows):
        choices = []
        for method in methods:
            for selector in ("largest_external_contour", "constrained"):
                result = saved_results[(method["name"], selector)]
                point = result["points"][index]
                choices.append(
                    dict(
                        method=method["name"],
                        selector=selector,
                        point_native_px=point.tolist() if np.isfinite(point).all() else None,
                        reason=str(result["reasons"][index]),
                        foreground_area_native_px=float(foreground[method["name"]][index].sum()),
                        selected_area_native_px=float(result["selected_masks"][index].sum()),
                    )
                )
        span = review.centroid_span(
            [saved_results[(m["name"], "largest_external_contour")]["points"][index] for m in methods]
        )
        per_frame.append(
            dict(
                tiff_suffix_0based=row["tiff_suffix_0based"],
                time_sec=row["time_sec"],
                segment_id=row["segment_id"],
                gap_before=row["gap_before"],
                centroid_span_native_px=span,
                choices=choices,
            )
        )
    max_span = max((r["centroid_span_native_px"] or 0 for r in per_frame), default=0)
    poster_index = max(range(n), key=lambda i: per_frame[i]["centroid_span_native_px"] or 0)
    name = f"sample{sample}_tiff_{rows[0]['tiff_suffix_0based']:05}_{rows[-1]['tiff_suffix_0based']:05}"
    video_mapping = []

    def frame(index, replay):
        results = {}
        for method in methods:
            for selector in ("largest_external_contour", "constrained"):
                result = saved_results[(method["name"], selector)]
                results[(method["name"], selector)] = dict(
                    point=result["points"][index],
                    selected=result["selected_masks"][index],
                    reason=result["reasons"][index],
                    all_coverage=foreground[method["name"]][index],
                )
        return review.review_frame(
            images[index], results, rows[index], methods, limits, context_limits, replay, repeats
        )

    poster = frame(poster_index, 1)
    if not cv2.imwrite(str(output / f"{name}.png"), poster):
        raise OSError("Cannot save review poster")

    def presentation():
        for replay in range(1, repeats + 1):
            if replay > 1:
                card = review.card(
                    poster.shape, [f"Replay {replay}/{repeats}", "Same source observations, not a continuous recording"]
                )
                for _ in range(5):
                    video_mapping.append(dict(kind="replay_card", replay=replay, source=None))
                    yield card
            for index, row in enumerate(rows):
                if index and row["gap_before"]:
                    before, after = rows[index - 1]["time_sec"], row["time_sec"]
                    duration = "unknown" if before is None or after is None else f"{after-before:.6f} s"
                    card = review.card(
                        poster.shape,
                        [
                            "RECORDING BOUNDARY: no observations across this interval",
                            f"Timestamp difference: {duration}",
                            "No coordinate interpolation or tracking across the gap",
                        ],
                    )
                    for _ in range(10):
                        video_mapping.append(dict(kind="gap_card", replay=replay, source=None))
                        yield card
                video_mapping.append(
                    dict(kind="observation", replay=replay, source=row["tiff_suffix_0based"], local_index=index)
                )
                yield frame(index, replay)

    review.encode_video(output / f"{name}.mp4", presentation())
    diagnostic.write_json(
        output / f"{name}.json",
        dict(
            source_kind="TIFF_suffix_0based; not a verified AVI frame mapping",
            methods=methods,
            display_only=dict(
                limits=limits, context_limits=context_limits, fps=10, nominal_slowdown=20, repeats=repeats
            ),
            frames=per_frame,
            presentation_frame_mapping=video_mapping,
            source_images=[diagnostic.fingerprint(row["tiff_path"]) for row in rows],
            source_results=source_results,
            limitations=[
                "Inter-method span is sensitivity, not true error",
                "No automatic adoption/rejection from mask separation",
                "Missing saved centroids remain missing",
                "Actual target ID/center truth remains unverified",
            ],
        ),
    )
    missing = {
        f"{method['name']}/{selector}": int(
            (~np.isfinite(saved_results[(method["name"], selector)]["points"]).all(axis=1)).sum()
        )
        for method in methods
        for selector in ("largest_external_contour", "constrained")
    }
    return dict(
        name=name,
        sample_no=sample,
        start_index=start,
        source_frames=n,
        max_centroid_span_native_px=max_span,
        poster_local_index=poster_index,
        missing_counts=missing,
        description="同一元TIFF上で比較。重心差は条件への感度であり、真の位置誤差ではありません。",
        review_status="awaiting_visual_review",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--manifest-root", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "centroid_evaluation")
    args = parser.parse_args(argv)
    try:
        output_root = diagnostic.private_output_root(ROOT, args.output_root)
    except ValueError as exc:
        parser.error(str(exc))
    if args.repeats < 1:
        parser.error("repeats must be positive")
    identity = json.loads((args.comparison / "identity.json").read_text())
    if identity["runtime"]["numpy"] != np.__version__ or identity["runtime"]["opencv"] != getattr(cv2, "__version__"):
        parser.error("Use the same NumPy/OpenCV versions as the saved comparison")
    for path in (ROOT / "utils/functions/centroid_comparison.py", ROOT / "utils/functions/centroid_diagnostics.py"):
        if identity["code"][str(path.relative_to(ROOT))] != diagnostic.fingerprint(path)["sha256"]:
            parser.error("Comparison implementation changed; foreground cannot be reproduced for this review")
    result = json.loads((args.comparison / "measured_comparison.json").read_text())
    output = output_root / ("review_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    diagnostic.write_json(
        output / "run.json",
        dict(status="running", comparison=diagnostic.fingerprint(args.comparison / "identity.json")),
    )
    print(output, flush=True)
    cv2.setNumThreads(1)
    entries = []
    try:
        manifests = {}
        for path in args.manifest_root.glob("*/manifest.json"):
            manifest = json.loads(path.read_text())
            manifests[manifest["sample_no"]] = json.loads((path.parent / "timeline.json").read_text())
        groups = sorted({(entry["sample_no"], entry["window_start"]) for entry in result["windows"]})
        for sample, start in groups:
            group = [
                entry for entry in result["windows"] if entry["sample_no"] == sample and entry["window_start"] == start
            ]
            entries.append(
                export_window(sample, start, group, manifests[sample], args.comparison, output, args.repeats)
            )
            print(f"review video {len(entries)}/{len(groups)}", flush=True)
        review.write_gallery(output, entries)
    except (Exception, KeyboardInterrupt) as exc:
        diagnostic.write_json(output / "run.json", dict(status="failed", error=str(exc)))
        raise
    diagnostic.write_json(
        output / "run.json",
        dict(status="complete", videos=len(entries), purpose="qualitative review only; criteria unchanged"),
    )


if __name__ == "__main__":
    main()
