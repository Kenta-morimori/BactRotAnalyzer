"""Compare registered centroid candidates and record an evidence-based decision."""

import argparse
import configparser
import copy
import json
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from tests.synthetic.images import CASES, generate_images  # noqa: E402
from utils.functions import centroid_comparison as comparison  # noqa: E402
from utils.functions import centroid_diagnostics as diagnostic  # noqa: E402
from utils.functions.get_centroid_coordinate import contours  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def protocol_identity(config):
    """Prevent validation after changes to methods, criteria, generator or scorer."""
    files = [
        Path(__file__),
        Path(comparison.__file__),
        Path(diagnostic.__file__),
        ROOT / "tests" / "synthetic" / "images.py",
    ]
    return dict(
        protocol=config,
        code={str(p.relative_to(ROOT)): diagnostic.fingerprint(p)["sha256"] for p in files},
        runtime={"python": sys.version.split()[0], "numpy": np.__version__, "opencv": getattr(cv2, "__version__")},
    )


def simulated_seeds(sequence):
    """Simulate an explicit foreground click at recording starts, never recovery."""
    seeds = {}
    for i in range(len(sequence.times)):
        if (i == 0 or sequence.segments[i] != sequence.segments[i - 1]) and sequence.observable[i]:
            y, x = np.nonzero(sequence.masks[i, 0])
            center = sequence.isolated_centroids[i, 0]
            nearest = np.argmin((x - center[0]) ** 2 + (y - center[1]) ** 2)
            seeds[i] = [int(x[nearest]), int(y[nearest])]
    return seeds


def run_synthetic(config, output, partition):
    """Compare conditions separately, using truth only in annotation and scoring."""
    started = time.perf_counter()
    summaries = []
    timing: dict[str, dict[str, Any]] = {}
    cache = output / "mask_cache"
    (output / "inputs").mkdir()
    (output / "examples").mkdir()
    seeds = config["synthetic_seeds"][partition]
    for case in CASES:
        pooled: dict[tuple[str, str], dict[str, Any]] = {}
        for seed in seeds:
            sequence = generate_images(case, seed)
            input_path = output / "inputs" / f"{case}_{seed}.npz"
            np.savez_compressed(input_path, **vars(sequence))
            clicks = simulated_seeds(sequence)
            for method in config["comparison"]["methods"]:
                name = method["name"]
                clock = time.perf_counter()
                with np.load(input_path, allow_pickle=False) as data:
                    images = data["images"]
                read_time = time.perf_counter() - clock
                clock = time.perf_counter()
                core_times: dict[str, float] = {}
                case_cache = cache / case / str(seed)
                masks, reused = comparison.cached_masks(images, method, case_cache, core_times)
                cold = time.perf_counter() - clock
                if reused:
                    raise ValueError("A new benchmark run unexpectedly reused a cold cache")
                clock = time.perf_counter()
                again, reused = comparison.cached_masks(images, method, case_cache)
                warm = time.perf_counter() - clock
                if not reused or not np.array_equal(masks, again):
                    raise ValueError("Cached mask reuse is not reproducible")
                del again
                results, descriptions = comparison.select_sequence(
                    masks,
                    method,
                    config["comparison"]["selector"],
                    sequence.source_frames,
                    sequence.times,
                    sequence.segments,
                    clicks,
                )
                separation = comparison.separation_metrics(descriptions, sequence.masks)
                timer = timing.setdefault(
                    name,
                    dict(
                        input_read_sec=0.0,
                        cold_segmentation_with_cache_io_sec=0.0,
                        cached_mask_load_sec=0.0,
                        frames=0,
                        native_input_pixels=0,
                        processed_pixels=0,
                        segmentation_core_sec=0.0,
                        selectors={},
                    ),
                )
                timer["input_read_sec"] += read_time
                timer["cold_segmentation_with_cache_io_sec"] += cold
                timer["cached_mask_load_sec"] += warm
                timer["segmentation_core_sec"] += core_times["segmentation_core_sec"]
                timer["frames"] += len(images)
                timer["native_input_pixels"] += images.size
                timer["processed_pixels"] += masks.size
                for selector, result in results.items():
                    _, details = comparison.score_sequence(
                        result,
                        sequence.masks,
                        sequence.isolated_centroids[:, 0],
                        sequence.observable,
                        sequence.source_frames,
                    )
                    item = pooled.setdefault(
                        (name, selector),
                        dict(
                            points=[],
                            truth=[],
                            observable=[],
                            radii=[],
                            ids=[],
                            frames=[],
                            ious=[],
                            merge_frames=0,
                            oversplit_frames=0,
                            target_loss_frames=0,
                            manual_seed_actions=0,
                        ),
                    )
                    item["points"].append(result["points"])
                    item["truth"].append(sequence.isolated_centroids[:, 0])
                    item["observable"].append(sequence.observable)
                    item["radii"].append(np.sqrt(sequence.masks[:, 0].sum(axis=(1, 2)) / np.pi))
                    item["ids"].extend(details["predicted_ids"])
                    item["frames"].extend(f"{seed}:{frame}" for frame in sequence.source_frames)
                    item["ious"].extend(v for v in details["mask_iou"] if v is not None)
                    for metric, value in separation.items():
                        item[metric] += value
                    item["manual_seed_actions"] += len(clicks) if selector == "constrained" else 0
                    timer["selectors"][selector] = timer["selectors"].get(selector, 0.0) + result["processing_sec"]
                    if seed == seeds[0]:
                        np.savez_compressed(
                            output / "examples" / f"{case}_{name}_{selector}.npz",
                            points=result["points"],
                            selected_mask_coverage=result["masks"],
                            reasons=result["reasons"],
                            source_frames=sequence.source_frames,
                            times=sequence.times,
                            segments=sequence.segments,
                        )
                del masks, results, descriptions
        for (method, selector), item in pooled.items():
            expected = [f"{seed}:{frame}" for seed in seeds for frame in range(1, 25)]
            metrics = diagnostic.position_metrics(
                np.concatenate(item["points"]),
                np.concatenate(item["truth"]),
                np.concatenate(item["observable"]),
                np.concatenate(item["radii"]),
                item["ids"],
                item["frames"],
                expected,
            )
            metrics.update(
                mask_iou_mean=float(np.mean(item["ious"])) if item["ious"] else None,
                separation={k: item[k] for k in ("merge_frames", "oversplit_frames", "target_loss_frames")},
                manual_seed_actions=item["manual_seed_actions"],
            )
            summaries.append(
                dict(
                    case=case,
                    method=method,
                    selector=selector,
                    metrics=metrics,
                    failures=comparison.acceptance_failures(metrics, config["acceptance"]),
                )
            )
        print(f"{partition}: {case}", flush=True)
    for timer in timing.values():
        timer["cold_fps_including_mask_cache_io"] = timer["frames"] / timer["cold_segmentation_with_cache_io_sec"]
        timer["warm_mask_load_fps"] = timer["frames"] / timer["cached_mask_load_sec"]
    candidates = []
    for method in config["comparison"]["methods"]:
        for selector in ("largest_external_contour", "constrained"):
            rows = [r for r in summaries if r["method"] == method["name"] and r["selector"] == selector]
            candidates.append(
                dict(
                    method=method["name"],
                    selector=selector,
                    passed=not any(row["failures"] for row in rows),
                    failed_conditions=[row["case"] for row in rows if row["failures"]],
                )
            )
    diagnostic.write_json(
        output / "comparison.json",
        dict(
            partition=partition,
            conditions=summaries,
            candidates=candidates,
            timings=timing,
            whole_run_wall_sec=time.perf_counter() - started,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            * (1 if sys.platform == "darwin" else 1024),
            memory_scope="Process lifetime high-water mark, not per-method memory",
            manual_work="Simulated foreground clicks; no claim of measured human time",
        ),
    )


def run_normal_regression(config, output):
    """Stream all normal test AVIs, keeping their existing binary masks fixed."""
    dataset = ROOT / "data" / "test_data"
    settings = configparser.ConfigParser()
    settings.read(dataset / "config.ini")
    fps = settings.getfloat("Settings", "FrameRate")
    expected = round(fps * settings.getfloat("Settings", "total_time"))
    summary = []
    for number in range(1, 11):
        avi = dataset / f"test_{number}.avi"
        capture = cv2.VideoCapture(str(avi))
        if not capture.isOpened():
            raise OSError(f"Cannot open {avi}")
        tracker = comparison.ConstrainedSelector(config["comparison"]["selector"])
        index, added_missing, changed, extra_finite, independent_mismatch = 0, 0, 0, 0, 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                binary, _, xy, agrees = diagnostic.audit_frame(frame, contours)
                objects = comparison.components(binary, 1)
                seed = xy if index == 0 and np.isfinite(xy).all() else None
                chosen, _ = tracker.step(objects, index + 1, index / fps, 0, seed)
                point = chosen.center if chosen is not None else np.full(2, np.nan)
                valid = np.isfinite(xy).all()
                finite = np.isfinite(point).all()
                independent_mismatch += not agrees
                added_missing += valid and not finite
                extra_finite += finite and not valid
                changed += valid and finite and not np.allclose(xy, point, atol=1e-10, rtol=0)
                index += 1
        finally:
            capture.release()
        summary.append(
            dict(
                sample_no=number,
                input=diagnostic.fingerprint(avi),
                frames=index,
                expected_frames=expected,
                added_missing=int(added_missing),
                changed_positions=int(changed),
                extra_finite=int(extra_finite),
                baseline_pixel_mean_mismatches=int(independent_mismatch),
            )
        )
        print(f"normal regression: {number}/10", flush=True)
    diagnostic.write_json(
        output / "normal_regression.json",
        dict(
            samples=summary,
            seed_policy="One simulated first-frame click from the current mask; no physical truth claim",
            passed=all(
                r["frames"] == expected
                and not any(
                    r[k]
                    for k in ("added_missing", "changed_positions", "extra_finite", "baseline_pixel_mean_mismatches")
                )
                for r in summary
            ),
        ),
    )


def run_measured(config, output, manifest_root, annotation_path):
    """Compare small registered windows, without claiming real positional truth."""
    annotations = json.loads(annotation_path.read_text())
    diagnostic.write_json(output / "annotations.json", annotations)
    summaries = []
    for sample_dir in sorted(manifest_root.glob("*/manifest.json")):
        manifest = json.loads(sample_dir.read_text())
        timeline = json.loads((sample_dir.parent / "timeline.json").read_text())
        sample = str(manifest["sample_no"])
        for window in annotations[sample]:
            start, stop = window["start_index"], window["stop_index_exclusive"]
            rows = timeline[start:stop]
            if not rows or len(rows) != stop - start:
                raise ValueError("Annotation window lies outside the source timeline")
            planes = []
            for row in rows:
                with Image.open(row["tiff_path"]) as image:
                    planes.append(np.array(image))
            images = np.stack(planes)
            frames = np.array([row["tiff_suffix_0based"] + 1 for row in rows])
            times = np.array([np.nan if row["time_sec"] is None else row["time_sec"] for row in rows])
            segments = np.array([row["segment_id"] for row in rows])
            clicks = {int(key): value for key, value in window["foreground_clicks"].items()}
            for registered in config["comparison"]["methods"]:
                method = copy.deepcopy(registered)
                if method["threshold_mode"] == "fixed":
                    method["threshold"] = config["comparison"]["measured_fixed_threshold"]
                masks, _ = comparison.cached_masks(images, method, output / "mask_cache")
                results, _ = comparison.select_sequence(
                    masks, method, config["comparison"]["selector"], frames, times, segments, clicks
                )
                for selector, result in results.items():
                    name = f"sample{sample}_{start}_{method['name']}_{selector}"
                    np.savez_compressed(
                        output / f"{name}.npz",
                        points=result["points"],
                        selected_masks=result["masks"],
                        reasons=result["reasons"],
                        source_frames=frames,
                        times=times,
                        segments=segments,
                    )
                    summaries.append(
                        dict(
                            sample_no=int(sample),
                            role=config["sample_roles"][sample],
                            window_start=start,
                            method=method,
                            selector=selector,
                            frames=len(rows),
                            accepted=int(np.isfinite(result["points"]).all(axis=1).sum()),
                            seed_actions=len(clicks) if selector == "constrained" else 0,
                            processing_sec=result["processing_sec"],
                            position_accuracy=None,
                            reason="No independent truth, verified AVI transform or complete per-frame target labels",
                        )
                    )
    diagnostic.write_json(
        output / "measured_comparison.json",
        dict(
            windows=summaries,
            accuracy_claim=False,
            annotation_input=diagnostic.fingerprint(annotation_path),
            saved_AVI_baseline="Retained separately in stage-A manifest; unverified axes are not compared numerically",
        ),
    )


def record_decision(config, output, paths):
    """Adopt nothing unless every required evidence gate is satisfied."""
    development, validation, normal, measured = paths
    dev = json.loads((development / "comparison.json").read_text())
    val = json.loads((validation / "comparison.json").read_text())
    normal_result = json.loads((normal / "normal_regression.json").read_text())
    measured_result = json.loads((measured / "measured_comparison.json").read_text())
    if not measured_result["windows"]:
        raise ValueError("Measured comparison contains no evidence windows")
    for directory in paths:
        if json.loads((directory / "identity.json").read_text()) != protocol_identity(config):
            raise ValueError("Decision inputs use different code or protocols")
        if json.loads((directory / "run.json").read_text())["status"] != "complete":
            raise ValueError("Decision requires completed evidence runs")
    if dev["partition"] != "development" or val["partition"] != "validation":
        raise ValueError("Decision needs both independent partitions")
    source = json.loads((validation / "validation_source.json").read_text())
    if source["sha256"] != diagnostic.fingerprint(development / "freeze.json")["sha256"]:
        raise ValueError("Validation does not refer to this development freeze")
    passed_dev = {(c["method"], c["selector"]) for c in dev["candidates"] if c["passed"]}
    passed_val = {(c["method"], c["selector"]) for c in val["candidates"] if c["passed"]}
    candidates = sorted(passed_dev & passed_val)
    diagnostic.write_json(
        output / "decision.json",
        dict(
            decision="no_adoption" if not candidates else "requires_independent_measured_review",
            synthetic_candidates_passing_both=candidates,
            constrained_normal_regression_passed=normal_result["passed"],
            reason=(
                "No method meets all agreed synthetic gates"
                if not candidates
                else "Synthetic success alone does not authorize replacing measured extraction"
            ),
            normal_analysis_change="none",
            preprocessing_integration="not applicable without an adopted method",
            downstream_input_policy="Use only independently quality-confirmed intervals; no input quality claim follows from this benchmark",
            limits=[
                "Fiji settings and coordinate mapping remain incomplete",
                "Synthetic scenes do not prove physical accuracy",
                "Measured windows lack independent center truth",
                "Simple candidates only; no universal impossibility claim",
            ],
            evidence_paths=[str(p.resolve()) for p in paths],
        ),
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("synthetic", "regression", "measured", "decision"))
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "centroid_evaluation.json")
    parser.add_argument("--partition", choices=("development", "validation"), default="development")
    parser.add_argument("--freeze", type=Path)
    parser.add_argument("--manifest-root", type=Path)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--evidence", type=Path, nargs=4)
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "centroid_evaluation")
    args = parser.parse_args(argv)
    try:
        output_root = diagnostic.private_output_root(ROOT, args.output_root)
    except ValueError as exc:
        parser.error(str(exc))
    config = json.loads(args.config.read_text())
    diagnostic.validate_config(config)
    if config["acceptance"]["status"] != "agreed":
        parser.error("Comparison requires agreed numerical criteria")
    identity = protocol_identity(config)
    if args.command == "synthetic" and args.partition == "validation":
        if args.freeze is None or json.loads(args.freeze.read_text())["identity"] != identity:
            parser.error("Validation requires an unchanged development freeze record")
    if args.command == "measured" and (args.manifest_root is None or args.annotations is None):
        parser.error("Measured comparison requires stage-A manifest and independent original-image annotations")
    if args.command == "decision" and args.evidence is None:
        parser.error("Decision requires development, validation, regression and measured directories")
    output = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    diagnostic.write_json(output / "identity.json", identity)
    if args.command == "synthetic" and args.partition == "validation":
        diagnostic.write_json(output / "validation_source.json", diagnostic.fingerprint(args.freeze))
    diagnostic.write_json(output / "run.json", dict(status="running", command=args.command))
    print(output, flush=True)
    cv2.setNumThreads(1)
    try:
        if args.command == "synthetic":
            run_synthetic(config, output, args.partition)
            if args.partition == "development":
                diagnostic.write_json(
                    output / "freeze.json",
                    dict(
                        identity=identity,
                        candidate_policy="All preregistered methods, unchanged after development",
                        created_after_development=True,
                    ),
                )
        elif args.command == "regression":
            run_normal_regression(config, output)
        elif args.command == "measured":
            run_measured(config, output, args.manifest_root, args.annotations)
        else:
            record_decision(config, output, args.evidence)
    except (Exception, KeyboardInterrupt) as exc:
        diagnostic.write_json(
            output / "run.json",
            dict(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc)),
        )
        raise
    diagnostic.write_json(output / "run.json", dict(status="complete", command=args.command))


if __name__ == "__main__":
    main()
