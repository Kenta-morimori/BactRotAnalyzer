"""Inspect image/centroid correspondence without changing normal analysis or caches."""

import argparse
import configparser
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
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from tests.synthetic.images import CASES, generate_images  # noqa: E402
from utils.functions import centroid_diagnostics as diagnostic  # noqa: E402
from utils.functions.get_centroid_coordinate import contours  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def inspect_sample(dataset, avi, sample_no, config, output):
    """Inventory timing, then select evidence independently of AVI centroids."""
    sample = output / avi.stem
    sample.mkdir()
    files = list((dataset / "tiff_data" / avi.stem).glob("*.tif"))
    if not files:
        raise FileNotFoundError(f"No source TIFF frames for {avi.stem}")
    rows = diagnostic.timeline(files, config["nominal_dt_sec"])
    chosen = {i: ["temporal_or_gap_coverage"] for i in diagnostic.representative_indices(rows)}
    # Fixed coarse coverage, independent of prior diagnostics or AVI positions.
    candidates = []
    for index in range(0, len(rows), config["coverage_stride"]):
        with Image.open(rows[index]["tiff_path"]) as image:
            values = np.array(image)
        binary = diagnostic.diagnostic_mask(values, config["diagnostic_threshold_native"])
        count, _, stats, centers = cv2.connectedComponentsWithStats(binary.astype(np.uint8), connectivity=8)
        ids = [i for i in range(1, count) if stats[i, cv2.CC_STAT_AREA] >= 3]
        separation = float("inf")
        if len(ids) >= 2:
            xy = centers[ids]
            distance = np.linalg.norm(xy[:, None] - xy[None, :], axis=2)
            np.fill_diagonal(distance, np.inf)
            separation = float(distance.min())
        candidates.append((index, float(np.median(values)), separation))
    if candidates:
        bright = max(candidates, key=lambda item: item[1])[0]
        chosen.setdefault(bright, []).append("background_high_candidate")
        finite = [item for item in candidates if np.isfinite(item[2])]
        if finite:
            chosen.setdefault(min(finite, key=lambda item: item[2])[0], []).append("proximity_candidate")
    raw_path = ROOT / "outputs" / dataset.relative_to(ROOT / "data") / "raw_centroid" / f"{avi.stem}.csv"
    raw = pd.read_csv(raw_path) if raw_path.exists() else None
    capture = cv2.VideoCapture(str(avi))
    if not capture.isOpened():
        raise OSError(f"Cannot open {avi}")
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # Index consistency is evidence, not a verified TIFF-to-AVI mapping.
    suffixes_contiguous = [r["tiff_suffix_0based"] for r in rows] == list(range(len(rows)))
    annotations = []
    try:
        for index, reasons in sorted(chosen.items()):
            row = rows[index]
            with Image.open(row["tiff_path"]) as image:
                values = np.array(image)
            binary = diagnostic.diagnostic_mask(values, config["diagnostic_threshold_native"])
            selected, _ = diagnostic.selected_component(binary)
            avi_frame = None
            xy = None
            agrees = None
            if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
                raise OSError(f"Cannot seek to AVI frame {index + 1}")
            ok, image = capture.read()
            if ok:
                avi_frame = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                _, _, xy, agrees = diagnostic.audit_frame(image, contours)
            # Persist both independent axes; never overlay empirical alignment.
            diagnostic.save_panel(
                sample / f"tiff_{row['tiff_suffix_0based']:05}.png", values, binary, selected, avi_frame, xy
            )
            np.savez_compressed(
                sample / f"tiff_{row['tiff_suffix_0based']:05}_masks.npz", all_regions=binary, selected_region=selected
            )
            cache_status = "unavailable"
            if raw is not None and index < len(raw):
                cache_status = "index_consistent_unverified"
                if int(raw.iloc[index].source_frame_1based) != index + 1:
                    cache_status = "frame_mismatch"
                elif xy is not None:
                    cached_xy = raw[["x_px", "y_px"]].iloc[index].to_numpy(dtype=float)
                    cache_status = (
                        "avi_centroid_matches"
                        if np.allclose(xy, cached_xy, atol=1e-10, rtol=0, equal_nan=True)
                        else "avi_centroid_mismatch"
                    )
            annotations.append(
                dict(
                    tiff_suffix_0based=row["tiff_suffix_0based"],
                    original_input=diagnostic.fingerprint(row["tiff_path"]),
                    candidate_avi_frame_1based=index + 1,
                    candidate_avi_decoded=ok,
                    selection_reasons=reasons,
                    target_annotation={"status": "unreviewed", "source": "original_TIFF_only", "object_id": None},
                    causes=["undetermined"],
                    evidence="Frame and crop correspondence unverified; no positional attribution possible.",
                    normal_pixel_mean_matches=agrees,
                    cached_centroid_status=cache_status,
                )
            )
    finally:
        capture.release()
    diagnostic.write_json(sample / "timeline.json", rows)
    diagnostic.write_json(
        sample / "manifest.json",
        dict(
            sample_no=sample_no,
            role=config["sample_roles"][str(sample_no)],
            avi=diagnostic.fingerprint(avi),
            raw_centroid=diagnostic.fingerprint(raw_path) if raw is not None else None,
            tiff_frames=len(rows),
            avi_frames=count,
            sequential_suffixes=suffixes_contiguous,
            frame_mapping={"status": "unverified", "equal_counts": count == len(rows)},
            transform={"status": "unknown", "avi_to_tiff_affine": None, "evidence": None},
            fiji={
                "scale_xy": [10, 10],
                "order": "resize_then_manual_threshold",
                "interpolation": None,
                "threshold": None,
                "threshold_changes": None,
                "crop": None,
                "fully_reproducible": False,
            },
            annotations=annotations,
            limitation="Sample 2 is inspected for coverage only, never used for method selection or tuning.",
        ),
    )
    return dict(
        sample_no=sample_no,
        tiff_frames=len(rows),
        avi_frames=count,
        gap_count=sum(r["gap_before"] for r in rows),
        evidence_frames=len(annotations),
        cause_status="undetermined",
        transform_status="unknown",
    )


def run_regression(dataset, output):
    """Audit all ten test AVIs with an optional read-only historical cache."""
    settings = configparser.ConfigParser()
    settings.read(dataset / "config.ini")
    fps = settings.getfloat("Settings", "FrameRate")
    calibration = np.array([settings.getfloat("Settings", f"px2um_{axis}") for axis in ("x", "y")])
    expected = round(fps * settings.getfloat("Settings", "total_time"))
    summary = []
    for number in range(1, 11):
        avi = dataset / f"test_{number}.avi"
        cache_path = ROOT / "outputs" / dataset.relative_to(ROOT / "data") / "raw_centroid" / f"test_{number}.csv"
        cached = None
        if cache_path.exists():
            cache = pd.read_csv(cache_path)
            cached = cache[["x_px", "y_px"]].to_numpy(dtype=float)
            if not np.array_equal(cache.source_frame_1based, np.arange(1, len(cache) + 1)):
                raise ValueError("Historical raw cache has an invalid source-frame axis")
            if not np.allclose(cache.time_sec, np.arange(len(cache)) / fps, atol=1e-9, rtol=0):
                raise ValueError("Historical raw cache has a different time axis")
        rows, result = diagnostic.regression(avi, contours, expected, fps, calibration, cached)
        np.savetxt(
            output / f"test_{number}_centroids.csv",
            rows,
            delimiter=",",
            header="source_frame_1based,time_sec,x_px,y_px,x_um,y_um",
            comments="",
        )
        result.update(input=diagnostic.fingerprint(avi), sample_no=number, calibration_um_per_px=calibration.tolist())
        summary.append(result)
    diagnostic.write_json(output / "regression.json", summary)
    return all(
        s["frame_count_matches"] and s["pixel_mean_mismatches"] == 0 and s["cached_coordinate_mismatches"] in (None, 0)
        for s in summary
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "regression", "synthetic"))
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "centroid_evaluation.json")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / "centroid_evaluation")
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text())
    diagnostic.validate_config(config)
    try:
        output_root = diagnostic.private_output_root(ROOT, args.output_root)
    except ValueError as exc:
        parser.error(str(exc))
    if args.command == "inspect" and args.dataset is None:
        parser.error("Image inspection requires an explicit --dataset")
    output = output_root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    diagnostic.write_json(output / "protocol.json", config)
    diagnostic.write_json(output / "run.json", {"status": "running", "command": args.command})
    print(output, flush=True)
    try:
        if args.command == "inspect":
            assert args.dataset is not None
            dataset = args.dataset.resolve()
            settings = configparser.ConfigParser()
            settings.read(dataset / "config.ini")
            names = [name.strip() for name in settings.get("Tiff_info", "tiff_data").split(",")]
            if len(names) != 2:
                raise ValueError("This registered protocol requires exactly two measured samples")
            result = [
                inspect_sample(dataset, dataset / f"{name}.avi", i, config, output) for i, name in enumerate(names, 1)
            ]
            diagnostic.write_json(output / "summary.json", result)
        elif args.command == "regression":
            if not run_regression((args.dataset or ROOT / "data" / "test_data").resolve(), output):
                raise ValueError("Frame count or independent pixel-mean regression failed")
        else:
            # Development seeds only: generating holdout evidence would expose it.
            for seed in config["synthetic_seeds"]["development"]:
                for case in CASES:
                    sequence = generate_images(case, seed)
                    np.savez_compressed(output / f"{case}_{seed}.npz", **vars(sequence))
            diagnostic.write_json(
                output / "synthetic.json",
                dict(
                    cases=list(CASES),
                    seeds=config["synthetic_seeds"]["development"],
                    stage="A",
                    performance_comparison=False,
                ),
            )
    except (Exception, KeyboardInterrupt) as exc:
        diagnostic.write_json(output / "run.json", {"status": "failed", "error": str(exc)})
        raise
    diagnostic.write_json(output / "run.json", {"status": "complete", "command": args.command})


if __name__ == "__main__":
    main()
