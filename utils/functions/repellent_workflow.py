"""Recalculate accepted center/angle measurements, then publish grouped outputs.

A measurement manifest contains input coordinates and acquisition metadata,
never precomputed centers or velocities. Results-bundle export is separate.
"""

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import tifffile

from utils import param
from utils.functions.orbit_center.completion import complete_centers
from utils.functions.orbit_center.provisional_center import (
    configuration,
    estimate_provisional_center,
)
from utils.functions.orbit_center.raw_edges import edge_omega, input_hash
from utils.functions.repellent_metrics import (
    primary_fluctuation_features,
    switching_windows,
)


def acquisition_policy(image):
    with tifffile.TiffFile(image) as tif:
        page = tif.pages[0]
        if not isinstance(page, tifffile.TiffPage):
            raise ValueError("first TIFF page must expose acquisition metadata")
        value = page.tags["XPComment"].value
        text = value.decode("utf-16le") if isinstance(value, bytes) else str(value)
        stamp = str(page.tags["DateTime"].value)
        periods = [
            float(row[3])
            for row in (line.split("\t") for line in text.splitlines())
            if len(row) > 3 and row[2] == "INTERNAL FRAME INTERVAL"
        ]
    if len(periods) != 1 or "." not in stamp:
        raise ValueError("explicit frame period and timestamp precision required")
    quantum = 10.0 ** (-len(stamp.rsplit(".", 1)[1]))
    if not np.isfinite(periods[0]) or periods[0] <= 0:
        raise ValueError("invalid acquisition period")
    return dict(frame_period_s=periods[0], timestamp_quantum_s=quantum)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _known_id(values):
    labels = pd.Series(values)
    return (
        labels.notna() & ~labels.astype(str).str.strip().str.lower().isin(["", "none", "nan", "unknown", "<na>"])
    ).to_numpy()


def analyze_sample(
    raw,
    *,
    sample_no,
    source_id,
    centroid_method,
    frame_period_s,
    timestamp_quantum_s,
    background=None,
    rise_source_frame=None,
    identity,
    parameters=None,
    completion_points=None
):
    """Fit every reporting anchor using all original observations in its window."""
    f = raw.source_frame_1based.to_numpy()
    t = raw.time_sec.to_numpy(float)
    if (
        len(raw) == 0
        or not np.isfinite(f).all()
        or (f < 1).any()
        or (np.diff(f) <= 0).any()
        or (f != np.floor(f)).any()
    ):
        raise ValueError("positive increasing original source frames required")
    p = raw[["x_um", "y_um"]].to_numpy(float)
    # Fit in original pixel space when available, preserving previous numerical input exactly.
    fit_xy = raw[["x_px", "y_px"]].to_numpy(float) if {"x_px", "y_px"}.issubset(raw) else p.copy()
    scale = np.ones(2)
    if {"x_px", "y_px"}.issubset(raw):
        for axis in range(2):
            use = np.isfinite(fit_xy[:, axis]) & np.isfinite(p[:, axis]) & (fit_xy[:, axis] != 0)
            if not use.any():
                # A missing/zero axis has no nonzero conversion to infer. The
                # consistency check below still rejects conflicting coordinates.
                scale[axis] = 1.0
                continue
            scale[axis] = np.median(p[use, axis] / fit_xy[use, axis])
        if (
            not np.isfinite(scale).all()
            or (scale <= 0).any()
            or not np.allclose(fit_xy * scale, p, atol=1e-12, rtol=1e-12, equal_nan=True)
        ):
            raise ValueError("inconsistent pixel/physical coordinates")
    gap = frame_period_s + timestamp_quantum_s
    valid = np.isfinite(p).all(1) & np.isfinite(fit_xy).all(1) & np.isfinite(t)
    if "detected" in raw:
        valid &= raw.detected.astype(str).str.lower().isin(["true", "1"]).to_numpy()
    if "reacquired" in raw:
        valid &= ~raw.reacquired.astype(str).str.lower().isin(["true", "1"]).to_numpy()
    segments = np.cumsum(
        np.r_[False, (np.diff(f) != 1) | ~np.isfinite(np.diff(t)) | (np.diff(t) <= 0) | (np.diff(t) > gap)]
    )
    if "segment_id" in raw:
        values = raw.segment_id.to_numpy()
        segments += np.cumsum(np.r_[False, values[1:] != values[:-1]])
        valid &= _known_id(values)
    if "target_id" in raw:
        values = raw.target_id.to_numpy()
        segments += np.cumsum(np.r_[False, values[1:] != values[:-1]])
        valid &= _known_id(values)
    if not identity:
        raise ValueError("explicit observed-track identity or declared assumption required")
    options = dict(param.repellent_center_parameters if parameters is None else parameters)
    result = estimate_provisional_center(
        f,
        t,
        fit_xy,
        valid=valid,
        segment_ids=segments,
        frame_period_s=frame_period_s,
        timestamp_quantum_s=timestamp_quantum_s,
        evaluation_step=options["default_evaluation_step_frames"],
        parameters=options,
    )
    baseline = result["completed_center"] * scale
    direct = np.isfinite(baseline).all(1) & ~result["interpolated"]
    completion = complete_centers(
        t,
        f,
        p,
        baseline,
        direct,
        result["run_id"],
        points=param.repellent_center_completion_points if completion_points is None else completion_points,
        max_gap_s=gap,
    )
    status = completion["status"].copy()
    status[np.isfinite(baseline).all(1) & ~result["interpolated"]] = "direct"
    status[np.isfinite(baseline).all(1) & result["interpolated"]] = "short_interpolation"
    before = result["computed_center"] * scale
    baseline_omega = edge_omega(f, t, p, baseline, max_gap_s=gap)
    baseline_omega[1:][(result["run_id"][1:] < 0) | (np.diff(result["run_id"]) != 0)] = np.nan
    frame = pd.DataFrame(
        dict(
            sample_no=sample_no,
            source_id=source_id,
            centroid_method=centroid_method,
            source_frame_1based=f,
            time_sec=t,
            raw_x_um=p[:, 0],
            raw_y_um=p[:, 1],
            center_x_um=completion["center"][:, 0],
            center_y_um=completion["center"][:, 1],
            center_status=status,
            angular_velocity_rad_s=completion["omega"],
            uncompleted_angular_velocity_rad_s=baseline_omega,
            completion_boundary_edge=completion["boundary"],
            source_run_id=completion["run"],
            max_gap_s=gap,
            centroid_missing_reason=np.where(valid, "", "missing_or_invalid_observation"),
            center_before_x_um=before[:, 0],
            center_before_y_um=before[:, 1],
            center_before_evaluated=result["evaluated"],
            center_before_review_model_switch=result["review_model_switch"],
        )
    )
    frame["center_estimate_status"] = [row["status"] for row in result["records"]]
    frame["center_estimate_reason"] = [row.get("reason") or "" for row in result["records"]]
    frame["center_review_required"] = result["review_model_switch"]
    frame["center_estimate_window_s"] = [row.get("selected_window_s", np.nan) for row in result["records"]]
    frame["center_estimate_degree"] = [row.get("selected_center_degree", np.nan) for row in result["records"]]
    for column in ("missing_reason", "reason"):
        if column in raw:
            frame.loc[~valid, "centroid_missing_reason"] = (
                raw.loc[~valid, column].fillna("missing_or_invalid_observation").to_numpy()
            )
            break
    if background is not None:
        if len(background) != len(raw):
            raise ValueError("background/source rows differ")
        frame["background_intensity"] = background
    supports = []
    for block in completion["blocks"]:
        for i in range(block["start_row"], block["end_row"] + 1):
            times = block["left_times_s"] + block["right_times_s"]
            supports.append(
                dict(
                    sample_no=sample_no,
                    source_frame_1based=f[i],
                    time_sec=t[i],
                    completion_status=block["status"],
                    block_id=block["block_id"],
                    left_support_count=len(block["left_rows"]),
                    right_support_count=len(block["right_rows"]),
                    requested_points=block["requested_points"],
                    support_shortfall=block["support_shortfall"],
                    left_frames=json.dumps(block["left_frames"]),
                    right_frames=json.dumps(block["right_frames"]),
                    left_mean_time_s=block["left_mean_time_s"],
                    right_mean_time_s=block["right_mean_time_s"],
                    block_start_s=block["start_s"],
                    block_end_s=block["end_s"],
                    block_duration_s=block["duration_s"],
                    distance_to_nearest_support_s=min(abs(t[i] - v) for v in times) if times else np.nan,
                )
            )
    for i in np.flatnonzero(result["run_id"] < 0):
        supports.append(
            dict(
                sample_no=sample_no,
                source_frame_1based=f[i],
                time_sec=t[i],
                completion_status="unavailable",
                left_support_count=0,
                right_support_count=0,
                left_frames="[]",
                right_frames="[]",
                distance_to_nearest_support_s=np.nan,
            )
        )
    for i in np.flatnonzero(np.isfinite(baseline).all(1)):
        a, b = result["left_support_row"][i], result["right_support_row"][i]
        supports.append(
            dict(
                sample_no=sample_no,
                source_frame_1based=f[i],
                time_sec=t[i],
                completion_status=status[i],
                left_support_count=1,
                right_support_count=1,
                left_frames=json.dumps([int(f[a])]),
                right_frames=json.dumps([int(f[b])]),
                left_mean_time_s=t[a],
                right_mean_time_s=t[b],
                block_start_s=t[a],
                block_end_s=t[b],
                block_duration_s=t[b] - t[a],
                distance_to_nearest_support_s=min(abs(t[i] - t[a]), abs(t[i] - t[b])),
            )
        )
    return (
        frame,
        pd.DataFrame(supports).sort_values("source_frame_1based"),
        dict(
            source_hash=input_hash(f, t, fit_xy),
            identity=identity,
            configuration=result["configuration"],
            frame_period_s=frame_period_s,
            timestamp_quantum_s=timestamp_quantum_s,
            rise_source_frame=rise_source_frame,
            centroid_method=centroid_method,
        ),
    )


def measurement_metrics(observations, rise_frames):
    fluctuations, spectra, frequencies, candidates, features = [], [], [], [], []
    for no, frame in observations.groupby("sample_no", sort=False):
        cut = rise_frames.get(str(no))
        if cut is None:
            continue
        # Same metric support as the approved comparison: split at baseline availability changes.
        # At a right-endpoint edge the baseline center can exist even if the previous edge is absent.
        baseline = frame.center_status.isin(["direct", "short_interpolation"]).to_numpy()
        runs = frame.source_run_id.to_numpy()
        segment = np.r_[0, np.cumsum((np.diff(runs) != 0) | (baseline[1:] != baseline[:-1]))]
        metrics = pd.DataFrame(
            dict(
                source_frame_1based=frame.source_frame_1based.to_numpy(),
                time_sec=frame.time_sec.to_numpy(),
                angular_velocity_rad_s=frame.angular_velocity_rad_s.to_numpy(),
                segment_id=segment,
                source_segment_id=runs,
            )
        )
        pre = metrics.loc[metrics.source_frame_1based.lt(cut)].copy()
        if (
            len(pre) >= 2
            and np.isfinite(pre.time_sec.iloc[[0, -1]]).all()
            and pre.time_sec.iloc[-1] > pre.time_sec.iloc[0]
        ):
            feat, sd, spectrum = primary_fluctuation_features(
                pre,
                comparison_range=[float(pre.time_sec.iloc[0]), float(pre.time_sec.iloc[-1])],
                comparison_source="background rise; strictly preceding source frames; completion joins split support",
            )
            for table in (feat, sd, spectrum):
                table.insert(0, "sample_no", no)
            meta = [
                "sample_no",
                "sd_window_sec",
                "source_start_frame_1based",
                "source_end_frame_1based",
                "fft_start_sec",
                "fft_end_sec",
                "fft_sample_count",
                "effective_interval_sec",
            ]
            spectrum = spectrum.merge(feat[meta], on=["sample_no", "sd_window_sec"], how="left", validate="many_to_one")
            features.append(feat)
            fluctuations.append(sd)
            spectra.append(spectrum)
        post = metrics.loc[metrics.source_frame_1based.ge(cut)].copy()
        if len(post):
            windows = switching_windows(post, width=param.repellent_diagnostic_switch_window_s)
            windows.insert(0, "sample_no", no)
            windows["time_sec"] = windows.window_center_sec
            windows["frequency_hz"] = np.nan
            windows["diagnostic_zero_crossing_rate_hz"] = windows.switch_frequency_hz
            windows["physical_reversal_assessment"] = "not_assessed"
            frequencies.append(windows)
        v, f, t = (
            frame.angular_velocity_rad_s.to_numpy(),
            frame.source_frame_1based.to_numpy(),
            frame.time_sec.to_numpy(),
        )
        ix = (
            np.flatnonzero(
                np.isfinite(v[1:])
                & np.isfinite(v[:-1])
                & (v[1:] != 0)
                & (v[:-1] != 0)
                & (np.sign(v[1:]) != np.sign(v[:-1]))
                & (np.diff(f) == 1)
                & (np.diff(runs) == 0)
                & (runs[:-1] >= 0)
                & (f[:-1] >= cut)
            )
            + 1
        )
        for i in ix:
            candidates.append(
                dict(
                    sample_no=no,
                    source_frame_1based=f[i],
                    time_sec=t[i],
                    previous_source_frame_1based=f[i - 1],
                    previous_time_sec=t[i - 1],
                    previous_omega_rad_s=v[i - 1],
                    omega_rad_s=v[i],
                    completion_join_involved=bool(
                        frame.completion_boundary_edge.iloc[i] or frame.completion_boundary_edge.iloc[i - 1]
                    ),
                    status="diagnostic_adjacent_sign_change",
                    physical_reversal_assessment="not_assessed",
                )
            )

    def join(tables):
        return pd.concat(tables, ignore_index=True) if tables else None

    return dict(
        fluctuation=join(fluctuations),
        spectrum=join(spectra),
        frequency=join(frequencies),
        candidates=pd.DataFrame(candidates) if candidates else None,
    ), join(features)


def recalculate_manifest(path):
    """Resolve paths relative to a local input manifest; verify inputs before/after."""
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if not manifest.get("samples"):
        raise ValueError("measurement manifest requires samples")
    observations, supports, methods, hashes, rise_frames = [], [], {}, {str(path): _sha(path)}, {}

    def source(name):
        value = (path.parent / name).resolve()
        hashes[str(value)] = _sha(value)
        return value

    seen = set()
    for sample in manifest["samples"]:
        no = sample["sample_no"]
        if type(no) is not int or no < 1 or no in seen:
            raise ValueError("unique positive original sample numbers required")
        seen.add(no)
        raw = pd.read_csv(source(sample["centroids"]), float_precision="round_trip")
        metadata = json.loads(source(sample["metadata"]).read_text())
        if metadata.get("sample_no") != no or metadata.get("frame_count") != len(raw):
            raise ValueError("centroid metadata/sample identity mismatch")
        if {"x_px", "y_px"}.issubset(raw):
            scale = [metadata["px2um_x"], metadata["px2um_y"]]
            if not np.allclose(
                raw[["x_px", "y_px"]].to_numpy() * scale,
                raw[["x_um", "y_um"]].to_numpy(),
                rtol=1e-12,
                atol=1e-12,
                equal_nan=True,
            ):
                raise ValueError("centroid calibration differs from metadata")
        policy = acquisition_policy(source(sample["acquisition_tiff"]))
        background = None
        if "background" in sample:
            bg = pd.read_csv(source(sample["background"]), float_precision="round_trip")
            if not np.array_equal(
                bg[sample["background_time_column"]].to_numpy(), raw.time_sec.to_numpy(), equal_nan=True
            ):
                raise ValueError("background/source timestamps differ")
            background = bg[sample["background_column"]].to_numpy()
        cut = sample.get("rise_source_frame")
        if cut is not None:
            if type(cut) is not int or cut not in raw.source_frame_1based.to_numpy():
                raise ValueError("rise source frame outside measurements")
            rise_frames[str(no)] = cut
        frame, support, method = analyze_sample(
            raw,
            sample_no=no,
            source_id=metadata.get("avi_filename", Path(sample["centroids"]).stem),
            centroid_method=metadata["method"],
            background=background,
            rise_source_frame=cut,
            identity=sample["identity"],
            **policy
        )
        observations.append(frame)
        supports.append(support)
        methods[str(no)] = method
    table = pd.concat(observations, ignore_index=True)
    metrics, feature = measurement_metrics(table, rise_frames)
    for name, value in hashes.items():
        if _sha(name) != value:
            raise RuntimeError("source changed during calculation")
    settings = workflow_settings(rise_frames, methods)
    settings.update(input_manifest_sha256=_sha(path), input_hashes=hashes)
    for key in ("details", "samples_per_page", "zoom_ranges"):
        if key in manifest:
            settings[key] = manifest[key]
    return (
        dict(observations=table, supports=pd.concat(supports, ignore_index=True), settings=settings, **metrics),
        feature,
    )


def workflow_settings(rises, methods):
    from importlib.metadata import version

    root = Path(__file__).resolve().parents[2]
    sources = list((Path(__file__).parent / "orbit_center").glob("*.py")) + [
        Path(__file__),
        Path(__file__).with_name("repellent_metrics.py"),
        Path(__file__).with_name("observation_timebase.py"),
        Path(__file__).with_name("frequency_analysis.py"),
        Path(param.__file__),
    ]
    return dict(
        runtime_versions={name: version(name) for name in ("numpy", "pandas", "scipy")},
        code_sha256={str(p.relative_to(root)): _sha(p) for p in sources},
        sign_convention="CCW_positive_image_y_down",
        series_role="provisional_center_angular_diagnostic; uncompleted reference saved separately",
        angular_sampling="adjacent_source_frames",
        center_estimator="local_conic",
        center_method=configuration()["method"],
        center_completion_points=param.repellent_center_completion_points,
        coordinate_smoothing="none",
        angular_smoothing="none",
        physical_reversal_assessment="not_assessed",
        rise_source_frames=rises,
        sample_methods=methods,
        center_completion_fallback="available direct candidates from adjacent valid block; actual count saved",
        one_sided_policy="mean hold within observed run only",
        before_correction_definition="computed candidates before diagnostic gating, review and interpolation",
    )


def recalculate_day(day, *, all_frames=False, baseline_ratio=0.5, sigma_threshold=3.0, min_consecutive=3):
    """Normal execution from source AVIs/TIFF clocks and configured background boundaries."""
    from utils.functions import raw_centroid, repellent_response

    clocks = repellent_response.ensure_time_list(day)
    raw_frames = raw_centroid.load_all(day, clocks)
    centroid_method = raw_centroid.METHOD
    backgrounds = repellent_response.get_background_intensity_time_series(day, time_list=clocks)
    clocks, series, ranges = repellent_response.apply_analysis_frame_ranges(
        time_list=clocks,
        series_lists=[[r.x_um.to_list() for r in raw_frames], [r.y_um.to_list() for r in raw_frames], backgrounds],
        frame_ranges=[] if all_frames else param.get_analysis_frame_ranges_config(day),
    )
    backgrounds = series[2]
    rises = repellent_response.detect_rise_points(
        background_list=backgrounds,
        baseline_ratio=baseline_ratio,
        sigma_threshold=sigma_threshold,
        min_consecutive=min_consecutive,
    )
    repellent_response.add_rise_time_to_results(day, rises, clocks)
    names = repellent_response.get_tiff_sample_names(day)
    frames, supports, methods, cuts = [], [], {}, {}
    for i, (clock, original, name) in enumerate(zip(clocks, raw_frames, names)):
        first = int(cast(Any, ranges[i]["source_start_frame_1based"]))
        raw = original.iloc[first - 1 : first - 1 + len(clock)].copy().reset_index(drop=True)
        tiffs = sorted((Path(param.input_dir_bef) / day / "tiff_data" / name).glob("*.tif*"))
        if not tiffs:
            raise ValueError("local conic analysis requires acquisition TIFF metadata")
        policy = acquisition_policy(tiffs[0])
        index = rises[i].get("rise_index", np.nan)
        cut = first + int(index) if np.isfinite(index) else None
        if cut is not None:
            cuts[str(i + 1)] = cut
        frame, support, method = analyze_sample(
            raw,
            sample_no=i + 1,
            source_id=name,
            centroid_method=centroid_method,
            background=backgrounds[i],
            rise_source_frame=cut,
            identity="source track assumed within continuous observed runs; not independently verified",
            **policy
        )
        frames.append(frame)
        supports.append(support)
        methods[str(i + 1)] = method
    observations = pd.concat(frames, ignore_index=True)
    metrics, features = measurement_metrics(observations, cuts)
    settings = workflow_settings(cuts, methods)
    settings.update(
        source_frame_ranges=ranges,
        rise_detection=dict(
            baseline_ratio=baseline_ratio, sigma_threshold=sigma_threshold, min_consecutive=min_consecutive
        ),
        rise_results=rises,
    )
    return (
        dict(observations=observations, supports=pd.concat(supports, ignore_index=True), settings=settings, **metrics),
        features,
    )
