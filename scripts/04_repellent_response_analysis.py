import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Optional, cast

import numpy as np

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import param  # noqa
from utils.functions import (  # noqa
    input_data,
    make_graph,
    raw_centroid,
    repellent_response,
    repellent_sign_reference,
    save2csv,
)
from utils.functions.repellent_output import (  # noqa
    metadata,
    response_root,
    response_run,
    set_ranges,
)
from utils.run_context import analysis_run
from utils.run_inputs import repellent_inputs, resolve_image_settings


def resolve_center_options(center_estimator, stop_measurement_enabled):
    center_estimator = center_estimator if center_estimator is not None else param.center_estimator
    stop_measurement_enabled = (
        param.stop_measurement_enabled if stop_measurement_enabled is None else stop_measurement_enabled
    )
    if center_estimator not in {"local_conic", "v2", "validated", "legacy"}:
        raise ValueError(f"Unknown center estimator: {center_estimator}")
    return center_estimator, stop_measurement_enabled


def _main(
    day: str,
    baseline_ratio: float,
    sigma_threshold: float,
    min_consecutive: int,
    post_rise_center_mode: int,
    stop_activity_ratio: float,
    stop_min_duration_rotations: float,
    output_suffix: Optional[str],
    center_estimator: Optional[str] = None,
    all_frames: bool = False,
    stop_measurement_enabled: Optional[bool] = None,
):
    center_estimator, stop_measurement_enabled = resolve_center_options(center_estimator, stop_measurement_enabled)
    param.post_rise_center_mode = post_rise_center_mode

    repellent_root = str(response_root(day))
    os.makedirs(f"{repellent_root}/00_all_rotational_analysis", exist_ok=True)
    os.makedirs(f"{repellent_root}/00_time_list", exist_ok=True)
    os.makedirs(f"{repellent_root}/01_brightness_change", exist_ok=True)
    os.makedirs(f"{repellent_root}/02_pre_rise_fluctuation", exist_ok=True)
    os.makedirs(f"{repellent_root}/03_post_rise_analysis/centroid_coordinate", exist_ok=True)

    # Keep time-list generation aligned with existing implementation.
    time_list = repellent_response.ensure_time_list(day)
    save2csv.save_repellent_avi_tiff_sample_map(input_data.get_tiff_avi_sample_map(day), day)

    # Both analyses consume the same source-frame measurements. The standard
    # center-relative centroid_coordinate.csv is a separate compatibility file.
    raw_frames = raw_centroid.load_all(day, time_list)
    x_list = [frame["x_um"].to_list() for frame in raw_frames]
    y_list = [frame["y_um"].to_list() for frame in raw_frames]

    # Phase 1: background intensity and rise-point detection.
    background_list = repellent_response.get_background_intensity_time_series(day, time_list=time_list)
    time_list, selected_series, range_rows = repellent_response.apply_analysis_frame_ranges(
        time_list=time_list,
        series_lists=[x_list, y_list, background_list],
        frame_ranges=[] if all_frames else param.get_analysis_frame_ranges_config(day),
    )
    x_list, y_list, background_list = selected_series
    set_ranges(day, range_rows)
    # The canonical outputs/<day>/time_list.csv remains the raw TIFF timeline;
    # every repellent-response artifact below uses this selected timeline.
    save2csv.save_repellent_time_list(time_list, day)
    save2csv.save_repellent_analysis_frame_ranges(range_rows, day)
    make_graph.plot_repellent_time_list(time_list, day)
    rise_results = repellent_response.detect_rise_points(
        background_list=background_list,
        baseline_ratio=baseline_ratio,
        sigma_threshold=sigma_threshold,
        min_consecutive=min_consecutive,
    )
    repellent_response.add_rise_time_to_results(day, rise_results, time_list)
    rise_indices = [result["rise_index"] for result in rise_results]

    save2csv.save_repellent_background_intensity(time_list, background_list, day)
    save2csv.save_repellent_rise_summary(rise_results, day)
    make_graph.plot_repellent_background_intensity(time_list, background_list, rise_indices, day)

    # Build all-time centroid components once, then split into pre/post later.
    all_comp_time_list, all_x_before_list, all_y_before_list = time_list, x_list, y_list
    automatic_stop_indices = repellent_response.detect_rotation_stop_indices(
        time_list=all_comp_time_list,
        x_raw_list=all_x_before_list,
        y_raw_list=all_y_before_list,
        rise_indices=rise_indices,
        activity_ratio=stop_activity_ratio,
        min_duration_rotations=stop_min_duration_rotations,
    )
    stop_indices, stop_sources = repellent_response.resolve_rotation_stop_indices(
        time_list=all_comp_time_list,
        rise_indices=rise_indices,
        automatic_stop_indices=automatic_stop_indices,
        manual_stop_indices=param.get_manual_stop_frame_indices_config(day),
    )
    for i, result in enumerate(rise_results):
        result_metadata = cast(dict[str, object], result)
        stop_idx = stop_indices[i] if i < len(stop_indices) else float("nan")
        result["rotation_stop_index"] = stop_idx
        result_metadata["rotation_stop_source"] = stop_sources[i] if i < len(stop_sources) else "none"
        if np.isfinite(stop_idx) and i < len(all_comp_time_list):
            sample_time = all_comp_time_list[i]
            index = int(stop_idx)
            result["rotation_stop_time"] = sample_time[index] if index < len(sample_time) else float("nan")
        else:
            result["rotation_stop_time"] = float("nan")
    save2csv.save_repellent_rise_summary(rise_results, day)
    if center_estimator == "v2":
        all_x_center_list, all_y_center_list, all_rot_v2 = repellent_sign_reference.analyze(
            day,
            all_comp_time_list,
            all_x_before_list,
            all_y_before_list,
            stop_indices,
            source_starts=[int(cast(Any, row["source_start_frame_1based"])) for row in range_rows],
            stop_measurement_enabled=stop_measurement_enabled,
        )
    elif center_estimator == "validated":
        all_x_center_list, all_y_center_list, center_assessments = (
            repellent_response.estimate_rotation_center_validated(
                time_list=all_comp_time_list,
                x_raw_list=all_x_before_list,
                y_raw_list=all_y_before_list,
                stop_indices=stop_indices,
            )
        )
        repellent_response.save_center_assessments(day, center_assessments)
    elif center_estimator == "legacy":
        all_center_x_standard_list, all_center_y_standard_list = (
            repellent_response.estimate_rotation_center_like_standard(
                time_list=all_comp_time_list,
                x_raw_list=all_x_before_list,
                y_raw_list=all_y_before_list,
                rise_indices=rise_indices,
            )
        )
        all_x_center_list, all_y_center_list = repellent_response.apply_post_rise_center_strategy(
            time_list=all_comp_time_list,
            x_raw_list=all_x_before_list,
            y_raw_list=all_y_before_list,
            center_x_standard_list=all_center_x_standard_list,
            center_y_standard_list=all_center_y_standard_list,
            rise_indices=rise_indices,
            post_rise_center_mode=post_rise_center_mode,
            stop_indices=stop_indices,
        )
    else:
        raise ValueError(f"Unknown center estimator: {center_estimator}")
    all_x_corr_list, all_y_corr_list = repellent_response.subtract_center_from_raw(
        x_raw_list=all_x_before_list,
        y_raw_list=all_y_before_list,
        center_x_list=all_x_center_list,
        center_y_list=all_y_center_list,
    )

    # Segment 0: all-time rotational analysis.
    repellent_response.save_repellent_segment_centroid_series(
        time_list=all_comp_time_list,
        x_list=all_x_corr_list,
        y_list=all_y_corr_list,
        day=day,
        segment_subdir="00_all_rotational_analysis",
    )
    repellent_response.save_segment_center_coordinate(
        day=day,
        segment_subdir="00_all_rotational_analysis",
        center_x_list=all_x_center_list,
        center_y_list=all_y_center_list,
    )
    if center_estimator == "v2":
        all_rot = all_rot_v2
    else:
        all_rot = repellent_response.run_segment_rotational_analysis(
            day=day,
            segment_subdir="00_all_rotational_analysis",
            time_list=all_comp_time_list,
            x_list=all_x_corr_list,
            y_list=all_y_corr_list,
            run_fluctuation=False,
            write_time_list=False,
        )
    all_rise_time_for_av = []
    for idx in all_rot["valid_indices"]:
        if idx < len(rise_results):
            all_rise_time_for_av.append(rise_results[idx].get("rise_time", float("nan")))
        else:
            all_rise_time_for_av.append(float("nan"))
    repellent_response.save_segment_angular_velocity_outputs(
        day=day,
        segment_subdir="00_all_rotational_analysis",
        time_list=all_rot["time_list"],
        angle_list=all_rot["angle_list"],
        angular_velocity_list=all_rot["angular_velocity_list"],
        original_sample_indices=all_rot["valid_indices"],
        rise_time_list=all_rise_time_for_av,
        write_time_list=center_estimator == "v2",
        source_frame_list=all_rot.get("source_frame_list"),
    )
    bg_for_av: list[list[float]] = []
    for idx in all_rot["valid_indices"]:
        if idx < len(background_list):
            if center_estimator == "v2":
                sources = all_rot["source_frame_list"][len(bg_for_av)]
                source_start = int(cast(Any, range_rows[idx]["source_start_frame_1based"]))
                background = np.asarray(background_list[idx], float)
                bg_for_av.append(
                    [
                        float(background[int(source) - source_start]) if np.isfinite(source) else np.nan
                        for source in sources
                    ]
                )
            else:
                bg_for_av.append(background_list[idx])
        else:
            bg_for_av.append([])

    switching_time_list, switching_count_list = repellent_response.calculate_angular_velocity_switching_count(
        time_list=all_rot["time_list"],
        angular_velocity_list=all_rot["signed_angular_velocity_list"],
        window_width_sec=1.0,
    )
    save2csv.save_angular_velocity_switching_count(
        switching_time_list,
        switching_count_list,
        day,
        original_sample_indices=all_rot["valid_indices"],
    )
    make_graph.plot_repellent_background_av_and_switching_count_stacked(
        time_list=all_rot["time_list"],
        background_list=bg_for_av,
        angular_velocity_list=all_rot["angular_velocity_list"],
        switching_time_list=switching_time_list,
        switching_count_list=switching_count_list,
        day=day,
        sample_indices=[idx + 1 for idx in all_rot["valid_indices"]],
        rise_time_list=all_rise_time_for_av,
    )
    cw_rate_time_list, cw_rate_list = repellent_response.calculate_angular_velocity_cw_rate(
        time_list=all_rot["time_list"],
        angular_velocity_list=all_rot["signed_angular_velocity_list"],
        window_width_sec=1.0,
        missing_breaks_window=center_estimator == "v2",
    )
    save2csv.save_angular_velocity_cw_rate(
        cw_rate_time_list,
        cw_rate_list,
        day,
        original_sample_indices=all_rot["valid_indices"],
    )
    make_graph.plot_repellent_background_and_av_stacked(
        time_list=all_rot["time_list"],
        background_list=bg_for_av,
        angular_velocity_list=all_rot["angular_velocity_list"],
        cw_rate_time_list=cw_rate_time_list,
        cw_rate_list=cw_rate_list,
        day=day,
        sample_indices=[idx + 1 for idx in all_rot["valid_indices"]],
        rise_time_list=all_rise_time_for_av,
    )
    make_graph.plot_angular_velocity_switching_count(
        switching_time_list,
        switching_count_list,
        day,
        sample_indices=[idx + 1 for idx in all_rot["valid_indices"]],
        rise_time_list=all_rise_time_for_av,
    )

    all_rise_time_for_centroid = [result.get("rise_time", float("nan")) for result in rise_results]
    if center_estimator != "v2":
        make_graph.plot_repellent_component_panels(
            all_comp_time_list,
            all_x_before_list,
            all_y_before_list,
            day,
            "All-time Centroid Before Correction",
            "centroid_before.png",
            save_subdir="00_all_rotational_analysis/centroid_coordinate",
            rise_time_list=all_rise_time_for_centroid,
            overlay_x_list=all_x_center_list,
            overlay_y_list=all_y_center_list,
        )
    if center_estimator != "v2":
        make_graph.plot_repellent_component_panels(
            all_comp_time_list,
            all_x_center_list,
            all_y_center_list,
            day,
            "All-time Rotation Center",
            "rotation_center.png",
            save_subdir="00_all_rotational_analysis/centroid_coordinate",
            rise_time_list=all_rise_time_for_centroid,
        )
    make_graph.plot_repellent_component_panels(
        all_comp_time_list,
        all_x_corr_list,
        all_y_corr_list,
        day,
        "All-time Centroid Corrected",
        "centroid_corrected.png",
        save_subdir="00_all_rotational_analysis/centroid_coordinate",
        rise_time_list=all_rise_time_for_centroid,
    )

    # Segment 1: pre-rise rotational + fluctuation analyses.
    (
        pre_time_list,
        pre_x_raw_list,
        pre_y_raw_list,
        pre_center_x_list,
        pre_center_y_list,
        pre_x_corr_list,
        pre_y_corr_list,
    ) = repellent_response.split_coordinate_components_by_rise(
        time_list=all_comp_time_list,
        x_before_list=all_x_before_list,
        y_before_list=all_y_before_list,
        x_center_list=all_x_center_list,
        y_center_list=all_y_center_list,
        x_corr_list=all_x_corr_list,
        y_corr_list=all_y_corr_list,
        rise_indices=rise_indices,
        mode="pre",
    )

    repellent_response.save_repellent_segment_centroid_series(
        time_list=pre_time_list,
        x_list=pre_x_corr_list,
        y_list=pre_y_corr_list,
        day=day,
        segment_subdir="02_pre_rise_fluctuation",
    )
    repellent_response.save_segment_center_coordinate(
        day=day,
        segment_subdir="02_pre_rise_fluctuation",
        center_x_list=pre_center_x_list,
        center_y_list=pre_center_y_list,
    )
    pre_av_time_list, pre_angle_list, pre_av_list = repellent_response.split_rotational_series_by_rise(
        time_list=all_rot["time_list"],
        angle_list=all_rot["angle_list"],
        angular_velocity_list=all_rot["angular_velocity_list"],
        valid_indices=all_rot["valid_indices"],
        rise_indices=rise_indices,
        mode="pre",
        rise_time_list=all_rise_time_for_centroid if center_estimator == "v2" else None,
    )
    pre_source_frames = (
        [frames[: len(t)] for frames, t in zip(all_rot["source_frame_list"], pre_av_time_list)]
        if center_estimator == "v2"
        else None
    )
    repellent_response.save_segment_angular_velocity_outputs(
        day=day,
        segment_subdir="02_pre_rise_fluctuation",
        time_list=pre_av_time_list,
        angle_list=pre_angle_list,
        angular_velocity_list=pre_av_list,
        original_sample_indices=all_rot["valid_indices"],
        source_frame_list=pre_source_frames,
    )
    repellent_response.run_pre_rise_fluctuation(
        pre_time_list=pre_av_time_list,
        pre_angular_velocity_list=pre_av_list,
        day=day,
        original_sample_indices=all_rot["valid_indices"],
    )

    if center_estimator != "v2":
        make_graph.plot_repellent_component_panels(
            pre_time_list,
            pre_x_raw_list,
            pre_y_raw_list,
            day,
            "Pre-rise Centroid Before Correction",
            "centroid_before.png",
            save_subdir="02_pre_rise_fluctuation/centroid_coordinate",
            overlay_x_list=pre_center_x_list,
            overlay_y_list=pre_center_y_list,
        )
    if center_estimator != "v2":
        make_graph.plot_repellent_component_panels(
            pre_time_list,
            pre_center_x_list,
            pre_center_y_list,
            day,
            "Pre-rise Rotation Center",
            "rotation_center.png",
            save_subdir="02_pre_rise_fluctuation/centroid_coordinate",
        )
    make_graph.plot_repellent_component_panels(
        pre_time_list,
        pre_x_corr_list,
        pre_y_corr_list,
        day,
        "Pre-rise Centroid Corrected",
        "centroid_corrected.png",
        save_subdir="02_pre_rise_fluctuation/centroid_coordinate",
    )

    # Segment 2: post-rise rotational analysis.
    (
        post_time_list,
        post_x_raw_list,
        post_y_raw_list,
        post_center_x_list,
        post_center_y_list,
        post_x_corr_list,
        post_y_corr_list,
    ) = repellent_response.split_coordinate_components_by_rise(
        time_list=all_comp_time_list,
        x_before_list=all_x_before_list,
        y_before_list=all_y_before_list,
        x_center_list=all_x_center_list,
        y_center_list=all_y_center_list,
        x_corr_list=all_x_corr_list,
        y_corr_list=all_y_corr_list,
        rise_indices=rise_indices,
        mode="post",
    )
    repellent_response.save_repellent_segment_centroid_series(
        time_list=post_time_list,
        x_list=post_x_corr_list,
        y_list=post_y_corr_list,
        day=day,
        segment_subdir="03_post_rise_analysis",
        csv_name="post_rise_centroid_time_series.csv",
    )
    repellent_response.save_segment_center_coordinate(
        day=day,
        segment_subdir="03_post_rise_analysis",
        center_x_list=post_center_x_list,
        center_y_list=post_center_y_list,
    )
    save2csv.save_repellent_post_rise_centroid(post_time_list, post_x_corr_list, post_y_corr_list, day)
    post_av_time_list, post_angle_list, post_av_list = repellent_response.split_rotational_series_by_rise(
        time_list=all_rot["time_list"],
        angle_list=all_rot["angle_list"],
        angular_velocity_list=all_rot["angular_velocity_list"],
        valid_indices=all_rot["valid_indices"],
        rise_indices=rise_indices,
        mode="post",
        rise_time_list=all_rise_time_for_centroid if center_estimator == "v2" else None,
    )
    post_source_frames = (
        [frames[len(frames) - len(t) :] for frames, t in zip(all_rot["source_frame_list"], post_av_time_list)]
        if center_estimator == "v2"
        else None
    )
    repellent_response.save_segment_angular_velocity_outputs(
        day=day,
        segment_subdir="03_post_rise_analysis",
        time_list=post_av_time_list,
        angle_list=post_angle_list,
        angular_velocity_list=post_av_list,
        original_sample_indices=all_rot["valid_indices"],
        source_frame_list=post_source_frames,
    )

    # Plot three modes as panel figures (x-y, x-t, y-t): before, center, corrected.
    if center_estimator != "v2":
        make_graph.plot_repellent_component_panels(
            post_time_list,
            post_x_raw_list,
            post_y_raw_list,
            day,
            "Centroid Before Correction",
            "centroid_before.png",
            overlay_x_list=post_center_x_list,
            overlay_y_list=post_center_y_list,
        )
    if center_estimator != "v2":
        make_graph.plot_repellent_component_panels(
            post_time_list,
            post_center_x_list,
            post_center_y_list,
            day,
            "Rotation Center",
            "rotation_center.png",
        )
    make_graph.plot_repellent_component_panels(
        post_time_list,
        post_x_corr_list,
        post_y_corr_list,
        day,
        "Centroid Corrected",
        "centroid_corrected.png",
    )

    if center_estimator == "v2":
        from utils.functions.repellent_plotting import phase_figures

        phase_figures(
            day,
            all_rot["reference_samples"],
            all_rise_time_for_centroid,
            rise_indices=rise_indices,
            source_starts=[int(cast(Any, row["source_start_frame_1based"])) for row in range_rows],
        )

    (Path(repellent_root) / "analysis_method.json").write_text(
        json.dumps(
            {
                **metadata(day),
                "center_estimator": center_estimator,
                "all_frames": all_frames,
                "stop_measurement_enabled": bool(stop_measurement_enabled and center_estimator == "v2"),
                "unknown": "NaN; no interpolation or carry-forward",
                "sign_convention": "CW +, CCW -",
            },
            indent=2,
        )
        + "\n"
    )
    from utils.functions.repellent_result_adapter import collect_results

    return collect_results(
        day,
        range_rows,
        raw_frames,
        all_comp_time_list,
        all_x_before_list,
        all_y_before_list,
        all_x_center_list,
        all_y_center_list,
        all_rot,
        background_list,
        rise_results,
        center_estimator,
        source_ids=[Path(path).stem for path in input_data.get_ordered_avi_paths(day)],
    )


def main(
    day: str,
    baseline_ratio: float,
    sigma_threshold: float,
    min_consecutive: int,
    post_rise_center_mode: int,
    stop_activity_ratio: float,
    stop_min_duration_rotations: float,
    output_suffix: Optional[str],
    center_estimator: Optional[str] = None,
    all_frames: bool = False,
    stop_measurement_enabled: Optional[bool] = None,
    results_bundle: Optional[str] = None,
    measurement_input: Optional[str] = None,
    input_settings: Optional[str] = None,
    input_source: Optional[str] = None,
    input_samples: Optional[list[int]] = None,
    force_extract: bool = False,
):
    from utils.functions.repellent_result_export import export_bundle, export_results

    if measurement_input is None and results_bundle is None:
        input_settings, input_source = resolve_image_settings(day, input_settings, input_source)
    base = Path(param.save_dir_bef)
    if input_settings is not None and output_suffix is None:
        output_suffix = f"reanalysis_{input_source}" if input_source in {"avi", "tiff"} else None
    if output_suffix:
        suffix = Path(output_suffix)
        if suffix.is_absolute() or ".." in suffix.parts:
            raise ValueError("output-suffix must be relative without parent traversal")
        base = base / suffix
    destination = base / day / "repellent_response"
    if input_settings is not None:
        if measurement_input is not None or results_bundle is not None:
            raise ValueError("input-settings, measurement-input and results-bundle are mutually exclusive")
        if input_source not in {"avi", "tiff"} or center_estimator not in (None, "local_conic"):
            raise ValueError("input-settings requires explicit input-source and local_conic estimator")
        from utils.functions.repellent_input import prepare_measurements

        measurement_input = str(
            prepare_measurements(
                input_settings,
                base / day,
                source=input_source,
                day=day,
                numbers=input_samples,
                all_frames=all_frames,
                force=force_extract,
                rise_detection=dict(
                    baseline_ratio=baseline_ratio, sigma_threshold=sigma_threshold, min_consecutive=min_consecutive
                ),
            )
        )
        all_frames = False  # The generated manifest now owns the resolved source scope.
    elif input_source is not None or input_samples is not None or force_extract:
        raise ValueError("input-source, input-samples and force-extract require input-settings")
    if results_bundle is not None and measurement_input is not None:
        raise ValueError("results-bundle and measurement-input are mutually exclusive")
    if results_bundle is not None:
        export_bundle(results_bundle, destination)
        print(f"Response outputs: {destination.resolve()}", flush=True)
        return
    estimator, stopped = resolve_center_options(center_estimator, stop_measurement_enabled)
    if measurement_input is not None or estimator == "local_conic":
        from utils.functions.repellent_workflow import (
            recalculate_day,
            recalculate_manifest,
        )

        if measurement_input is not None and (all_frames or center_estimator not in (None, "local_conic")):
            raise ValueError(
                "measurement-input defines source scope; do not mix centroid bundle, all-frames or another estimator"
            )
        if measurement_input is not None:
            bundle, features = recalculate_manifest(measurement_input)
        else:
            bundle, features = recalculate_day(
                day,
                all_frames=all_frames,
                baseline_ratio=baseline_ratio,
                sigma_threshold=sigma_threshold,
                min_consecutive=min_consecutive,
            )
        for source in bundle["settings"].get("input_hashes", {}):
            if Path(source).resolve().is_relative_to(destination.resolve()):
                raise ValueError("output contains a measurement input; choose a separate output-suffix")
        export_results(destination, fft_features=features, **bundle)
        print(f"Response outputs: {destination.resolve()}", flush=True)
        return
    options = dict(
        center_estimator=estimator,
        stop_measurement_enabled=bool(stopped and estimator == "v2"),
        input_centroid_method=raw_centroid.METHOD,
    )
    previous_mode = param.post_rise_center_mode
    try:
        # Existing backends may create diagnostic/per-sample intermediates. Keep
        # those temporary and publish only the approved phase-grouped layout.
        destination.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="repellent_export_") as temporary:
            with response_run(
                day,
                all_frames=all_frames,
                output_suffix=output_suffix,
                options=options,
                root_override=Path(temporary) / "analysis",
            ):
                bundle = _main(
                    day,
                    baseline_ratio,
                    sigma_threshold,
                    min_consecutive,
                    post_rise_center_mode,
                    stop_activity_ratio,
                    stop_min_duration_rotations,
                    output_suffix,
                    estimator,
                    all_frames,
                    stopped,
                )
            bundle["settings"] = {**options, **bundle["settings"], "all_frames": all_frames}
            export_results(destination, **bundle)
            print(f"Response outputs: {destination.resolve()}", flush=True)
    except Exception as error:
        (destination / "settings.json").write_text(
            json.dumps(dict(options, status="failed", error_type=type(error).__name__)) + "\n"
        )
        raise
    finally:
        param.post_rise_center_mode = previous_mode


def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--day", type=str, default="repellent-response/20260312")
    parser.add_argument("--baseline-ratio", type=float, default=0.5)
    parser.add_argument("--sigma-threshold", type=float, default=3.0)
    parser.add_argument("--min-consecutive", type=int, default=3)
    parser.add_argument("--post-rise-center-mode", type=int, choices=[1, 2, 3], default=param.post_rise_center_mode)
    parser.add_argument("--center-estimator", choices=["local_conic", "v2", "validated", "legacy"], default=None)
    parser.add_argument(
        "--stop-measurement",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save separate stopped-orbit measurements for V2 (default: param)",
    )
    parser.add_argument(
        "--all-frames", action="store_true", help="Analyze every saved frame, overriding configured frame ranges"
    )
    parser.add_argument("--stop-activity-ratio", type=float, default=param.stop_activity_ratio)
    parser.add_argument("--stop-min-duration-rotations", type=float, default=param.stop_min_duration_rotations)
    parser.add_argument("--output-suffix", type=str, default=None)
    parser.add_argument("--input-settings", type=str, help="Confirmed local TIFF/AVI settings; generate measurements")
    parser.add_argument("--input-source", choices=["tiff", "avi"], help="Explicit source matching input-settings")
    parser.add_argument("--input-samples", nargs="+", type=int, help="Original sample numbers; never renumber")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract this input's isolated centroid cache")
    parser.add_argument(
        "--measurement-input",
        type=str,
        default=None,
        help="Recalculate centers/angles/metrics from an explicit local source manifest",
    )
    parser.add_argument(
        "--results-bundle",
        type=str,
        default=None,
        help="Export existing source-aligned results without recomputing measurements",
    )

    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    with analysis_run(parameters=vars(args), inputs=repellent_inputs(args), outputs=[param.save_dir_bef]):
        main(
            day=args.day,
            baseline_ratio=args.baseline_ratio,
            sigma_threshold=args.sigma_threshold,
            min_consecutive=args.min_consecutive,
            post_rise_center_mode=args.post_rise_center_mode,
            stop_activity_ratio=args.stop_activity_ratio,
            stop_min_duration_rotations=args.stop_min_duration_rotations,
            output_suffix=args.output_suffix,
            center_estimator=args.center_estimator,
            all_frames=args.all_frames,
            stop_measurement_enabled=args.stop_measurement,
            results_bundle=args.results_bundle,
            measurement_input=args.measurement_input,
            input_settings=args.input_settings,
            input_source=args.input_source,
            input_samples=args.input_samples,
            force_extract=args.force_extract,
        )
