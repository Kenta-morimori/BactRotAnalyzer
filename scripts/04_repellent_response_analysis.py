import argparse
import os
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils import param  # noqa
from utils.functions import input_data, make_graph, repellent_response, save2csv  # noqa


def main(
    day: str,
    baseline_ratio: float,
    sigma_threshold: float,
    min_consecutive: int,
):
    repellent_root = f"{param.save_dir_bef}/{day}/repellent_response"
    repellent_response.cleanup_legacy_repellent_outputs(day)
    os.makedirs(f"{repellent_root}/00_all_rotational_analysis", exist_ok=True)
    os.makedirs(f"{repellent_root}/00_time_list", exist_ok=True)
    os.makedirs(f"{repellent_root}/01_brightness_change", exist_ok=True)
    os.makedirs(f"{repellent_root}/02_pre_rise_fluctuation", exist_ok=True)
    os.makedirs(f"{repellent_root}/03_post_rise_analysis/centroid_coordinate", exist_ok=True)

    # Default repellent dataset may not have config.ini; create a minimal one when missing.
    repellent_response.ensure_repellent_config(day)

    # Keep time-list generation aligned with existing implementation.
    time_list = repellent_response.ensure_time_list(day)
    save2csv.save_repellent_time_list(time_list, day)
    make_graph.plot_repellent_time_list(time_list, day)

    # Reuse existing centroid / angular-velocity pipeline.
    centroid_csv = f"{param.save_dir_bef}/{day}/centroid_coordinate.csv"
    if not os.path.isfile(centroid_csv):
        try:
            subprocess.run(["python3", "utils/functions/get_centroid_coordinate.py", day], check=True)
        except subprocess.CalledProcessError:
            repellent_response.generate_centroid_coordinate_simple(day)
    x_list, y_list = input_data.input_centroid_coordinate(day)
    x_list, y_list = repellent_response.align_coordinate_series_to_time(x_list, y_list, time_list)

    # Phase 1: background intensity and rise-point detection.
    background_list = repellent_response.get_background_intensity_time_series(day)
    rise_results = repellent_response.detect_rise_points(
        background_list=background_list,
        baseline_ratio=baseline_ratio,
        sigma_threshold=sigma_threshold,
        min_consecutive=min_consecutive,
    )
    repellent_response.add_rise_time_to_results(rise_results, time_list)
    rise_indices = [result["rise_index"] for result in rise_results]

    save2csv.save_repellent_background_intensity(time_list, background_list, day)
    save2csv.save_repellent_rise_summary(rise_results, day)
    make_graph.plot_repellent_background_intensity(time_list, background_list, rise_indices, day)

    # Segment 0: all-time rotational analysis.
    repellent_response.save_repellent_segment_centroid_series(
        time_list=time_list,
        x_list=x_list,
        y_list=y_list,
        day=day,
        segment_subdir="00_all_rotational_analysis",
    )
    all_rot = repellent_response.run_segment_rotational_analysis(
        day=day,
        segment_subdir="00_all_rotational_analysis",
        time_list=time_list,
        x_list=x_list,
        y_list=y_list,
        run_fluctuation=False,
    )
    all_rise_time_list = []
    for idx in all_rot["valid_indices"]:
        if idx < len(rise_results):
            all_rise_time_list.append(rise_results[idx].get("rise_time", float("nan")))
        else:
            all_rise_time_list.append(float("nan"))
    repellent_response.save_segment_angular_velocity_outputs(
        day=day,
        segment_subdir="00_all_rotational_analysis",
        time_list=all_rot["time_list"],
        angle_list=all_rot["angle_list"],
        angular_velocity_list=all_rot["angular_velocity_list"],
        original_sample_indices=all_rot["valid_indices"],
        rise_time_list=all_rise_time_list,
    )
    repellent_response.copy_center_coordinate_to_segment(day, "00_all_rotational_analysis")
    (
        all_comp_time_list,
        all_x_before_list,
        all_y_before_list,
        all_x_center_list,
        all_y_center_list,
        all_x_corr_list,
        all_y_corr_list,
    ) = repellent_response.build_post_rise_coordinate_components(
        day=day,
        time_list=time_list,
        corrected_x_list=x_list,
        corrected_y_list=y_list,
        rise_indices=[0.0] * len(time_list),
    )
    make_graph.plot_repellent_component_panels(
        all_comp_time_list,
        all_x_before_list,
        all_y_before_list,
        day,
        "All-time Centroid Before Correction",
        "centroid_before.png",
        save_subdir="00_all_rotational_analysis/centroid_coordinate",
        rise_time_list=all_rise_time_list,
    )
    make_graph.plot_repellent_component_panels(
        all_comp_time_list,
        all_x_center_list,
        all_y_center_list,
        day,
        "All-time Rotation Center",
        "rotation_center.png",
        save_subdir="00_all_rotational_analysis/centroid_coordinate",
        rise_time_list=all_rise_time_list,
    )
    make_graph.plot_repellent_component_panels(
        all_comp_time_list,
        all_x_corr_list,
        all_y_corr_list,
        day,
        "All-time Centroid Corrected",
        "centroid_corrected.png",
        save_subdir="00_all_rotational_analysis/centroid_coordinate",
        rise_time_list=all_rise_time_list,
    )

    # Segment 1: pre-rise rotational + fluctuation analyses.
    pre_time_list, pre_x_raw_list, pre_y_raw_list = repellent_response.build_pre_rise_raw_centroid_series(
        day=day,
        time_list=time_list,
        corrected_x_list=x_list,
        corrected_y_list=y_list,
        rise_indices=rise_indices,
    )
    pre_center_x_list, pre_center_y_list = repellent_response.estimate_rotation_center_like_standard(
        time_list=pre_time_list,
        x_raw_list=pre_x_raw_list,
        y_raw_list=pre_y_raw_list,
    )
    pre_x_list, pre_y_list = repellent_response.subtract_center_from_raw(
        x_raw_list=pre_x_raw_list,
        y_raw_list=pre_y_raw_list,
        center_x_list=pre_center_x_list,
        center_y_list=pre_center_y_list,
    )

    repellent_response.save_repellent_segment_centroid_series(
        time_list=pre_time_list,
        x_list=pre_x_list,
        y_list=pre_y_list,
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
    )
    repellent_response.save_segment_angular_velocity_outputs(
        day=day,
        segment_subdir="02_pre_rise_fluctuation",
        time_list=pre_av_time_list,
        angle_list=pre_angle_list,
        angular_velocity_list=pre_av_list,
        original_sample_indices=all_rot["valid_indices"],
    )
    repellent_response.run_pre_rise_fluctuation(
        pre_time_list=pre_av_time_list,
        pre_angular_velocity_list=pre_av_list,
        day=day,
        original_sample_indices=all_rot["valid_indices"],
    )

    make_graph.plot_repellent_component_panels(
        pre_time_list,
        pre_x_raw_list,
        pre_y_raw_list,
        day,
        "Pre-rise Centroid Before Correction",
        "centroid_before.png",
        save_subdir="02_pre_rise_fluctuation/centroid_coordinate",
    )
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
        pre_x_list,
        pre_y_list,
        day,
        "Pre-rise Centroid Corrected",
        "centroid_corrected.png",
        save_subdir="02_pre_rise_fluctuation/centroid_coordinate",
    )

    # Segment 2: post-rise rotational analysis.
    x_corrected_list, y_corrected_list = repellent_response.load_centroid_coordinate_with_nan(day)
    post_time_list, post_x_list, post_y_list = repellent_response.build_post_rise_centroid_series(
        time_list=time_list,
        x_list=x_corrected_list,
        y_list=y_corrected_list,
        rise_indices=rise_indices,
    )
    repellent_response.save_repellent_segment_centroid_series(
        time_list=post_time_list,
        x_list=post_x_list,
        y_list=post_y_list,
        day=day,
        segment_subdir="03_post_rise_analysis",
        csv_name="post_rise_centroid_time_series.csv",
    )
    save2csv.save_repellent_post_rise_centroid(post_time_list, post_x_list, post_y_list, day)
    post_av_time_list, post_angle_list, post_av_list = repellent_response.split_rotational_series_by_rise(
        time_list=all_rot["time_list"],
        angle_list=all_rot["angle_list"],
        angular_velocity_list=all_rot["angular_velocity_list"],
        valid_indices=all_rot["valid_indices"],
        rise_indices=rise_indices,
        mode="post",
    )
    repellent_response.save_segment_angular_velocity_outputs(
        day=day,
        segment_subdir="03_post_rise_analysis",
        time_list=post_av_time_list,
        angle_list=post_angle_list,
        angular_velocity_list=post_av_list,
        original_sample_indices=all_rot["valid_indices"],
    )

    # Plot three modes as panel figures (x-y, x-t, y-t): before, center, corrected.
    (
        comp_time_list,
        x_before_list,
        y_before_list,
        x_center_list,
        y_center_list,
        x_corr_list,
        y_corr_list,
    ) = repellent_response.build_post_rise_coordinate_components(
        day=day,
        time_list=time_list,
        corrected_x_list=x_corrected_list,
        corrected_y_list=y_corrected_list,
        rise_indices=rise_indices,
    )
    make_graph.plot_repellent_component_panels(
        comp_time_list, x_before_list, y_before_list, day, "Centroid Before Correction", "centroid_before.png"
    )
    make_graph.plot_repellent_component_panels(
        comp_time_list, x_center_list, y_center_list, day, "Rotation Center", "rotation_center.png"
    )
    make_graph.plot_repellent_component_panels(
        comp_time_list, x_corr_list, y_corr_list, day, "Centroid Corrected", "centroid_corrected.png"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--day", type=str, default="repellent-response/20260312")
    parser.add_argument("--baseline-ratio", type=float, default=0.5)
    parser.add_argument("--sigma-threshold", type=float, default=3.0)
    parser.add_argument("--min-consecutive", type=int, default=3)

    args = parser.parse_args()

    main(
        day=args.day,
        baseline_ratio=args.baseline_ratio,
        sigma_threshold=args.sigma_threshold,
        min_consecutive=args.min_consecutive,
    )
