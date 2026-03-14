import argparse
import os
import subprocess
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.functions import (  # noqa
    get_angular_velocity,
    input_data,
    make_graph,
    repellent_response,
    rot_df_manage,
    save2csv,
)
from utils import param  # noqa


def main(
    day: str,
    baseline_ratio: float,
    sigma_threshold: float,
    min_consecutive: int,
):
    repellent_root = f"{param.save_dir_bef}/{day}/repellent_response"
    os.makedirs(f"{repellent_root}/00_time_list", exist_ok=True)
    os.makedirs(f"{repellent_root}/01_brightness_change", exist_ok=True)
    os.makedirs(f"{repellent_root}/02_pre_rise_fluctuation", exist_ok=True)
    os.makedirs(f"{repellent_root}/03_post_rise_analysis/centroid_coordinate", exist_ok=True)
    repellent_response.cleanup_legacy_repellent_outputs(day)

    # Default repellent dataset may not have config.ini; create a minimal one when missing.
    repellent_response.ensure_repellent_config(day)

    # Keep time-list generation aligned with existing implementation.
    time_list = repellent_response.ensure_time_list(day)
    rot_df_manage.create_rot_df(day)
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
    motion_time_list = [time_list[i][: len(x_list[i])] for i in range(min(len(time_list), len(x_list), len(y_list)))]
    try:
        save2csv.save_time_list(motion_time_list, day)
        _, angular_velocity_list = get_angular_velocity.get_angular_velocity(x_list, y_list, day)
    finally:
        save2csv.save_time_list(time_list, day)

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

    # Phase 2-a: pre-rise fluctuation analysis (equivalent flow as --fluc on sliced interval).
    pre_time_list, pre_av_list = repellent_response.build_pre_rise_fluctuation_inputs(
        time_list=time_list,
        angular_velocity_list=angular_velocity_list,
        rise_indices=rise_indices,
    )
    repellent_response.run_pre_rise_fluctuation(pre_time_list, pre_av_list, day)

    # Phase 2-b: post-rise centroid time series.
    x_corrected_list, y_corrected_list = repellent_response.load_centroid_coordinate_with_nan(day)
    post_time_list, post_x_list, post_y_list = repellent_response.build_post_rise_centroid_series(
        time_list=time_list,
        x_list=x_corrected_list,
        y_list=y_corrected_list,
        rise_indices=rise_indices,
    )
    save2csv.save_repellent_post_rise_centroid(post_time_list, post_x_list, post_y_list, day)

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
