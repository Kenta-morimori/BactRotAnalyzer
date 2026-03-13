import argparse
import os
import subprocess
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from utils.functions import (  # noqa
    get_angular_velocity,
    input_data,
    make_graph,
    repellent_response,
    save2csv,
)
from utils import param  # noqa


def main(
    day: str,
    baseline_ratio: float,
    sigma_threshold: float,
    min_consecutive: int,
):
    os.makedirs(f"{param.save_dir_bef}/{day}/repellent_response", exist_ok=True)

    # Keep time-list generation aligned with existing implementation.
    time_list = repellent_response.ensure_time_list(day)

    # Reuse existing centroid / angular-velocity pipeline.
    centroid_csv = f"{param.save_dir_bef}/{day}/centroid_coordinate.csv"
    if not os.path.isfile(centroid_csv):
        subprocess.run(["python3", "utils/functions/get_centroid_coordinate.py", day], check=True)
    x_list, y_list = input_data.input_centroid_coordinate(day)
    _, angular_velocity_list = get_angular_velocity.get_angular_velocity(x_list, y_list, day)

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
    x_raw_list, y_raw_list = repellent_response.load_centroid_coordinate_with_nan(day)
    post_time_list, post_x_list, post_y_list = repellent_response.build_post_rise_centroid_series(
        time_list=time_list,
        x_list=x_raw_list,
        y_list=y_raw_list,
        rise_indices=rise_indices,
    )
    save2csv.save_repellent_post_rise_centroid(post_time_list, post_x_list, post_y_list, day)
    make_graph.plot_repellent_post_rise_centroid(post_time_list, post_x_list, post_y_list, day)


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
