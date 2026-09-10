import numpy as np

import matplotlib

matplotlib.use("Agg")

from utils import param
from utils.functions import make_graph


def test_page_save_name_preserves_legacy_name_for_one_page():
    assert make_graph._page_save_name("trajectory.png", 10, 1) == "trajectory.png"


def test_page_save_name_adds_suffix_for_multiple_pages():
    assert list(make_graph._sample_pages(21)) == [(1, 0, 10), (2, 10, 20), (3, 20, 21)]
    assert make_graph._page_save_name("trajectory.png", 21, 2) == "trajectory_part02.png"


def test_component_panels_split_and_keep_original_sample_numbers(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    values = [np.linspace(0, 1, 5) + i for i in range(11)]
    make_graph.plot_repellent_component_panels(
        values,
        values,
        values,
        "day",
        "Component",
        "components.png",
        rise_time_list=[0.5] * 11,
        overlay_x_list=values,
        overlay_y_list=values,
    )
    out_dir = tmp_path / "day" / "repellent_response" / "03_post_rise_analysis" / "centroid_coordinate"
    assert (out_dir / "components_part01.png").is_file()
    assert (out_dir / "components_part02.png").is_file()


def test_coordinate_with_center_splits_into_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(param, "save_dir_bef", str(tmp_path))
    monkeypatch.setattr(param, "get_config", lambda _day: (11, [], []))
    monkeypatch.setattr(make_graph.read_csv, "get_timelist", lambda _day: [list(range(5)) for _ in range(11)])
    values = [np.linspace(i, i + 1, 5) for i in range(11)]

    make_graph.plot_coordinate_with_center(values, values, values, values, "day")

    out_dir = tmp_path / "day" / "centroid_coordinate"
    assert (out_dir / "trajectory_with_center_part01.png").is_file()
    assert (out_dir / "trajectory_with_center_part02.png").is_file()
    assert (out_dir / "x_centroid_center_part01.png").is_file()
    assert (out_dir / "x_centroid_center_part02.png").is_file()
    assert (out_dir / "y_centroid_center_part01.png").is_file()
    assert (out_dir / "y_centroid_center_part02.png").is_file()
