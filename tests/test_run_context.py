"""Execution provenance uses temporary Git trees and synthetic inputs only."""

import hashlib
import json
import logging
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from utils.run_context import (
    REPOSITORY,
    analysis_run,
    input_inventory,
    require_clean_git,
)
from utils.run_inputs import diagnostic_inputs, manifest_inputs, repellent_inputs


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True, stderr=subprocess.PIPE).strip()


def commit(root):
    git(root, "add", ".")
    git(root, "-c", "user.name=Synthetic Test", "-c", "user.email=test@example.invalid", "commit", "-m", "Fixture")


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, "init", "-b", "codex/synthetic-runtime")
    (tmp_path / ".gitignore").write_text("outputs/\ndata/\n__pycache__/\n")
    (tmp_path / "analysis.py").write_text("# synthetic source\n")
    commit(tmp_path)
    return tmp_path


@pytest.mark.parametrize("change", ["modified", "staged", "untracked", "deleted"])
def test_dirty_execution_stops_before_creating_outputs(repository, change):
    source = repository / "analysis.py"
    if change == "deleted":
        source.unlink()
    elif change == "untracked":
        (repository / "new source with spaces.py").write_text("# untracked\n")
    else:
        source.write_text("# modified\n")
        if change == "staged":
            git(repository, "add", "analysis.py")
    with pytest.raises(RuntimeError, match="未コミット"):
        with analysis_run(repository=repository, parameters={}):
            pytest.fail("analysis must not start")
    assert not (repository / "outputs").exists()


def test_git_unavailable_or_unversioned_fails_closed(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="Git状態"):
        require_clean_git(tmp_path)
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="Git状態"):
        require_clean_git(tmp_path)


def test_empty_git_repository_and_nested_directory_rejected(repository, tmp_path):
    nested = repository / "data" / "nested"
    nested.mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Git状態"):
        require_clean_git(nested)
    empty = tmp_path / "empty"
    empty.mkdir()
    git(empty, "init")
    with pytest.raises(RuntimeError, match="Git状態"):
        require_clean_git(empty)


def test_success_records_revision_inputs_environment_and_streams(repository, capsys):
    data = repository / "data"
    data.mkdir()
    source = data / "synthetic.csv"
    source.write_text("frame,value\n1,2\n")
    original_handlers = list(logging.getLogger().handlers)
    with analysis_run(repository=repository, parameters={"day": "synthetic"}, inputs=[data, source]) as root:
        print("progress message")
        print("stderr message", file=sys.stderr)
        sys.stdout.writelines(["stdout line one\n", "stdout line two\n"])
        sys.stderr.writelines(["stderr line one\n", "stderr line two\n"])
        logging.warning("logging message")
        assert json.loads((root / "manifest.json").read_text())["status"] == "running"
    saved = json.loads((root / "manifest.json").read_text())
    assert saved["status"] == "complete"
    assert saved["git"]["branch"] == "codex/synthetic-runtime"
    assert saved["git"]["commit"] == git(repository, "rev-parse", "HEAD")
    assert saved["parameters"] == {"day": "synthetic"}
    assert saved["started_at"].endswith("+09:00") and saved["finished_at"].endswith("+09:00")
    assert saved["elapsed_seconds"] >= 0
    assert saved["environment"]["packages"] and saved["environment"]["executable"] == sys.executable
    assert saved["inputs"]["files"] == [
        {
            "path": str(source),
            "size_bytes": source.stat().st_size,
            "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        }
    ]
    log = (root / "run.log").read_text()
    assert all(message in log for message in ("progress message", "stderr message", "logging message", "Run complete"))
    assert "stdout line one\nstdout line two\n" in log
    assert "stderr line one\nstderr line two\n" in log
    console = capsys.readouterr()
    assert "progress message" in console.out and "stderr message" in console.err
    assert logging.getLogger().handlers == original_handlers
    assert require_clean_git(repository)["is_clean"]


@pytest.mark.parametrize(
    "exception,status",
    [(ValueError("synthetic failure"), "failed"), (KeyboardInterrupt(), "interrupted"), (SystemExit(2), "failed")],
)
def test_failure_and_interruption_are_durable(repository, exception, status):
    with pytest.raises(type(exception)):
        with analysis_run(repository=repository, parameters={}) as root:
            raise exception
    saved = json.loads((root / "manifest.json").read_text())
    assert saved["status"] == status and saved["error"]["type"] == type(exception).__name__
    assert type(exception).__name__ in (root / "run.log").read_text()
    assert saved["finished_at"]


@pytest.mark.parametrize("exit_code", [None, 0])
def test_successful_system_exit_records_completion_without_an_error(repository, exit_code):
    with pytest.raises(SystemExit) as caught:
        with analysis_run(repository=repository, parameters={}) as root:
            raise SystemExit(exit_code)
    assert caught.value.code == exit_code
    saved = json.loads((root / "manifest.json").read_text())
    assert saved["status"] == "complete" and "error" not in saved
    assert "SystemExit" not in (root / "run.log").read_text()


@pytest.mark.parametrize("change", ["uncommitted", "committed"])
def test_zero_exit_cannot_skip_the_final_revision_check(repository, change):
    with pytest.raises(RuntimeError):
        with analysis_run(repository=repository, parameters={}) as root:
            (repository / "analysis.py").write_text("# changed before exit\n")
            if change == "committed":
                commit(repository)
            raise SystemExit(0)
    saved = json.loads((root / "manifest.json").read_text())
    assert saved["status"] == "failed" and saved["error"]["type"] == "RuntimeError"


def test_bad_input_inventory_records_failure_before_analysis(repository):
    descriptor = repository / "data" / "bad.json"
    descriptor.parent.mkdir()
    descriptor.write_text("malformed")
    with pytest.raises(json.JSONDecodeError):
        with analysis_run(repository=repository, parameters={}, inputs=manifest_inputs(descriptor, "measurement")):
            pytest.fail("must not start")
    record = next((repository / "outputs/_runs").glob("*/*/manifest.json"))
    assert json.loads(record.read_text())["status"] == "failed"


def test_runs_do_not_collide_and_detached_worktree_is_supported(repository):
    git(repository, "checkout", "--detach")
    with analysis_run(repository=repository, parameters={}) as first:
        pass
    with analysis_run(repository=repository, parameters={}) as second:
        pass
    assert first != second and first.exists()
    assert json.loads((second / "manifest.json").read_text())["git"]["detached_head"]
    worktree = repository / "data" / "worktree"
    worktree.parent.mkdir(exist_ok=True)
    git(repository, "worktree", "add", "--detach", str(worktree))
    assert require_clean_git(worktree)["commit"] == git(repository, "rev-parse", "HEAD")


def test_source_edit_during_run_is_not_reported_complete(repository):
    with pytest.raises(RuntimeError, match="未コミット"):
        with analysis_run(repository=repository, parameters={}) as root:
            (repository / "analysis.py").write_text("# changed during run\n")
    assert json.loads((root / "manifest.json").read_text())["status"] == "failed"


@pytest.mark.parametrize(
    "kind,document,filenames",
    [
        (
            "measurement",
            {
                "samples": [
                    {
                        "centroids": "xy.csv",
                        "metadata": "xy.json",
                        "acquisition_tiff": "frame.tif",
                        "background": "bg.csv",
                    }
                ]
            },
            ["xy.csv", "xy.json", "frame.tif", "bg.csv"],
        ),
        ("results", {"observations": "xy.csv", "spectrum": "fft.csv"}, ["xy.csv", "fft.csv"]),
    ],
)
def test_manifest_transitive_inputs_are_hashed(tmp_path, kind, document, filenames):
    descriptor = tmp_path / "inputs.json"
    descriptor.write_text(json.dumps(document))
    for name in filenames:
        (tmp_path / name).write_bytes(b"synthetic")
    saved = input_inventory(manifest_inputs(descriptor, kind))
    assert {Path(item["path"]).name for item in saved["files"]} == {"inputs.json", *filenames}


@pytest.mark.parametrize("kind,expected_value", [("measurement", 1), ("results", 2)])
def test_descriptor_symlink_uses_the_same_relative_references_as_its_loader(
    tmp_path, monkeypatch, kind, expected_value
):
    from utils.functions import repellent_result_export

    original = tmp_path / "original"
    selected = tmp_path / "selected"
    original.mkdir()
    selected.mkdir()
    descriptor = original / "bundle.json"
    descriptor.write_text(
        json.dumps({"observations": "rows.csv", "settings": {}, "samples": [{"centroids": "rows.csv"}]})
    )
    (original / "rows.csv").write_text("marker\n1\n")
    (selected / "rows.csv").write_text("marker\n2\n")
    alias = selected / "bundle.json"
    alias.symlink_to(descriptor)
    saved = input_inventory(manifest_inputs(alias, kind))
    expected = (original if kind == "measurement" else selected) / "rows.csv"
    assert {item["path"] for item in saved["files"]} == {str(descriptor), str(expected)}
    assert (
        next(item["sha256"] for item in saved["files"] if item["path"] == str(expected))
        == hashlib.sha256(expected.read_bytes()).hexdigest()
    )
    if kind == "results":
        monkeypatch.setattr(
            repellent_result_export, "export_results", lambda root, *, settings, **tables: tables["observations"]
        )
        actual = repellent_result_export.export_bundle(alias, tmp_path / "out")
        assert actual.marker.tolist() == [expected_value]


def test_rerender_does_not_inventory_unrelated_raw_dataset(tmp_path):
    descriptor = tmp_path / "bundle.json"
    descriptor.write_text(json.dumps({"observations": "rows.csv"}))
    args = Namespace(measurement_input=None, results_bundle=str(descriptor), day="unused")
    assert list(repellent_inputs(args)) == [descriptor, tmp_path / "rows.csv"]


def test_diagnostic_inventory_includes_external_tiff_and_missing_cache(tmp_path):
    sample = tmp_path / "manifests" / "sample1"
    sample.mkdir(parents=True)
    frame = tmp_path / "external.tif"
    frame.write_bytes(b"synthetic")
    (sample / "timeline.json").write_text(json.dumps([{"tiff_path": str(frame)}]))
    args = Namespace(manifest_root=sample.parent, command="regression")
    saved = input_inventory(diagnostic_inputs(args, tmp_path))
    assert str(frame) in {item["path"] for item in saved["files"]}
    assert str(tmp_path / "data/test_data") in saved["missing"]


def test_inventory_follows_external_directory_symlinks_without_cycles(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    (external / "synthetic.csv").write_text("1,2\n")
    (source / "frames").symlink_to(external, target_is_directory=True)
    (external / "cycle").symlink_to(source, target_is_directory=True)
    saved = input_inventory([source])
    assert [item["path"] for item in saved["files"]] == [str(external / "synthetic.csv")]


@pytest.fixture
def cli_repository(repository):
    shutil.copytree(REPOSITORY / "utils", repository / "utils", ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copytree(
        REPOSITORY / "scripts", repository / "scripts", ignore=shutil.ignore_patterns("__pycache__", "dev_*")
    )
    (repository / "tests").mkdir()
    (repository / "tests/__init__.py").write_text("")
    shutil.copytree(
        REPOSITORY / "tests/synthetic", repository / "tests/synthetic", ignore=shutil.ignore_patterns("__pycache__")
    )
    commit(repository)
    return repository


def run_cli(repository, script, *args):
    return subprocess.run(
        [sys.executable, str(repository / script), *map(str, args)],
        cwd=repository,
        text=True,
        capture_output=True,
    )


@pytest.mark.parametrize(
    "script,arguments",
    [
        ("scripts/01_standard_analyisis/rotation_analysis_main.py", ["--day", "absent"]),
        ("scripts/02_compare_fluctuation_property/compare_fluctuation_main.py", ["--days", "absent"]),
        ("scripts/03_av_analysis/make_dataset.py", []),
        ("scripts/04_repellent_response_analysis.py", ["--day", "absent"]),
        (
            "scripts/04_repellent_response_analysis.py",
            ["--day", "absent", "--input-settings", "absent", "--input-source", "tiff"],
        ),
        ("scripts/tiff_roi_centroid.py", ["--input", "absent", "--output", "outputs/synthetic"]),
        ("scripts/05_centroid_diagnostics.py", ["synthetic"]),
        ("scripts/06_centroid_comparison.py", ["synthetic"]),
        ("scripts/07_centroid_review.py", ["--comparison", "absent", "--manifest-root", "absent"]),
        (
            "scripts/08_registered_centroid_review.py",
            ["--dataset", "absent", "--diagnostics", "absent", "--prior-review", "absent", "--raw-centroids", "absent"],
        ),
        ("utils/functions/get_centroid_coordinate.py", ["absent"]),
    ],
)
def test_every_analysis_cli_refuses_dirty_code_but_allows_help(cli_repository, script, arguments):
    (cli_repository / "untracked.py").write_text("# uncommitted source\n")
    result = run_cli(cli_repository, script, *arguments)
    assert result.returncode != 0 and "未コミット" in result.stderr, result.stderr
    assert not (cli_repository / "outputs").exists()
    result = run_cli(cli_repository, script, "--help")
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_real_repellent_cli_logs_synthetic_run_and_rejects_dirty_repeat(cli_repository):
    data = cli_repository / "data"
    data.mkdir()
    observations = data / "observations.csv"
    observations.write_text(
        "sample_no,source_frame_1based,time_sec,raw_x_um,raw_y_um,center_x_um,center_y_um,angular_velocity_rad_s\n"
        "1,1,0.0,1,0,0,0,\n1,2,0.005,0,1,0,0,1\n1,3,0.01,-1,0,0,0,1\n"
    )
    descriptor = data / "bundle.json"
    descriptor.write_text(
        json.dumps(
            {
                "observations": "observations.csv",
                "settings": {
                    "sign_convention": "CCW_positive_image_y_down",
                    "series_role": "synthetic",
                    "angular_sampling": "adjacent_source_frames",
                },
            }
        )
    )
    script = "scripts/04_repellent_response_analysis.py"
    arguments = ["--day", "synthetic", "--results-bundle", descriptor]
    result = run_cli(cli_repository, script, *arguments)
    assert result.returncode == 0, result.stderr
    records = list((cli_repository / "outputs/_runs").glob("*/*/manifest.json"))
    assert len(records) == 1
    saved = json.loads(records[0].read_text())
    assert saved["status"] == "complete" and saved["parameters"]["results_bundle"] == str(descriptor)
    assert {Path(item["path"]).name for item in saved["inputs"]["files"]} == {"bundle.json", "observations.csv"}
    assert "Response outputs:" in records[0].with_name("run.log").read_text()
    assert (cli_repository / "outputs/synthetic/repellent_response/output_manifest.csv").exists()
    (cli_repository / "analysis.py").write_text("# dirty\n")
    result = run_cli(cli_repository, script, *arguments)
    assert result.returncode != 0 and "未コミット" in result.stderr
    assert len(list((cli_repository / "outputs/_runs").glob("*/*/manifest.json"))) == 1


@pytest.mark.parametrize("with_brightness,implicit", [(False, False), (True, False), (True, True)])
def test_image_input_cli_logs_original_tiffs_and_exports_grouped_measurements(
    cli_repository, with_brightness, implicit
):
    from tests.test_repellent_image_input import mean_csv, settings_file

    draft, _ = settings_file(cli_repository)
    settings = cli_repository / ("data/day/repellent_input.json" if implicit else "data/settings.json")
    draft.replace(settings)
    if with_brightness:
        mean_csv(cli_repository)
        selection = ["--day", "day"] if implicit else ["--settings", settings]
        registered = run_cli(cli_repository, "scripts/prepare_repellent_input.py", "--attach-brightness", *selection)
        assert registered.returncode == 0, registered.stderr
    selection = [] if implicit else ["--input-settings", settings, "--input-source", "tiff"]
    result = run_cli(
        cli_repository,
        "scripts/04_repellent_response_analysis.py",
        "--day",
        "day",
        *selection,
        "--rise-method",
        "legacy",
        "--baseline-ratio",
        "0.4",
        "--sigma-threshold",
        "2.5",
        "--min-consecutive",
        "4",
    )
    assert result.returncode == 0, result.stderr
    records = list((cli_repository / "outputs/_runs").glob("*/*/manifest.json"))
    assert len(records) == 1
    saved = json.loads(records[0].read_text())
    assert saved["status"] == "complete"
    assert saved["parameters"]["input_source"] == (None if implicit else "tiff")
    files = {item["path"] for item in saved["inputs"]["files"]}
    assert str(settings) in files and len([p for p in files if p.endswith(".tif")]) == 80
    output = cli_repository / "outputs/reanalysis_tiff/day/repellent_response"
    assert (output / "output_manifest.csv").exists() and (output / "index.html").exists()
    assert (output / "00_all_rotational_analysis/centroid_coordinate/centroid_time_series.csv").exists()
    if with_brightness:
        import csv

        exported = json.loads((output / "settings.json").read_text())
        detection = exported["image_input"]["rise_results"]["7"]
        assert detection["method"] == "auto" and detection["status"] == "detected"
        assert detection["parameters"] == dict(baseline_ratio=0.4, sigma_threshold=2.5, min_consecutive=4)
        with (output / "01_brightness_change/rise_summary.csv").open() as stream:
            summary = next(csv.DictReader(stream))
        assert summary["rise_detection_method"] == "auto" and summary["rise_detection_status"] == "detected"
        assert int(summary["rise_source_frame_1based"]) == detection["rise_source_frame"]
        assert str(cli_repository / "data/day/brightness_data/recording.csv") in files
        assert not (cli_repository / "data/day/config.ini").exists()


def test_settings_cli_creates_default_json_without_overwriting_prior_choices(cli_repository):
    from tests.test_repellent_image_input import recording

    recording(cli_repository, frames=10)
    arguments = ["--init", "--day", "day", "--source", "tiff", "--px2um", "0.2", "0.2"]
    result = run_cli(cli_repository, "scripts/prepare_repellent_input.py", *arguments)
    assert result.returncode == 0, result.stderr
    path = cli_repository / "data/day/repellent_input.json"
    saved = path.read_bytes()
    document = json.loads(saved)
    assert document["source"] == "tiff" and not document["order_confirmed"]
    assert document["samples"][0]["target_confirmed"] is False
    repeated = run_cli(cli_repository, "scripts/prepare_repellent_input.py", *arguments)
    assert repeated.returncode != 0 and path.read_bytes() == saved


def test_batch_review_cli_approves_subset_then_day_only_analysis_keeps_original_number(cli_repository):
    import copy
    import re
    import time
    from urllib.request import Request, urlopen

    import pandas as pd

    from tests.test_repellent_image_input import settings_file

    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    draft, settings = settings_file(cli_repository, frames=24)
    settings["samples"][0].update(threshold=None, roi=None, seeds={}, target_confirmed=False)
    held = copy.deepcopy(settings["samples"][0])
    held.update(sample_no=9, input_id="not-visible", threshold=20000)
    settings["samples"].append(held)
    path = cli_repository / "data/day/repellent_input.json"
    path.write_text(json.dumps(settings))
    draft.unlink()
    code = (
        "import runpy,sys,webbrowser;webbrowser.open=lambda _:None;"
        "sys.argv=['scripts/prepare_repellent_input.py','--day','day','--configure'];"
        "runpy.run_path(sys.argv[0],run_name='__main__')"
    )
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        cwd=cli_repository,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        line = process.stdout.readline()
        matched = re.search(r"http://127.0.0.1:\d+", line)
        assert matched is not None, line
        base = matched.group(0)
        with urlopen(base, timeout=5) as response:
            matched = re.search(r"const token='([^']+)'", response.read().decode())
            assert matched is not None
            token = matched.group(1)

        def post(route, body):
            with urlopen(
                Request(base + route, data=json.dumps(body).encode(), headers={"X-Editor-Token": token}), timeout=20
            ) as response:
                return json.load(response)

        initial = post("/settings", {})
        before = path.read_bytes()
        post("/review/start", {})
        for _ in range(400):
            status = post("/review/status", {})
            if status["status"] not in {"running", "queued", "finalizing"}:
                break
            time.sleep(0.05)
        assert status["status"] == "complete", status
        assert path.read_bytes() == before
        assert status["entries"]["7"]["eligible"] and not status["entries"]["9"]["eligible"]
        post("/review/approve", dict(samples=[7], revision=initial["revision"]))
        post("/exit", {})
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == 0, stdout + stderr
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
    result = run_cli(cli_repository, "scripts/04_repellent_response_analysis.py", "--day", "day")
    assert result.returncode == 0, result.stderr
    output = cli_repository / "outputs/reanalysis_tiff/day/repellent_response"
    table = pd.read_csv(output / "00_all_rotational_analysis/centroid_coordinate/centroid_time_series.csv")
    assert table.sample_no.unique().tolist() == [7] and len(table) == 24
    saved = json.loads(path.read_text())
    assert saved["review"]["held_samples"] == [9]
    movie = Path(saved["samples"][0]["review_artifact"]).parent / "review.mp4"
    assert movie.is_file()
    records = [json.loads(p.read_text()) for p in (cli_repository / "outputs/_runs").glob("*/*/manifest.json")]
    assert len(records) == 2 and all(r["status"] == "complete" for r in records)
    assert any(r["parameters"].get("operation") == "central-target-fixed-roi-batch-review-v1" for r in records)
    inventory = {f["path"] for r in records for f in r["inputs"]["files"]}
    assert str(movie) in inventory
