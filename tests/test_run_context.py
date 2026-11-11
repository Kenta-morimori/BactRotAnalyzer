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
