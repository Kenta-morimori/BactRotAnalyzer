"""Synthetic batch review, exact movie/measurement pairing and explicit subset approval."""

import copy
import json
import re
import shutil
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import cv2
import numpy as np
import pandas as pd
import pytest

from tests.test_repellent_image_input import settings_file
from utils.functions import repellent_input as inputs
from utils.functions import repellent_review as review
from utils.functions.repellent_editor import Editor, editor_server, saved_readiness
from utils.run_inputs import approved_image_numbers


@pytest.fixture
def fake_encoder(monkeypatch):
    counts = []

    def encode(path, frames, fps):
        count = 0
        for image in frames:
            assert image.shape == (740, 1280, 3) and image.dtype == np.uint8
            count += 1
        Path(path).write_bytes(b"synthetic-movie-" + str(count).encode())
        counts.append((count, fps))

    monkeypatch.setattr(review, "encode_video", encode)
    return counts


def draft(tmp_path, **kwargs):
    path, settings = settings_file(tmp_path, **kwargs)
    settings["samples"][0].update(threshold=None, roi=None, seeds={}, target_confirmed=False)
    inputs._write(path, settings)
    return path, settings


def finish(manager):
    assert manager.thread is not None
    manager.thread.join(20)
    assert not manager.thread.is_alive(), manager.status()
    assert manager.status()["status"] == "complete", manager.status()


def test_central_region_beats_brighter_decoy_and_seed_stays_in_ring_foreground():
    image = np.zeros((64, 64), np.uint16)
    cv2.circle(image, (32, 32), 6, 6000, -1)
    cv2.circle(image, (32, 32), 3, 0, -1)
    cv2.circle(image, (12, 12), 8, 20000, -1)
    seed, reason = review.central_seed(image, 2000)
    assert reason == "central_candidate" and seed is not None
    assert np.linalg.norm(np.asarray(seed) - [32, 32]) < 7
    tracker = review.Tracker(review.ROI(0, 0, 64, 64), 2000)
    result = tracker.step(image, 1, 0, seed=seed)
    assert result.point is not None
    np.testing.assert_allclose(result.point, [32, 32], atol=0.1)
    assert np.linalg.norm(np.asarray(seed) - result.point) > 2


def test_ambiguous_absent_and_edge_candidates_are_not_forced():
    image = np.zeros((64, 64), np.uint16)
    assert review.central_seed(image, 2000)[1] == "no_central_candidate"
    cv2.circle(image, (20, 31), 3, 10000, -1)
    cv2.circle(image, (43, 31), 3, 10000, -1)
    assert review.central_seed(image, 2000)[1] == "ambiguous_central_candidate"
    image[:] = 10000
    assert review.central_seed(image, 2000)[1] == "central_candidate_on_image_edge"


def test_insufficient_roi_support_keeps_full_field_preview_unapproved(tmp_path, fake_encoder):
    path, settings = draft(tmp_path, frames=10)
    result = review.generate(settings["samples"][0], path, tmp_path / "review", threading.Event(), lambda *_: None)
    assert not result["eligible"] and result["conditions"]["roi"] is None
    assert "automatic_roi_unavailable" in result["issues"]
    assert pd.read_csv(tmp_path / "review/centroids.csv").x_px.isna().all()
    assert fake_encoder == [(10, 50)]


def test_full_field_binary_context_is_visible_outside_the_fixed_roi():
    image = np.zeros((64, 64), np.uint16)
    cv2.circle(image, (32, 32), 3, 10000, -1)
    cv2.circle(image, (8, 8), 3, 10000, -1)
    bounds = [20, 20, 24, 24]
    sample = dict(input_id="synthetic", roi=bounds, threshold=2000, seeds={"1": [32, 32]})
    row = dict(source_frame_1based=1, time_sec=0.0, segment_id=0)
    found = review.Tracker(review.ROI(*bounds), 2000).step(image, 1, 0.0, seed=[32, 32])
    canvas = review.movie_frame(image, row, found, sample, [0, 10000], [])
    np.testing.assert_array_equal(canvas[100 + 8 * 10, 640 + 8 * 10], [255, 255, 255])
    assert found.point is not None
    np.testing.assert_allclose(found.point, [32, 32], atol=0.1)


def test_proposals_keep_saved_choices_and_reseed_each_independent_run(tmp_path):
    path, settings = draft(tmp_path, gap=True)
    sample = settings["samples"][0]
    rows = review.images.directory_rows(sample["tiff_directory"])
    candidate, issues, origins = review.propose(sample, rows)
    assert not issues and candidate["roi"] is not None
    assert set(candidate["seeds"]) == {"1", "32", "42"}
    assert candidate["threshold"] == 2000
    assert set(origins["seed_origins"].values()) == {"central_candidate"}
    sample.update(threshold=2200, roi=[15, 15, 34, 34], seeds={"1": [37, 32]})
    saved, issues, origins = review.propose(sample, rows)
    assert saved["threshold"] == 2200 and saved["roi"] == sample["roi"]
    assert saved["seeds"]["1"] == [37, 32] and origins["seed_origins"]["1"] == "saved"
    assert json.loads(path.read_text())["samples"][0]["roi"] is None


def test_movie_csv_pairing_gaps_and_approved_cache_reuse(tmp_path, fake_encoder, monkeypatch):
    path, settings = draft(tmp_path, gap=True)
    before = path.read_bytes()
    manager = review.ReviewManager(path, root=tmp_path / "reviews")
    manager.start()
    finish(manager)
    assert path.read_bytes() == before and fake_encoder == [(80, 50)]
    artifact = manager.artifact(7)
    detail = manager.details(7)
    assert detail["candidate"]["eligible"]
    mapping = detail["presentation"]["frames"]
    assert len(mapping) == 80 and mapping[-1]["source_frame_1based"] == 81 and mapping[30]["time_sec"] is None
    raw = pd.read_csv(artifact.parent / "centroids.csv", float_precision="round_trip")
    effective = dict(settings["samples"][0], **detail["candidate"]["conditions"])
    rows = review.images.directory_rows(effective["tiff_directory"])
    expected = inputs._extract(effective, rows, "tiff", path)
    pd.testing.assert_frame_equal(raw, expected, check_dtype=False)
    assert raw.iloc[30][["x_px", "y_px"]].isna().all()
    assert raw.target_id.iloc[0] != raw.target_id.iloc[31] != raw.target_id.iloc[40]
    approved = manager.approve([7], inputs._hash(path))
    assert approved["review"]["accepted_samples"] == [7]
    monkeypatch.setattr(inputs, "_extract", lambda *args: pytest.fail("approved measurements should be reused"))
    output = inputs.prepare_measurements(path, tmp_path / "analysis", source="tiff", day="day")
    entry = json.loads(output.read_text())["samples"][0]
    assert entry["sample_no"] == 7
    actual = pd.read_csv(entry["centroids"], float_precision="round_trip")
    pd.testing.assert_frame_equal(actual, raw)
    restarted = review.ReviewManager(path, root=tmp_path / "reviews")
    restarted.start()
    finish(restarted)
    assert restarted.artifact(7) == artifact and len(fake_encoder) == 1
    assert (artifact.parent.parent / "index.html").is_file()


def test_partial_approval_holds_failed_roi_and_preserves_original_numbers(tmp_path, fake_encoder):
    path, settings = draft(tmp_path)
    other = copy.deepcopy(settings["samples"][0])
    other.update(sample_no=9, input_id="not-visible", threshold=20000)
    settings["samples"].append(other)
    inputs._write(path, settings)
    manager = review.ReviewManager(path, root=tmp_path / "reviews")
    manager.start()
    finish(manager)
    assert manager.details(7)["candidate"]["eligible"]
    failed = manager.details(9)["candidate"]
    assert not failed["eligible"] and failed["conditions"]["roi"] is None
    assert (manager.artifact(9).parent / "review.mp4").is_file()
    assert pd.read_csv(manager.artifact(9).parent / "centroids.csv").x_px.isna().all()
    before = path.read_bytes()
    with pytest.raises(ValueError, match="未確定"):
        manager.approve([7, 9], inputs._hash(path))
    assert path.read_bytes() == before
    settings = manager.approve([7], inputs._hash(path))
    assert approved_image_numbers(settings) == [7]
    assert settings["review"]["held_samples"] == [9]
    with pytest.raises(ValueError, match="未承認"):
        approved_image_numbers(settings, [9])
    manifest = inputs.prepare_measurements(path, tmp_path / "analysis", source="tiff", day="day")
    assert [s["sample_no"] for s in json.loads(manifest.read_text())["samples"]] == [7]
    assert approved_image_numbers({"samples": settings["samples"]}) is None


def test_revision_edit_and_source_or_artifact_changes_require_new_review(tmp_path, fake_encoder):
    path, settings = draft(tmp_path)
    manager = review.ReviewManager(path, root=tmp_path / "reviews")
    manager.start()
    finish(manager)
    revision = inputs._hash(path)
    settings["details"] = True
    inputs._write(path, settings)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="別操作"):
        manager.approve([7], revision)
    assert path.read_bytes() == before
    manager.start()
    finish(manager)
    artifact = manager.artifact(7)
    approved = manager.approve([7], inputs._hash(path))
    changed = json.loads(path.read_text())
    changed["samples"][0]["threshold"] = 2300
    inputs._write(path, changed)
    with pytest.raises(ValueError, match="変わりました"):
        inputs.prepare_measurements(path, tmp_path / "analysis", source="tiff", day="day")
    inputs._write(path, approved)
    extra = Path(settings["samples"][0]["tiff_directory"]) / "recording_00100.tif"
    shutil.copy2(next(extra.parent.glob("*.tif")), extra)
    with pytest.raises(ValueError, match="元TIFF"):
        review.verify_artifact(artifact)
    extra.unlink()
    source = next(extra.parent.glob("*.tif"))
    original = source.read_bytes()
    source.write_bytes(b"changed source")
    with pytest.raises(ValueError, match="元TIFF"):
        review.verify_artifact(artifact)
    source.write_bytes(original)
    movie = artifact.parent / "review.mp4"
    movie.write_bytes(b"changed")
    with pytest.raises(ValueError, match="確認動画"):
        review.verify_artifact(artifact)


def test_regeneration_changes_only_selected_movie_and_preserves_manual_request(tmp_path, fake_encoder):
    path, settings = draft(tmp_path)
    other = copy.deepcopy(settings["samples"][0])
    other.update(sample_no=9, input_id="second")
    settings["samples"].append(other)
    inputs._write(path, settings)
    manager = review.ReviewManager(path, root=tmp_path / "reviews")
    manager.start()
    finish(manager)
    first, second = manager.artifact(7), manager.artifact(9)
    manager.start([7], {"7": {"threshold": 2200}})
    finish(manager)
    assert manager.artifact(7) != first and manager.artifact(9) == second and first.is_file()
    assert len(fake_encoder) == 3
    old = manager.approve([7, 9], inputs._hash(path))
    manager.start([7], {"7": {"threshold": 2300}})
    finish(manager)
    status = saved_readiness(old, manager.status())
    assert status["pending_review_samples"] == [7] and "前回承認済みの条件" in status["summary"]
    result = manager.approve([7, 9], inputs._hash(path))
    assert result["samples"][0]["threshold"] == 2300
    assert result["samples"][1]["threshold"] == 2000
    editor = Editor(path, review_root=tmp_path / "reviews")
    sample = result["samples"][0]
    editor.save(
        dict(
            sample_no=7,
            revision=inputs._hash(path),
            threshold=2400,
            roi=sample["roi"],
            seeds=sample["seeds"],
            confirmed=True,
            order_confirmed=True,
        )
    )
    edited = json.loads(path.read_text())
    assert edited["review"]["accepted_samples"] == [9]
    assert edited["samples"][0]["target_confirmed"] is False
    with pytest.raises(ValueError, match="未承認"):
        approved_image_numbers(edited, [7])


def test_interruption_is_not_approvable_and_completed_artifacts_survive(tmp_path, monkeypatch):
    path, _ = draft(tmp_path)
    ready, resume = threading.Event(), threading.Event()

    def encode(path, frames, fps):
        iterator = iter(frames)
        next(iterator)
        Path(path).write_bytes(b"partial")
        ready.set()
        assert resume.wait(5)
        for _ in iterator:
            pass

    monkeypatch.setattr(review, "encode_video", encode)
    manager = review.ReviewManager(path, root=tmp_path / "reviews")
    before = path.read_bytes()
    manager.start([7], {"7": {"threshold": 2100}})
    assert ready.wait(5)
    manager.interrupt()
    resume.set()
    manager.close()
    assert manager.status()["status"] == "interrupted" and path.read_bytes() == before
    with pytest.raises(ValueError, match="未完了"):
        manager.approve([7], inputs._hash(path))
    restored = review.ReviewManager(path, root=tmp_path / "reviews")
    assert restored.status()["entries"]["7"]["request"]["threshold"] == 2100

    def completed(path, frames, fps):
        for _ in frames:
            pass
        Path(path).write_bytes(b"complete")

    monkeypatch.setattr(review, "encode_video", completed)
    restored.start()
    finish(restored)
    assert restored.details(7)["candidate"]["conditions"]["threshold"] == 2100
    assert restored.approve([7], inputs._hash(path))["samples"][0]["threshold"] == 2100


def test_real_mp4_contains_every_frame_and_roi_panels_align(tmp_path):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg unavailable")
    path, settings = draft(tmp_path, frames=24)
    sample = settings["samples"][0]
    folder = tmp_path / "movie"
    result = review.generate(sample, path, folder, threading.Event(), lambda *_: None)
    assert result["eligible"] and result["frame_count"] == 24
    capture = cv2.VideoCapture(str(folder / "review.mp4"))
    count = 0
    while True:
        ok, image = capture.read()
        if not ok:
            break
        assert image.shape == (740, 1280, 3)
        count += 1
    capture.release()
    assert count == 24
    effective = dict(sample, **result["conditions"])
    rows = review.images.directory_rows(sample["tiff_directory"])
    record, plane, found = next(inputs._observations(effective, rows, "tiff", path))
    image = review.movie_frame(plane, rows[0], found, effective, [0, 10000], [])
    x, y, w, h = effective["roi"]
    left = image[100 + y * 10, x * 10]
    right = image[100 + y * 10, 640 + x * 10]
    np.testing.assert_array_equal(left, [40, 220, 40])
    np.testing.assert_array_equal(left, right)
    assert np.isfinite(record["x_px"])


def test_http_batch_approval_and_video_range_are_local_and_persistent(tmp_path, fake_encoder):
    path, _ = draft(tmp_path)
    server = editor_server(path, review_root=tmp_path / "reviews")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        page = urlopen(base).read().decode()
        matched = re.search(r"const token='([^']+)'", page)
        assert matched is not None
        token = matched.group(1)

        def post(route, body):
            with urlopen(
                Request(base + route, data=json.dumps(body).encode(), headers={"X-Editor-Token": token})
            ) as response:
                return json.load(response)

        initial = post("/settings", {})
        post("/review/start", {})
        finish(server.review)
        status = post("/review/status", {})
        assert status["entries"]["7"]["eligible"]
        url = base + "/review/media?sample_no=7&token=" + token
        with pytest.raises(HTTPError) as denied:
            urlopen(base + "/review/media?sample_no=7")
        assert denied.value.code == 403
        with urlopen(Request(url, headers={"Range": "bytes=0-3"})) as response:
            assert response.status == 206 and response.read() == b"synt"
        with pytest.raises(HTTPError) as invalid:
            urlopen(Request(url, headers={"Range": "bytes=10000-20000"}))
        assert invalid.value.code == 416
        result = post("/review/approve", dict(samples=[7], revision=initial["revision"]))
        assert result["readiness"]["executable_command"].endswith("--day day")
        video = server.review.artifact(7).parent / "review.mp4"
        post("/exit", {})
        thread.join(5)
        assert not thread.is_alive() and video.is_file()
    finally:
        server.review.close()
        server.shutdown()
        server.server_close()
        thread.join(5)
