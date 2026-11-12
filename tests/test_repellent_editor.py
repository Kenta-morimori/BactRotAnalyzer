"""Synthetic-only HTML configuration, full-record browsing and local HTTP."""

import copy
import importlib.util
import json
import re
import socket
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from tests.test_repellent_image_input import settings_file
from utils.functions.repellent_editor import Editor, editor_server
from utils.functions.repellent_input import _write


def test_unset_threshold_uses_report_candidate_without_writing_or_replacing_saved_roi(tmp_path):
    path, settings = settings_file(tmp_path)
    sample = settings["samples"][0]
    sample.update(threshold=None, roi=None, seeds={}, target_confirmed=False)
    _write(path, settings)
    before = path.read_bytes()
    editor = Editor(path)
    assert editor.settings()["defaults"] == dict(threshold=2000)
    default = editor.preview(dict(sample_no=7))
    explicit = editor.preview(dict(sample_no=7, threshold=2000))
    assert default["mask"] == explicit["mask"]
    assert editor.settings()["samples"][0]["roi"] is None
    assert path.read_bytes() == before
    sample.update(threshold=12000, roi=[15, 15, 34, 34])
    _write(path, settings)
    saved = path.read_bytes()
    reused = editor.preview(dict(sample_no=7))
    explicit = editor.preview(dict(sample_no=7, threshold=12000, roi=sample["roi"]))
    assert reused["mask"] == explicit["mask"] and reused["mask"] != default["mask"]
    assert editor.settings()["samples"][0]["threshold"] == 12000
    assert editor.settings()["samples"][0]["roi"] == sample["roi"]
    assert path.read_bytes() == saved
    with pytest.raises(ValueError, match="Positive threshold"):
        editor.preview(dict(sample_no=7, threshold=0))


def test_full_record_slider_keeps_source_gaps_and_missing_clock(tmp_path):
    path, _ = settings_file(tmp_path, gap=True)
    editor = Editor(path)
    before = path.read_bytes()
    first = editor.preview(dict(sample_no=7, index=0))
    last = editor.preview(dict(sample_no=7, index=79))
    missing = editor.preview(dict(sample_no=7, index=30))
    assert first["count"] == last["count"] == 80
    assert first["source_frame"] == 1 and last["source_frame"] == 81
    assert missing["time_sec"] is None and first["required"] == [1, 32, 42]
    assert editor.locate(dict(sample_no=7, source_frame=42)) == dict(index=40)
    assert last["image"].startswith("data:image/png;base64,")
    with pytest.raises(ValueError, match="outside"):
        editor.preview(dict(sample_no=7, index=80))
    assert path.read_bytes() == before


def test_save_changes_only_selected_no_and_preserves_other_settings(tmp_path):
    path, settings = settings_file(tmp_path, gap=True)
    other = copy.deepcopy(settings["samples"][0])
    other.update(sample_no=9, input_id="second-target")
    settings["samples"].append(other)
    _write(path, settings)
    editor = Editor(path, numbers=[7])
    sample = settings["samples"][0]
    result = editor.save(
        dict(
            sample_no=7,
            revision=editor.settings()["revision"],
            threshold=2200,
            roi=sample["roi"],
            seeds=sample["seeds"],
            confirmed=True,
            order_confirmed=True,
        )
    )
    saved = json.loads(path.read_text())
    assert saved["samples"][0]["threshold"] == 2200
    assert saved["samples"][1] == other and saved["samples"][0]["background"] == sample["background"]
    assert saved["order_confirmed"] and saved["samples"][0]["target_confirmed"]
    assert [s["sample_no"] for s in result["samples"]] == [7]
    with pytest.raises(ValueError, match="unknown"):
        editor.preview(dict(sample_no=9))


def test_readiness_uses_persisted_settings_and_includes_hidden_unconfirmed_samples(tmp_path):
    path, settings = settings_file(tmp_path)
    other = copy.deepcopy(settings["samples"][0])
    other.update(sample_no=9, input_id="second-target", target_confirmed=False, seeds={})
    settings["samples"].append(other)
    _write(path, settings)
    before = path.read_bytes()
    status = Editor(path, numbers=[7]).settings()
    assert status["settings_path"] == str(path.resolve())
    ready = status["readiness"]
    assert ready["ready_samples"] == [7]
    assert ready["executable_command"].endswith("--input-samples 7")
    assert ready["missing_samples"] == [dict(sample_no=9, missing=["確認済みとして保存", "対象点"])]
    assert "現在は全件実行不可" in ready["summary"]
    assert path.read_bytes() == before
    settings["order_confirmed"] = False
    _write(path, settings)
    status = Editor(path, confirm_order=True).settings()
    assert status["order_confirmed"]  # The checkbox hint is not a persisted confirmation.
    assert status["readiness"]["ready_samples"] == []
    assert status["readiness"]["executable_command"] is None
    assert "撮影順の確認・保存" in status["readiness"]["summary"]
    settings["order_confirmed"] = True
    other.update(target_confirmed=True, seeds=settings["samples"][0]["seeds"])
    _write(path, settings)
    status = Editor(path).settings()["readiness"]
    assert status["executable_command"] == "python scripts/04_repellent_response_analysis.py --day day"
    assert status["missing_samples"] == []
    assert status["sample_count"] == 2


def test_invalid_or_stale_save_does_not_replace_settings(tmp_path):
    path, settings = settings_file(tmp_path, gap=True)
    editor = Editor(path)
    sample = settings["samples"][0]
    request = dict(
        sample_no=7,
        revision=editor.settings()["revision"],
        threshold=2000,
        roi=sample["roi"],
        seeds=sample["seeds"],
        confirmed=True,
        order_confirmed=True,
    )
    before = path.read_bytes()
    with pytest.raises(ValueError, match="未指定"):
        editor.save(dict(request, seeds={"1": sample["seeds"]["1"]}))
    with pytest.raises(ValueError, match="対象点"):
        editor.save(dict(request, seeds=dict(sample["seeds"], **{"1": [0, 0]})))
    assert path.read_bytes() == before
    _write(path, dict(settings, details=True))
    changed = path.read_bytes()
    with pytest.raises(ValueError, match="再読込"):
        editor.save(request)
    assert path.read_bytes() == changed


def test_draft_save_does_not_confirm_and_auto_roi_reuses_existing_inference(tmp_path):
    path, settings = settings_file(tmp_path)
    editor = Editor(path)
    sample = settings["samples"][0]
    auto = editor.auto_roi(dict(sample_no=7, seeds=sample["seeds"], threshold=2000))
    assert len(auto["roi"]) == 4 and auto["diagnostics"]["calibration_frames"] == 30
    editor.save(
        dict(
            sample_no=7,
            revision=editor.settings()["revision"],
            threshold=2000,
            roi=sample["roi"],
            seeds={},
            confirmed=False,
            order_confirmed=True,
        )
    )
    assert json.loads(path.read_text())["samples"][0]["target_confirmed"] is False
    status = editor.settings()["readiness"]
    assert status["ready_samples"] == [] and status["executable_command"] is None
    assert status["missing_samples"] == [dict(sample_no=7, missing=["確認済みとして保存", "対象点"])]
    assert "終了操作は保存しません" in status["summary"]


def test_http_requires_local_token_and_serves_full_record_preview(tmp_path):
    path, _ = settings_file(tmp_path)
    server = editor_server(path)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"
    try:
        with urlopen(base) as response:
            page = response.read().decode()
        matched = re.search(r"const token='([^']+)'", page)
        assert matched is not None
        token = matched.group(1)
        assert 'id="frame"' in page and 'id="sample"' in page and "https://" not in page
        with pytest.raises(HTTPError) as denied:
            urlopen(Request(base + "/save", data=b"{}", method="POST"))
        assert denied.value.code == 403
        request = Request(
            base + "/preview",
            data=b'{"sample_no":7,"index":79}',
            headers={"X-Editor-Token": token, "Content-Type": "application/json"},
        )
        with urlopen(request) as response:
            assert json.load(response)["source_frame"] == 80
        wrong_origin = Request(
            base + "/settings", data=b"{}", headers={"X-Editor-Token": token, "Origin": "https://example.invalid"}
        )
        with pytest.raises(HTTPError) as denied:
            urlopen(wrong_origin)
        assert denied.value.code == 403
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_configure_opens_html_for_only_requested_samples(tmp_path, monkeypatch):
    path, _ = settings_file(tmp_path)
    script = Path(__file__).resolve().parents[1] / "scripts/prepare_repellent_input.py"
    spec = importlib.util.spec_from_file_location("prepare_editor", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from utils.functions import repellent_editor

    opened = []
    monkeypatch.setattr(repellent_editor, "serve_editor", lambda *args: opened.append(args))
    monkeypatch.setattr(sys, "argv", [str(script), "--configure", "--settings", str(path), "--samples", "7"])
    module.main()
    assert opened == [(path, [7], False)]


@pytest.mark.parametrize("confirmed", [True, False])
def test_exit_returns_to_caller_closes_port_and_preserves_json(tmp_path, monkeypatch, capsys, confirmed):
    from utils.functions import repellent_editor

    path, settings = settings_file(tmp_path)
    if not confirmed:
        settings["samples"][0].update(target_confirmed=False, seeds={})
        _write(path, settings)
    before = path.read_bytes()
    opened = threading.Event()
    addresses = []

    def browser(url):
        addresses.append(url)
        opened.set()

    monkeypatch.setattr(repellent_editor.webbrowser, "open", browser)
    thread = threading.Thread(target=repellent_editor.serve_editor, args=(path,), daemon=True)
    thread.start()
    assert opened.wait(5)
    base = addresses[0].rstrip("/")
    with urlopen(base, timeout=5) as response:
        page = response.read().decode()
    matched = re.search(r"const token='([^']+)'", page)
    assert matched is not None
    with pytest.raises(HTTPError) as denied:
        urlopen(Request(base + "/exit", data=b"{}"), timeout=5)
    assert denied.value.code == 403 and thread.is_alive()
    with urlopen(
        Request(base + "/exit", data=b"{}", headers={"X-Editor-Token": matched.group(1)}), timeout=5
    ) as response:
        result = json.load(response)
    assert result["command"] == "python scripts/04_repellent_response_analysis.py --day day"
    if confirmed:
        assert result["ready_samples"] == [7] and result["missing_samples"] == []
    else:
        assert result["ready_samples"] == [] and result["executable_command"] is None
        assert "No.7 不足: 確認済みとして保存・対象点" in result["summary"]
    thread.join(5)
    assert not thread.is_alive() and path.read_bytes() == before
    port = int(base.rsplit(":", 1)[1])
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=1)
    assert "設定画面を終了しました" in capsys.readouterr().out


def test_ctrl_c_remains_available_without_saving(tmp_path, monkeypatch):
    from utils.functions import repellent_editor

    path, _ = settings_file(tmp_path)
    before = path.read_bytes()
    closed = []

    def interrupted():
        raise KeyboardInterrupt

    server = SimpleNamespace(server_port=12345, serve_forever=interrupted, server_close=lambda: closed.append(True))
    monkeypatch.setattr(repellent_editor, "editor_server", lambda *args: server)
    monkeypatch.setattr(repellent_editor.webbrowser, "open", lambda url: None)
    repellent_editor.serve_editor(path)
    assert closed == [True] and path.read_bytes() == before
