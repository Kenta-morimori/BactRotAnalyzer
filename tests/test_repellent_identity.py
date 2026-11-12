"""Read-only migration of audited full-file image evidence, with fail-closed checks."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.test_repellent_image_input import prepare, settings_file
from tests.test_repellent_review import draft, fake_encoder, finish  # noqa: F401
from utils.functions import repellent_identity as identity
from utils.functions import repellent_input as inputs
from utils.functions import repellent_review as review


def legacy_review(current):
    old = copy.deepcopy(current)
    old.pop("identity")
    old["code"].update({"repellent_input.py": identity.LEGACY_INPUT, "repellent_review.py": identity.LEGACY_REVIEW})
    return old


def test_audited_image_scopes_match_and_background_only_edit_is_independent(tmp_path):
    folder = Path(inputs.__file__).parent
    for name, digest in identity.LEGACY_SCOPES.items():
        assert identity.image_scope(folder / name) == digest
    path = tmp_path / "repellent_input.py"
    source = Path(inputs.__file__).read_text()
    path.write_text(source + "\n# A formatting-only change\n")
    original = identity.image_scope(path)
    start, end = source.index("def _background_rise("), source.index("def prepare_measurements(")
    path.write_text(source[:start] + "def _background_rise(*args):\n    return 'another detector'\n\n\n" + source[end:])
    assert identity.image_scope(path) == original
    path.write_text(path.read_text().replace("TIFF_INITIAL_THRESHOLD = 2000", "TIFF_INITIAL_THRESHOLD = 2100"))
    assert identity.image_scope(path) != original


@pytest.mark.parametrize("change", ["input", "review", "tracker", "runtime", "unknown_legacy", "extra_field"])
def test_legacy_review_migration_never_accepts_changed_image_contract(change):
    current = review.implementation()
    saved = legacy_review(current)
    assert identity.compatible_review(saved, current)
    if change in {"input", "review", "tracker"}:
        name = {"input": "repellent_input.py", "review": "repellent_review.py", "tracker": "tiff_roi.py"}[change]
        current["code"][name] = "changed"
    elif change == "runtime":
        current["runtime"]["numpy"] = "different"
    elif change == "unknown_legacy":
        saved["code"]["repellent_input.py"] = "unknown older implementation"
    else:
        saved["unrecognized"] = True
    assert not identity.compatible_review(saved, current)


@pytest.mark.usefixtures("fake_encoder")
def test_legacy_approved_movie_and_measurements_reuse_without_rewriting(tmp_path, monkeypatch, request):
    path, _ = draft(tmp_path, gap=True)
    manager = review.ReviewManager(path, root=tmp_path / "reviews")
    manager.start()
    finish(manager)
    artifact = manager.artifact(7)
    record = json.loads(artifact.read_text())
    record["implementation"] = legacy_review(record["implementation"])
    inputs._write(artifact, record)
    manager.approve([7], inputs._hash(path))
    before = {p: p.read_bytes() for p in artifact.parent.iterdir() if p.is_file()}
    monkeypatch.setattr(inputs, "_extract", lambda *a: pytest.fail("legacy approved measurements must be reused"))
    monkeypatch.setattr(review, "generate", lambda *a: pytest.fail("legacy approved movie must be reused"))
    manifest = prepare(path)
    document = json.loads(manifest.read_text())
    assert len(document["samples"]) == 1
    restarted = review.ReviewManager(path, root=tmp_path / "reviews")
    restarted.start()
    finish(restarted)
    assert restarted.artifact(7) == artifact
    assert all(p.read_bytes() == content for p, content in before.items())
    assert len(request.getfixturevalue("fake_encoder")) == 1


def test_unreviewed_legacy_cache_reuses_exact_csv_with_original_provenance(tmp_path, monkeypatch):
    path, _ = settings_file(tmp_path)
    first = json.loads(prepare(path).read_text())["samples"][0]
    metadata = json.loads(Path(first["metadata"]).read_text())
    provenance = metadata["provenance"]
    provenance.pop("code_identity")
    provenance["code_sha256"]["repellent_input.py"] = identity.LEGACY_INPUT
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    csv = Path(first["centroids"]).with_name(key + ".csv")
    meta = csv.with_suffix(".json")
    Path(first["centroids"]).rename(csv)
    inputs._write(meta, metadata)
    Path(first["metadata"]).unlink()
    before = {p: p.read_bytes() for p in (csv, meta)}
    monkeypatch.setattr(inputs, "_extract", lambda *a: pytest.fail("audited legacy cache must be reused"))
    second = json.loads(prepare(path).read_text())["samples"][0]
    assert second["centroids"] == str(csv)
    assert all(p.read_bytes() == content for p, content in before.items())
    # A content mismatch cannot be laundered through the legacy migration.
    csv.write_text(csv.read_text() + "tampered")
    with pytest.raises(BaseException, match="audited legacy cache"):
        prepare(path)
