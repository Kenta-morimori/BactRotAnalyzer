"""Versioned image-measurement identities, independent of background analysis.

The one legacy identity is audited against the scoped AST of the same code.
Legacy evidence is accepted read-only only while every image-producing scope,
other dependency, runtime, input, condition and artifact still matches.
"""

import ast
import hashlib
import json
from pathlib import Path

# Audited image implementation at the first batch-review contract (ADR 0008).
LEGACY_INPUT = "8b505ca226831a716a7bc9ee04b032da34516ff42ff4554c97fc2d3501b45c33"
LEGACY_REVIEW = "d4e193d80c843e719ea7569f0398d6ad7064f1f96fc7b189abf3f3c106f43737"
LEGACY_SCOPES = {
    "repellent_input.py": "9dcb21a3d6a1f76e759f4cdd49bd54e4d05c84895aac33064a0dc8b7291e63f6",
    "repellent_review.py": "773c1fd729a15fc4e2cf9d588fedc4781533ec2ee6d523127bf7812ad6fb5015",
}
EXCLUDED = {
    "repellent_input.py": {"prepare_measurements", "_background_series", "_background_rise"},
    "repellent_review.py": {"implementation", "verify_artifact"},
}


def image_scope(path):
    """Hash module semantics except background and cache orchestration functions.

    All image reading, observation mapping, extraction, calibration, scoping and
    review generation/rendering remain in scope. New top-level definitions and
    imports participate too. AST identity ignores formatting and comments.
    """
    path = Path(path)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if getattr(node, "name", None) not in EXCLUDED[path.name]]
    if path.name == "repellent_review.py":
        tree.body = [
            node for node in tree.body if not (isinstance(node, ast.ImportFrom) and node.module == "importlib.metadata")
        ]
    return hashlib.sha256(ast.dump(tree, include_attributes=False).encode()).hexdigest()


def measurement_code(paths):
    from utils.functions.repellent_input import _hash

    return {p.name: image_scope(p) if p.name == "repellent_input.py" else _hash(p) for p in paths}


def review_implementation():
    from importlib.metadata import version

    from utils.functions.repellent_input import _hash
    from utils.functions.repellent_review import METHOD

    folder = Path(__file__).parent
    return dict(
        method=METHOD,
        identity="image-scope-ast-v1",
        code={
            name: image_scope(folder / name) if name in EXCLUDED else _hash(folder / name)
            for name in ("repellent_review.py", "repellent_input.py", "tiff_roi.py", "tiff_roi_io.py")
        },
        runtime={name: version(name) for name in ("numpy", "opencv-python", "pandas", "Pillow", "tifffile")},
    )


def compatible_review(saved, current):
    if saved == current:
        return True
    if not isinstance(saved, dict):
        return False
    if any(current["code"].get(name) != digest for name, digest in LEGACY_SCOPES.items()):
        return False
    legacy = dict(current)
    legacy.pop("identity")
    legacy["code"] = dict(current["code"], **{"repellent_input.py": LEGACY_INPUT, "repellent_review.py": LEGACY_REVIEW})
    return saved == legacy


def measurement_cache(output_root, method, provenance, metadata_fields, *, force=False):
    """Find a fully verified prior cache under the audited whole-file identity.

    Do not rewrite evidence, forge a current full-file hash, or relax content
    verification. The caller checks all metadata and CSV bytes again on reuse.
    """
    from utils.functions.repellent_input import _hash

    def location(record):
        key = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()
        return Path(output_root) / "raw_centroid" / method / key

    cache = location(provenance)
    if force or provenance["code_sha256"].get("repellent_input.py") != LEGACY_SCOPES["repellent_input.py"]:
        return cache, provenance
    legacy = dict(provenance, code_sha256=dict(provenance["code_sha256"], **{"repellent_input.py": LEGACY_INPUT}))
    legacy.pop("code_identity", None)
    candidate = location(legacy)
    csv, metadata = candidate.with_suffix(".csv"), candidate.with_suffix(".json")
    try:
        saved = json.loads(metadata.read_text())
        expected = dict(metadata_fields, provenance=legacy)
        if all(saved.get(k) == v for k, v in expected.items()) and saved.get("csv_sha256") == _hash(csv):
            return candidate, legacy
    except (ValueError, OSError, AttributeError):
        pass
    return cache, provenance
