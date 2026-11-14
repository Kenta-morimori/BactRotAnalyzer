"""Declare CLI input scopes without importing the numerical analysis stack."""

import json
from pathlib import Path

from utils.run_context import REPOSITORY


def default_image_settings(day, data_root=None):
    relative = Path(day)
    if relative.is_absolute() or ".." in relative.parts or relative.parts[:1] == ("outputs",):
        raise ValueError("day must be a data identifier without parent traversal")
    return Path(data_root or Path.cwd() / "data") / relative / "repellent_input.json"


def resolve_image_settings(day, path=None, source=None):
    """An explicit descriptor wins; otherwise use the dataset's single active settings file."""
    candidate = Path(path) if path is not None else default_image_settings(day)
    if path is None and not candidate.is_file():
        if source is not None:
            raise ValueError("create dataset settings with prepare_repellent_input.py --init first")
        return None, None
    document = json.loads(candidate.read_text(encoding="utf-8"))
    selected = document.get("source")
    if selected not in {"tiff", "avi"} or document.get("day") != day:
        raise ValueError("input settings must identify the requested day and TIFF/AVI source")
    if source is not None and source != selected:
        raise ValueError("CLI source differs from settings source")
    return str(candidate.resolve()), selected


def dataset_inputs(day, repository=REPOSITORY):
    """Raw dataset plus existing shared caches that may supply observations."""
    root = Path(repository)
    yield root / "data" / day
    cache = root / "outputs" / day
    for name in (
        "raw_centroid",
        "time_list.csv",
        "centroid_coordinate.csv",
        "centroid_coordinate_sample_map.csv",
        "centroid_coordinate_source.json",
        "centroid_coordinate/rotation_axes.csv",
    ):
        yield cache / name


def manifest_inputs(path, kind):
    """Include files referenced by supported portable input descriptors."""
    # The measurement loader resolves the descriptor first; the result exporter
    # opens relative tables beside the supplied path, including a file symlink.
    path = Path(path).resolve() if kind == "measurement" else Path(path)
    yield path
    document = json.loads(path.read_text(encoding="utf-8"))
    if kind == "measurement":
        for sample in document["samples"]:
            for key in ("centroids", "metadata", "acquisition_tiff", "background"):
                if key in sample:
                    yield path.parent / sample[key]
        for name in document.get("image_input", {}).get("input_hashes", {}):
            yield path.parent / name
    elif kind == "results":
        for key in ("observations", "supports", "fluctuation", "spectrum", "candidates", "frequency", "fft_features"):
            if key in document:
                yield path.parent / document[key]
    else:
        raise ValueError(f"Unsupported input descriptor: {kind}")


def approved_image_numbers(document, numbers=None):
    """A persisted batch approval owns the default analysis scope, never the editor scope."""
    review = document.get("review")
    if review is None:
        return numbers
    accepted = review.get("accepted_samples")
    ids = [s["sample_no"] for s in document["samples"]]
    if (
        not isinstance(accepted, list)
        or not accepted
        or any(type(n) is not int for n in accepted)
        or len(set(accepted)) != len(accepted)
        or not set(accepted) <= set(ids)
    ):
        raise ValueError("invalid or empty batch approval; review and approve the preprocessing first")
    if numbers is not None and (not numbers or not set(numbers) <= set(accepted)):
        raise ValueError("未承認・保留のNo.が含まれます。HTMLで再生成・承認してください")
    return accepted if numbers is None else numbers


def repellent_inputs(args):
    settings_path = getattr(args, "input_settings", None)
    selected_source = getattr(args, "input_source", None)
    if not getattr(args, "measurement_input", None) and not getattr(args, "results_bundle", None):
        settings_path, selected_source = resolve_image_settings(args.day, settings_path, selected_source)
    if settings_path:
        path = Path(settings_path).resolve()
        yield path
        document = json.loads(path.read_text(encoding="utf-8"))
        numbers = approved_image_numbers(document, getattr(args, "input_samples", None))
        for sample in document["samples"]:
            if numbers is not None and sample["sample_no"] not in numbers:
                continue
            yield path.parent / sample["tiff_directory"]
            if "avi" in sample:
                yield path.parent / sample["avi"]
            if sample.get("background") is not None:
                yield path.parent / sample["background"]["csv"]
            if sample.get("review_artifact"):
                artifact = (path.parent / sample["review_artifact"]).resolve()
                yield artifact
                saved = json.loads(artifact.read_text())
                for name in saved.get("artifact_hashes", {}):
                    yield artifact.parent / name
        suffix = getattr(args, "output_suffix", None)
        if suffix is None:
            suffix = f"reanalysis_{selected_source}"
        yield Path.cwd() / "outputs" / suffix / args.day / "raw_centroid"
    elif args.measurement_input:
        yield from manifest_inputs(args.measurement_input, "measurement")
    elif args.results_bundle:
        yield from manifest_inputs(args.results_bundle, "results")
    else:
        yield from dataset_inputs(args.day, Path.cwd())


def tiff_inputs(args):
    """Standalone TIFF extraction uses descriptor-relative source image paths."""
    if args.manifest:
        path = Path(args.manifest)
        yield path
        for sample in json.loads(path.read_text(encoding="utf-8"))["samples"]:
            for row in sample["frames"]:
                yield path.parent / row["path"]
    else:
        yield args.input


def diagnostic_inputs(args, repository=REPOSITORY):
    """Declare selected diagnostic inputs and external TIFF timeline references."""
    for name in ("config", "dataset", "freeze", "annotations", "comparison", "prior_review", "raw_centroids"):
        value = getattr(args, name, None)
        if value is not None:
            yield Path(value)
    for value in getattr(args, "evidence", None) or []:
        yield Path(value)
    command = getattr(args, "command", None)
    if command == "regression":
        yield from dataset_inputs("test_data", repository)
    dataset = getattr(args, "dataset", None)
    if dataset is not None:
        try:
            day = Path(dataset).resolve().relative_to(Path(repository) / "data")
        except ValueError:
            pass
        else:
            yield from dataset_inputs(day, repository)
    for name in ("manifest_root", "diagnostics"):
        value = getattr(args, name, None)
        if value is None:
            continue
        root = Path(value)
        yield root
        for timeline in sorted(root.glob("*/timeline.json")):
            for row in json.loads(timeline.read_text(encoding="utf-8")):
                # Diagnostic timelines store paths interpreted relative to cwd.
                yield Path(row["tiff_path"])
