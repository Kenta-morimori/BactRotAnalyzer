"""Extract one selected TIFF bead with an automatic fixed ROI, CSV and movie."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.functions.centroid_diagnostics import private_output_root
from utils.functions.tiff_roi_io import directory_rows, process


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Directory of original single-plane TIFFs with suffix/time tags")
    source.add_argument("--manifest", type=Path, help="Batch samples with explicit original frame/time rows and seeds")
    parser.add_argument("--output", type=Path, required=True, help="New private directory under outputs/")
    parser.add_argument("--seed", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--roi", nargs=4, type=int, metavar=("X", "Y", "WIDTH", "HEIGHT"))
    parser.add_argument("--threshold", type=float, default=2000)
    parser.add_argument(
        "--interactive", action="store_true", help="Click target and confirm/edit ROI; reselect at recording gaps"
    )
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=int, default=50)
    args = parser.parse_args()
    output = private_output_root(Path(__file__).resolve().parents[1], args.output)
    if args.manifest:
        if args.seed is not None or args.roi is not None:
            parser.error("Batch seeds and ROIs belong in the manifest")
        manifest = json.loads(args.manifest.read_text())
        for sample in manifest["samples"]:
            for row in sample["frames"]:
                row["path"] = str((args.manifest.parent / row["path"]).resolve())
        interactive = args.interactive
    else:
        rows = directory_rows(args.input)
        manifest = dict(
            samples=[
                dict(
                    input_id=args.input.name,
                    frames=rows,
                    threshold=args.threshold,
                    roi=args.roi,
                    seeds={str(rows[0]["source_frame_1based"]): args.seed} if args.seed is not None else {},
                )
            ]
        )
        interactive = args.interactive or args.seed is None
    process(manifest, output, interactive=interactive, video=not args.no_video, video_fps=args.video_fps)
    print(output.resolve())


if __name__ == "__main__":
    main()
