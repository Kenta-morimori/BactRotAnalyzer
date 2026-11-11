"""Create/confirm local TIFF or AVI input settings; never run a full analysis."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.functions.repellent_input import (
    _write,
    attach_brightness_settings,
    configure_settings,
    draft_settings,
)
from utils.run_inputs import default_image_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--init", action="store_true")
    action.add_argument("--configure", action="store_true")
    action.add_argument("--attach-brightness", action="store_true", help="Register same-name brightness_data Mean CSVs")
    parser.add_argument("--settings", type=Path, help="Default: data/<day>/repellent_input.json")
    parser.add_argument("--day")
    parser.add_argument("--source", choices=["tiff", "avi"])
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--px2um", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--samples", nargs="+", type=int)
    parser.add_argument("--confirm-order", action="store_true", help="Confirm the reviewed sample-number mapping")
    parser.add_argument(
        "--brightness-root", type=Path, help="Override the recording's sibling brightness_data directory"
    )
    args = parser.parse_args()
    if args.settings is None:
        if not args.day:
            parser.error("provide --day to use the default settings location")
        args.settings = default_image_settings(args.day, args.data_root)
    if not args.init and args.day:
        if json.loads(args.settings.read_text())["day"] != args.day:
            parser.error("settings day differs from --day")
    if args.brightness_root is not None and not args.attach_brightness:
        parser.error("--brightness-root requires --attach-brightness")
    if args.init:
        if not args.day or not args.source:
            parser.error("--init requires --day and --source")
        if args.samples or args.confirm_order:
            parser.error("review the complete draft before choosing samples or confirming its order")
        if args.settings.exists():
            parser.error("settings already exist; edit/configure them without replacing prior choices")
        settings = draft_settings(args.day, args.source, args.data_root, px2um=args.px2um)
        _write(args.settings, settings)
    elif args.configure:
        if args.source or args.px2um:
            parser.error("--configure uses the source, day and calibration saved in settings")
        settings = configure_settings(args.settings, numbers=args.samples, confirm_order=args.confirm_order)
    else:
        if args.source or args.px2um or args.confirm_order:
            parser.error("--attach-brightness preserves the existing source, order and calibration")
        settings = attach_brightness_settings(args.settings, numbers=args.samples, brightness_root=args.brightness_root)
    for sample in settings["samples"]:
        print(f"No.{sample['sample_no']}: {sample['input_id']}")
    print(json.dumps(dict(settings=str(args.settings.resolve()), order_confirmed=settings["order_confirmed"])))


if __name__ == "__main__":
    main()
