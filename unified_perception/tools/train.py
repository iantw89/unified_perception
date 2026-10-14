#!/usr/bin/env python
"""Train the unified-perception model."""

from __future__ import annotations

import argparse
import os
import sys

# Register FFmpeg DLLs before any torchcodec import.
# add_dll_directory covers the main process; PATH is inherited by spawned
# worker sub-processes (which import torchcodec before worker_init_fn runs).
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _ffmpeg_bin = os.environ.get("FFMPEG_SHARED_BIN", "")
    if _ffmpeg_bin and os.path.isdir(_ffmpeg_bin):
        os.add_dll_directory(_ffmpeg_bin)
        os.environ["PATH"] = _ffmpeg_bin + os.pathsep + os.environ.get("PATH", "")

from unified_perception.config import load_config  # noqa: E402
from unified_perception.engine import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument(
        "--resume",
        nargs="?",
        const="auto",
        default=None,
        metavar="CHECKPOINT",
        help=(
            "resume from a checkpoint. "
            "Omit the path to auto-resume from last.pt in the output directory."
        ),
    )
    ap.add_argument("overrides", nargs="*", help="key.path=value config overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)

    if args.resume is not None:
        if args.resume == "auto":
            # Resolve lazily: output_dir is already set from config/overrides.
            cfg.train.resume = os.path.join(cfg.train.output_dir, "last.pt")
        else:
            cfg.train.resume = args.resume

    train(cfg)


if __name__ == "__main__":
    main()
