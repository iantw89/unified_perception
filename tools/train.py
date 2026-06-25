#!/usr/bin/env python
"""Train the unified-perception model."""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from unified_perception.config import load_config  # noqa: E402
from unified_perception.engine import train  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="optional YAML config")
    ap.add_argument("overrides", nargs="*", help="key.path=value config overrides")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    train(cfg)


if __name__ == "__main__":
    main()
