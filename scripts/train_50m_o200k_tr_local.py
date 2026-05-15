#!/usr/bin/env python3
"""Compatibility entrypoint for the 50M o200k Token-Routed profile."""

from __future__ import annotations

import sys

from complexity.training.o200k_pretrain import main


if __name__ == "__main__":
    if "--profile" not in sys.argv:
        sys.argv[1:1] = ["--profile", "50m"]
    main()
