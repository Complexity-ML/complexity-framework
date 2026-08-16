#!/usr/bin/env python3
"""Generic entry point for the maintained 500M/32k TR-HASH SFT runner."""

if __package__:
    from scripts.sft_500m_32k_tr import main
else:
    from sft_500m_32k_tr import main


if __name__ == "__main__":
    main()
