#!/usr/bin/env python3
"""Generic TR-MoE supervised fine-tuning entry point.

The implementation remains in the historical module for import compatibility,
but this neutral entry point makes clear that model size and tokenizer are read
from the supplied checkpoint rather than fixed to the original 100M/o200k
experiment.
"""

if __package__:
    from scripts.sft_100m_o200k_tr_local import main
else:
    from sft_100m_o200k_tr_local import main


if __name__ == "__main__":
    main()
