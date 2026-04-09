#!/usr/bin/env python3
"""Backward-compatible wrapper — batch processing is now in run_batch.py.

run_batch.py already processes all frames per sol by default.
This script exists only so old invocations still work.
"""

from run_batch import main

if __name__ == "__main__":
    main()
