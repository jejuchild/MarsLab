#!/usr/bin/env python3
from __future__ import annotations

import os
import subprocess
from pathlib import Path


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def main() -> None:
    repo = Path(os.getenv("MARSLAB_COLAB_REPO", "/content/MarsLab"))
    if not repo.exists():
        run(["git", "clone", "https://github.com/csparkresearch/MarsLab.git", str(repo)])
    else:
        run(["git", "-C", str(repo), "pull"])

    os.environ["MARSLAB_ROOT"] = str(repo)
    print(f"MARSLAB_ROOT={repo}")
    print("Run this next:")
    print("python scripts/marslandform_v2/train_v3_tile_classifier.py --device cuda --epochs 100 --batch-size 512 --lr 1e-4 --patience 15")


if __name__ == "__main__":
    main()
