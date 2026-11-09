"""TASK 2. Out-of-fold RMSE from the four fold validation files.

    python scripts/oof.py models/<name>

Each fold run (`python -m src.train --fold k ...`) writes a `<name>_val.csv`
with one row per validation photograph: image_id, usability, eval_ok, family,
pred. This script pools the four files, keeps the scoreable rows
(eval_ok == 1) and prints the RMSE of each file and the pooled out-of-fold
RMSE -- the one number used to compare experiments.

The pooled RMSE is computed over all rows at once, not as the mean of the four
fold RMSEs: the folds differ in size.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from src.data import TARGET
from src.evaluate import rmse


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: python scripts/oof.py models/<name>")
    folder = Path(sys.argv[1])

    val_files = sorted(folder.glob("*_val.csv"))
    if len(val_files) != 4:
        raise SystemExit(f"expected four *_val.csv files in {folder}, found {len(val_files)}")

    frames = []
    for f in val_files:
        df = pd.read_csv(f)
        ok = df[df["eval_ok"] == 1]
        print(f"{f.name}: RMSE {rmse(ok[TARGET], ok['pred']):.4f} on {len(ok)}")
        frames.append(df)

    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled[pooled["eval_ok"] == 1]
    print(f"out-of-fold RMSE {rmse(pooled[TARGET], pooled['pred']):.4f} on {len(pooled)} photographs")


if __name__ == "__main__":
    main()
