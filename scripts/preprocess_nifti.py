#!/usr/bin/env python3
"""Convert a NIfTI tumour mask to a compact voxel/adjacency CSV representation."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mask")
    parser.add_argument("--label", type=int, default=2)
    parser.add_argument("--out", default="data/clinical_complex")
    args = parser.parse_args()
    try:
        import nibabel as nib
    except ImportError as exc:
        raise SystemExit("Install nibabel: pip install nibabel") from exc
    img = nib.load(args.mask)
    arr = np.asarray(img.dataobj)
    mask = arr == args.label
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    vox = np.argwhere(mask)
    ids = {tuple(v): i for i, v in enumerate(vox)}
    pd.DataFrame(vox, columns=["i", "j", "k"]).assign(cell_id=np.arange(len(vox))).to_csv(out / "cells.csv", index=False)
    rows = []
    for a, v in enumerate(vox):
        for axis in range(3):
            w = v.copy(); w[axis] += 1
            b = ids.get(tuple(w))
            if b is not None:
                rows.append((len(rows), a, b, axis))
    pd.DataFrame(rows, columns=["edge_id", "tail", "head", "axis"]).to_csv(out / "interfaces.csv", index=False)
    print(f"wrote {len(vox)} cells and {len(rows)} interfaces to {out}")


if __name__ == "__main__":
    main()
