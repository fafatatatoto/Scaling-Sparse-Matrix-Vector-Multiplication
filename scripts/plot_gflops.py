#!/usr/bin/env python3
"""Bar chart of serial CSR GFLOP/s from a spmv CSV."""

import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: plot_gflops.py results/spmv.csv plots/gflops.png", file=sys.stderr)
        return 2

    with open(sys.argv[1], newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("csv has no rows", file=sys.stderr)
        return 1

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed", file=sys.stderr)
        return 1

    names = [Path(row["matrix"]).name for row in rows]
    gflops = [float(row["gflops"]) for row in rows]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(names)), gflops, color="#3d5a80")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Serial matvec")
    fig.tight_layout()

    out = Path(sys.argv[2])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
