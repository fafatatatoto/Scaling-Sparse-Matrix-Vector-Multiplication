#!/usr/bin/env python3
"""Single-thread roofline for serial CSR SpMV and dense matvec."""

import csv
import sys
from pathlib import Path


def load_rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> int:
    if len(sys.argv) != 4:
        print(
            "usage: plot_roofline.py results/m1.csv results/machine.csv plots/roofline.png",
            file=sys.stderr,
        )
        return 2

    rows = load_rows(sys.argv[1])
    machine = load_rows(sys.argv[2])
    if not rows or not machine:
        print("csv has no rows", file=sys.stderr)
        return 1

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed", file=sys.stderr)
        return 1

    peak = float(machine[0]["peak_gflops"])
    bandwidth = float(machine[0]["stream_gbps"])
    if peak <= 0.0 or bandwidth <= 0.0:
        print("machine roofs must be positive", file=sys.stderr)
        return 1

    intensities = [10 ** (exponent / 20) for exponent in range(-40, 41)]
    roof = [min(peak, bandwidth * intensity) for intensity in intensities]

    fig, ax = plt.subplots(figsize=(8, 5.5))
    ax.loglog(intensities, roof, color="#1b1b1b", linewidth=1.5, label="single-thread roof")

    for row in rows:
        name = Path(row["matrix"]).stem
        gflops = float(row["gflops"])
        ai_low = float(row["ai_x_reloaded"])
        ai_high = float(row["ai_x_reused"])
        dense = row["symmetry"] == "dense"
        color = "#c44536" if dense else "#3d5a80"
        ax.plot([ai_low, ai_high], [gflops, gflops], color=color, linewidth=1.2, alpha=0.9)
        ax.scatter([ai_low, ai_high], [gflops, gflops], color=color, s=28, zorder=3)
        ax.annotate(name, (ai_high, gflops), textcoords="offset points", xytext=(4, 4), fontsize=8)

    ax.set_xlabel("Arithmetic intensity (FLOP/byte)")
    ax.set_ylabel("GFLOP/s")
    ax.set_title("Serial matvec on a single-thread roofline")
    ax.grid(True, which="both", linestyle=":", alpha=0.5)
    ax.legend(frameon=False)
    fig.tight_layout()

    out = Path(sys.argv[3])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
