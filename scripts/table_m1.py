#!/usr/bin/env python3
"""Write Table 1: matrix size, intensity, and serial performance."""

import csv
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: table_m1.py results/m1.csv results/table1.csv", file=sys.stderr)
        return 2

    with open(sys.argv[1], newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        print("csv has no rows", file=sys.stderr)
        return 1

    out_path = Path(sys.argv[2])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "matrix",
        "symmetry",
        "nrows",
        "nnz",
        "nnz_per_row",
        "gflops",
        "gbps_x_reused",
        "gbps_x_reloaded",
        "ai_x_reused",
        "ai_x_reloaded",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            nrows = float(row["nrows"])
            nnz = float(row["nnz"])
            writer.writerow(
                {
                    "matrix": Path(row["matrix"]).stem,
                    "symmetry": row["symmetry"],
                    "nrows": row["nrows"],
                    "nnz": row["nnz"],
                    "nnz_per_row": f"{(nnz / nrows) if nrows else 0.0:.6g}",
                    "gflops": row["gflops"],
                    "gbps_x_reused": row["gbps_x_reused"],
                    "gbps_x_reloaded": row["gbps_x_reloaded"],
                    "ai_x_reused": row["ai_x_reused"],
                    "ai_x_reloaded": row["ai_x_reloaded"],
                }
            )

    print(
        f"{'matrix':<22} {'n':>10} {'nnz':>12} {'nnz/row':>8} "
        f"{'GFLOP/s':>10} {'AI reused':>10} {'AI reload':>10}"
    )
    for row in rows:
        nrows = float(row["nrows"])
        nnz = float(row["nnz"])
        per_row = nnz / nrows if nrows else 0.0
        print(
            f"{Path(row['matrix']).stem:<22} {int(nrows):>10} {int(nnz):>12} {per_row:>8.2f} "
            f"{float(row['gflops']):>10.4f} {float(row['ai_x_reused']):>10.4f} "
            f"{float(row['ai_x_reloaded']):>10.4f}"
        )
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
