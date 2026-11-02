# SpMV Performance Study: Roofline, OpenMP, and MPI

Sparse matrix-vector multiplication (`y = Ax`, double precision), built step by step to answer the questions in lecture 12 with measurements from one machine. Each nonzero costs one multiply-add (2 flops) but requires loading a value and a column index, so the kernel is expected to be memory-bound, not compute-bound.

The project uses SpMV to cover three parts of the course:

- single-node locality and the roofline model (lectures 4, 5, 12)
- shared-memory parallelism with OpenMP (lectures 6, 12)
- distributed-memory partitioning and communication cost with MPI (lectures 9–12)

Goals and milestones are in [docs/goals.md](docs/goals.md). The kernel, byte model, and timing rules are fixed in [docs/methodology.md](docs/methodology.md); later milestones reuse the same definitions.

## Status

| Milestone | Question | State |
| --- | --- | --- |
| M1 serial CSR + roofline | Is SpMV memory-bound on this machine? | done |
| M2 OpenMP row-parallel | Why does adding cores stop helping? | next |
| M3 register blocking | Once bandwidth is saturated, does reading less data help? | planned |
| M4 MPI 1D row / 1D column / 2D | How much does each partitioning pay in communication? | planned |
| M5 graph partitioning | Does reordering reduce off-block nonzeros and communication? | planned |

## Build

Inside WSL:

```bash
cd ~/projects/spMV
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

Performance numbers come from Release only (`-O3 -march=native`, no `-ffast-math`, no LTO).

## Run

```bash
./build/spmv benchmarks/tiny.mtx --check
./build/spmv benchmarks/tiny.mtx --check --csv-out results/tiny.csv
./build/spmv --dense 4096
./build/spmv --machine
```

`--check` compares CSR against a triplet reference using an `x` that varies by column. Timed runs use `x = 1`.

Output columns that map to the milestone exit conditions:

| Column | Meaning |
| --- | --- |
| `matrix`, `nrows`, `nnz` | matrix, row count, CSR nonzero count |
| `median_s` | median time of one SpMV |
| `gflops` | `2 * nnz / time / 1e9` |
| `gbps_x_reused`, `gbps_x_reloaded` | estimated bandwidth under the two `x` traffic models |

`benchmarks/tiny.mtx` only validates the pipeline. Its GFLOP/s reflects call overhead and is not a performance result.

Symmetric matrices are expanded to full CSR. `stored_nnz` matches the nnz shown on the SuiteSparse website; `nnz` is the number of nonzeros the kernel actually processes.

## SuiteSparse

```bash
bash scripts/download_matrices.sh
./build/spmv data/west0479.mtx --check --csv-out results/spmv.csv
./build/spmv data/bcsstk17.mtx --csv-out results/spmv.csv
```

The list is in `scripts/matrices.txt`. The structures are deliberately varied: a small unsymmetric chemistry matrix, banded structural matrices, protein, FEM, circuit, economics, and graph. Downloaded `.mtx` files go to `data/` and are not tracked in git. If `sparse.tamu.edu` returns 403, the script falls back to the old UF mirror.

With matplotlib installed, the full M1 set of figures and tables:

```bash
bash scripts/run_m1.sh
```

This downloads the matrices in `scripts/matrices.txt`, measures the single-thread FMA peak and STREAM triad, runs serial CSR and `--dense 4096`, and writes:

```text
results/m1.csv        one row per matrix
results/machine.csv   peak GFLOP/s and STREAM GB/s
results/table1.csv    n, nnz, nnz/row, GFLOP/s, both arithmetic intensities
plots/gflops.png      Figure 1
plots/roofline.png    Figure 2
```

## M1 results (i5-13420H, single thread)

Measured roofs: 39.05 GFLOP/s peak (AVX2 FMA), 10.37 GB/s STREAM triad, ridge point ≈ 3.8 FLOP/byte. All matrices have arithmetic intensity between 0.10 and 0.25 FLOP/byte and run at 0.9–2.1 GFLOP/s, on the memory slope of the roofline. `scircuit` falls below the STREAM line because of irregular access to `x`.

## Layout

```text
include/     headers for CSR, loader, timing
src/         serial kernel, Matrix Market loader, byte model, benchmark
benchmarks/  small matrices for tests
tests/       correctness
scripts/     matrix download, runs, plots
docs/        goals and frozen methodology
data/        SuiteSparse matrices (after download)
results/     CSV
plots/       figures
```
