# Locality-Aware Parallel SpMV

研究問題：SpMV 裡 memory locality 和 parallelism 怎麼互相影響，以及 data layout / reordering 在 CPU 和 GPU 上什麼時候真的有用。

這個目錄現在有 serial CSR benchmark，以及第一步的 dense matvec 和 single-thread roofline。kernel、byte model、計時方式寫在 [docs/methodology.md](docs/methodology.md)。OpenMP、reordering、SELL、GPU 會沿用同一組定義。

## Build

在 WSL 裡：

```bash
cd ~/projects/spMV
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

效能數字只用 Release（`-O3 -march=native`，沒有 `-ffast-math`，沒有 LTO）。

## Run

```bash
./build/spmv benchmarks/tiny.mtx --check
./build/spmv benchmarks/tiny.mtx --check --csv-out results/tiny.csv
```

`--check` 用一組會隨 column 變化的 `x`，對 CSR 和 triplet reference。計時本身的 `x` 是全 1。

輸出裡和 milestone exit condition 對應的欄位：

| 欄位 | 意義 |
| --- | --- |
| `matrix`, `nrows`, `nnz` | 矩陣、列數、CSR 非零元數 |
| `median_s` | 一次 SpMV 的 median 時間 |
| `gflops` | `2 * nnz / time / 1e9` |
| `gbps_x_reused`, `gbps_x_reloaded` | 兩種 `x` 流量模型下的估計頻寬 |

`benchmarks/tiny.mtx` 只驗證管線。它的 GFLOP/s 是函式呼叫的開銷，不能拿來當性能結論。

對稱矩陣會展開成完整 CSR。`stored_nnz` 對得上 SuiteSparse 網頁上的 nnz；`nnz` 是 kernel 真正做的非零元數。

## SuiteSparse

```bash
bash scripts/download_matrices.sh
./build/spmv data/west0479.mtx --check --csv-out results/spmv.csv
./build/spmv data/bcsstk17.mtx --csv-out results/spmv.csv
```

清單在 `scripts/matrices.txt`。結構故意岔開：小的非對稱化學矩陣、帶狀結構矩陣、蛋白質、FEM、電路、經濟模型、圖。下載的 `.mtx` 在 `data/`，不進 git。

有 matplotlib 的話，第一步整組圖表：

```bash
bash scripts/run_m1.sh
```

這會下載 `scripts/matrices.txt` 裡的矩陣，量 single-thread 的 FMA peak 和 STREAM triad，跑 serial CSR 和 `--dense 4096`，並寫出：

```text
results/m1.csv        每個矩陣一列
results/machine.csv   peak GFLOP/s 和 STREAM GB/s
results/table1.csv    n、nnz、nnz/row、GFLOP/s、兩種 arithmetic intensity
plots/gflops.png      Figure 1
plots/roofline.png    Figure 2
```

## Layout

```text
include/     CSR、loader、計時的 header
src/         serial kernel、Matrix Market、byte model、benchmark
benchmarks/  給測試用的小矩陣
tests/       正確性
scripts/     下載矩陣、畫 GFLOP/s
docs/        凍結的方法論
data/        SuiteSparse（下載後才有）
results/     CSV
plots/       圖
```

## 目標

整個 project 對應 CSE 6230 lecture 12。做到哪一步、每一步要交出什麼，寫在 [docs/goals.md](docs/goals.md)。下一步是 OpenMP 列並行。
